// the attention core (osmos/gpu/attn_core.h): a desc checked whole at create, a step launched on its family's kernels -
// Inkling's (attn_core_kernel.h: F32, f16, Q8 records, Q8 keys with Q5 values), a heads block's (Qwen's, and apart
// osmos1.7b's and q35's: attn_core_heads.h, attn_core_q35.h), Hy4's MLA (attn_core_mla.h), DeepSeek's latent
// (attn_core_latent.h) and the drafters' blocks (attn_core_draft.h) - each moved whole, its bits (docs/attention-core.md
// C1 - C3); one order and one precision across them next, a family at a time, by KL
#include <hip/hip_runtime.h>

#include <cstdlib>

#include "attn_core_draft.h"
#include "attn_core_heads.h"
#include "attn_core_latent.h"
#include "attn_core_mla.h"
#include "attn_core_q35.h"
#include "attn_core_kernel.h"

using namespace attn;

struct AttnCore {
  AttnCoreDesc d;
  float* part;  // the borrowed scratch: a split decode's units' partials
};

namespace {

bool refuse(char* err, const char* what) { return osmos_fail(err, "the attention core: %s", what); }

// the keys' and values' type in memory for the desc's rows (the kernels' KV); false: a pairing not made
enum class Kv { F32, F16, KV8, KV85, NONE };
Kv kv_of(const AttnCoreDesc& d) {
  if (d.k_rows == ATTN_ROWS_F32 && d.v_rows == ATTN_ROWS_F32) return Kv::F32;
  if (d.k_rows == ATTN_ROWS_F16 && d.v_rows == ATTN_ROWS_F16) return Kv::F16;
  if (d.k_rows == ATTN_ROWS_KV8 && d.v_rows == ATTN_ROWS_KV8) return Kv::KV8;
  if (d.k_rows == ATTN_ROWS_KV8 && d.v_rows == ATTN_ROWS_KV5) return Kv::KV85;
  return Kv::NONE;
}

template <class KV>
bool launch(const AttnCore& c, const AttnStep& s, const KV* k, const KV* v, hipStream_t stream) {
  const AttnCoreDesc& d = c.d;
  const AttnArgs a = {d.heads, d.kv_heads, d.d_rel, d.extent, d.window, d.scaling};
  const uint32_t n_tok = s.n_tok, n_keys = s.keys.n_keys, pos0 = s.keys.key0;
  const float *q = s.q, *rel = s.rel, *sink = s.sink, *gate = s.gate, *rel_proj = s.rel_proj;
  float *out = s.out, *scratch = c.part;
  const uint32_t G = d.heads / d.kv_heads;
  if (n_tok <= (uint32_t) kFaTile) {  // a decode's: a block a unit and KV head, then the fold
    const uint32_t qlo = n_keys - n_tok, qhi = n_keys - 1;
    const uint32_t klo = d.window && qlo + 1 > d.window ? (qlo + 1 - d.window) / kFaTile * kFaTile : 0;
    const uint32_t u0 = (pos0 + klo) / kFaUnit, units = (pos0 + qhi) / kFaUnit - u0 + 1;
    const dim3 grid(1, d.kv_heads, units);
    const uint32_t rows = s.x16 ? (n_tok + mm::kTile - 1) / mm::kTile * mm::kTile : n_tok;  // x16's: a whole tile
    // the waves the (token, head) rows fill (n_tok G of them, 16 a wave), then more while the call's blocks leave
    // the GPU idle: a decode's blocks are latency-bound (waiting on memory at one wave), and a block's waves past
    // its rows only load its tiles - each row's sums its own, so the same bits at any W
    const uint32_t need = (n_tok * G + kFaTile - 1) / kFaTile, blocks = d.kv_heads * units;
    uint32_t W = need <= 1 ? 1 : need <= 2 ? 2 : need <= 4 ? 4 : 8;
    while (W < 8 && blocks * W * 2 <= kFaDecodeWaves) W *= 2;
#define FSW(g, w)                                                                                                 \
  if (G == g && W == w) {                                                                                         \
    fa_kernel<g, 1, KV, true, w><<<grid, w * 32, 0, stream>>>(a, q, rel, sink, gate, n_tok, k, v, n_keys, pos0,   \
                                                               rel_proj, out, u0, scratch);                       \
    fa_fold_kernel<<<rows * d.heads, kFaHead, 0, stream>>>(scratch, units, n_tok, d.heads, sink, gate, out,       \
                                                           (_Float16*) s.x16, s.overflow);                        \
    return hipGetLastError() == hipSuccess;                                                                       \
  }
#define FS(g) FSW(g, 1) FSW(g, 2) FSW(g, 4) FSW(g, 8)
    FS(1) FS(2) FS(3) FS(4) FS(6) FS(8)
#undef FS
#undef FSW
    return false;
  }
  // a block's waves G R: Mini's 3 heads a KV head 8 row tiles (24 waves, Small's 6 x 4) - every key tile decoded once
  // for twice the rows; the tiles a query's keys are cut into are the same in any block, so R changes no bit
  const uint32_t R = G == 3 ? 8u : kFaRowTiles;
  const dim3 grid_fa((n_tok + kFaTile * R - 1) / (kFaTile * R), d.kv_heads);
#define FA(g, r)                                                                                                 \
  if (G == g && R == r) {                                                                                        \
    fa_kernel<g, r, KV, false><<<grid_fa, g * r * 32, 0, stream>>>(a, q, rel, sink, gate, n_tok, k, v, n_keys,    \
                                                                   pos0, rel_proj, out, 0, nullptr);             \
    return hipGetLastError() == hipSuccess;                                                                      \
  }
  FA(1, 1) FA(2, 1) FA(3, 1) FA(4, 1) FA(6, 1) FA(8, 1) FA(1, 2) FA(2, 2) FA(3, 2) FA(4, 2) FA(6, 2) FA(8, 2) FA(1, 4)
  FA(2, 4) FA(3, 4) FA(4, 4) FA(6, 4) FA(3, 8)
#undef FA
  return false;
}

// MLA's rows (Hy4's: one latent row for every head, keys the row, values its latent)
bool is_mla(const AttnCoreDesc& d) { return d.k_rows == ATTN_ROWS_MLA_F32 || d.k_rows == ATTN_ROWS_MLA_Q8; }

// an MLA desc (hy4_mla_attn's bounds): its refusal, or nullptr
const char* mla_bad(const AttnCoreDesc& d) {
  const bool q8 = d.k_rows == ATTN_ROWS_MLA_Q8;
  const uint32_t rope = d.dk > d.dv ? d.dk - d.dv : 0;
  return d.v_rows != d.k_rows ? "v_rows: MLA's the keys' rows (its latent)"
         : !d.heads || d.kv_heads != 1 ? "kv_heads: MLA's one latent row for every head (1)"
         : !d.dv || d.dv > HY4_MLA_MAX_LATENT || d.dv % (q8 ? 32 : 4) ? "dv: the latent 4 .. HY4_MLA_MAX_LATENT (Q8: whole blocks)"
         : !rope || rope > mla::kMaxRope || rope % 4 ? "dk: the latent and a rope part of 4 .. 128"
         : q8 && mla::mla_row_bytes(d.dv, rope, true) % 16 ? "dk: a Q8 row not whole 16 bytes"
         : d.term != ATTN_TERM_NONE ? "term: MLA's none"
         : d.sink != ATTN_SINK_HEAD ? "sink: MLA's a head's logit"
         : d.gate != ATTN_GATE_NONE ? "gate: MLA's none (hy4_mla_out's, after)"
         : d.unit != attn_hy4::kSplitKeys ? "unit: MLA's 32 (its splits: its bits)"
         : !d.apart ? "apart: MLA's prompt and decode read apart"
         : d.window > HY4_MLA_MAX_KEYS ? "window: past HY4_MLA_MAX_KEYS"
         : d.slots && d.slots < d.window ? "slots: a ring short of its window"
         : d.ratio && d.window ? "ratio: groups under no window"
         : d.max_keys > HY4_MLA_MAX_KEYS ? "max_keys: past HY4_MLA_MAX_KEYS"
                                          : nullptr;
}

// an MLA step: hy4_mla_attn's, as its callers had it - a read by mla_mm, else the splits
bool mla_launch(const AttnCore& c, const AttnStep& s, hipStream_t stream) {
  const AttnCoreDesc& d = c.d;
  const AttnKeys& k = s.keys;
  if (!s.pos || !s.sink || k.k != k.v || k.lists || (k.sel != nullptr) != (d.ratio != 0) ||
      (k.sel && (!k.npick || !k.stride || (uint64_t) k.stride * d.ratio > d.max_keys)) || s.x16 || !s.n_tok ||
      s.n_tok > d.max_tokens)
    return false;
  return attn_hy4::run(s.q, k.k, d.slots, d.k_rows == ATTN_ROWS_MLA_Q8, s.sink, d.heads, d.dv, d.dk - d.dv, d.window,
                       k.sel, k.npick, k.stride, d.ratio, d.scaling, s.pos, s.n_tok, s.out, c.part, s.read, stream);
}

// a latent's rows (DeepSeek's: one row a key, its value the same row, for every head)
bool is_latent(const AttnCoreDesc& d) { return d.k_rows == ATTN_ROWS_FP8; }

// a latent desc (ops_sparse_attn's bounds): its refusal, or nullptr
const char* latent_bad(const AttnCoreDesc& d) {
  return d.kv_heads != 1 ? "kv_heads: a latent's one row for every head (1)"
         : !d.heads || d.heads % attn_ds4::kHeads ? "heads: whole blocks of 64"
         : d.dv != d.dk ? "dv: a latent's value its key's row"
         : !d.dk || d.dk % attn_ds4::kVdims || d.dk % attn_ds4::kStep ? "dk: whole blocks of 256"
         : d.v_rows != ATTN_ROWS_FP8 ? "v_rows: a latent's its keys' (FP8)"
         : !d.window || d.window > OPS_MAX_WINDOW ? "window: 1 .. OPS_MAX_WINDOW"
         : d.slots < d.window ? "slots: a ring short of its window"
         : d.max_sel > OPS_MAX_SELECT ? "max_sel: past OPS_MAX_SELECT"
         : d.s_rows != ATTN_ROWS_FP4 && d.s_rows != ATTN_ROWS_F32 ? "s_rows: a latent's selected rows FP4 or F32"
         : d.term != ATTN_TERM_NONE ? "term: a latent's none"
         : d.sink != ATTN_SINK_HEAD ? "sink: a latent's a head's logit"
         : d.gate != ATTN_GATE_NONE ? "gate: a latent's none"
         : d.unit ? "unit: a latent's 0 (one softmax over a query's keys: its bits)"
         : d.apart ? "apart: a latent's one kernel for its prompt and its decode"
         : d.ratio ? "ratio: a latent's selection its rows', not blocks"
         : d.max_keys < d.window + d.max_sel ? "max_keys: short of the window and its selected rows"
                                             : nullptr;
}

// a latent step: ops_sparse_attn's, as its callers had it (the rows unpacked a step's choice: the same bits)
bool latent_launch(const AttnCore& c, const AttnStep& s, hipStream_t stream) {
  const AttnCoreDesc& d = c.d;
  const AttnKeys& k = s.keys;
  if (!s.pos || !s.sink || k.k != k.v || k.lists || !s.n_tok || s.n_tok > d.max_tokens || k.floor < 0 || s.x16 ||
      (k.rows && (!k.sel || !k.npick || k.stride < d.max_sel || !d.max_sel)) || (!k.rows && k.sel))
    return false;
  const uint32_t rfmt = k.k_wide ? KV_F32 : KV_FP8;
  const uint32_t fmt = k.rows_wide || d.s_rows == ATTN_ROWS_F32 ? KV_F32 : KV_FP4_E4M3;
  return attn_ds4::run(s.q, k.k, rfmt, d.slots, s.pos, d.window, k.floor, k.rows, fmt, k.rows ? k.sel : nullptr,
                       k.rows ? k.npick : nullptr, k.rows ? k.stride : 0, k.rows ? d.max_sel : 0, s.sink, d.scaling,
                       d.heads, d.dk, s.n_tok, c.part, s.out, stream);
}

// a drafter's block desc (Inkling's MTP: an f16 ring; Qwen's DFlash: an F32 ring): its refusal, or nullptr
const char* draft_bad(const AttnCoreDesc& d) {
  const bool mtp = d.k_rows == ATTN_ROWS_F16;
  const uint32_t G = d.kv_heads && !(d.heads % d.kv_heads) ? d.heads / d.kv_heads : 0;
  return d.dk != d.dv || d.dk != INK2_MTP_HEAD || d.dk != Q4X_DF_HEAD ? "dk / dv: a drafter's heads of 128"
         : !G ? "kv_heads: not a group of heads"
         : (d.k_rows != ATTN_ROWS_F16 && d.k_rows != ATTN_ROWS_F32) || d.v_rows != d.k_rows
             ? "k_rows / v_rows: a drafter's ring f16 (Inkling's MTP) or F32 (Qwen's DFlash)"
         : !d.max_tokens ? "max_tokens: 0"
         : mtp && (uint64_t) d.max_tokens * G > attn_mtp::kRows ? "max_tokens: a block's rows past the MTP's 48"
         : !d.slots ? "slots: the ring's rows"
         : !mtp && (!d.window || d.window < d.max_tokens) ? "window: DFlash's at least its block"
         : !mtp && ((uint64_t) d.slots + d.max_tokens) * sizeof(float) > 65536 ? "slots: past the LDS DFlash's scores take"
         : d.term != ATTN_TERM_NONE ? "term: a drafter's none"
         : d.sink != ATTN_SINK_NONE ? "sink: a drafter's none"
         : d.gate != ATTN_GATE_NONE ? "gate: a drafter's none"
         : d.unit ? "unit: a drafter's 0 (its family's split: its bits)"
         : d.ratio ? "ratio: a drafter's no picks"
                   : nullptr;
}

// a drafter's step: its family's kernel as its caller had it - the context from keys.key0 to the block, then its own
bool draft_launch(const AttnCore& c, const AttnStep& s, hipStream_t stream) {
  const AttnCoreDesc& d = c.d;
  const AttnKeys& k = s.keys;
  if (!k.own_k || !k.own_v || !s.n_tok || s.n_tok > d.max_tokens || k.key0 > s.pos0 || k.lists || k.sel || s.x16 ||
      s.pos0 > (uint32_t) INT32_MAX)
    return false;
  if (d.k_rows == ATTN_ROWS_F16)
    return attn_mtp::run(s.q, s.n_tok, (int32_t) s.pos0, k.own_k, k.own_v, (const uint16_t*) k.k, (const uint16_t*) k.v,
                         d.slots, (int32_t) k.key0, d.heads, d.kv_heads, d.window, c.part, s.out, stream);
  return attn_df::run(s.q, s.n_tok, (int32_t) s.pos0, k.own_k, k.own_v, (const float*) k.k, (const float*) k.v, d.slots,
                      (int32_t) k.key0, d.heads, d.kv_heads, d.window, s.out, stream);
}

// a heads block's desc (Qwen's; osmos1.7b's and q35's, apart): its refusal, or nullptr
const char* heads_bad(const AttnCoreDesc& d) {
  const uint32_t G = d.kv_heads && !(d.heads % d.kv_heads) ? d.heads / d.kv_heads : 0;
  const bool decode_g = G == 1 || G == 2 || G == 4 || G == 6 || G == 8;  // (attn_q35's instances)
  return d.dk != d.dv || (d.dk != Q4X_HEAD_DIM && d.dk != Q4X_QK_HEAD_NARROW) ? "dk / dv: a heads block's head 256 or 128"
         : !G ? "kv_heads: not a group of heads"
         : G > (uint32_t) attn_q4x::kTile ? "group: a heads block's query heads a KV head 16 at most (a tile's rows)"
         : d.apart && !decode_g ? "group: an apart family's decode's 1, 2, 4, 6 or 8 a KV head"
         : d.k_rows != ATTN_ROWS_Q8_0 || d.v_rows != ATTN_ROWS_Q8_0 ? "k_rows / v_rows: a heads block's Q8_0"
         : d.term != ATTN_TERM_NONE ? "term: a heads block's none"
         : d.sink != ATTN_SINK_NONE ? "sink: a heads block's none"
         : d.gate == ATTN_GATE_NONE ? "gate: a heads block's a value's or a head's"
         : d.slots && (!d.window || d.slots < d.window + d.max_tokens - 1) ? "slots: a ring of the window and a step past it"
         : d.ratio == 1 || (d.ratio && attn_q4x::kChunk % d.ratio) ? "ratio: picks' blocks of 2 or more, a chunk's divisor"
         : d.ratio && d.window ? "ratio: picks under no window"
         : d.unit != (uint32_t) attn_q4x::kChunk ? "unit: a heads block's 128 (its chunks: the families' bits)"
         : d.max_keys > OPS_ATTN_MAX_KEYS ? "max_keys: past OPS_ATTN_MAX_KEYS (its offsets', its chunks' grid)"
                                                 : nullptr;
}

// a heads block's step: the prompt's kernels (q4x_attn's - its picks or a window and ring - and q4x_attn_lists' for
// LSA's lists) or, apart and not a read, the decode's (q35_attn_decode's in steps of Q35_ATTN_DECODE_MAX tokens,
// q35_attn_read's a token at a time) - each as its caller had it, its bits
bool heads_launch(const AttnCore& c, const AttnStep& s, hipStream_t stream) {
  const AttnCoreDesc& d = c.d;
  const AttnKeys& k = s.keys;
  const uint32_t span = d.gate == ATTN_GATE_HEAD ? d.dv : 1, gstride = d.heads * d.dv / span;  // a token's gates
  const AttnShape sh = {span, d.window, d.slots};
  const uint64_t qs = (uint64_t) d.heads * d.dk, os = (uint64_t) d.heads * d.dv;
  if (k.lists) {
    if (d.apart && !s.read) {
      for (uint32_t t = 0; t < s.n_tok; t++)
        if (!attn_q35::run(s.q + t * qs, k.k, k.v, 1, d.heads, d.kv_heads, d.dk, &sh, 0, k.lists + (uint64_t) t * k.stride,
                           (const uint32_t*) k.counts + t, k.stride, s.gate + (uint64_t) t * gstride, c.part,
                           s.out + t * os, stream))
          return false;
      return true;
    }
    if (d.dk != Q4X_QK_HEAD_NARROW || d.heads / d.kv_heads > (uint32_t) attn_q4x::kListGqa || d.window || d.slots)
      return false;
    attn_q4x::fa_list_kernel<<<dim3(s.n_tok, d.kv_heads), 32, 0, stream>>>(
        s.q, (const uint8_t*) k.k, (const uint8_t*) k.v, d.heads, d.kv_heads, k.lists, k.counts, k.stride, s.gate, span,
        d.scaling, s.out);
    return hipGetLastError() == hipSuccess;
  }
  if (d.apart && !s.read) {
    for (uint32_t t0 = 0; t0 < s.n_tok; t0 += Q35_ATTN_DECODE_MAX) {
      const uint32_t m = s.n_tok - t0 < Q35_ATTN_DECODE_MAX ? s.n_tok - t0 : Q35_ATTN_DECODE_MAX;
      if (!attn_q35::run(s.q + t0 * qs, k.k, k.v, m, d.heads, d.kv_heads, d.dk, &sh, s.pos0 + t0, nullptr, nullptr, 0,
                         s.gate + (uint64_t) t0 * gstride, c.part, s.out + t0 * os, stream))
        return false;
    }
    return true;
  }
  const Q4xPick pick = {k.sel, k.npick, d.ratio, k.stride};
  const Q4xX16 side = {s.x16, s.overflow, d.heads * d.dv, s.x16_row0};
  return d.dk == Q4X_HEAD_DIM
             ? attn_q4x::attn<Q4X_HEAD_DIM>(s.q, k.k, k.v, s.n_tok, d.heads, d.kv_heads, s.pos0, s.gate,
                                            k.sel ? &pick : nullptr, sh, c.part, s.out, s.x16 ? &side : nullptr, stream)
             : attn_q4x::attn<Q4X_QK_HEAD_NARROW>(s.q, k.k, k.v, s.n_tok, d.heads, d.kv_heads, s.pos0, s.gate,
                                                  k.sel ? &pick : nullptr, sh, c.part, s.out, s.x16 ? &side : nullptr,
                                                  stream);
}

}  // namespace

extern "C" uint64_t attn_core_scratch(const AttnCoreDesc* d) {
  if (!d) return 0;
  if (d->block == ATTN_BLOCK_DRAFT)  // the MTP's spans' partials; DFlash's in its LDS
    return d->k_rows == ATTN_ROWS_F16 ? attn_mtp::scratch_bytes(d->max_tokens, d->heads, d->kv_heads) : 0;
  if (d->block == ATTN_BLOCK_HEADS && is_mla(*d)) return attn_hy4::scratch_floats(d->heads) * sizeof(float);
  if (d->block == ATTN_BLOCK_HEADS && is_latent(*d))
    return OPS_ATTN_SCRATCH(d->max_tokens, d->heads, (uint64_t) d->window + d->max_sel) * sizeof(float);
  if (d->block == ATTN_BLOCK_HEADS) {  // the prompt's (q4x_attn's) and, apart, the decode's (q35_attn's): one at a time
    const uint64_t p = attn_q4x::scratch_floats(d->heads, d->kv_heads, d->dk, d->max_keys),
                   q = d->apart ? attn_q35::scratch_floats(d->heads, Q35_ATTN_DECODE_MAX) : 0;
    return (p > q ? p : q) * sizeof(float);
  }
  const uint32_t U = d->unit ? d->unit : kFaUnit;
  return ((uint64_t) d->max_keys / U + 2) * kFaTile * d->heads * (kFaHead + 2) * sizeof(float);
}

extern "C" AttnCore* attn_core_create(const AttnCoreDesc* d, void* scratch, uint64_t scratch_bytes,
                                      char err[OSMOS_ERR]) {
  if (!d) return refuse(err, "no desc"), nullptr;
  if (d->block == ATTN_BLOCK_DRAFT) {
    const char* bad = draft_bad(*d);
    const uint64_t need = bad ? 0 : attn_core_scratch(d);
    if (!bad && need && (!scratch || scratch_bytes < need)) bad = "max_tokens: past the scratch given";
    if (bad) return refuse(err, bad), nullptr;
    AttnCore* c = (AttnCore*) std::calloc(1, sizeof *c);
    if (!c) return refuse(err, "out of memory"), nullptr;
    c->d = *d, c->part = (float*) scratch;
    return c;
  }
  if (d->block == ATTN_BLOCK_HEADS) {
    const char* bad = is_mla(*d) ? mla_bad(*d) : is_latent(*d) ? latent_bad(*d) : heads_bad(*d);
    if (!bad && !d->max_tokens) bad = "max_tokens: 0";
    if (!bad && !d->max_keys) bad = "max_keys: 0";
    if (!bad && (!scratch || scratch_bytes < attn_core_scratch(d))) bad = "max_keys: past the scratch given";
    if (bad) return refuse(err, bad), nullptr;
    AttnCore* c = (AttnCore*) std::calloc(1, sizeof *c);
    if (!c) return refuse(err, "out of memory"), nullptr;
    c->d = *d, c->part = (float*) scratch;
    return c;
  }
  const uint32_t G = d->kv_heads && !(d->heads % d->kv_heads) ? d->heads / d->kv_heads : 0;
  const bool group_made = G == 1 || G == 2 || G == 3 || G == 4 || G == 6 || G == 8;  // (the launches' instances)
  const char* bad = d->ratio ? "ratio: picks a heads block's alone (a tile's rows one token's)"
                    : d->unit && d->unit != ATTN_UNIT ? "unit: Inkling's ATTN_UNIT (its bits)"
                    : d->dk != ATTN_HEAD || d->dv != ATTN_HEAD ? "dk / dv: a head of ATTN_HEAD alone"
                    : !d->heads || !d->kv_heads || d->heads % d->kv_heads ? "kv_heads: not a group of heads"
                    : !group_made ? "group: query heads a KV head 1, 2, 3, 4, 6 or 8"
                    : kv_of(*d) == Kv::NONE ? "k_rows / v_rows: a pairing not made (F32, f16, Q8, Q8 keys and Q5 values)"
                    : d->term != ATTN_TERM_REL ? "term: ATTN_TERM_REL alone (B2: Inkling's)"
                    : d->d_rel != ATTN_REL ? "d_rel: ATTN_REL alone"
                    : !d->extent || d->extent > ATTN_EXTENT_MAX ? "extent: 1 .. ATTN_EXTENT_MAX"
                    : d->window % ATTN_TILE ? "window: a whole number of tiles"
                    : d->sink != ATTN_SINK_TOKEN ? "sink: a (token, head)'s logit alone (B2: Inkling's)"
                    : d->gate == ATTN_GATE_HEAD ? "gate: a value's or none (B2: Inkling's)"
                    : d->block != ATTN_BLOCK_TOKENS ? "block: not a block made"
                    : !d->max_tokens ? "max_tokens: 0"
                    : !d->max_keys ? "max_keys: 0"
                    : !scratch || scratch_bytes < attn_core_scratch(d) ? "max_keys: past the scratch given"
                                                                       : nullptr;
  if (bad) return refuse(err, bad), nullptr;
  AttnCore* c = (AttnCore*) std::calloc(1, sizeof *c);
  if (!c) return refuse(err, "out of memory"), nullptr;
  c->d = *d, c->part = (float*) scratch;
  return c;
}

extern "C" void attn_core_destroy(AttnCore* c) { std::free(c); }

extern "C" bool attn_core_run(const AttnCore* c, const AttnStep* s, void* stream) {
  if (!c || !s || !s->q || !s->out || !s->keys.k || !s->keys.v) return false;
  const AttnCoreDesc& d = c->d;
  const AttnKeys& k = s->keys;
  if (d.block == ATTN_BLOCK_DRAFT) return draft_launch(*c, *s, (hipStream_t) stream);
  if (d.block == ATTN_BLOCK_HEADS && is_mla(d)) return mla_launch(*c, *s, (hipStream_t) stream);
  if (d.block == ATTN_BLOCK_HEADS && is_latent(d)) return latent_launch(*c, *s, (hipStream_t) stream);
  if (d.block == ATTN_BLOCK_HEADS) {  // positions from key0 (the cache's rows), or picks of them
    if (!s->n_tok || s->n_tok > d.max_tokens || k.key0 || k.n_keys > d.max_keys || !s->gate ||
        (!k.lists && !d.slots && s->pos0 + s->n_tok > k.n_keys) || (k.lists && (!k.counts || !k.stride || k.sel)) ||
        (k.sel != nullptr) != (k.npick != nullptr) || (k.sel && (!d.ratio || !k.stride)) || (s->x16 && !s->overflow) ||
        (s->x16 && (k.lists || (d.apart && !s->read))))  // (the cache's rows from 0: key0 0; a gate; a side the prompt's)
      return false;
    return heads_launch(*c, *s, (hipStream_t) stream);
  }
  if (!s->rel || !s->rel_proj || !s->sink) return false;
  if (!s->n_tok || s->n_tok > d.max_tokens || k.n_keys > d.max_keys || s->n_tok > k.n_keys || k.key0 % ATTN_TILE ||
      k.lists || k.sel || k.npick || s->x16_row0 || s->pos0 != k.key0 + k.n_keys - s->n_tok)  // (B2: positions, the step's tokens its keys' last)
    return false;
  if (s->x16 && (s->n_tok > (uint32_t) kFaTile || !s->overflow)) return false;  // a split decode's alone
  hipStream_t st = (hipStream_t) stream;
  switch (kv_of(d)) {
    case Kv::F32: return launch<float>(*c, *s, (const float*) k.k, (const float*) k.v, st);
    case Kv::F16: return launch<_Float16>(*c, *s, (const _Float16*) k.k, (const _Float16*) k.v, st);
    case Kv::KV8: return launch<Kv8>(*c, *s, (const Kv8*) k.k, (const Kv8*) k.v, st);
    case Kv::KV85: return launch<Kv85>(*c, *s, (const Kv85*) k.k, (const Kv85*) k.v, st);
    default: return false;
  }
}
