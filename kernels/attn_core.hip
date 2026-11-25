// the attention core (osmos/gpu/attn_core.h): a desc checked whole at create, a step launched on its rows' kernels
// (attn_core_kernel.h) - B2: Inkling's formats (F32, f16, Q8 records, Q8 keys with Q5 values)
#include <hip/hip_runtime.h>

#include <cstdlib>

#include "attn_core_heads.h"
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
         : d.sink ? "sink: a heads block's none"
         : d.gate == ATTN_GATE_NONE ? "gate: a heads block's a value's or a head's"
         : d.slots && (!d.window || d.slots < d.window + d.max_tokens - 1) ? "slots: a ring of the window and a step past it"
         : d.ratio == 1 || (d.ratio && attn_q4x::kChunk % d.ratio) ? "ratio: picks' blocks of 2 or more, a chunk's divisor"
         : d.ratio && d.window ? "ratio: picks under no window"
         : d.unit != (uint32_t) attn_q4x::kChunk ? "unit: a heads block's 128 (its chunks: the families' bits)"
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
  if (d->block == ATTN_BLOCK_HEADS) {  // the prompt's (q4x_attn's) and, apart, the decode's (q35_attn's): one at a time
    const uint64_t p = q4x_attn_scratch_dim(d->heads, d->kv_heads, d->dk, d->max_keys),
                   q = d->apart ? q35_attn_decode_scratch(d->heads, Q35_ATTN_DECODE_MAX) : 0;
    return (p > q ? p : q) * sizeof(float);
  }
  const uint32_t U = d->unit ? d->unit : kFaUnit;
  return ((uint64_t) d->max_keys / U + 2) * kFaTile * d->heads * (kFaHead + 2) * sizeof(float);
}

extern "C" AttnCore* attn_core_create(const AttnCoreDesc* d, void* scratch, uint64_t scratch_bytes,
                                      char err[OSMOS_ERR]) {
  if (!d) return refuse(err, "no desc"), nullptr;
  if (d->block == ATTN_BLOCK_HEADS) {
    const char* bad = heads_bad(*d);
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
                    : !d->sink ? "sink: a sink logit alone (B2: Inkling's)"
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
  if (d.block == ATTN_BLOCK_HEADS) {  // positions from key0 (the cache's rows), or picks of them
    if (!s->n_tok || s->n_tok > d.max_tokens || k.key0 || k.n_keys > d.max_keys || !s->gate ||
        (!k.lists && s->pos0 + s->n_tok > k.n_keys) || (k.lists && (!k.counts || !k.stride || k.sel)) ||
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

// q4x_attn and q4x_attn_lists (q4x_ops.h) - osmos1.7b's and q35's callers until C2 moves them onto the core: the same
// code the core's heads block runs
namespace attn_q4x {
extern "C" uint64_t q4x_attn_scratch(uint32_t heads, uint32_t max_ctx) {
  const uint64_t chunks = (max_ctx + kChunk - 1) / kChunk, pairs = chunks < kPieceChunks ? kPieceChunks : chunks;
  return pairs * heads * (kHead + 2);  // a piece of at least one token at the context's end
}

extern "C" uint64_t q4x_attn_scratch_dim(uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t max_ctx) {
  if (dim != Q4X_QK_HEAD_NARROW || !kv_heads || heads % kv_heads) return q4x_attn_scratch(heads, max_ctx);
  const uint64_t a = span_pairs(heads, kv_heads, max_ctx) * heads * (dim + 2), b = q4x_attn_scratch(heads, max_ctx);
  return a > b ? a : b;
}

extern "C" bool q4x_attn(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                         uint32_t kv_heads, uint32_t dim, uint32_t pos0, const float* gate, const Q4xPick* pick,
                         const AttnShape* shape, float* scratch, float* out, const Q4xX16* side, void* stream) {
  const AttnShape sh = shape ? *shape : AttnShape{1, 0, 0};
  if (!n_tok || !heads || !kv_heads || heads % kv_heads || heads / kv_heads > kTile ||
      (dim != (uint32_t) kHead && dim != Q4X_QK_HEAD_NARROW) || (sh.gate_span != 1 && sh.gate_span != dim) ||
      (sh.slots && (!sh.window || sh.slots < sh.window + n_tok - 1)) || (uint64_t) pos0 + n_tok > OPS_ATTN_MAX_KEYS)
    return false;
  if (!q || !kc || !vc || !gate || !scratch || !out ||
      (pick && (!pick->npick || !pick->sel || pick->ratio < 2 || !pick->blocks || sh.window)))
    return false;
  return dim == (uint32_t) kHead
             ? attn<Q4X_HEAD_DIM>(q, kc, vc, n_tok, heads, kv_heads, pos0, gate, pick, sh, scratch, out, side, stream)
             : attn<Q4X_QK_HEAD_NARROW>(q, kc, vc, n_tok, heads, kv_heads, pos0, gate, pick, sh, scratch, out, side,
                                        stream);
}

extern "C" bool q4x_attn_lists(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                               uint32_t kv_heads, uint32_t dim, const uint32_t* lists, const int32_t* counts,
                               uint32_t list_stride, const float* gate, const AttnShape* shape, float* out,
                               void* stream) {
  const AttnShape sh = shape ? *shape : AttnShape{1, 0, 0};
  if (!q || !kc || !vc || !lists || !counts || !gate || !out || !n_tok || !kv_heads || heads % kv_heads ||
      heads / kv_heads > (uint32_t) kListGqa || dim != Q4X_QK_HEAD_NARROW || !list_stride || sh.window || sh.slots ||
      (sh.gate_span != 1 && sh.gate_span != dim))
    return false;
  fa_list_kernel<<<dim3(n_tok, kv_heads), 32, 0, st(stream)>>>(
      q, (const uint8_t*) kc, (const uint8_t*) vc, heads, kv_heads, lists, counts, list_stride, gate, sh.gate_span,
      1.0f / sqrtf((float) dim), out);
  return ok();
}


}  // namespace attn_q4x

// q35_attn_decode and q35_attn_read (q35_ops.h) - osmos1.7b's and q35's callers until C2 moves them onto the core
namespace attn_q35 {
extern "C" uint64_t q35_attn_decode_scratch(uint32_t heads, uint32_t max_tokens) {
  return (uint64_t) max_tokens * heads * Q35_ATTN_PARTS * (kHeadWide + 2);  // the widest head's
}

extern "C" bool q35_attn_decode(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                                uint32_t kv_heads, uint32_t dim, const AttnShape* shape, uint32_t pos0,
                                const float* gate, float* scratch, float* out, void* stream) {
  return run(q, kc, vc, n_tok, heads, kv_heads, dim, shape, pos0, nullptr, nullptr, 0, gate, scratch, out, stream);
}

extern "C" bool q35_attn_read(const float* q, const void* kc, const void* vc, uint32_t heads, uint32_t kv_heads,
                              uint32_t dim, const AttnShape* shape, const uint32_t* list, const uint32_t* list_len,
                              uint32_t list_max, const float* gate, float* scratch, float* out, void* stream) {
  return run(q, kc, vc, 1, heads, kv_heads, dim, shape, 0, list, list_len, list_max, gate, scratch, out, stream);
}

}  // namespace attn_q35
