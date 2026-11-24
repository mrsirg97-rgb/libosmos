// the attention core (osmos/gpu/attn_core.h): a desc checked whole at create, a step launched on its rows' kernels
// (attn_core_kernel.h) - B2: Inkling's formats (F32, f16, Q8 records, Q8 keys with Q5 values)
#include <hip/hip_runtime.h>

#include <cstdlib>

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

}  // namespace

extern "C" uint64_t attn_core_scratch(const AttnCoreDesc* d) {
  return d ? ((uint64_t) d->max_keys / kFaUnit + 2) * kFaTile * d->heads * (kFaHead + 2) * sizeof(float) : 0;
}

extern "C" AttnCore* attn_core_create(const AttnCoreDesc* d, void* scratch, uint64_t scratch_bytes,
                                      char err[OSMOS_ERR]) {
  if (!d) return refuse(err, "no desc"), nullptr;
  const uint32_t G = d->kv_heads && !(d->heads % d->kv_heads) ? d->heads / d->kv_heads : 0;
  const bool group_made = G == 1 || G == 2 || G == 3 || G == 4 || G == 6 || G == 8;  // (the launches' instances)
  const char* bad = d->dk != ATTN_HEAD || d->dv != ATTN_HEAD ? "dk / dv: a head of ATTN_HEAD alone"
                    : !d->heads || !d->kv_heads || d->heads % d->kv_heads ? "kv_heads: not a group of heads"
                    : !group_made ? "group: query heads a KV head 1, 2, 3, 4, 6 or 8"
                    : kv_of(*d) == Kv::NONE ? "k_rows / v_rows: a pairing not made (F32, f16, Q8, Q8 keys and Q5 values)"
                    : d->term != ATTN_TERM_REL ? "term: ATTN_TERM_REL alone (B2: Inkling's)"
                    : d->d_rel != ATTN_REL ? "d_rel: ATTN_REL alone"
                    : !d->extent || d->extent > ATTN_EXTENT_MAX ? "extent: 1 .. ATTN_EXTENT_MAX"
                    : d->window % ATTN_TILE ? "window: a whole number of tiles"
                    : !d->sink ? "sink: a sink logit alone (B2: Inkling's)"
                    : d->gate == ATTN_GATE_HEAD ? "gate: a value's or none (B2: Inkling's)"
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
  if (!c || !s || !s->q || !s->out || !s->keys.k || !s->keys.v || !s->rel || !s->rel_proj || !s->sink) return false;
  const AttnCoreDesc& d = c->d;
  const AttnKeys& k = s->keys;
  if (!s->n_tok || s->n_tok > d.max_tokens || k.n_keys > d.max_keys || s->n_tok > k.n_keys || k.key0 % ATTN_TILE ||
      k.lists || s->pos0 != k.key0 + k.n_keys - s->n_tok)  // (B2: positions, the step's tokens its keys' last)
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
