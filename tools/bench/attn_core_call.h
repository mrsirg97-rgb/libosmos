// the bench tools' one call of the attention core (osmos/gpu/attn_core.h) for Inkling's layer: a desc from its
// shape, a core over one scratch kept for the process (grown to the largest desc's), the step - n_tok queries at
// the last n_tok of n_keys keys from key0
#pragma once
#include <hip/hip_runtime.h>

extern "C" {
#include "osmos/gpu/attn_core.h"
}

inline bool attn_core_call(uint32_t heads, uint32_t kv_heads, uint32_t head_dim, uint32_t d_rel, uint32_t extent,
                           uint32_t window, float scaling, AttnRows k_rows, AttnRows v_rows, const float* q,
                           const float* rel, const float* sink, const float* gate, uint32_t n_tok, const void* k,
                           const void* v, uint32_t n_keys, uint32_t key0, const float* rel_proj, float* out,
                           void* stream = nullptr) {
  static void* scratch = nullptr;
  static uint64_t have = 0;
  AttnCoreDesc d{};
  d.heads = heads, d.kv_heads = kv_heads, d.dk = d.dv = head_dim, d.k_rows = k_rows, d.v_rows = v_rows;
  d.window = window, d.scaling = scaling, d.term = ATTN_TERM_REL, d.d_rel = d_rel, d.extent = extent, d.sink = ATTN_SINK_TOKEN;
  d.gate = ATTN_GATE_VALUE, d.block = ATTN_BLOCK_TOKENS, d.max_tokens = n_tok, d.max_keys = key0 + n_keys;
  const uint64_t need = attn_core_scratch(&d);
  if (need > have) {
    if (scratch) (void) hipFree(scratch);
    if (hipMalloc(&scratch, need) != hipSuccess) return scratch = nullptr, have = 0, false;
    have = need;
  }
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, have, err);
  if (!c) return false;
  const AttnStep s = {q,    n_tok,    key0 + n_keys - n_tok, nullptr, {k, v, key0, n_keys, nullptr, nullptr, nullptr, nullptr, 0},
                      rel,  rel_proj, sink, gate, out, nullptr, nullptr, 0, false};
  const bool ok = attn_core_run(c, &s, stream);
  attn_core_destroy(c);
  return ok;
}

// the heads block's calls for the tests and tools that called the families' kernels by name (q4x_attn, q4x_attn_lists,
// q35_attn_decode, q35_attn_read - their arguments, those names' meanings): a core of the call's desc over a scratch
// kept for the process; the scratch an old call passed is not needed (the core's own). a desc the core refuses, false
extern "C" {
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/kvfmt.h"
}

inline bool attn_core_heads_call(const AttnCoreDesc& d, const AttnStep& s, void* stream) {
  static void* scratch = nullptr;
  static uint64_t have = 0;
  const uint64_t need = attn_core_scratch(&d);
  if (need > have) {
    if (scratch) (void) hipFree(scratch);
    if (hipMalloc(&scratch, need) != hipSuccess) return scratch = nullptr, have = 0, false;
    have = need;
  }
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, have, err);
  if (!c) return false;
  const bool ok = attn_core_run(c, &s, stream);
  attn_core_destroy(c);
  return ok;
}

// a shape's gate span the core takes: a value's (1) or a head's (dim) - the old entries refused any other
inline bool attn_core_span_ok(const AttnShape* shape, uint32_t dim) {
  return !shape || shape->gate_span == 1 || shape->gate_span == dim;
}

inline AttnCoreDesc attn_core_heads_desc(uint32_t heads, uint32_t kv_heads, uint32_t dim, const AttnShape* shape,
                                         bool apart, uint32_t max_tokens, uint32_t max_keys, uint32_t ratio) {
  const AttnShape sh = shape ? *shape : AttnShape{1, 0, 0};
  AttnCoreDesc d{};
  d.heads = heads, d.kv_heads = kv_heads, d.dk = d.dv = dim, d.k_rows = d.v_rows = ATTN_ROWS_Q8_0;
  d.window = sh.window, d.slots = sh.slots, d.scaling = 1.0f / sqrtf((float) dim), d.term = ATTN_TERM_NONE;
  d.gate = sh.gate_span == dim && dim > 1 ? ATTN_GATE_HEAD : ATTN_GATE_VALUE, d.block = ATTN_BLOCK_HEADS;
  d.unit = 128, d.apart = apart, d.ratio = ratio, d.max_tokens = max_tokens, d.max_keys = max_keys;
  return d;
}

// q4x_attn's: a prompt's kernel (its picks, or a window and ring)
inline bool core_q4x_attn(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                          uint32_t kv_heads, uint32_t dim, uint32_t pos0, const float* gate, const Q4xPick* pick,
                          const AttnShape* shape, float*, float* out, const Q4xX16* side, void* stream) {
  if (!attn_core_span_ok(shape, dim)) return false;
  const uint32_t slots = shape ? shape->slots : 0, keys = slots ? slots : pos0 + n_tok;
  const AttnCoreDesc d = attn_core_heads_desc(heads, kv_heads, dim, shape, false, n_tok, keys, pick ? pick->ratio : 0);
  AttnStep s{};
  s.q = q, s.n_tok = n_tok, s.pos0 = pos0, s.gate = gate, s.out = out, s.read = true;
  s.keys.k = kc, s.keys.v = vc, s.keys.n_keys = keys;
  if (pick) s.keys.sel = pick->sel, s.keys.npick = pick->npick, s.keys.stride = pick->blocks;
  if (side) {
    if (side->cols != heads * dim) return false;
    s.x16 = side->x16, s.overflow = side->overflow, s.x16_row0 = side->row0;
  }
  return attn_core_heads_call(d, s, stream);
}

// q4x_attn_lists': a prompt's LSA lists
inline bool core_q4x_attn_lists(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                                uint32_t kv_heads, uint32_t dim, const uint32_t* lists, const int32_t* counts,
                                uint32_t stride, const float* gate, const AttnShape* shape, float* out, void* stream) {
  if (!attn_core_span_ok(shape, dim)) return false;
  const AttnCoreDesc d = attn_core_heads_desc(heads, kv_heads, dim, shape, true, n_tok, stride, 0);
  AttnStep s{};
  s.q = q, s.n_tok = n_tok, s.gate = gate, s.out = out, s.read = true;
  s.keys.k = kc, s.keys.v = vc, s.keys.n_keys = stride, s.keys.lists = lists, s.keys.counts = counts, s.keys.stride = stride;
  return attn_core_heads_call(d, s, stream);
}

// q35_attn_decode's: a decode's (apart, not a read)
inline bool core_q35_attn_decode(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                                 uint32_t kv_heads, uint32_t dim, const AttnShape* shape, uint32_t pos0,
                                 const float* gate, float*, float* out, void* stream) {
  if (!attn_core_span_ok(shape, dim)) return false;
  const uint32_t slots = shape ? shape->slots : 0, keys = slots ? slots : pos0 + n_tok;
  const AttnCoreDesc d = attn_core_heads_desc(heads, kv_heads, dim, shape, true, n_tok, keys, 0);
  AttnStep s{};
  s.q = q, s.n_tok = n_tok, s.pos0 = pos0, s.gate = gate, s.out = out;
  s.keys.k = kc, s.keys.v = vc, s.keys.n_keys = keys;
  return attn_core_heads_call(d, s, stream);
}

// q35_attn_read's: one token's LSA read (apart, not a read)
inline bool core_q35_attn_read(const float* q, const void* kc, const void* vc, uint32_t heads, uint32_t kv_heads,
                               uint32_t dim, const AttnShape* shape, const uint32_t* list, const uint32_t* list_len,
                               uint32_t list_max, const float* gate, float*, float* out, void* stream) {
  if (!attn_core_span_ok(shape, dim)) return false;
  const AttnCoreDesc d = attn_core_heads_desc(heads, kv_heads, dim, shape, true, 1, list_max, 0);
  AttnStep s{};
  s.q = q, s.n_tok = 1, s.gate = gate, s.out = out;
  s.keys.k = kc, s.keys.v = vc, s.keys.n_keys = list_max, s.keys.lists = list, s.keys.counts = (const int32_t*) list_len;
  s.keys.stride = list_max;
  return attn_core_heads_call(d, s, stream);
}

// hy4_mla_attn's: Hy4's MLA (one latent row for every head; the step's read its prompt's kernel). the old call's
// scratch not needed (the core's own); its sel_cap the keys' stride
inline AttnCoreDesc attn_core_mla_desc(uint32_t heads, uint32_t latent, uint32_t rope_dim, bool q8, uint32_t window,
                                       uint32_t slots, uint32_t pool, float scale, uint32_t max_tokens) {
  AttnCoreDesc d{};
  d.heads = heads, d.kv_heads = 1, d.dk = latent + rope_dim, d.dv = latent;
  d.k_rows = d.v_rows = q8 ? ATTN_ROWS_MLA_Q8 : ATTN_ROWS_MLA_F32;
  d.window = window, d.slots = slots, d.scaling = scale, d.term = ATTN_TERM_NONE, d.sink = ATTN_SINK_HEAD;
  d.gate = ATTN_GATE_NONE, d.block = ATTN_BLOCK_HEADS, d.ratio = pool, d.unit = 32, d.apart = true;
  d.max_tokens = max_tokens, d.max_keys = HY4_MLA_MAX_KEYS;
  return d;
}
inline bool core_hy4_mla_attn(const float* qa, const void* cache, uint32_t slots, bool q8, const float* sinks,
                              uint32_t heads, uint32_t latent, uint32_t rope_dim, uint32_t window, const int32_t* sel,
                              const int32_t* n_sel, uint32_t sel_cap, uint32_t pool, float scale, const int32_t* pos,
                              uint32_t n_tok, float* o_lat, float*, bool read, void* stream) {
  const AttnCoreDesc d = attn_core_mla_desc(heads, latent, rope_dim, q8, window, slots, sel ? pool : 0, scale, n_tok);
  AttnStep s{};
  s.q = qa, s.n_tok = n_tok, s.pos = pos, s.sink = sinks, s.out = o_lat, s.read = read;
  s.keys.k = s.keys.v = cache, s.keys.sel = sel, s.keys.npick = n_sel, s.keys.stride = sel_cap;
  return attn_core_heads_call(d, s, stream);
}

// ops_sparse_attn's: DeepSeek's latent (the window's ring, then the selected rows), its old arguments' meanings. the
// old call's scores not needed (the core's own scratch)
inline bool core_ops_sparse_attn(const float* q, const void* ring, uint32_t ring_fmt, uint32_t slots, const int32_t* pos,
                                 uint32_t window, int32_t floor, const void* ckv, uint32_t ckv_fmt, const int32_t* sel,
                                 const int32_t* n_sel, uint32_t sel_cap, uint32_t max_sel, const float* sink, float scale,
                                 uint32_t heads, uint32_t head_dim, uint32_t n_tok, float*, float* o, void* stream) {
  if ((ring_fmt != KV_FP8 && ring_fmt != KV_F32) || (ckv && ckv_fmt != KV_FP4_E4M3 && ckv_fmt != KV_F32)) return false;
  AttnCoreDesc d{};
  d.heads = heads, d.kv_heads = 1, d.dk = d.dv = head_dim, d.k_rows = d.v_rows = ATTN_ROWS_FP8;
  d.s_rows = ckv_fmt == KV_F32 ? ATTN_ROWS_F32 : ATTN_ROWS_FP4, d.max_sel = ckv ? max_sel : 0;
  d.window = window, d.slots = slots, d.scaling = scale, d.term = ATTN_TERM_NONE, d.sink = ATTN_SINK_HEAD;
  d.gate = ATTN_GATE_NONE, d.block = ATTN_BLOCK_HEADS, d.max_tokens = n_tok, d.max_keys = window + d.max_sel;
  AttnStep s{};
  s.q = q, s.n_tok = n_tok, s.pos = pos, s.sink = sink, s.out = o;
  s.keys.k = s.keys.v = ring, s.keys.k_wide = ring_fmt == KV_F32, s.keys.floor = floor;
  if (ckv) s.keys.rows = ckv, s.keys.sel = sel, s.keys.npick = n_sel, s.keys.stride = sel_cap;
  return attn_core_heads_call(d, s, stream);
}

// ink2_mtp_attn's and q4x_df_attn's: a drafter's block (the ring's context from ctx_from, then the block's own), their
// old arguments' meanings; the MTP's old scratch not needed (the core's own)
inline bool attn_core_draft_call(bool f16, const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb,
                                 const void* kr, const void* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads,
                                 uint32_t kv_heads, uint32_t window, float* out, void* stream) {
  if (pos0 < 0 || ctx_from < 0) return false;
  AttnCoreDesc d{};
  d.heads = heads, d.kv_heads = kv_heads, d.dk = d.dv = 128, d.k_rows = d.v_rows = f16 ? ATTN_ROWS_F16 : ATTN_ROWS_F32;
  d.window = window, d.slots = ring_rows, d.scaling = 1.0f / sqrtf(128.0f), d.term = ATTN_TERM_NONE;
  d.sink = ATTN_SINK_NONE, d.gate = ATTN_GATE_NONE, d.block = ATTN_BLOCK_DRAFT, d.max_tokens = n_q;
  d.max_keys = ring_rows + n_q;
  AttnStep s{};
  s.q = q, s.n_tok = n_q, s.pos0 = (uint32_t) pos0, s.out = out;
  s.keys.k = kr, s.keys.v = vr, s.keys.key0 = (uint32_t) ctx_from, s.keys.own_k = kb, s.keys.own_v = vb;
  return attn_core_heads_call(d, s, stream);
}
inline bool core_ink2_mtp_attn(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb,
                               const uint16_t* kr, const uint16_t* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads,
                               uint32_t kv_heads, uint32_t window, float*, float* out, void* stream) {
  return attn_core_draft_call(true, q, n_q, pos0, kb, vb, kr, vr, ring_rows, ctx_from, heads, kv_heads, window, out, stream);
}
inline bool core_q4x_df_attn(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb, const float* kr,
                             const float* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                             uint32_t window, float* out, void* stream) {
  return attn_core_draft_call(false, q, n_q, pos0, kb, vb, kr, vr, ring_rows, ctx_from, heads, kv_heads, window, out, stream);
}
