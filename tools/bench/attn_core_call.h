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
  d.window = window, d.scaling = scaling, d.term = ATTN_TERM_REL, d.d_rel = d_rel, d.extent = extent, d.sink = true;
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
  const AttnStep s = {q,    n_tok,    key0 + n_keys - n_tok, {k, v, key0, n_keys, nullptr, nullptr, nullptr, nullptr, 0},
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
