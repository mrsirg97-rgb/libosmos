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
  const AttnCoreDesc d = {heads, kv_heads, head_dim, head_dim, k_rows, v_rows, window, scaling, ATTN_TERM_REL, d_rel,
                          extent, true, ATTN_GATE_VALUE, n_tok, key0 + n_keys};
  const uint64_t need = attn_core_scratch(&d);
  if (need > have) {
    if (scratch) (void) hipFree(scratch);
    if (hipMalloc(&scratch, need) != hipSuccess) return scratch = nullptr, have = 0, false;
    have = need;
  }
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, have, err);
  if (!c) return false;
  const AttnStep s = {q, n_tok, key0 + n_keys - n_tok, {k, v, key0, n_keys, nullptr, nullptr, 0}, rel, rel_proj, sink,
                      gate, out, nullptr, nullptr};
  const bool ok = attn_core_run(c, &s, stream);
  attn_core_destroy(c);
  return ok;
}
