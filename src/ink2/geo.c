#include <string.h>

#include "../model/gguf_meta.h"
#include "osmos/ink2/weights.h"

#define ARCH "inkling2"
#define DIM_MAX (1 << 24)
#define CTX_MAX (1 << 30)
#define GATING_SIGMOID 2 /* llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID */
_Static_assert(INK2_ERR == GGUF_META_ERR && INK2_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");

/* the sliding layers a list; the rest full - at least one (the log scaling and the long relative bias are theirs) */
static bool kinds(GgufMeta *m, Ink2Geometry *g)
{
  int n = 0;
  if (!meta_layers(m, ARCH ".attention.sliding_layers", g->n_layers, g->sliding, &n))
    return false;
  return n < g->n_layers || meta_fail(m, ARCH ".attention.sliding_layers", "every layer sliding: no full layer");
}

/* the router: sigmoid scores, the picked ones' weights normalized - what the MoE implements, nothing else */
static bool router(GgufMeta *m, Ink2Geometry *g)
{
  return meta_uint_is(m, ARCH ".expert_gating_func", GATING_SIGMOID) &&
         (meta_flag_set(m, ARCH ".expert_weights_norm") ||
          meta_fail(m, ARCH ".expert_weights_norm", "not set: the picked experts' weights not normalized")) &&
         meta_real(m, ARCH ".expert_weights_scale", &g->route_scale);
}

/* llama.cpp's inkling2 keys (unsloth's GGUFs): the sliding layers a flag a layer, the relative bias's sizes and clips
   their own names, q's and k's clip one threshold; the router's sigmoid and normalized weights the architecture's
   (no keys) - a full layer's extent theirs, a sliding layer's its window's (rel_extent_swa: refused if not) */
static bool llama_keys(GgufMeta *m, Ink2Geometry *g)
{
  bool sw[INK2_MAX_LAYERS] = {0};
  int ext_swa = 0, full = 0;
  bool ok = meta_flags(m, ARCH ".attention.sliding_window_pattern", (size_t)g->n_layers, sw) &&
            meta_uint_in(m, ARCH ".unpadded_vocab_size", 1, DIM_MAX, &g->n_vocab) &&
            meta_uint_in(m, ARCH ".d_rel", 1, DIM_MAX, &g->d_rel) &&
            meta_uint_in(m, ARCH ".rel_extent", 1, CTX_MAX, &g->rel_extent) &&
            meta_uint_in(m, ARCH ".rel_extent_swa", 1, CTX_MAX, &ext_swa) &&
            (ext_swa == g->window || meta_fail(m, ARCH ".rel_extent_swa", "not the sliding window's")) &&
            meta_uint_in(m, ARCH ".log_scaling_n_floor", 1, CTX_MAX, &g->log_floor) &&
            meta_real(m, ARCH ".log_scaling_alpha", &g->log_alpha) &&
            meta_real(m, ARCH ".qk_clip_threshold", &g->q_clip) &&
            meta_real(m, ARCH ".rel_clip_threshold", &g->rel_clip) &&
            meta_uint_in(m, ARCH ".shortconv_kernel", 1, INK2_CONV_TAPS_MAX, &g->conv) &&
            meta_real(m, ARCH ".expert_weights_scale", &g->route_scale) &&
            meta_real(m, ARCH ".mlp_gate_softcap", &g->gate_softcap) &&
            meta_real(m, ARCH ".mlp_up_softcap", &g->up_softcap) &&
            (meta_flag_set(m, ARCH ".attn_res_normalize_values") ||
             meta_fail(m, ARCH ".attn_res_normalize_values", "not set: attn_res's values not normalized")) &&
            (meta_flag_set(m, ARCH ".value_norm") || meta_fail(m, ARCH ".value_norm", "not set: v not normalized"));
  for (int l = 0; ok && l < g->n_layers; l++)
    g->sliding[l] = sw[l], full += !sw[l];
  g->k_clip = g->q_clip, g->llama_names = true;
  return ok && (full || meta_fail(m, ARCH ".attention.sliding_window_pattern", "every layer sliding: no full layer"));
}

static bool ours(GgufMeta *m, Ink2Geometry *g);

bool ink2_geometry(const Gguf *model, Ink2Geometry *g, char err[INK2_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int ctx = 0;
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 1, INK2_MAX_LAYERS, &g->n_layers))
    return false;
  const bool llama = meta_present(&m, ARCH ".d_rel");
  const bool common =
      meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
      meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
      meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
      meta_uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
      meta_uint_in(&m, ARCH ".attention.head_count_kv", 1, g->n_heads, &g->n_kv_heads) &&
      (g->n_heads % g->n_kv_heads == 0 || meta_fail(&m, ARCH ".attention.head_count_kv", "not a heads' divisor")) &&
      meta_uint_in(&m, ARCH ".attention.key_length", 1, DIM_MAX, &g->head_dim) &&
      meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
      meta_uint_in(&m, ARCH ".attention.sliding_window", 1, CTX_MAX, &g->window) &&
      meta_uint_in(&m, ARCH ".attn_res.block_size", 1, g->n_layers, &g->res_block) &&
      meta_uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
      meta_uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
      meta_uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->ff_exp) &&
      meta_real(&m, ARCH ".final_logit_softcapping", &g->logit_softcap);
  const bool ok = common && (llama ? llama_keys(&m, g) : ours(&m, g));
  g->native_ctx = (uint32_t)ctx;
  return ok;
}

/* ink2_ref_gguf.py's keys */
static bool ours(GgufMeta *m, Ink2Geometry *g)
{
  return meta_uint_in(m, ARCH ".vocab_size", 1, DIM_MAX, &g->n_vocab) && kinds(m, g) &&
      meta_uint_in(m, ARCH ".attention.relative_dim", 1, DIM_MAX, &g->d_rel) &&
      meta_uint_in(m, ARCH ".attention.relative_extent", 1, CTX_MAX, &g->rel_extent) &&
      meta_uint_in(m, ARCH ".attention.log_scaling_floor", 1, CTX_MAX, &g->log_floor) &&
      meta_real(m, ARCH ".attention.log_scaling_alpha", &g->log_alpha) &&
      meta_real(m, ARCH ".attention.q_clip", &g->q_clip) && meta_real(m, ARCH ".attention.k_clip", &g->k_clip) &&
      meta_real(m, ARCH ".attention.relative_clip", &g->rel_clip) &&
      meta_uint_in(m, ARCH ".conv_kernel", 1, INK2_CONV_TAPS_MAX, &g->conv) &&
      router(m, g) && meta_real(m, ARCH ".expert_gate_softcap", &g->gate_softcap) &&
      meta_real(m, ARCH ".expert_up_softcap", &g->up_softcap);
}
