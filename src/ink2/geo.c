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

bool ink2_geometry(const Gguf *model, Ink2Geometry *g, char err[INK2_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int ctx = 0;
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 1, INK2_MAX_LAYERS, &g->n_layers))
    return false;
  const bool ok =
      meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
      meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
      meta_uint_in(&m, ARCH ".vocab_size", 1, DIM_MAX, &g->n_vocab) &&
      meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
      meta_uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
      meta_uint_in(&m, ARCH ".attention.head_count_kv", 1, g->n_heads, &g->n_kv_heads) &&
      (g->n_heads % g->n_kv_heads == 0 || meta_fail(&m, ARCH ".attention.head_count_kv", "not a heads' divisor")) &&
      meta_uint_in(&m, ARCH ".attention.key_length", 1, DIM_MAX, &g->head_dim) &&
      meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
      meta_uint_in(&m, ARCH ".attention.sliding_window", 1, CTX_MAX, &g->window) && kinds(&m, g) &&
      meta_uint_in(&m, ARCH ".attention.relative_dim", 1, DIM_MAX, &g->d_rel) &&
      meta_uint_in(&m, ARCH ".attention.relative_extent", 1, CTX_MAX, &g->rel_extent) &&
      meta_uint_in(&m, ARCH ".attention.log_scaling_floor", 1, CTX_MAX, &g->log_floor) &&
      meta_real(&m, ARCH ".attention.log_scaling_alpha", &g->log_alpha) &&
      meta_real(&m, ARCH ".attention.q_clip", &g->q_clip) && meta_real(&m, ARCH ".attention.k_clip", &g->k_clip) &&
      meta_real(&m, ARCH ".attention.relative_clip", &g->rel_clip) &&
      meta_uint_in(&m, ARCH ".conv_kernel", 1, INK2_CONV_TAPS_MAX, &g->conv) &&
      meta_uint_in(&m, ARCH ".attn_res.block_size", 1, g->n_layers, &g->res_block) &&
      meta_uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
      meta_uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
      meta_uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->ff_exp) && router(&m, g) &&
      meta_real(&m, ARCH ".expert_gate_softcap", &g->gate_softcap) &&
      meta_real(&m, ARCH ".expert_up_softcap", &g->up_softcap) &&
      meta_real(&m, ARCH ".final_logit_softcapping", &g->logit_softcap);
  g->native_ctx = (uint32_t)ctx;
  return ok;
}
