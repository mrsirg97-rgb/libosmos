#include <string.h>

#include "../model/gguf_meta.h"
#include "osmos/ink2/weights.h"

#define ARCH "inkling2"
#define DIM_MAX (1 << 24)
#define CTX_MAX (1 << 30)
#define PASSES_MAX 64
#define GATING_SIGMOID 2 /* llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID */
_Static_assert(OSMOS_ERR == OSMOS_ERR && INK2_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");

/* a value a slot - one integer, or an array of one a slot (upstream's): each slot's its layer's, every layer's the
   same (one count implemented) */
static bool slot_uint(GgufMeta *m, const Ink2Geometry *g, const char *key, int lo, int hi, int *out)
{
  const GgufKv *kv = gguf_kv(m->g, key);
  if (!kv || kv->type != GGUF_ARRAY)
    return meta_uint_in(m, key, lo, hi, out);
  int64_t v[INK2_MAX_LAYERS];
  size_t n = 0;
  if (!meta_ints(m, key, (size_t)g->n_slots, (size_t)g->n_slots, v, &n))
    return false;
  for (int s = 0; s < g->n_slots; s++)
  {
    if (v[s] != v[ink2_slot_layer(g, s)])
      return meta_fail(m, key, "a slot's value not its layer's");
    if (v[s] != v[0])
      return meta_fail(m, key, "not the same in every layer: one count implemented");
  }
  if (v[0] < lo || v[0] > hi)
    return meta_fail(m, key, "out of range");
  *out = (int)v[0];
  return true;
}

/* the sliding layers a flag a slot, a slot's its layer's; at least one full (the log scaling and the long relative
   bias are theirs) */
static bool slot_flags(GgufMeta *m, Ink2Geometry *g)
{
  const char *key = ARCH ".attention.sliding_window_pattern";
  bool sw[INK2_MAX_LAYERS] = {0};
  int full = 0;
  if (!meta_flags(m, key, (size_t)g->n_slots, sw))
    return false;
  for (int s = 0; s < g->n_slots; s++)
    if (sw[s] != sw[ink2_slot_layer(g, s)])
      return meta_fail(m, key, "a slot another kind than its layer");
  for (int l = 0; l < g->n_layers; l++)
    g->sliding[l] = sw[l], full += !sw[l];
  return full || meta_fail(m, key, "every layer sliding: no full layer");
}

/* the loop (unsloth's keys: inkling2.num_loops, inkling2.loop.*): block_count its slots, the core whole attn_res
   blocks within the layers. none: the slots the layers */
static bool loop(GgufMeta *m, Ink2Geometry *g)
{
  Ink2Loop *p = &g->loop;
  if (!meta_present(m, ARCH ".num_loops"))
    return g->n_layers = g->n_slots, true;
  if (!meta_uint_in(m, ARCH ".num_loops", 2, PASSES_MAX, &p->passes) ||
      !meta_uint_in(m, ARCH ".loop.core_start", 1, g->n_slots, &p->start) ||
      !meta_uint_in(m, ARCH ".loop.core_end", p->start + 1, g->n_slots, &p->end) ||
      !meta_uint_in(m, ARCH ".loop.gate_rank", 1, DIM_MAX, &p->gate_rank))
    return false;
  const int more = (p->passes - 1) * (p->end - p->start);
  if (more >= g->n_slots)
    return meta_fail(m, ARCH ".block_count", "fewer slots than the passes after the first need");
  g->n_layers = g->n_slots - more;
  if (p->end > g->n_layers)
    return meta_fail(m, ARCH ".loop.core_end", "past the layers");
  if (p->start % g->res_block || p->end % g->res_block)
    return meta_fail(m, p->start % g->res_block ? ARCH ".loop.core_start" : ARCH ".loop.core_end",
                     "not on an attn_res block's edge");
  return true;
}

/* the router: sigmoid scores, the picked ones' weights normalized - what the MoE implements, nothing else */
static bool router(GgufMeta *m, Ink2Geometry *g)
{
  return meta_uint_is(m, ARCH ".expert_gating_func", GATING_SIGMOID) &&
         (meta_flag_set(m, ARCH ".expert_weights_norm") ||
          meta_fail(m, ARCH ".expert_weights_norm", "not set: the picked experts' weights not normalized")) &&
         meta_real(m, ARCH ".expert_weights_scale", &g->route_scale);
}

/* unsloth's Inkling-2-Small keys: the sliding layers a flag a layer, the relative bias's sizes and clips their own
   names, q's and k's clip one threshold; the router's sigmoid and normalized weights the architecture's (no keys) - a
   full layer's extent theirs, a sliding layer's its window's (rel_extent_swa: refused if not) */
static bool unsloth_small_keys(GgufMeta *m, Ink2Geometry *g)
{
  int ext_swa = 0;
  g->keys = INK2_KEYS_UNSLOTH_SMALL;
  const bool ok = slot_flags(m, g) && meta_uint_in(m, ARCH ".unpadded_vocab_size", 1, DIM_MAX, &g->n_vocab) &&
         meta_uint_in(m, ARCH ".d_rel", 1, DIM_MAX, &g->d_rel) &&
         meta_uint_in(m, ARCH ".rel_extent", 1, CTX_MAX, &g->rel_extent) &&
         meta_uint_in(m, ARCH ".rel_extent_swa", 1, CTX_MAX, &ext_swa) &&
         (ext_swa == g->window || meta_fail(m, ARCH ".rel_extent_swa", "not the sliding window's")) &&
         meta_uint_in(m, ARCH ".log_scaling_n_floor", 1, CTX_MAX, &g->log_floor) &&
         meta_real(m, ARCH ".log_scaling_alpha", &g->log_alpha) && meta_real(m, ARCH ".qk_clip_threshold", &g->q_clip) &&
         meta_real(m, ARCH ".rel_clip_threshold", &g->rel_clip) &&
         meta_uint_in(m, ARCH ".shortconv_kernel", 1, INK2_CONV_TAPS_MAX, &g->conv) &&
         meta_real(m, ARCH ".expert_weights_scale", &g->route_scale) &&
         meta_real(m, ARCH ".mlp_gate_softcap", &g->gate_softcap) &&
         meta_real(m, ARCH ".mlp_up_softcap", &g->up_softcap) &&
         (meta_flag_set(m, ARCH ".attn_res_normalize_values") ||
          meta_fail(m, ARCH ".attn_res_normalize_values", "not set: attn_res's values not normalized")) &&
         (meta_flag_set(m, ARCH ".value_norm") || meta_fail(m, ARCH ".value_norm", "not set: v not normalized"));
  g->k_clip = g->q_clip;
  return ok;
}

/* upstream llama.cpp's keys (unsloth's Inkling-2-Mini): the relative bias's sizes, the log scale and the
   clips under attention., q's and k's clip one; the MoE's activation SiTU, its betas the softcaps (beta tanh(g / beta)
   sigmoid(g) * lb tanh(u / lb)); the router's sigmoid and normalized weights, the normalized values the architecture's
   (no keys); no RoPE; one head width for the sliding and the full layers */
static bool upstream_keys(GgufMeta *m, Ink2Geometry *g)
{
  g->keys = INK2_KEYS_UPSTREAM;
  const bool ok = slot_flags(m, g) && meta_uint_in(m, ARCH ".vocab_size_unpadded", 1, DIM_MAX, &g->n_vocab) &&
         meta_uint_in(m, ARCH ".attention.relative_length", 1, DIM_MAX, &g->d_rel) &&
         meta_uint_in(m, ARCH ".attention.relative_extent", 1, CTX_MAX, &g->rel_extent) &&
         meta_uint_in(m, ARCH ".attention.log_scale_floor", 1, CTX_MAX, &g->log_floor) &&
         meta_real(m, ARCH ".attention.log_scale_alpha", &g->log_alpha) &&
         meta_real(m, ARCH ".attention.qk_clip", &g->q_clip) &&
         meta_real(m, ARCH ".attention.relative_clip", &g->rel_clip) &&
         meta_uint_in(m, ARCH ".shortconv.l_cache", 2, INK2_CONV_TAPS_MAX, &g->conv) &&
         meta_real(m, ARCH ".activation.situ_beta", &g->gate_softcap) &&
         meta_real(m, ARCH ".activation.situ_linear_beta", &g->up_softcap) &&
         meta_real(m, ARCH ".expert_weights_scale", &g->route_scale) &&
         (!meta_present(m, ARCH ".rope.dimension_count") || meta_uint_is(m, ARCH ".rope.dimension_count", 0)) &&
         (!meta_present(m, ARCH ".attention.key_length_swa") ||
          meta_uint_is(m, ARCH ".attention.key_length_swa", (uint64_t)g->head_dim)) &&
         (!meta_present(m, ARCH ".attention.value_length_swa") ||
          meta_uint_is(m, ARCH ".attention.value_length_swa", (uint64_t)g->head_dim));
  g->k_clip = g->q_clip;
  return ok;
}

static bool ours(GgufMeta *m, Ink2Geometry *g);

bool ink2_geometry(const Gguf *model, Ink2Geometry *g, char err[OSMOS_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int ctx = 0;
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 1, INK2_MAX_LAYERS, &g->n_slots) ||
      !meta_uint_in(&m, ARCH ".attn_res.block_size", 1, g->n_slots, &g->res_block) || !loop(&m, g))
    return false;
  const bool common =
      meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
      meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
      meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
      slot_uint(&m, g, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
      slot_uint(&m, g, ARCH ".attention.head_count_kv", 1, g->n_heads, &g->n_kv_heads) &&
      (g->n_heads % g->n_kv_heads == 0 || meta_fail(&m, ARCH ".attention.head_count_kv", "not a heads' divisor")) &&
      meta_uint_in(&m, ARCH ".attention.key_length", 1, DIM_MAX, &g->head_dim) &&
      meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
      meta_uint_in(&m, ARCH ".attention.sliding_window", 1, CTX_MAX, &g->window) &&
      (g->res_block <= g->n_layers || meta_fail(&m, ARCH ".attn_res.block_size", "past the layers")) &&
      meta_uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
      meta_uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
      meta_uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->ff_exp) &&
      meta_real(&m, ARCH ".final_logit_softcapping", &g->logit_softcap);
  const bool ok = common && (meta_present(&m, ARCH ".attention.relative_length") ? upstream_keys(&m, g)
                             : meta_present(&m, ARCH ".d_rel")                   ? unsloth_small_keys(&m, g)
                                                                                 : ours(&m, g));
  g->native_ctx = (uint32_t)ctx;
  return ok;
}

/* the sliding layers a list; the rest full - at least one */
static bool kinds(GgufMeta *m, Ink2Geometry *g)
{
  int n = 0;
  if (!meta_layers(m, ARCH ".attention.sliding_layers", g->n_layers, g->sliding, &n))
    return false;
  return n < g->n_layers || meta_fail(m, ARCH ".attention.sliding_layers", "every layer sliding: no full layer");
}

/* ink2_ref_gguf.py's keys */
static bool ours(GgufMeta *m, Ink2Geometry *g)
{
  g->keys = INK2_KEYS_OURS;
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
