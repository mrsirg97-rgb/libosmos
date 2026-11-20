#include <stdio.h>
#include <string.h>

#include "gguf_meta.h"
#include "osmos/hy4/weights.h"

#define ARCH "hy_v4_flash"
#define DIM_MAX (1 << 24)
#define CTX_MAX (1 << 30)
#define SIGMOID_GATING 2 /* llama.cpp's expert_gating_func: 1 softmax, 2 sigmoid */
_Static_assert(HY4_ERR == GGUF_META_ERR && HY4_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");

/* a float a layer, at least n of them (the list covers block_count: the trunk and the nextn layers) */
static bool floats(GgufMeta *m, const char *key, int n, float *out)
{
  const GgufKv *k = meta_find(m, key, GGUF_ARRAY);
  if (!k)
    return false;
  if (k->item != GGUF_FLOAT32 || k->n < (uint64_t)n)
    return meta_fail(m, key, "not a float a layer");
  for (int i = 0; i < n; i++)
    out[i] = ((const float *)k->v.items)[i];
  return true;
}

/* a layer's heads, the attention kinds and the indexer's layers: a sliding layer's window, a sparse one's indexer -
   one or the other, every trunk layer */
static bool layer_kinds(GgufMeta *m, Hy4Geometry *g, int blocks)
{
  int64_t heads[HY4_MAX_LAYERS];
  bool sliding[HY4_MAX_LAYERS], indexed[HY4_MAX_LAYERS];
  size_t n = 0;
  if (!meta_ints(m, ARCH ".attention.head_count", (size_t)blocks, HY4_MAX_LAYERS, heads, &n) ||
      !meta_flags(m, ARCH ".attention.sliding_window_pattern", (size_t)blocks, sliding))
    return false;
  const GgufKv *ix = meta_find(m, ARCH ".attention.indexer.types", GGUF_ARRAY);
  if (!ix)
    return false;
  if (ix->item != GGUF_BOOL || ix->n < (uint64_t)g->n_layers || ix->n > (uint64_t)blocks)
    return meta_fail(m, ARCH ".attention.indexer.types", "not a flag a trunk layer");
  for (int l = 0; l < g->n_layers; l++)
  {
    indexed[l] = ((const uint8_t *)ix->v.items)[l] != 0;
    if (heads[l] < 1 || heads[l] > DIM_MAX)
      return meta_fail(m, ARCH ".attention.head_count", "a layer's heads out of range");
    if (indexed[l] == sliding[l])
      return meta_fail(m, ARCH ".attention.indexer.types", "a layer both sliding and indexed, or neither");
    g->n_heads[l] = (int)heads[l];
    g->sliding[l] = sliding[l];
  }
  return true;
}

static bool engram(GgufMeta *m, Hy4Geometry *g)
{
  int pad = 0;
  if (!meta_layers(m, ARCH ".ple.layers", g->n_layers, g->engram, &g->n_engram) ||
      !meta_uint_in(m, ARCH ".ple.ngram_size", 2, ENGRAM_MAX_NGRAM, &g->engram_ngram) ||
      !meta_uint_in(m, ARCH ".ple.heads_per_ngram", 1, DIM_MAX, &g->engram_heads) ||
      !meta_uint_in(m, ARCH ".embedding_length_per_layer_input", 1, DIM_MAX, &g->engram_dim) ||
      !meta_uint_in(m, ARCH ".engram.pad_token_id", 0, g->n_vocab - 1, &pad))
    return false;
  if (g->n_engram < 1 || g->n_engram > HY4_MAX_ENGRAM)
    return meta_fail(m, ARCH ".ple.layers", "no Engram layer, or more than this build keeps");
  g->engram_pad = pad;
  return true;
}

bool hy4_geometry(const Gguf *model, Hy4Geometry *g, char err[HY4_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int blocks = 0, ctx = 0, key = 0, key_mla = 0, gating = 0;
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 2, HY4_MAX_LAYERS, &blocks) ||
      (meta_present(&m, ARCH ".nextn_predict_layers") &&
       !meta_uint_in(&m, ARCH ".nextn_predict_layers", 0, blocks - 1, &g->n_nextn)))
    return false;
  g->n_layers = blocks - g->n_nextn;
  const bool ok =
      meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
      meta_uint_in(&m, ARCH ".vocab_size", 1, DIM_MAX, &g->n_vocab) &&
      meta_uint_in(&m, ARCH ".feed_forward_length", 1, DIM_MAX, &g->n_ff) &&
      meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
      meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
      meta_uint_in(&m, ARCH ".hyper_connection.count", 1, HY4_MAX_HC, &g->hc) &&
      meta_real(&m, ARCH ".hyper_connection.epsilon", &g->hc_eps) &&
      meta_real(&m, ARCH ".hyper_connection.magnitude", &g->hc_magnitude) &&
      /* MLA: one latent geometry, the cache's row the latent and its rope (MQA: one KV head) */
      meta_uint_is(&m, ARCH ".attention.head_count_kv", 1) &&
      meta_uint_in(&m, ARCH ".attention.q_lora_rank", 1, DIM_MAX, &g->q_lora) &&
      meta_uint_in(&m, ARCH ".attention.kv_lora_rank", 1, DIM_MAX, &g->kv_lora) &&
      meta_uint_in(&m, ARCH ".rope.dimension_count", 2, DIM_MAX, &g->rope_dim) &&
      (g->rope_dim % 2 == 0 || meta_fail(&m, ARCH ".rope.dimension_count", "odd: not pairs")) &&
      meta_uint_in(&m, ARCH ".attention.key_length", 1, DIM_MAX, &key) &&
      (key == g->kv_lora + g->rope_dim || meta_fail(&m, ARCH ".attention.key_length", "not the latent and its rope")) &&
      meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->kv_lora) &&
      meta_uint_in(&m, ARCH ".attention.key_length_mla", g->rope_dim + 1, DIM_MAX, &key_mla) &&
      meta_uint_in(&m, ARCH ".attention.value_length_mla", 1, DIM_MAX, &g->v_dim) &&
      (g->nope_dim = key_mla - g->rope_dim) > 0 &&
      meta_uint_in(&m, ARCH ".attention.sliding_window", 1, CTX_MAX, &g->window) &&
      meta_real(&m, ARCH ".rope.freq_base", &g->rope_base) &&
      meta_real(&m, ARCH ".rope.freq_base_swa", &g->rope_base_swa) &&
      layer_kinds(&m, g, blocks) &&
      /* the indexer: its queries' and keys' rope on their last rope_dim dims; a pooled group dividing its keys */
      meta_uint_in(&m, ARCH ".attention.indexer.head_count", 1, DIM_MAX, &g->ix_heads) &&
      meta_uint_in(&m, ARCH ".attention.indexer.key_length", g->rope_dim, DIM_MAX, &g->ix_dim) &&
      meta_uint_in(&m, ARCH ".attention.indexer.top_k", 1, CTX_MAX, &g->ix_topk) &&
      meta_uint_in(&m, ARCH ".attention.indexer.kpool", 2, g->ix_topk, &g->ix_pool) &&
      (g->ix_topk % g->ix_pool == 0 ||
       meta_fail(&m, ARCH ".attention.indexer.kpool", "a pooled group not dividing the indexer's keys")) &&
      /* the MLP: dense first, the latent MoE after - sigmoid routing, the picked weights normalized */
      meta_uint_in(&m, ARCH ".leading_dense_block_count", 0, g->n_layers, &g->n_dense) &&
      meta_uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
      meta_uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
      meta_uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->ff_exp) &&
      meta_uint_in(&m, ARCH ".expert_latent_length", 1, DIM_MAX, &g->latent) &&
      meta_uint_in(&m, ARCH ".expert_shared_count", 1, DIM_MAX, &g->n_shexp) &&
      meta_uint_in(&m, ARCH ".expert_gating_func", 0, DIM_MAX, &gating) &&
      (gating == SIGMOID_GATING || meta_fail(&m, ARCH ".expert_gating_func", "not the sigmoid router")) &&
      (meta_flag_set(&m, ARCH ".expert_weights_norm") ||
       meta_fail(&m, ARCH ".expert_weights_norm", "the picked experts' weights not normalized")) &&
      meta_real(&m, ARCH ".expert_weights_scale", &g->expert_scale) &&
      floats(&m, ARCH ".swiglu_clamp_exp", g->n_layers, g->clamp) &&
      engram(&m, g);
  if (!ok)
    return false;
  g->native_ctx = (uint32_t)ctx;
  return true;
}
