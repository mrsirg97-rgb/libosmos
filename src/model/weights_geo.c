#include <stdio.h>
#include <string.h>

#include "weights_geo.h"

#define ARCH "deepseek41"
#define DIM_MAX (1 << 24)
#define LIST_MAX WEIGHTS_MAX_LAYERS
#define GATING_SQRTSOFTPLUS 4 /* deepseek41.expert_gating_func, the one this build routes by */
#define NGRAM_MIN 2

typedef struct Meta
{
  const Gguf *g;
  char *err;
} Meta;

static bool fail(Meta *m, const char *key, const char *why)
{
  snprintf(m->err, WEIGHTS_ERR, "%s: %s", key, why);
  return false;
}

static const GgufKv *find(Meta *m, const char *key, GgufValueType type)
{
  const GgufKv *kv = gguf_kv(m->g, key);
  if (!kv)
    return fail(m, key, "missing"), NULL;
  if (kv->type != type)
    return fail(m, key, "of another value type"), NULL;
  return kv;
}

static bool uint_in(Meta *m, const char *key, int lo, int hi, int *out)
{
  const GgufKv *kv = find(m, key, GGUF_UINT32);
  if (!kv)
    return false;
  if (kv->v.u < (uint64_t)lo || kv->v.u > (uint64_t)hi)
    return fail(m, key, "out of range");
  *out = (int)kv->v.u;
  return true;
}

static bool uint_is(Meta *m, const char *key, uint64_t want)
{
  const GgufKv *kv = find(m, key, GGUF_UINT32);
  return kv && (kv->v.u == want || fail(m, key, "a value this build does not run"));
}

static bool flag_set(Meta *m, const char *key)
{
  const GgufKv *kv = find(m, key, GGUF_BOOL);
  return kv && (kv->v.u || fail(m, key, "false; this build runs it true"));
}

static bool real(Meta *m, const char *key, float *out)
{
  const GgufKv *kv = find(m, key, GGUF_FLOAT32);
  if (!kv)
    return false;
  if (!(kv->v.f > 0))
    return fail(m, key, "not positive");
  *out = (float)kv->v.f;
  return true;
}

static bool text_is(Meta *m, const char *key, const char *want)
{
  const GgufKv *kv = find(m, key, GGUF_STRING);
  return kv && (strcmp(kv->v.s, want) == 0 || fail(m, key, "a value this build does not run"));
}

/* an array of integers: n (at least min, at most cap) into out, each read at its own width */
static bool ints(Meta *m, const char *key, size_t min, size_t cap, int64_t *out, size_t *n)
{
  const GgufKv *kv = find(m, key, GGUF_ARRAY);
  if (!kv)
    return false;
  if (kv->n < min || kv->n > cap)
    return fail(m, key, "of a length this model does not have");
  for (size_t i = 0; i < kv->n; i++)
    switch (kv->item)
    {
    case GGUF_INT32: out[i] = ((const int32_t *)kv->v.items)[i]; break;
    case GGUF_UINT32: out[i] = ((const uint32_t *)kv->v.items)[i]; break;
    case GGUF_INT64: out[i] = ((const int64_t *)kv->v.items)[i]; break;
    case GGUF_UINT64:
      if (((const uint64_t *)kv->v.items)[i] > INT64_MAX)
        return fail(m, key, "out of range");
      out[i] = (int64_t)((const uint64_t *)kv->v.items)[i];
      break;
    default: return fail(m, key, "not of integers");
    }
  *n = kv->n;
  return true;
}

/* a list of layers, each within them and once, as flags */
static bool layers(Meta *m, const char *key, int n_layers, bool *set, int *count)
{
  int64_t v[LIST_MAX];
  size_t n = 0;
  if (!ints(m, key, 1, LIST_MAX, v, &n))
    return false;
  for (size_t i = 0; i < n; i++)
  {
    if (v[i] < 0 || v[i] >= n_layers)
      return fail(m, key, "a layer past the model's");
    if (set[v[i]])
      return fail(m, key, "a layer twice");
    set[v[i]] = true;
  }
  if (count)
    *count = (int)n;
  return true;
}

static bool clamps(Meta *m, const char *key, int n_layers, float *out)
{
  const GgufKv *kv = find(m, key, GGUF_ARRAY);
  if (!kv)
    return false;
  if (kv->item != GGUF_FLOAT32 || kv->n < (uint64_t)n_layers)
    return fail(m, key, "not a float a layer");
  for (int i = 0; i < n_layers; i++)
    if (!((out[i] = ((const float *)kv->v.items)[i]) > 0))
      return fail(m, key, "not positive");
  return true;
}

/* compression: a KV source compresses; every compressed layer reads the last source of its ratio at or before it */
static bool sources(Meta *m, Geometry *g)
{
  int64_t r[LIST_MAX];
  size_t n = 0;
  if (!ints(m, ARCH ".attention.compress_ratios", (size_t)g->n_layers, LIST_MAX, r, &n))
    return false;
  for (int l = 0; l < g->n_layers; l++)
  {
    if (r[l] < 0 || r[l] > DIM_MAX)
      return fail(m, ARCH ".attention.compress_ratios", "out of range");
    g->compress[l] = (int)r[l];
    if (g->kv_source[l] && !g->compress[l])
      return fail(m, ARCH ".kv_source_layer_ids", "an uncompressed layer");
    if (g->ix_source[l] && !g->compress[l])
      return fail(m, ARCH ".index_source_layer_ids", "an uncompressed layer");
    g->kv_from[l] = -1;
    for (int s = l; g->compress[l] && s >= 0 && g->kv_from[l] < 0; s--)
      if (g->kv_source[s] && g->compress[s] == g->compress[l])
        g->kv_from[l] = s;
    if (g->compress[l] && g->kv_from[l] < 0)
      return fail(m, ARCH ".attention.compress_ratios", "a compressed layer with no KV source of its ratio before it");
  }
  if (!g->kv_source[g->cand_source])
    return fail(m, ARCH ".attention.candidate_source_layer_id", "not a KV source");
  return true;
}

static bool engram(Meta *m, const Gguf *file, Geometry *g)
{
  if (!flag_set(m, ARCH ".engram.present"))
    return false;
  if (!file)
    return fail(m, ARCH ".engram.file", "no Engram file given");
  int n_eg = 0;
  int64_t rows[LIST_MAX];
  size_t n_rows = 0;
  if (!layers(m, ARCH ".engram.layer_ids", g->n_layers, g->engram, &n_eg) ||
      !ints(m, ARCH ".engram.num_embeddings", (size_t)n_eg, (size_t)n_eg, rows, &n_rows) ||
      !uint_in(m, ARCH ".engram_head_dim", 1, DIM_MAX, &g->engram_head_dim) ||
      !uint_in(m, ARCH ".engram_n_heads", 1, DIM_MAX, &g->engram_heads) ||
      !uint_in(m, ARCH ".engram_max_ngram_size", NGRAM_MIN, LIST_MAX, &g->engram_max_ngram))
    return false;
  for (int l = 0, i = 0; l < g->n_layers; l++)
    if (g->engram[l] && (g->engram_rows[l] = (uint64_t)rows[i++]) == 0)
      return fail(m, ARCH ".engram.num_embeddings", "an empty table");
  g->engram_cols = (g->engram_max_ngram - 1) * g->engram_heads;
  const GgufKv *src = find(m, ARCH ".engram.source", GGUF_STRING);
  const GgufKv *theirs = gguf_kv(file, ARCH ".engram.source");
  if (!src || !theirs || theirs->type != GGUF_STRING || strcmp(src->v.s, theirs->v.s) != 0)
    return fail(m, ARCH ".engram.source", "the Engram file is not this model's");
  const GgufKv *arch = gguf_kv(file, "general.architecture");
  if (!arch || arch->type != GGUF_STRING || strcmp(arch->v.s, ARCH "-engram") != 0)
    return fail(m, "general.architecture", "the Engram file is not " ARCH "-engram");
  return true;
}

bool weights_geometry(const Gguf *model, const Gguf *file, Geometry *g, char err[WEIGHTS_ERR])
{
  Meta m = {model, err};
  memset(g, 0, sizeof *g);
  const int L = WEIGHTS_MAX_LAYERS;
  if (!text_is(&m, "general.architecture", ARCH) || !uint_in(&m, ARCH ".block_count", 1, L, &g->n_layers))
    return false;
  const int n = g->n_layers;
  return uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
         uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
         uint_is(&m, ARCH ".attention.head_count_kv", 1) &&
         uint_in(&m, ARCH ".attention.key_length", 1, DIM_MAX, &g->head_dim) &&
         uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
         uint_in(&m, ARCH ".rope.dimension_count", 1, g->head_dim, &g->rope_dim) &&
         uint_in(&m, ARCH ".attention.q_lora_rank", 1, DIM_MAX, &g->q_lora) &&
         uint_in(&m, ARCH ".attention.output_lora_rank", 1, DIM_MAX, &g->o_lora) &&
         uint_in(&m, ARCH ".attention.output_group_count", 1, g->n_heads, &g->o_groups) &&
         (g->n_heads % g->o_groups == 0 || fail(&m, ARCH ".attention.output_group_count", "not a heads' divisor")) &&
         uint_in(&m, ARCH ".attention.sliding_window", 1, DIM_MAX, &g->window) &&
         uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
         uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
         uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->expert_ff) &&
         uint_is(&m, ARCH ".expert_shared_count", 1) && uint_is(&m, ARCH ".hash_layer_count", 0) &&
         uint_is(&m, ARCH ".expert_gating_func", GATING_SQRTSOFTPLUS) &&
         text_is(&m, ARCH ".expert_scoring_func", "sqrtsoftplus") &&
         flag_set(&m, ARCH ".expert_weights_norm") &&
         real(&m, ARCH ".expert_weights_scale", &g->expert_scale) &&
         uint_in(&m, ARCH ".hyper_connection.count", 1, DIM_MAX, &g->hc) &&
         uint_in(&m, ARCH ".hyper_connection.sinkhorn_iterations", 1, DIM_MAX, &g->sinkhorn_iters) &&
         real(&m, ARCH ".hyper_connection.epsilon", &g->hc_eps) &&
         uint_in(&m, ARCH ".attention.indexer.head_count", 1, DIM_MAX, &g->ix_heads) &&
         uint_in(&m, ARCH ".attention.indexer.key_length", 1, DIM_MAX, &g->ix_head_dim) &&
         uint_in(&m, ARCH ".attention.indexer.top_k", 1, DIM_MAX, &g->ix_topk) &&
         uint_in(&m, ARCH ".attention.candidate_topk_blocks", 1, DIM_MAX, &g->cand_blocks) &&
         uint_in(&m, ARCH ".attention.candidate_block_size", 1, DIM_MAX, &g->cand_block) &&
         uint_in(&m, ARCH ".attention.candidate_source_layer_id", 0, n - 1, &g->cand_source) &&
         layers(&m, ARCH ".kv_source_layer_ids", n, g->kv_source, NULL) &&
         layers(&m, ARCH ".index_source_layer_ids", n, g->ix_source, NULL) &&
         layers(&m, ARCH ".dspark_target_layer_ids", n, g->dspark_target, NULL) && sources(&m, g) &&
         clamps(&m, ARCH ".swiglu_clamp_exp", n, g->clamp_exp) &&
         clamps(&m, ARCH ".swiglu_clamp_shexp", n, g->clamp_shexp) &&
         real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
         real(&m, ARCH ".rope.freq_base", &g->rope_base) &&
         real(&m, ARCH ".attention.compress_rope_freq_base", &g->compress_rope_base) &&
         text_is(&m, ARCH ".rope.scaling.type", "yarn") && real(&m, ARCH ".rope.scaling.factor", &g->rope_factor) &&
         uint_in(&m, ARCH ".rope.scaling.original_context_length", 1, DIM_MAX, &g->rope_original) &&
         real(&m, ARCH ".rope.scaling.yarn_beta_fast", &g->yarn_beta_fast) &&
         real(&m, ARCH ".rope.scaling.yarn_beta_slow", &g->yarn_beta_slow) && engram(&m, file, g);
}
