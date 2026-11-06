#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf_meta.h"
#include "weights_geo.h"

#define ARCH "deepseek41"
#define DIM_MAX (1 << 24)
#define LIST_MAX WEIGHTS_MAX_LAYERS
_Static_assert(WEIGHTS_ERR == GGUF_META_ERR && WEIGHTS_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");
#define GATING_SQRTSOFTPLUS 4 /* deepseek41.expert_gating_func, the one this build routes by */
#define NGRAM_MIN 2

/* a field's two spellings: ds1017's (maya's pack), then llama.cpp's (unsloth's UD). one geometry, read from either;
   both present and unequal refuses */
static const char *const kAliases[][2] = {
    {ARCH ".kv_source_layer_ids", ARCH ".attention.kv_source_layers"},
    {ARCH ".index_source_layer_ids", ARCH ".attention.indexer.source_layers"},
    {ARCH ".dspark_target_layer_ids", ARCH ".dspark.target_layers"},
    {ARCH ".attention.candidate_topk_blocks", ARCH ".attention.indexer.candidate_top_k_blocks"},
    {ARCH ".attention.candidate_block_size", ARCH ".attention.indexer.candidate_block_size"},
    {ARCH ".attention.candidate_source_layer_id", ARCH ".attention.indexer.candidate_source_layer"},
};
#define N_ALIASES (sizeof kAliases / sizeof kAliases[0])

static bool clamps(GgufMeta *m, const char *key, int n_layers, float *out)
{
  const GgufKv *kv = meta_find(m, key, GGUF_ARRAY);
  if (!kv)
    return false;
  if (kv->item != GGUF_FLOAT32 || kv->n < (uint64_t)n_layers)
    return meta_fail(m, key, "not a float a layer");
  for (int i = 0; i < n_layers; i++)
    if (!((out[i] = ((const float *)kv->v.items)[i]) > 0))
      return meta_fail(m, key, "not positive");
  return true;
}

/* compression: a KV source compresses; every compressed layer reads the last source of its ratio at or before it */
static bool sources(GgufMeta *m, Geometry *g)
{
  int64_t r[LIST_MAX];
  size_t n = 0;
  if (!meta_ints(m, ARCH ".attention.compress_ratios", (size_t)g->n_layers, LIST_MAX, r, &n))
    return false;
  for (int l = 0; l < g->n_layers; l++)
  {
    if (r[l] < 0 || r[l] > DIM_MAX)
      return meta_fail(m, ARCH ".attention.compress_ratios", "out of range");
    g->compress[l] = (int)r[l];
    if (g->kv_source[l] && !g->compress[l])
      return meta_fail(m, ARCH ".kv_source_layer_ids", "an uncompressed layer");
    if (g->ix_source[l] && !g->compress[l])
      return meta_fail(m, ARCH ".index_source_layer_ids", "an uncompressed layer");
    g->kv_from[l] = -1;
    for (int s = l; g->compress[l] && s >= 0 && g->kv_from[l] < 0; s--)
      if (g->kv_source[s] && g->compress[s] == g->compress[l])
        g->kv_from[l] = s;
    if (g->compress[l] && g->kv_from[l] < 0)
      return meta_fail(m, ARCH ".attention.compress_ratios", "a compressed layer with no KV source of its ratio before it");
  }
  if (!g->kv_source[g->cand_source])
    return meta_fail(m, ARCH ".attention.candidate_source_layer_id", "not a KV source");
  return true;
}

static bool engram(GgufMeta *m, const Gguf *file, Geometry *g)
{
  if (!meta_flag_set(m, ARCH ".engram.present"))
    return false;
  if (!file)
    return meta_fail(m, ARCH ".engram.file", "no Engram file given");
  int n_eg = 0;
  int64_t rows[LIST_MAX];
  size_t n_rows = 0;
  if (!meta_layers(m, ARCH ".engram.layer_ids", g->n_layers, g->engram, &n_eg) ||
      !meta_ints(m, ARCH ".engram.num_embeddings", (size_t)n_eg, (size_t)n_eg, rows, &n_rows) ||
      !meta_uint_in(m, ARCH ".engram_head_dim", 1, DIM_MAX, &g->engram_head_dim) ||
      !meta_uint_in(m, ARCH ".engram_n_heads", 1, DIM_MAX, &g->engram_heads) ||
      !meta_uint_in(m, ARCH ".engram_max_ngram_size", NGRAM_MIN, LIST_MAX, &g->engram_max_ngram) ||
      !meta_uint_in(m, ARCH ".engram_pad_token_id", 0, DIM_MAX, &g->engram_pad) ||
      !meta_uint_in(m, ARCH ".engram_compressed_vocab_size", 1, DIM_MAX, &g->engram_vocab))
    return false;
  for (int l = 0, i = 0; l < g->n_layers; l++)
    if (g->engram[l] && (g->engram_rows[l] = (uint64_t)rows[i++]) == 0)
      return meta_fail(m, ARCH ".engram.num_embeddings", "an empty table");
  g->engram_cols = (g->engram_max_ngram - 1) * g->engram_heads;
  const GgufKv *src = meta_find(m, ARCH ".engram.source", GGUF_STRING);
  const GgufKv *theirs = gguf_kv(file, ARCH ".engram.source");
  if (!src || !theirs || theirs->type != GGUF_STRING || strcmp(src->v.s, theirs->v.s) != 0)
    return meta_fail(m, ARCH ".engram.source", "the Engram file is not this model's");
  const GgufKv *arch = gguf_kv(file, "general.architecture");
  if (!arch || arch->type != GGUF_STRING || strcmp(arch->v.s, ARCH "-engram") != 0)
    return meta_fail(m, "general.architecture", "the Engram file is not " ARCH "-engram");
  return true;
}

/* llama.cpp's Engram (unsloth's UD): its keys, the tables the model's own (per_layer_token_embd, weights.c), a layer's
   rows its columns' span. the columns tile the tensor: each one's offset the sizes before it, the layers in order */
static bool ple_engram(GgufMeta *m, const Gguf *file, Geometry *g)
{
  if (file)
    return meta_fail(m, ARCH ".ple.layers", "the Engram is the model's own: no Engram file");
  int64_t ids[LIST_MAX];
  size_t n_eg = 0;
  if (!meta_ints(m, ARCH ".ple.layers", 1, LIST_MAX, ids, &n_eg) ||
      !meta_uint_in(m, ARCH ".ple.heads_per_ngram", 1, DIM_MAX, &g->engram_heads) ||
      !meta_uint_in(m, ARCH ".ple.ngram_size", NGRAM_MIN, LIST_MAX, &g->engram_max_ngram) ||
      !meta_uint_in(m, ARCH ".embedding_length_per_layer_input", 1, DIM_MAX, &g->engram_head_dim) ||
      !meta_uint_in(m, ARCH ".engram.pad_token_id", 0, DIM_MAX, &g->engram_pad) ||
      (meta_present(m, ARCH ".engram.compressed_vocab_size") && /* ISTA's GGUFs: absent, 0 - the map's (engram.c) */
       !meta_uint_in(m, ARCH ".engram.compressed_vocab_size", 1, DIM_MAX, &g->engram_vocab)))
    return false;
  for (size_t i = 0; i < n_eg; i++)
  {
    if (ids[i] < 0 || ids[i] >= g->n_layers)
      return meta_fail(m, ARCH ".ple.layers", "a layer past the model's");
    if (i && ids[i] <= ids[i - 1])
      return meta_fail(m, ARCH ".ple.layers", "not in ascending order");
    g->engram[ids[i]] = true;
  }
  g->engram_cols = (g->engram_max_ngram - 1) * g->engram_heads;
  g->engram_own = true;
  const size_t n = n_eg * (size_t)g->engram_cols;
  int64_t *off = malloc(n * sizeof *off), *size = malloc(n * sizeof *size);
  size_t n_off = 0, n_size = 0;
  bool ok = off && size && meta_ints(m, ARCH ".ple.head_offsets", n, n, off, &n_off) &&
            meta_ints(m, ARCH ".ple.head_vocab_sizes", n, n, size, &n_size);
  if (!off || !size)
    meta_fail(m, ARCH ".ple.head_offsets", "out of memory");
  uint64_t at = 0;
  for (size_t i = 0; ok && i < n; i++)
  {
    if (size[i] <= 0)
      ok = meta_fail(m, ARCH ".ple.head_vocab_sizes", "an empty column");
    else if ((uint64_t)off[i] != at)
      ok = meta_fail(m, ARCH ".ple.head_offsets", "a column not right after the one before");
    else
      at += (uint64_t)size[i];
  }
  for (size_t i = 0; ok && i < n_eg; i++)
  {
    const size_t c0 = i * (size_t)g->engram_cols, c1 = c0 + (size_t)g->engram_cols - 1;
    g->engram_rows[ids[i]] = (uint64_t)(off[c1] + size[c1] - off[c0]);
  }
  free(off);
  free(size);
  return ok;
}

bool weights_geometry(const Gguf *model, const Gguf *file, Geometry *g, char err[WEIGHTS_ERR])
{
  GgufMeta m = {model, err, kAliases, N_ALIASES};
  memset(g, 0, sizeof *g);
  const int L = WEIGHTS_MAX_LAYERS;
  if (!meta_text_is(&m, "general.architecture", ARCH) || !meta_uint_in(&m, ARCH ".block_count", 1, L, &g->n_layers))
    return false;
  const int n = g->n_layers;
  return meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
         meta_uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
         meta_uint_is(&m, ARCH ".attention.head_count_kv", 1) &&
         meta_uint_in(&m, ARCH ".attention.key_length", 1, DIM_MAX, &g->head_dim) &&
         meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
         meta_uint_in(&m, ARCH ".rope.dimension_count", 1, g->head_dim, &g->rope_dim) &&
         meta_uint_in(&m, ARCH ".attention.q_lora_rank", 1, DIM_MAX, &g->q_lora) &&
         meta_uint_in(&m, ARCH ".attention.output_lora_rank", 1, DIM_MAX, &g->o_lora) &&
         meta_uint_in(&m, ARCH ".attention.output_group_count", 1, g->n_heads, &g->o_groups) &&
         (g->n_heads % g->o_groups == 0 || meta_fail(&m, ARCH ".attention.output_group_count", "not a heads' divisor")) &&
         meta_uint_in(&m, ARCH ".attention.sliding_window", 1, DIM_MAX, &g->window) &&
         meta_uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
         meta_uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
         meta_uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->expert_ff) &&
         meta_uint_is(&m, ARCH ".expert_shared_count", 1) && meta_uint_is(&m, ARCH ".hash_layer_count", 0) &&
         meta_uint_is(&m, ARCH ".expert_gating_func", GATING_SQRTSOFTPLUS) &&
         (!gguf_kv(model, ARCH ".expert_scoring_func") || meta_text_is(&m, ARCH ".expert_scoring_func", "sqrtsoftplus")) &&
         meta_flag_set(&m, ARCH ".expert_weights_norm") &&
         meta_real(&m, ARCH ".expert_weights_scale", &g->expert_scale) &&
         meta_uint_in(&m, ARCH ".hyper_connection.count", 1, DIM_MAX, &g->hc) &&
         meta_uint_in(&m, ARCH ".hyper_connection.sinkhorn_iterations", 1, DIM_MAX, &g->sinkhorn_iters) &&
         meta_real(&m, ARCH ".hyper_connection.epsilon", &g->hc_eps) &&
         meta_uint_in(&m, ARCH ".attention.indexer.head_count", 1, DIM_MAX, &g->ix_heads) &&
         meta_uint_in(&m, ARCH ".attention.indexer.key_length", 1, DIM_MAX, &g->ix_head_dim) &&
         meta_uint_in(&m, ARCH ".attention.indexer.top_k", 1, DIM_MAX, &g->ix_topk) &&
         meta_uint_in(&m, ARCH ".attention.candidate_topk_blocks", 1, DIM_MAX, &g->cand_blocks) &&
         meta_uint_in(&m, ARCH ".attention.candidate_block_size", 1, DIM_MAX, &g->cand_block) &&
         meta_uint_in(&m, ARCH ".attention.candidate_source_layer_id", 0, n - 1, &g->cand_source) &&
         meta_layers(&m, ARCH ".kv_source_layer_ids", n, g->kv_source, NULL) &&
         meta_layers(&m, ARCH ".index_source_layer_ids", n, g->ix_source, NULL) &&
         (!meta_present(&m, ARCH ".dspark_target_layer_ids") || /* ISTA's GGUFs name none: the drafter's then */
          meta_layers(&m, ARCH ".dspark_target_layer_ids", n, g->dspark_target, NULL)) &&
         sources(&m, g) &&
         clamps(&m, ARCH ".swiglu_clamp_exp", n, g->clamp_exp) &&
         clamps(&m, ARCH ".swiglu_clamp_shexp", n, g->clamp_shexp) &&
         meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
         meta_real(&m, ARCH ".rope.freq_base", &g->rope_base) &&
         meta_real(&m, ARCH ".attention.compress_rope_freq_base", &g->compress_rope_base) &&
         meta_text_is(&m, ARCH ".rope.scaling.type", "yarn") && meta_real(&m, ARCH ".rope.scaling.factor", &g->rope_factor) &&
         meta_uint_in(&m, ARCH ".rope.scaling.original_context_length", 1, DIM_MAX, &g->rope_original) &&
         meta_real(&m, ARCH ".rope.scaling.yarn_beta_fast", &g->yarn_beta_fast) &&
         meta_real(&m, ARCH ".rope.scaling.yarn_beta_slow", &g->yarn_beta_slow) &&
         (gguf_kv(model, ARCH ".ple.layers")       ? ple_engram(&m, file, g)
          : gguf_kv(model, ARCH ".engram.present") ? engram(&m, file, g)
                                                   : meta_fail(&m, ARCH ".ple.layers (or " ARCH ".engram.present)", "missing"));
}
