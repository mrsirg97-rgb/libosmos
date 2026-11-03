#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "substrate/model/gguf.h"
#include "weights_geo.h"

/* the geometry from unsloth's UD-IQ1_M (llama.cpp's spelling, the Engram in the model's own files): equal to one
   written by hand from its metadata (DESIGN.md, step 4a); the same geometry with its keys renamed to ds1017's
   spelling; a key in both spellings equal is taken, unequal refused; a key missing refused. needs the UD file;
   skipped without it */

#define A "deepseek41."
#define UD_LAYERS 40

static Gguf *model;
static int failures;

static void check(bool ok, const char *what)
{
  if (!ok)
  {
    fprintf(stderr, "  %s\n", what);
    failures++;
  }
}

static GgufKv *kv(const char *key) { return (GgufKv *)gguf_kv(model, key); }

/* the geometry by hand: the UD file's metadata (all 62 deepseek41.* keys read 2026-11-03) */
static Geometry by_hand(void)
{
  Geometry g;
  memset(&g, 0, sizeof g);
  g.n_layers = UD_LAYERS, g.n_embd = 5120, g.n_heads = 64, g.head_dim = 512, g.rope_dim = 64, g.q_lora = 1280;
  g.o_lora = 1024, g.o_groups = 8, g.window = 128, g.n_experts = 384, g.n_used = 6, g.expert_ff = 2304, g.hc = 4;
  g.sinkhorn_iters = 20, g.ix_heads = 32, g.ix_head_dim = 128, g.ix_topk = 512, g.cand_blocks = 2048;
  g.cand_block = 8, g.cand_source = 20;
  const int ratios[UD_LAYERS] = {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int l = 0; l < UD_LAYERS; l++)
  {
    g.compress[l] = ratios[l], g.clamp_exp[l] = g.clamp_shexp[l] = 10.0f;
    /* the last KV source of the layer's ratio at or before it: 2 for ratio 2 (2 .. 7), 8, 14, 20 */
    g.kv_from[l] = !ratios[l] ? -1 : ratios[l] == 2 ? (l < 8 ? 2 : l < 14 ? 8 : 14) : 20;
  }
  for (int l = 2; l <= 20; l += 6)
    g.kv_source[l] = true;
  for (int l = 2; l <= 20; l += 6)
    g.ix_source[l] = true;
  for (int l = 24; l <= 36; l += 4)
    g.ix_source[l] = true;
  g.dspark_target[37] = g.dspark_target[38] = g.dspark_target[39] = true;
  g.engram[1] = g.engram[14] = true;
  g.engram_rows[1] = 384006168, g.engram_rows[14] = 384016682; /* each layer's 24 columns' span; 768,022,850 in all */
  g.engram_heads = 8, g.engram_head_dim = 256, g.engram_max_ngram = 4, g.engram_cols = 24, g.engram_pad = 2;
  g.engram_vocab = 99092, g.engram_own = true;
  g.rms_eps = 1e-20f, g.hc_eps = 1e-6f, g.rope_base = 10000.0f, g.compress_rope_base = 160000.0f;
  g.rope_factor = 16.0f, g.rope_original = 65536, g.yarn_beta_fast = 32.0f, g.yarn_beta_slow = 1.0f;
  g.expert_scale = 1.5f;
  return g;
}

static bool read(Geometry *g, char err[WEIGHTS_ERR])
{
  err[0] = '\0';
  return weights_geometry(model, NULL, g, err);
}

static void same(const Geometry *a, const Geometry *b, const char *what)
{
  if (memcmp(a, b, sizeof *a) == 0)
    return;
  const int *x = (const int *)a, *y = (const int *)b; /* the first differing word, for the reader */
  size_t i = 0;
  while (x[i] == y[i])
    i++;
  fprintf(stderr, "  %s: the geometries differ at byte %zu (%d against %d)\n", what, i * sizeof(int), x[i], y[i]);
  failures++;
}

static void refused(const char *what, const char *needle)
{
  Geometry g;
  char err[WEIGHTS_ERR];
  const bool ok = read(&g, err);
  check(!ok && strstr(err, needle), what);
  if (!ok)
    printf("    %-56s -> %s\n", what, err);
}

/* the pairs of spellings: ds1017's (maya's pack), then the UD file's (llama.cpp's) */
static const char *const kPairs[][2] = {
    {A "kv_source_layer_ids", A "attention.kv_source_layers"},
    {A "index_source_layer_ids", A "attention.indexer.source_layers"},
    {A "dspark_target_layer_ids", A "dspark.target_layers"},
    {A "attention.candidate_topk_blocks", A "attention.indexer.candidate_top_k_blocks"},
    {A "attention.candidate_block_size", A "attention.indexer.candidate_block_size"},
    {A "attention.candidate_source_layer_id", A "attention.indexer.candidate_source_layer"},
};
#define N_PAIRS (sizeof kPairs / sizeof kPairs[0])

static void rename_key(const char *from, const char *to)
{
  GgufKv *k = kv(from);
  if (!k)
  {
    fprintf(stderr, "  no key %s to rename\n", from);
    exit(1);
  }
  free(k->key);
  k->key = strdup(to);
}

int main(void)
{
  printf("[test_geo] start\n");
  char path[512], err[GGUF_ERR];
  snprintf(path, sizeof path, "%s/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf",
           getenv("HOME") ? getenv("HOME") : "");
  if (!(model = gguf_create(path, err)))
  {
    printf("[test_geo] skipped: no UD-IQ1_M (%s)\n", err);
    return 0;
  }
  const Geometry want = by_hand();
  Geometry got;
  char why[WEIGHTS_ERR];
  const bool ok = read(&got, why);
  check(ok, why);
  if (ok)
    same(&got, &want, "UD-IQ1_M's geometry against the hand-written one");
  if (!failures)
    printf("  UD-IQ1_M: the geometry by hand, the Engram's from ple.* (rows 384,006,168 + 384,016,682)\n");

  /* each pair renamed to ds1017's spelling: the same geometry */
  for (size_t i = 0; i < N_PAIRS; i++)
    rename_key(kPairs[i][1], kPairs[i][0]);
  check(read(&got, why), why);
  same(&got, &want, "ds1017's spelling");
  for (size_t i = 0; i < N_PAIRS; i++)
    rename_key(kPairs[i][0], kPairs[i][1]);
  printf("  ds1017's spelling of the %zu pairs: the same geometry\n", N_PAIRS);

  /* a scalar in both spellings: equal taken, unequal refused (an unread key renamed into the other spelling) */
  GgufKv *ctx = kv(A "context_length");
  const uint64_t ctx_was = ctx->v.u;
  rename_key(A "context_length", A "attention.candidate_source_layer_id");
  ctx->v.u = 20;
  check(read(&got, why), "both spellings, equal: refused");
  same(&got, &want, "both spellings, equal");
  ctx->v.u = 14;
  refused("candidate source: both spellings, unequal", "attention.candidate_source_layer_id");
  rename_key(A "attention.candidate_source_layer_id", A "context_length");
  ctx->v.u = ctx_was;
  /* an array in both: a list of another length */
  rename_key("tokenizer.ggml.token_type", A "kv_source_layer_ids");
  refused("KV sources: both spellings, unequal", "kv_source_layer_ids");
  rename_key(A "kv_source_layer_ids", "tokenizer.ggml.token_type");

  /* each key the geometry reads, missing: refused, the key named */
  const char *const needed[] = {A "attention.kv_source_layers", A "attention.indexer.candidate_source_layer",
                                A "ple.layers", A "ple.head_offsets", A "ple.head_vocab_sizes", A "ple.ngram_size",
                                A "ple.heads_per_ngram", A "embedding_length_per_layer_input",
                                A "engram.pad_token_id", A "engram.compressed_vocab_size", A "expert_gating_func"};
  for (size_t i = 0; i < sizeof needed / sizeof needed[0]; i++)
  {
    char gone[160];
    snprintf(gone, sizeof gone, "x.%s", needed[i]);
    rename_key(needed[i], gone);
    char what[200];
    snprintf(what, sizeof what, "missing %s", needed[i] + strlen(A));
    refused(what, needed[i] + strlen(A));
    rename_key(gone, needed[i]);
  }

  /* the scoring function where present: sqrtsoftplus only; the gating function 4 only */
  GgufKv *gate = kv(A "expert_gating_func");
  gate->v.u = 2;
  refused("gating function 2 (sigmoid)", "expert_gating_func");
  gate->v.u = 4;
  /* the Engram's columns' span: an offset past its layer's columns refused */
  GgufKv *off = kv(A "ple.head_offsets");
  uint64_t *o = off->v.items, was = o[24];
  o[24] = o[23]; /* the second layer's first column overlapping the first layer's last */
  refused("Engram: a layer's columns overlapping the one before", "ple.head_offsets");
  o[24] = was;
  check(read(&got, why), "the file restored: refused");
  same(&got, &want, "the file restored");

  gguf_destroy(model);
  if (failures)
  {
    fprintf(stderr, "[test_geo] %d failures\n", failures);
    return 1;
  }
  printf("[test_geo] ok\n");
  return 0;
}
