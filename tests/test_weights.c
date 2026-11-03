#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "substrate/model/gguf.h"
#include "substrate/model/qtype.h"
#include "substrate/model/weights.h"

#define T_BF16 30
#define T_Q8_0 8
#define T_IQ2_XXS 16
#define FUZZ 300

static Gguf *model, *engram;

static Gguf *open_model(const char *name)
{
  char p[512], err[GGUF_ERR];
  snprintf(p, sizeof p, "%s/models/ds1017/%s", getenv("HOME"), name);
  return gguf_create(p, err);
}

static GgufTensor *tensor(Gguf *g, const char *name) { return (GgufTensor *)gguf_tensor(g, name); }
static GgufKv *kv(const char *key) { return (GgufKv *)gguf_kv(model, key); }

static bool accepted(void)
{
  char err[WEIGHTS_ERR];
  Weights *w = weights_create(model, engram, err);
  weights_destroy(w);
  return w != NULL;
}

/* refused, and why names `needle` */
static void refused(const char *what, const char *needle)
{
  char err[WEIGHTS_ERR] = {0};
  Weights *w = weights_create(model, engram, err);
  if (w || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, w ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-44s -> %s\n", what, err);
}

static void test_null(void)
{
  char err[WEIGHTS_ERR];
  assert(weights_create(NULL, NULL, err) == NULL && err[0]);
  weights_destroy(NULL);
  printf("  null: no model refused; destroy(NULL) nothing\n");
}

static void test_real(void)
{
  char err[WEIGHTS_ERR];
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  Weights *w = weights_create(model, engram, err);
  clock_gettime(CLOCK_MONOTONIC, &b);
  if (!w)
    printf("  real: refused: %s\n", err);
  assert(w);
  const Geometry *g = &w->geo;
  assert(g->n_layers == 40 && g->n_vocab == 129280 && g->n_embd == 5120 && g->n_heads == 64 && g->head_dim == 512);
  assert(g->n_experts == 384 && g->n_used == 6 && g->expert_ff == 2304 && g->hc == 4 && g->engram_cols == 24);
  assert(g->compress[1] == 0 && g->compress[2] == 2 && g->compress[19] == 2 && g->compress[20] == 1);
  assert(g->kv_from[0] == -1 && g->kv_from[7] == 2 && g->kv_from[8] == 8 && g->kv_from[39] == 20);
  assert(g->ix_source[24] && !g->kv_source[24] && g->kv_source[20] && g->engram[1] && g->engram[14] && !g->engram[2]);
  assert(g->engram_rows[1] == 384006168u && g->engram_rows[14] == 384016682u && g->dspark_target[39]);
  assert(g->rms_eps > 0 && g->rms_eps < 1e-19f && g->expert_scale == 1.5f);

  const LayerWeights *l = w->layer;
  assert(l[2].comp_gate && !l[20].comp_gate && l[20].comp_kv && !l[3].comp_kv);
  assert(l[24].ix_q_b && !l[24].ix_k && l[2].ix_k && !l[25].ix_q_b);
  assert(l[1].eg_embd && l[1].eg_embd == gguf_tensor(engram, "blk.1.engram_embd.weight") && !l[0].eg_q);
  assert(l[0].exp_gate == gguf_tensor(model, "blk.0.ffn_gate_exps.weight") && w->embed && w->eg_multipliers);
  for (int i = 0; i < g->n_layers; i++)
    assert(l[i].q_a && l[i].hc_ffn_fn && l[i].exp_down && l[i].sh_down && l[i].router_bias);

  /* the experts: every slice where its tensor says, the slices tiling the tensors exactly */
  const int fds[4] = {40, 41, 42, 43};
  const size_t n = weights_experts(w, fds, NULL, 0);
  assert(n == 40u * 384u);
  PoolUnit *u = calloc(n, sizeof *u);
  assert(u && weights_experts(w, fds, u, 10) == n && u[10].n_parts == 0);
  assert(weights_experts(w, fds, u, n) == n);
  uint64_t got = 0, want = 0;
  for (size_t i = 0; i < n; i++)
  {
    const LayerWeights *li = &l[i / 384];
    const GgufTensor *t[3] = {li->exp_gate, li->exp_up, li->exp_down};
    assert(u[i].n_parts == 3);
    for (int k = 0; k < 3; k++)
    {
      const uint64_t slice = t[k]->bytes / 384;
      assert(u[i].part[k].fd == fds[t[k]->file] && u[i].part[k].bytes == slice);
      assert(u[i].part[k].offset == t[k]->offset + (i % 384) * slice);
      got += u[i].part[k].bytes;
    }
  }
  for (int i = 0; i < 40; i++)
    want += l[i].exp_gate->bytes + l[i].exp_up->bytes + l[i].exp_down->bytes;
  assert(got == want);
  free(u);
  weights_destroy(w);
  printf("  real: ds1017 whole - 40 layers, every role, %zu experts tiling their tensors (%.1f GB); in %.2f ms\n",
         n, (double)got / 1e9, (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) / 1e6);
}

static void test_tensors(void)
{
  printf("  tensors refused:\n");
  GgufTensor *t = tensor(model, "blk.5.attn_q_b.weight");
  t->name[0] = 'X';
  refused("a layer's q_b missing", "blk.5.attn_q_b");
  t->name[0] = 'b';

  t = tensor(model, "blk.2.attn_compressor_gate.weight");
  t->name[4] = '3'; /* blk.3: compressed, but reading layer 2's KV - no compressor of its own */
  refused("a compressor gate moved off its source", "compressor_gate");
  t->name[4] = '2';

  t = tensor(model, "blk.5.attn_q_b.weight");
  t->dims[1]--;
  refused("q_b one row short", "blk.5.attn_q_b");
  t->dims[1]++;
  t->n_dims = 3;
  refused("q_b of 3 dims", "blk.5.attn_q_b");
  t->n_dims = 2;

  t = tensor(model, "blk.5.attn_norm.weight");
  t->type = T_BF16;
  refused("a norm in BF16", "blk.5.attn_norm");
  t->type = 0;

  t = tensor(model, "blk.9.ffn_up_exps.weight");
  const uint32_t ty = t->type;
  t->type = T_Q8_0;
  refused("experts in Q8_0", "blk.9.ffn_up_exps");
  t->type = T_IQ2_XXS; /* allowed for the role, but not of these bytes */
  refused("experts of a type their bytes are not", "blk.9.ffn_up_exps");
  t->type = ty;

  t = tensor(engram, "blk.14.engram_embd.weight");
  t->dims[1]++;
  refused("an Engram table a row long", "blk.14.engram_embd");
  t->dims[1]--;

  t = tensor(model, "output_norm.weight");
  t->name[0] = 'X';
  refused("output_norm missing", "output_norm");
  t->name[0] = 'o';
  assert(accepted());
}

static void test_kv(void)
{
  printf("  metadata refused:\n");
  GgufKv *k = kv("deepseek41.block_count");
  k->v.u = 41;
  refused("41 layers (40 clamps)", "swiglu_clamp_exp");
  k->v.u = 65;
  refused("65 layers", "block_count");
  k->v.u = 40;

  k = kv("deepseek41.expert_count");
  k->key[0] = 'X';
  refused("no expert_count", "expert_count");
  k->key[0] = 'd';

  k = kv("deepseek41.expert_scoring_func");
  const char c = k->v.s[0];
  k->v.s[0] = 'X';
  refused("another scoring function", "expert_scoring_func");
  k->v.s[0] = c;

  k = kv("deepseek41.kv_source_layer_ids");
  uint32_t *ids = k->v.items;
  ids[0] = 1;
  refused("an uncompressed KV source", "kv_source_layer_ids");
  ids[0] = 99;
  refused("a KV source past the layers", "kv_source_layer_ids");
  ids[0] = 2;

  k = kv("deepseek41.index_source_layer_ids");
  ids = k->v.items;
  ids[1] = 1;
  refused("an uncompressed index source", "index_source_layer_ids");
  ids[1] = 8;

  k = kv("deepseek41.engram.layer_ids");
  ids = k->v.items;
  ids[1] = 15; /* layer 14's Engram tensors where the metadata says none */
  refused("Engram tensors on a layer without", "blk.14.engram_q.weight: present");
  ids[1] = 14;

  k = kv("deepseek41.attention.compress_ratios");
  int32_t *r = k->v.items;
  r[1] = 2; /* layer 1 compressed, with no KV source at or before it */
  refused("a compressed layer with no KV source", "compress_ratios");
  r[1] = 0;

  k = kv("deepseek41.attention.head_count_kv");
  k->v.u = 2;
  refused("two KV heads", "head_count_kv");
  k->v.u = 1;

  k = kv("deepseek41.engram.source");
  k->v.s[0] ^= 1;
  refused("an Engram file of another model", "engram.source");
  k->v.s[0] ^= 1;

  char err[WEIGHTS_ERR];
  assert(weights_create(model, NULL, err) == NULL && strstr(err, "engram"));
  printf("    %-44s -> %s\n", "no Engram file", err);
  assert(accepted());
}

/* a tensor no role names, with nothing missing */
static void test_unknown(void)
{
  GgufTensor *grown = realloc(engram->tensors, (engram->n_tensors + 1) * sizeof *grown);
  assert(grown);
  engram->tensors = grown;
  grown[engram->n_tensors] = grown[0];
  grown[engram->n_tensors].name = strdup("blk.1.engram_extra.weight");
  engram->n_tensors++;
  printf("  unknown:\n");
  refused("a tensor no role names", "blk.1.engram_extra.weight: a tensor this build does not know");
  engram->n_tensors--;
  free(grown[engram->n_tensors].name);
  assert(accepted());
}

/* one random change to a tensor (a name, a dim, a type), the model refused every time; then put back */
static void test_fuzz(void)
{
  srand(1017);
  for (int i = 0; i < FUZZ; i++)
  {
    Gguf *g = rand() % 8 ? model : engram;
    GgufTensor *t = &g->tensors[(size_t)rand() % g->n_tensors];
    const GgufTensor keep = *t;
    char name0 = t->name[rand() % strlen(t->name)];
    char *at = strchr(t->name, name0);
    switch (rand() % 4)
    {
    case 0: *at ^= 0x20; break;
    case 1: t->dims[rand() % t->n_dims] += 1 + (uint64_t)(rand() % 3); break;
    case 2: t->type = t->type == T_BF16 ? T_Q8_0 : T_BF16; break;
    default: t->n_dims = t->n_dims == 1 ? 2 : t->n_dims - 1; break;
    }
    char err[WEIGHTS_ERR];
    Weights *w = weights_create(model, engram, err);
    if (w)
      printf("  fuzz %d: ACCEPTED a changed %s\n", i, keep.name);
    assert(!w);
    *at = name0;
    *t = keep;
  }
  assert(accepted());
  printf("  fuzz: %d random changes to a tensor, every one refused\n", FUZZ);
}

/* unsloth's UD-IQ1_M: llama.cpp's names and its own types, the Engram in the model's files - every role by the path
   that reads it (DESIGN.md, step 4b) */
static void test_ud(void)
{
  char p[512], err[WEIGHTS_ERR];
  snprintf(p, sizeof p, "%s/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf",
           getenv("HOME"));
  Gguf *ud = gguf_create(p, err);
  if (!ud)
  {
    printf("  UD-IQ1_M: not on this machine, skipped\n");
    return;
  }
  model = ud, engram = NULL;
  Weights *w = weights_create(model, NULL, err);
  if (!w)
    printf("  UD-IQ1_M: refused: %s\n", err);
  assert(w);
  const Geometry *g = &w->geo;
  assert(g->engram_own && g->n_vocab == 129280 && g->engram[1] && g->engram[14]);
  const LayerWeights *l = w->layer;
  assert(w->embed->type == QTYPE_Q8_0 && w->output->type == QTYPE_Q6_K && !w->eg_token_map && !w->eg_primes);
  assert(l[0].router->type == QTYPE_F32 && l[0].q_a->type == QTYPE_Q6_K && l[0].out_a->n_dims == 2);
  assert(l[0].exp_gate->type == QTYPE_IQ1_S && l[20].exp_down->type == QTYPE_IQ2_S);
  assert(l[1].eg_gate && !l[1].eg_q && !l[1].eg_k && l[1].eg_wkv == gguf_tensor(model, "blk.1.engram_kv.weight"));
  assert(!l[0].eg_gate && !l[0].eg_embd && !l[2].eg_wkv);
  for (int i = 0; i < g->n_layers; i++)
    assert(l[i].q_a && l[i].out_a && l[i].hc_ffn_fn && l[i].exp_down && l[i].sh_down && l[i].router_bias);
  /* the layers' tables: slices of per_layer_token_embd in layer order, tiling it */
  const GgufTensor *ple = gguf_tensor(model, "per_layer_token_embd.weight");
  const uint64_t row = gguf_bytes(ple->type, ple->dims[0]);
  const GgufTensor *a = l[1].eg_embd, *b = l[14].eg_embd;
  assert(a && b && a != ple && a->type == ple->type && a->file == ple->file && b->file == ple->file);
  assert(a->dims[0] == ple->dims[0] && a->dims[1] == g->engram_rows[1] && b->dims[1] == g->engram_rows[14]);
  assert(a->offset == ple->offset && b->offset == a->offset + a->dims[1] * row && a->bytes == a->dims[1] * row);
  assert(a->bytes + b->bytes == ple->bytes);
  assert(weights_experts(w, (const int[5]){0}, NULL, 0) == 40u * 384u);
  weights_destroy(w);
  printf("  UD-IQ1_M: every role by its path (Q8_0 embedding, F32 router, 2-dim out_a, IQ1_S / IQ2_S experts), the "
         "gate folded, the tables two slices tiling per_layer_token_embd\n");

  printf("  UD-IQ1_M refused:\n");
  GgufTensor *t = tensor(model, "blk.3.attn_output_a.weight");
  t->type = QTYPE_BF16; /* the matmul takes no BF16 */
  refused("out_a in BF16 (no prompt matmul)", "blk.3.attn_output_a");
  t->type = QTYPE_Q6_K;
  t = tensor(model, "blk.14.engram_gate.weight");
  t->type = QTYPE_BF16;
  refused("the folded gate in BF16", "blk.14.engram_gate");
  t->type = QTYPE_F32;
  t = tensor(model, "per_layer_token_embd.weight");
  t->dims[1]++;
  refused("the Engram tables a row longer than the layers'", "per_layer_token_embd");
  t->dims[1]--;
  t = tensor(model, "blk.6.ffn_gate_inp.weight");
  t->type = QTYPE_I32;
  refused("a router of I32 (no matvec)", "blk.6.ffn_gate_inp");
  t->type = QTYPE_F32;
  t = tensor(model, "blk.0.ffn_up_exps.weight");
  t->type = QTYPE_IQ2_XXS; /* a type the experts' paths take, not of these bytes */
  refused("IQ1_S experts called IQ2_XXS", "blk.0.ffn_up_exps");
  t->type = QTYPE_IQ1_S;
  t = tensor(model, "blk.7.exp_probs_b_vl.bias");
  t->dims[0]--;
  refused("the image routing bias a row short", "blk.7.exp_probs_b_vl");
  t->dims[0]++;
  t = tensor(model, "blk.2.engram_kv.weight");
  assert(!t); /* not an Engram layer: none */
  t = tensor(model, "blk.1.engram_kv.weight");
  t->name[4] = '2';
  refused("engram_kv on a layer with no Engram", "engram_kv");
  t->name[4] = '1';
  assert(accepted());
  gguf_destroy(ud);
  model = NULL;
}

int main(void)
{
  printf("[test_weights] start\n");
  test_null();
  model = open_model("DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf");
  engram = open_model("DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf");
  if (!model || !engram)
    printf("  ds1017: not on this machine, skipped\n");
  else
  {
    test_real();
    test_tensors();
    test_kv();
    test_unknown();
    test_fuzz();
  }
  gguf_destroy(model);
  gguf_destroy(engram);
  test_ud();
  printf("[test_weights] ok\n");
  return 0;
}
