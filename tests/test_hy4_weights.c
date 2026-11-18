/* Hy4-flash's weights (osmos/hy4/weights.h) on its GGUF (llama.cpp PR #32517's layout): the geometry the release's
   config.json states (tencent/Hy4-flash: 42 layers, swa_pattern, MLA's dims, the indexer, the latent MoE, Engram),
   each layer's kind its tensors, then the file edited in memory - a key out of range or another router, a tensor
   missing, of another shape or type, a tensor no role takes - each refused, its cause named. OSMOS_TEST_HY4: the
   GGUF, else ~/Models/hy4-flash/Hy4-flash-SPARK-4Bit.gguf. the tensors' data is never read: a sparse file of the
   header alone is enough (`head -c 6000000 FILE > H; truncate -s SIZE H`). skipped without */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/hy4/weights.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"

#define T_Q8_0 8
#define T_BF16 30
#define MODEL "%s/Models/hy4-flash/Hy4-flash-SPARK-4Bit.gguf"
#define ARCH "hy_v4_flash."

/* config.json's text_config: swa_pattern (1 a sliding layer), engram_layer_ids (1-based: before 0-based id - 1) */
static const int swa_pattern[42] = {0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
                                    0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
static const int engram_layers[4] = {1, 11, 21, 31};

static Gguf *model;

static GgufTensor *tensor(const char *name)
{
  GgufTensor *t = (GgufTensor *)gguf_tensor(model, name);
  assert(t);
  return t;
}

static GgufKv *kv(const char *key)
{
  GgufKv *k = (GgufKv *)gguf_kv(model, key);
  assert(k);
  return k;
}

static bool accepted(void)
{
  char err[HY4_ERR] = "";
  Hy4Weights *w = hy4_weights_create(model, err);
  if (!w)
    printf("    refused: %s\n", err);
  hy4_weights_destroy(w);
  return w != NULL;
}

static void refused(const char *what, const char *needle)
{
  char err[HY4_ERR] = "";
  Hy4Weights *w = hy4_weights_create(model, err);
  if (w || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, w ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-52s -> %s\n", what, err);
}

static bool is_engram(int l)
{
  for (int i = 0; i < 4; i++)
    if (engram_layers[i] == l)
      return true;
  return false;
}

static void test_real(const Hy4Weights *w)
{
  const Hy4Geometry *g = &w->geo;
  assert(g->n_layers == 42 && g->n_nextn == 1 && g->n_embd == 3072 && g->n_vocab == 120832 && g->n_ff == 13824);
  assert(g->native_ctx == 1048576 && g->rms_eps > 0.99e-5f && g->rms_eps < 1.01e-5f);
  assert(g->hc == 4 && g->hc_magnitude == 2.0f && g->hc_eps > 0.99e-6f && g->hc_eps < 1.01e-6f);
  assert(g->q_lora == 1024 && g->kv_lora == 512 && g->nope_dim == 192 && g->rope_dim == 64 && g->v_dim == 256);
  assert(g->window == 512 && g->rope_base == 1e7f && g->rope_base_swa == 1e4f);
  assert(g->ix_heads == 16 && g->ix_dim == 128 && g->ix_topk == 2048 && g->ix_pool == 4);
  assert(g->n_dense == 1 && g->n_experts == 512 && g->n_used == 16 && g->ff_exp == 1536 && g->latent == 1536);
  assert(g->n_shexp == 1 && g->expert_scale > 3.9979f && g->expert_scale < 3.9981f);
  assert(g->n_engram == 4 && g->engram_ngram == 3 && g->engram_heads == 8 && g->engram_dim == 96);
  assert(g->engram_pad == 120002);
  int sliding = 0;
  for (int l = 0; l < g->n_layers; l++)
  {
    const Hy4Layer *L = &w->layer[l];
    assert(g->sliding[l] == (bool)swa_pattern[l] && g->n_heads[l] == (swa_pattern[l] ? 64 : 32));
    assert(g->engram[l] == is_engram(l) && g->clamp[l] == 10.0f);
    sliding += g->sliding[l];
    /* every layer: iHC twice, MLA with its gate, sinks and sandwich norms */
    assert(L->hc_attn_fn && L->hc_attn_base && L->hc_attn_scale && L->hc_ffn_fn && L->hc_ffn_base && L->hc_ffn_scale);
    assert(L->attn_norm && L->q_a && L->q_a_norm && L->q_b && L->kv_a && L->kv_a_norm && L->k_b && L->v_b);
    assert(L->gate && L->out && L->sinks && L->post_attn_norm && L->ffn_norm && L->post_ffn_norm);
    assert(L->k_b->dims[2] == (uint64_t)g->n_heads[l] && L->sinks->dims[0] == (uint64_t)g->n_heads[l]);
    /* a sparse layer's indexer, a sliding one's none */
    const bool ix = L->ix_q_b && L->ix_k && L->ix_k_norm && L->ix_proj && L->ix_pool_value && L->ix_pool_gate &&
                    L->ix_pool_ape && L->ix_pool_norm;
    assert(g->sliding[l] ? !L->ix_q_b && !L->ix_k && !L->ix_pool_ape : ix);
    /* layer 0 dense, the rest the latent MoE */
    if (l < g->n_dense)
      assert(L->ffn_gate && L->ffn_up && L->ffn_down && !L->router && !L->exp_gate);
    else
      assert(!L->ffn_gate && L->router && L->router_bias && L->exp_gate && L->exp_up && L->exp_down &&
             L->routed_down && L->routed_up && L->sh_gate && L->sh_up && L->sh_down &&
             L->exp_up->type == L->exp_gate->type);
    /* Engram's tensors on its layers alone */
    const bool eg = L->eg_kv && L->eg_kv_b && L->eg_gate && L->eg_lookup_norm && L->eg_pre_gamma && L->eg_hc_scale;
    assert(g->engram[l] ? eg : !L->eg_kv && !L->eg_gate);
  }
  assert(sliding == 30);
  assert(w->embed && w->output && w->output_norm && w->out_hc_fn && w->out_hc_base && w->out_hc_scale);
  assert(w->engram_table && w->engram_table->dims[0] == 96u);
  printf("  %d layers (%d sliding, %d sparse) + %d nextn, %d wide, MLA %d/%d latent, heads 32/64; %d experts of %d "
         "(latent %d), %d a token; Engram before layers 1, 11, 21, 31\n",
         g->n_layers, sliding, g->n_layers - sliding, g->n_nextn, g->n_embd, g->q_lora, g->kv_lora, g->n_experts,
         g->ff_exp, g->latent, g->n_used);
}

static void edit_u(const char *key, uint64_t bad, const char *what)
{
  GgufKv *k = kv(key);
  const uint64_t was = k->v.u;
  k->v.u = bad;
  refused(what, key + strlen(ARCH));
  k->v.u = was;
}

static void test_kv(void)
{
  printf("  metadata refused:\n");
  GgufKv *k = kv("general.architecture");
  k->v.s[0] = 'X';
  refused("another architecture", "general.architecture");
  k->v.s[0] = 'h';
  edit_u(ARCH "block_count", 65, "65 blocks");
  edit_u(ARCH "expert_gating_func", 1, "a softmax router");
  edit_u(ARCH "expert_weights_norm", 0, "the picked experts' weights not normalized");
  edit_u(ARCH "hyper_connection.count", 5, "five residual streams");
  edit_u(ARCH "attention.indexer.kpool", 3, "a pooled group not dividing the indexer's keys");
  edit_u(ARCH "attention.key_length", 577, "a cache row not the latent and its rope");
  k = kv(ARCH "leading_dense_block_count");
  k->v.u = 2;
  refused("two dense layers (layer 1's tensors a MoE's)", "blk.1.ffn_gate.weight: missing");
  k->v.u = 1;
  k = kv(ARCH "ple.layers");
  int32_t *items = (int32_t *)k->v.items, was = items[3];
  items[3] = 42;
  refused("an Engram layer past the trunk", "ple.layers");
  items[3] = was;
  assert(accepted());
}

static void test_tensors(void)
{
  printf("  tensors refused:\n");
  GgufTensor *t = tensor("blk.0.attn_q_a.weight");
  t->name[0] = 'X';
  refused("a layer's q_a missing", "blk.0.attn_q_a.weight: missing");
  t->name[0] = 'b';

  t = tensor("blk.2.attn_q_b.weight");
  t->dims[1] /= 2;
  refused("a sliding layer's q_b of a sparse one's heads", "blk.2.attn_q_b.weight: of another shape");
  t->dims[1] *= 2;

  t = tensor("blk.1.indexer.attn_k.weight");
  t->name[0] = 'X';
  refused("a sparse layer's indexer missing", "blk.1.indexer.attn_k.weight: missing");
  t->name[0] = 'b';

  t = tensor("blk.0.attn_norm.weight");
  uint32_t ty = t->type;
  t->type = T_Q8_0;
  refused("a norm in Q8_0", "blk.0.attn_norm.weight: a type its reader");
  t->type = ty;

  t = tensor("blk.3.ffn_gate_exps.weight");
  ty = t->type;
  t->type = T_BF16;
  refused("the routed experts in a type their path does not read", "blk.3.ffn_gate_exps.weight: a type");
  t->type = ty;

  t = tensor("blk.3.ffn_routed_down.weight");
  t->dims[1]++;
  refused("the latent's down projection past the latent", "blk.3.ffn_routed_down.weight: of another shape");
  t->dims[1]--;

  t = tensor("blk.5.exp_probs_b.bias");
  t->name[0] = 'X';
  refused("a router's bias missing", "blk.5.exp_probs_b.bias: missing");
  t->name[0] = 'b';

  /* a tensor no role takes: the drafter's (left to it) renamed as an indexer norm a sliding layer does not have */
  t = tensor("blk.42.nextn.enorm.weight");
  char *was = t->name;
  t->name = (char *)"blk.2.indexer.k_norm.weight";
  refused("an indexer tensor on a sliding layer", "blk.2.indexer.k_norm.weight: a tensor no role");
  t->name = was;

  t = tensor("per_layer_token_embd.weight");
  t->dims[1]--;
  refused("the Engram table a row short of its heads'", "per_layer_token_embd.weight");
  t->dims[1]++;

  t = tensor("output_hc_fn.weight");
  t->name[0] = 'X';
  refused("the head's iHC missing", "output_hc_fn.weight: missing");
  t->name[0] = 'o';
  assert(accepted());
}

int main(void)
{
  printf("[test_hy4_weights] start\n");
  char err[HY4_ERR] = "", path[512], gerr[GGUF_ERR] = "";
  assert(!hy4_weights_create(NULL, err) && err[0]);
  hy4_weights_destroy(NULL);
  if (getenv("OSMOS_TEST_HY4"))
    snprintf(path, sizeof path, "%s", getenv("OSMOS_TEST_HY4"));
  else
    snprintf(path, sizeof path, MODEL, getenv("HOME") ? getenv("HOME") : "");
  if (!(model = gguf_create(path, gerr)))
  {
    printf("  Hy4-flash's GGUF: not on this machine, skipped (%s)\n", gerr);
    printf("[test_hy4_weights] ok\n");
    return 0;
  }
  Hy4Weights *w = hy4_weights_create(model, err);
  if (!w)
    printf("  refused: %s\n", err);
  assert(w);
  test_real(w);
  hy4_weights_destroy(w);
  test_kv();
  test_tensors();
  gguf_destroy(model);
  printf("[test_hy4_weights] ok\n");
  return 0;
}
