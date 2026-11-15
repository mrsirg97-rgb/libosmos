/* Inkling-2's weights (osmos/ink2/weights.h) on its reference GGUF (tools/convert/ink2_ref_gguf.py): the geometry its
   metadata says, each layer's kind its tensors, each NVFP4 expert tensor's scale beside it, then the file edited in
   memory - a key out of range or another router, a tensor missing, of another shape or type, a scale gone - each
   refused, its cause named. OSMOS_TEST_INK2: the GGUF (its first shard), else ~/Models/inkling2/ref's; tools/convert/
   test_oracle.py --out DIR then ink2_ref_gguf.py --src DIR --out DIR/ref writes a small one (DIR/ref.gguf). skipped
   without */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"

#define T_Q8_0 8
#define REF "%s/Models/inkling2/ref/ink2-ref-00001-of-00009.gguf"

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
  char err[INK2_ERR] = "";
  Ink2Weights *w = ink2_weights_create(model, err);
  if (!w)
    printf("    refused: %s\n", err);
  ink2_weights_destroy(w);
  return w != NULL;
}

static void refused(const char *what, const char *needle)
{
  char err[INK2_ERR] = "";
  Ink2Weights *w = ink2_weights_create(model, err);
  if (w || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, w ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-48s -> %s\n", what, err);
}

/* the first layer whose experts are of this type, or -1 */
static int layer_of(const Ink2Weights *w, uint32_t type)
{
  for (int l = 0; l < w->geo.n_layers; l++)
    if (w->layer[l].exp_gate->type == type)
      return l;
  return -1;
}

static void test_real(const Ink2Weights *w)
{
  const Ink2Geometry *g = &w->geo;
  int full = 0, nv = 0, mx = 0;
  for (int l = 0; l < g->n_layers; l++)
  {
    const Ink2Layer *L = &w->layer[l];
    full += !g->sliding[l];
    /* a layer's kind its tensors: a full one's three input norms and its convs' projections, a sliding one's one */
    assert(g->sliding[l] ? L->attn_norm && L->k_conv && !L->q_norm && !L->k_conv_proj
                         : L->q_norm && L->k_norm && L->v_norm && L->k_conv_proj && L->v_conv_static && !L->attn_norm);
    assert(l == 0 ? !L->attn_res_query : L->attn_res_query && L->attn_res_query->n_dims == (g->sliding[l] ? 1u : 2u));
    const bool is_nv = L->exp_gate->type == QTYPE_NVFP4;
    assert(is_nv == (L->exp_gate_scale && L->exp_up_scale && L->exp_down_scale));
    assert(L->exp_up->type == L->exp_gate->type);
    nv += is_nv, mx += L->exp_gate->type == QTYPE_MXFP8;
  }
  assert(full > 0 && g->n_vocab <= g->n_vocab_rows);
  assert(ink2_weights_experts(w, NULL, NULL, 0) == (size_t)g->n_layers * (size_t)g->n_experts);
  printf("  %d layers (%d full), %d wide, %d heads (%d KV) of %d, window %d, relative %d x %d; %d experts of %d, %d "
         "a token: %d layers NVFP4 (a scale an expert), %d MXFP8; vocabulary %d of %d rows\n",
         g->n_layers, full, g->n_embd, g->n_heads, g->n_kv_heads, g->head_dim, g->window, g->d_rel, g->rel_extent,
         g->n_experts, g->ff_exp, g->n_used, nv, mx, g->n_vocab, g->n_vocab_rows);
}

static void test_kv(void)
{
  printf("  metadata refused:\n");
  GgufKv *k = kv("general.architecture");
  k->v.s[0] = 'X';
  refused("another architecture", "general.architecture");
  k->v.s[0] = 'i';

  const struct
  {
    const char *key, *what;
    uint64_t bad;
  } bad[] = {
      {"inkling2.block_count", "65 layers", 65},
      {"inkling2.expert_gating_func", "a softmax router", 1},
      {"inkling2.conv_kernel", "a convolution's taps past 8", 9},
      {"inkling2.attention.head_count_kv", "KV heads not the heads' divisor", 7},
      {"inkling2.expert_used_count", "more experts a token than there are", 100000},
  };
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
  {
    k = kv(bad[i].key);
    const uint64_t was = k->v.u;
    k->v.u = bad[i].bad;
    refused(bad[i].what, bad[i].key + strlen("inkling2."));
    k->v.u = was;
  }
  k = kv("inkling2.attention.value_length");
  k->v.u++;
  refused("v's heads another width than q's and k's", "value_length");
  k->v.u--;
  k = kv("inkling2.expert_weights_norm");
  k->v.u = 0;
  refused("the picked experts' weights not normalized", "expert_weights_norm");
  k->v.u = 1;
  k = kv("inkling2.attention.sliding_layers");
  int32_t *items = (int32_t *)k->v.items, was = items[0];
  items[0] = items[1];
  refused("a sliding layer listed twice", "sliding_layers");
  items[0] = was;
  assert(accepted());
}

static void test_tensors(const Ink2Weights *w)
{
  printf("  tensors refused:\n");
  GgufTensor *t = tensor("blk.0.attn_q.weight");
  t->name[0] = 'X';
  refused("a layer's q missing", "blk.0.attn_q.weight: missing");
  t->name[0] = 'b';
  t->dims[1]--;
  refused("q a row short", "blk.0.attn_q.weight: of another shape");
  t->dims[1]++;

  int full = 0;
  while (w->geo.sliding[full])
    full++;
  char name[96];
  snprintf(name, sizeof name, "blk.%d.attn_k_conv_proj.weight", full);
  t = tensor(name);
  t->name[0] = 'X';
  refused("a full layer's k convolution projection missing", "attn_k_conv_proj.weight: missing");
  t->name[0] = 'b';

  t = tensor("blk.0.attn_rel_proj");
  t->dims[0]++;
  refused("a sliding layer's relative bias past its window", "blk.0.attn_rel_proj: of another shape");
  t->dims[0]--;

  t = tensor("blk.0.ffn_norm.weight");
  const uint32_t ty = t->type;
  t->type = T_Q8_0;
  refused("a norm in Q8_0", "blk.0.ffn_norm.weight: a type its reader");
  t->type = ty;

  const int nv = layer_of(w, QTYPE_NVFP4), mx = layer_of(w, QTYPE_MXFP8);
  if (nv >= 0)
  {
    snprintf(name, sizeof name, "blk.%d.ffn_up_exps.scale", nv);
    t = tensor(name);
    t->name[0] = 'X';
    refused("an NVFP4 expert tensor without its scale", "ffn_up_exps.scale: missing");
    t->name[0] = 'b';
    t->dims[0]--;
    refused("a scale for fewer experts", "ffn_up_exps.scale: of another shape");
    t->dims[0]++;
    snprintf(name, sizeof name, "blk.%d.ffn_down_exps.weight", nv);
    t = tensor(name);
    t->type = QTYPE_MXFP8;
    refused("NVFP4's experts read as MXFP8", "ffn_down_exps.weight: its bytes");
    t->type = QTYPE_NVFP4;
  }
  if (mx >= 0)
  {
    snprintf(name, sizeof name, "blk.%d.ffn_gate_exps.weight", mx);
    t = tensor(name);
    t->type = QTYPE_NVFP4;
    refused("MXFP8's experts read as NVFP4", "ffn_gate_exps.weight: its bytes");
    t->type = QTYPE_MXFP8;
  }
  t = tensor("output.weight");
  t->name[0] = 'X';
  refused("the head missing", "output.weight: missing");
  t->name[0] = 'o';
  assert(accepted());
}

int main(void)
{
  printf("[test_ink2_weights] start\n");
  char err[INK2_ERR] = "", path[512], gerr[GGUF_ERR] = "";
  assert(!ink2_weights_create(NULL, err) && err[0]);
  ink2_weights_destroy(NULL);
  if (getenv("OSMOS_TEST_INK2"))
    snprintf(path, sizeof path, "%s", getenv("OSMOS_TEST_INK2"));
  else
    snprintf(path, sizeof path, REF, getenv("HOME") ? getenv("HOME") : "");
  if (!(model = gguf_create(path, gerr)))
  {
    printf("  Inkling-2's reference: not on this machine, skipped (%s)\n", gerr);
    printf("[test_ink2_weights] ok\n");
    return 0;
  }
  Ink2Weights *w = ink2_weights_create(model, err);
  if (!w)
    printf("  refused: %s\n", err);
  assert(w);
  test_real(w);
  test_kv();
  test_tensors(w);
  ink2_weights_destroy(w);
  gguf_destroy(model);
  printf("[test_ink2_weights] ok\n");
  return 0;
}
