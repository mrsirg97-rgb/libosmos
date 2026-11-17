/* LFM3's weights (osmos/lfm3/forward.h) on LFM3-1.7B's BF16 and NVFP4 files: the geometry its metadata says, each
   layer's kind its tensors, each NVFP4 matrix's scale its .scale's, then the file edited in memory - another
   architecture, a role's tensor missing, of another shape or type, a convolution past the kernel's taps, an NVFP4
   matrix without its .scale, a layer the metadata calls a convolution with attention's tensors - each refused, its
   cause named. OSMOS_TEST_LFM3 / OSMOS_TEST_LFM3_NVFP4 (the BF16 and NVFP4 files; vibes-flash's own pipeline writes
   both: incarnate's convert and student export); skipped without */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"

#define BF16_FILE "/mnt/data/models/lfm3-orig-bf16.gguf"
#define NVFP4_FILE "/mnt/data/models/lfm3-orig-nvfp4-rtn.gguf"
#define T_F32 0
#define T_BF16 30
#define T_NVFP4 40

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

static void refused(const char *what, const char *needle)
{
  char err[LFM3_ERR] = "";
  Lfm3Weights *w = lfm3_weights_create(model, err);
  if (w || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, w ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-52s -> %s\n", what, err);
}

static Gguf *open_file(const char *env, const char *fallback)
{
  const char *path = getenv(env) ? getenv(env) : fallback;
  char err[GGUF_ERR] = "";
  Gguf *g = gguf_create(path, err);
  if (!g)
    printf("  %s: not on this machine, skipped (%s)\n", path, err);
  return g;
}

/* the file's own facts, read here apart from lfm3_geometry: the layers' kinds by head_count_kv and the window
   pattern, the rotated pairs by rope_freqs */
static void test_geometry(const Lfm3Weights *w)
{
  const Lfm3Geometry *g = &w->geo;
  assert(g->n_layers == 24 && g->n_embd == 2048 && g->n_ff == 7680 && g->n_vocab == 128000);
  assert(g->n_heads == 16 && g->n_kv_heads == 4 && g->head_dim == 128);
  assert(g->conv_taps == 7 && g->window == 256 && g->rope_pairs == 32 && g->native_ctx == 262144);
  assert(fabsf(g->rms_eps - 1e-5f) < 1e-9f && g->rope_base == 1e6f && g->rope_base_swa == 1e4f);
  int conv = 0, full = 0, sliding = 0;
  for (int l = 0; l < g->n_layers; l++)
  {
    const Lfm3Kind want = l % 4 == 2 ? LFM3_FULL : l % 4 == 0 && l ? LFM3_SLIDING : LFM3_CONV;
    assert(g->kind[l] == want);
    conv += want == LFM3_CONV, full += want == LFM3_FULL, sliding += want == LFM3_SLIDING;
  }
  assert(conv == 13 && full == 6 && sliding == 5);
  printf("  geometry: 24 layers (13 conv, 6 full, 5 sliding), D 2048, ff 7680, 16 / 4 heads of 128, conv 7 taps, "
         "window 256, 32 of 64 pairs rotated\n");
}

/* each layer's tensors its kind's, the other kind's NULL; the head the embedding (tied); scales 1 but NVFP4's */
static void test_roles(const Lfm3Weights *w, bool nvfp4)
{
  const Lfm3Geometry *g = &w->geo;
  assert(w->embd && w->out_norm && w->rope_freqs && w->embd->type == T_BF16);
  for (int l = 0; l < g->n_layers; l++)
  {
    const Lfm3Layer *L = &w->layer[l];
    assert(L->attn_norm && L->post_attn_norm && L->ffn_norm && L->post_ffn_norm && L->ffn_gate && L->ffn_up &&
           L->ffn_down);
    const bool conv = g->kind[l] == LFM3_CONV;
    assert(conv == !!L->in_proj && conv == !!L->conv && conv == !!L->out_proj);
    assert(!conv == !!L->q && !conv == !!L->k && !conv == !!L->v && !conv == !!L->q_norm && !conv == !!L->k_norm &&
           !conv == !!L->gate && !conv == !!L->out);
    const float s[] = {L->s_ffn_gate, L->s_ffn_up, L->s_ffn_down, conv ? L->s_in_proj : L->s_q,
                       conv ? L->s_out_proj : L->s_out};
    for (size_t i = 0; i < sizeof s / sizeof *s; i++)
      assert(nvfp4 ? s[i] > 0.0f && s[i] != 1.0f : s[i] == 1.0f);
  }
  if (nvfp4)
  { /* the scale beside a matrix read as stored: blk.0.ffn_up's */
    assert(w->layer[0].ffn_up->type == T_NVFP4);
    printf("  NVFP4: every matrix with its .scale (blk.0.ffn_up %.6g)\n", (double)w->layer[0].s_ffn_up);
  }
  printf("  roles: each layer its kind's tensors, the other's none; the head the embedding\n");
}

static void test_refusals(void)
{
  GgufKv *arch = kv("general.architecture");
  arch->v.s[3] = '2';
  refused("another architecture (lfm2)", "architecture");
  arch->v.s[3] = '3';

  GgufTensor *t = tensor("blk.2.attn_gate.weight");
  t->name[0] = 'X';
  refused("a role's tensor missing (blk.2.attn_gate)", "blk.2.attn_gate.weight");
  t->name[0] = 'b';

  t = tensor("blk.4.attn_k.weight");
  const uint64_t d1 = t->dims[1];
  t->dims[1] = d1 / 2;
  refused("a matrix of another shape (blk.4.attn_k)", "blk.4.attn_k.weight");
  t->dims[1] = d1;

  t = tensor("blk.0.ffn_norm.weight");
  t->type = T_BF16;
  refused("a norm not F32 (blk.0.ffn_norm)", "blk.0.ffn_norm.weight");
  t->type = T_F32;

  GgufKv *taps = kv("lfm3.shortconv.l_cache");
  const uint64_t taps0 = taps->v.u;
  taps->v.u = 9;
  refused("a convolution past the kernel's taps (9)", "taps");
  taps->v.u = taps0;

  GgufKv *heads_kv = kv("lfm3.attention.head_count_kv");
  if (heads_kv->type == GGUF_ARRAY)
  {
    int32_t *items = heads_kv->v.items;
    const int32_t old = items[2];
    items[2] = 0;
    refused("layer 2 called a convolution, attention's tensors", "blk.2.");
    items[2] = old;
  }

  char err[LFM3_ERR] = "";
  Lfm3Weights *w = lfm3_weights_create(model, err);
  if (!w)
  {
    printf("  restored file refused: %s\n", err);
    exit(1);
  }
  lfm3_weights_destroy(w);
}

static void test_nvfp4_scale_missing(void)
{
  GgufTensor *t = tensor("blk.0.ffn_up.scale");
  t->name[0] = 'X';
  refused("an NVFP4 matrix without its .scale (blk.0.ffn_up)", ".scale");
  t->name[0] = 'b';
}

int main(void)
{
  printf("[test_lfm3_weights] start\n");
  model = open_file("OSMOS_TEST_LFM3", BF16_FILE);
  if (!model)
    return 77;
  char err[LFM3_ERR] = "";
  Lfm3Weights *w = lfm3_weights_create(model, err);
  if (!w)
  {
    printf("  refused: %s\n", err);
    return 1;
  }
  test_geometry(w);
  test_roles(w, false);
  lfm3_weights_destroy(w);
  test_refusals();
  gguf_destroy(model);

  model = open_file("OSMOS_TEST_LFM3_NVFP4", NVFP4_FILE);
  if (model)
  {
    w = lfm3_weights_create(model, err);
    if (!w)
    {
      printf("  NVFP4 refused: %s\n", err);
      return 1;
    }
    test_geometry(w);
    test_roles(w, true);
    lfm3_weights_destroy(w);
    test_nvfp4_scale_missing();
    gguf_destroy(model);
  }
  printf("[test_lfm3_weights] ok\n");
  return 0;
}
