/* LFM3's weights (osmos/lfm3/forward.h) on vibes1.7b's Q4_0 and BF16 files: the geometry its metadata says, each
   layer's kind its tensors, every matrix the file's one type; then the Q4_0 file edited in memory - another
   architecture, a role's tensor missing, of another shape or type, one matrix of another type than the rest, a
   convolution past the kernel's taps, a layer the metadata calls a convolution with attention's tensors - each
   refused, its cause named. OSMOS_TEST_LFM3 / OSMOS_TEST_LFM3_BF16 (default ~/Models/vibes1.7b's); skipped without
   (said, then 0) */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"

#define Q4_0_FILE "Models/vibes1.7b/vibes1.7b-Q4_0.gguf"
#define BF16_FILE "Models/vibes1.7b/vibes1.7b-BF16.gguf"
#define T_F32 0
#define T_Q4_0 2
#define T_Q8_0 8
#define T_BF16 30

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
  char home[512];
  snprintf(home, sizeof home, "%s/%s", getenv("HOME") ? getenv("HOME") : "", fallback);
  const char *path = getenv(env) ? getenv(env) : home;
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

/* each layer's tensors its kind's, the other kind's NULL; every matrix `type`; the head the embedding (tied, BF16) */
static void test_roles(const Lfm3Weights *w, uint32_t type)
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
    const GgufTensor *m[] = {L->ffn_gate, L->ffn_up, L->ffn_down, conv ? L->in_proj : L->q,
                             conv ? L->out_proj : L->out, conv ? L->out_proj : L->k, conv ? L->out_proj : L->gate};
    for (size_t i = 0; i < sizeof m / sizeof *m; i++)
      assert(m[i]->type == type);
  }
  printf("  roles: each layer its kind's tensors, the other's none; every matrix type %u; the head the embedding\n",
         type);
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

  t = tensor("blk.6.ffn_down.weight");
  t->type = T_Q8_0;
  refused("one matrix of another type than the rest (blk.6.ffn_down)", "blk.6.ffn_down.weight");
  t->type = T_Q4_0;

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

int main(void)
{
  printf("[test_lfm3_weights] start\n");
  model = open_file("OSMOS_TEST_LFM3", Q4_0_FILE);
  if (!model)
    return printf("[test_lfm3_weights] ok\n"), 0; /* a host test's skip: its runner takes 0 alone (test_ink2_weights') */
  char err[LFM3_ERR] = "";
  Lfm3Weights *w = lfm3_weights_create(model, err);
  if (!w)
  {
    printf("  refused: %s\n", err);
    return 1;
  }
  test_geometry(w);
  test_roles(w, T_Q4_0);
  lfm3_weights_destroy(w);
  test_refusals();
  gguf_destroy(model);

  model = open_file("OSMOS_TEST_LFM3_BF16", BF16_FILE);
  if (model)
  {
    w = lfm3_weights_create(model, err);
    if (!w)
    {
      printf("  BF16 refused: %s\n", err);
      return 1;
    }
    test_geometry(w);
    test_roles(w, T_BF16);
    lfm3_weights_destroy(w);
    gguf_destroy(model);
  }
  printf("[test_lfm3_weights] ok\n");
  return 0;
}
