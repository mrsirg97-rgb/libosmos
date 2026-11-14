/* DFlash2's drafter (osmos/q4x/dflash.h) for Qwen3.8 UD-IQ3_XXS: its geometry as its metadata says (5 layers, 32 / 8
   heads of 128, a block of 8, the taps 5 .. 45), then its file edited in memory - a key past its kernels or another
   model's, a tensor missing, of another shape or type, one no role takes - each refused, its cause named. needs the
   Qwen set and ~/Models/qwen38/DFlash2/dflash2-Q8_0.gguf (tools/convert/dflash2_gguf.py); skipped without */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/gguf.h"
#include "osmos/q4x/dflash.h"
#include "osmos/q4x/weights.h"

#define QWEN "%s/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf"
#define DRAFT "%s/Models/qwen38/DFlash2/dflash2-Q8_0.gguf"
#define T_F32 0
#define T_BF16 30

static Gguf *draft;
static Q4xWeights *target;

static GgufKv *kv(const char *key)
{
  GgufKv *k = (GgufKv *)gguf_kv(draft, key);
  assert(k);
  return k;
}

static GgufTensor *tensor(const char *name)
{
  GgufTensor *t = (GgufTensor *)gguf_tensor(draft, name);
  assert(t);
  return t;
}

static void refused(const char *what, const char *needle)
{
  char err[Q4X_ERR] = "";
  Q4xDflash *d = q4x_dflash_create(draft, target, err);
  if (d || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, d ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-40s -> %s\n", what, err);
}

static void accepted(void)
{
  char err[Q4X_ERR] = "";
  Q4xDflash *d = q4x_dflash_create(draft, target, err);
  if (!d)
    printf("  refused: %s\n", err), exit(1);
  q4x_dflash_destroy(d);
}

/* a scalar key set to v, refused for it, then back */
static void key(const char *name, uint64_t v, const char *needle)
{
  GgufKv *k = kv(name);
  const uint64_t old = k->v.u;
  k->v.u = v;
  refused(name, needle);
  k->v.u = old;
  accepted();
}

int main(void)
{
  printf("[test_q4x_dflash] start\n");
  char err[Q4X_ERR] = "", path[512], gerr[GGUF_ERR] = "";
  const char *home = getenv("HOME") ? getenv("HOME") : "";
  assert(!q4x_dflash_create(NULL, NULL, err) && err[0]);
  q4x_dflash_destroy(NULL);
  snprintf(path, sizeof path, QWEN, home);
  Gguf *model = gguf_create(path, gerr);
  snprintf(path, sizeof path, DRAFT, home);
  draft = model ? gguf_create(path, gerr) : NULL;
  if (!model || !draft)
  {
    gguf_destroy(model); /* (the model whole but no drafter: its shards freed - 0.7.3, a leak under ASan) */
    printf("  Qwen3.8 UD-IQ3_XXS or the converted DFlash2: not on this machine, skipped (%s)\n", gerr);
    printf("[test_q4x_dflash] ok\n");
    return 0;
  }
  target = q4x_weights_create(model, NULL, err);
  assert(target);

  Q4xDflash *d = q4x_dflash_create(draft, target, err);
  if (!d)
    printf("  refused: %s\n", err), exit(1);
  const Q4xDflashGeometry *g = &d->geo;
  assert(g->n_layers == 5 && g->n_embd == 2560 && g->n_heads == 32 && g->n_kv_heads == 8 && g->head_dim == 128);
  assert(g->ff == 7680 && g->n_vocab == 248320 && g->window == 2048 && g->block == 8 && g->mask_token == 248077);
  assert(g->n_taps == 5 && g->taps[0] == 5 && g->taps[4] == 45 && g->conv_group == 16);
  assert(g->sel_rank == 256 && g->sel_top_k == 16 && g->eps > 0 && g->eps < 1e-5f && g->rope_base == 10000.0f);
  assert(d->fc && d->sel_pred && d->layer[4].ffn_conv_proj && d->layer[0].attn_conv);
  q4x_dflash_destroy(d);
  printf("  DFlash2: 5 layers, 32 / 8 heads of 128, ff 7680, a block of 8, window 2048, taps 5 .. 45\n");

  printf("  its keys, each past its kernels or another model's:\n");
  key("dflash2.embedding_length", 2048, "the target's");
  key("dflash2.vocab_size", 248000, "the target's");
  key("dflash2.target_block_count", 47, "the target's");
  key("dflash2.attention.key_length", 64, "head dim");
  key("dflash2.attention.head_count_kv", 7, "multiple");
  key("dflash2.conv.kernel", 3, "taps");
  key("dflash2.conv.group", 7, "divide");
  key("dflash2.selector.top_k", 17, "candidates");
  key("dflash2.block_size", 9, "block");
  key("dflash2.mask_token_id", 248320, "vocabulary");
  key("dflash2.attention.sliding_window", Q4X_DFLASH_WINDOW_MAX + 1, "window");
  key("dflash2.block_count", Q4X_DFLASH_MAX_LAYERS + 1, "layers");
  key("dflash2.block_count", 4, "no role"); /* its fifth layer's tensors left over: refused, not ignored */
  GgufKv *taps = kv("dflash2.target_layers");
  int32_t *tl = (int32_t *)taps->v.items, t4 = tl[4], t1 = tl[1];
  tl[4] = 48, refused("a tap past the target's layers", "tap"), tl[4] = t4;
  tl[1] = 2, refused("taps out of order", "tap"), tl[1] = t1;
  char *arch = kv("general.architecture")->v.s, a0 = arch[0];
  arch[0] = 'x', refused("another architecture", "architecture"), arch[0] = a0;

  printf("  its tensors:\n");
  GgufTensor *q = tensor("blk.2.attn_q.weight");
  char n0 = q->name[0];
  q->name[0] = 'x', refused("a tensor missing (and one no role takes)", "blk.2.attn_q.weight"), q->name[0] = n0;
  const uint64_t d1 = q->dims[1];
  q->dims[1] = 2048, refused("a tensor of another shape", "shape"), q->dims[1] = d1;
  const uint32_t ty = q->type;
  q->type = T_BF16, refused("a matrix the matmul does not read", "type"), q->type = ty;
  GgufTensor *cv = tensor("blk.0.attn_conv.weight");
  cv->type = 8, refused("a conv kernel not F32", "type"), cv->type = T_F32;
  accepted();
  printf("  each refused, its cause named; the file as it was accepted again\n");
  q4x_weights_destroy(target);
  gguf_destroy(draft), gguf_destroy(model);
  printf("[test_q4x_dflash] ok\n");
  return 0;
}
