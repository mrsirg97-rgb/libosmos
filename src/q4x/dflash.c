#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/qtype.h"
#include "osmos/q4x/dflash.h"
#include "../model/gguf_meta.h"

/* DFlash2's GGUF (tools/convert/dflash2_gguf.py) read and checked: its keys against the target and its kernels' fixed
   shapes, its tensors each a role (q4x_weights_create's checks), none left over */

#define KEY(s) "dflash2." s
#define MAT (QTYPE_MV | QTYPE_MM) /* a matrix: a decode-sized step's matvec, a prompt's (the taps' fc) matmul */
#define ROWS QTYPE_EMBED          /* the codebooks: a token's row gathered */
#define F32 0                     /* read as its F32 values */
#define T_F32 0
#define NAME_MAX_LEN 64

typedef enum Dim { D_NONE, D_EMBD, D_TAPS, D_Q, D_KV, D_HEAD, D_FF, D_CONV, D_CONV_PROJ, D_RANK, D_VOCAB, D_TWO } Dim;

typedef struct Role
{
  const char *name;
  size_t field;
  uint32_t path;
  Dim dims[3];
} Role;

#define TOP(f) offsetof(Q4xDflash, f)
#define LAY(f) offsetof(Q4xDflashLayer, f)

static const Role tops[] = {
    {"fc.weight", TOP(fc), MAT, {D_TAPS, D_EMBD}},        {"hidden_norm.weight", TOP(hidden_norm), F32, {D_EMBD}},
    {"output_norm.weight", TOP(output_norm), F32, {D_EMBD}}, {"sel_proj.weight", TOP(sel_proj), MAT, {D_EMBD, D_RANK}},
    {"sel_pred.weight", TOP(sel_pred), ROWS, {D_RANK, D_VOCAB}}, {"sel_succ.weight", TOP(sel_succ), ROWS, {D_RANK, D_VOCAB}},
};

static const Role layers[] = {
    {"attn_norm.weight", LAY(attn_norm), F32, {D_EMBD}},
    {"attn_q.weight", LAY(q), MAT, {D_EMBD, D_Q}},
    {"attn_k.weight", LAY(k), MAT, {D_EMBD, D_KV}},
    {"attn_v.weight", LAY(v), MAT, {D_EMBD, D_KV}},
    {"attn_output.weight", LAY(out), MAT, {D_Q, D_EMBD}},
    {"attn_q_norm.weight", LAY(q_norm), F32, {D_HEAD}},
    {"attn_k_norm.weight", LAY(k_norm), F32, {D_HEAD}},
    {"ffn_norm.weight", LAY(ffn_norm), F32, {D_EMBD}},
    {"ffn_gate.weight", LAY(gate), MAT, {D_EMBD, D_FF}},
    {"ffn_up.weight", LAY(up), MAT, {D_EMBD, D_FF}},
    {"ffn_down.weight", LAY(down), MAT, {D_FF, D_EMBD}},
    {"attn_conv.weight", LAY(attn_conv), F32, {D_EMBD, D_CONV, D_TWO}},
    {"attn_conv_proj.weight", LAY(attn_conv_proj), MAT, {D_EMBD, D_CONV_PROJ}},
    {"ffn_conv.weight", LAY(ffn_conv), F32, {D_EMBD, D_CONV, D_TWO}},
    {"ffn_conv_proj.weight", LAY(ffn_conv_proj), MAT, {D_EMBD, D_CONV_PROJ}},
};

static bool fail(char *err, const char *name, const char *why)
{
  snprintf(err, Q4X_ERR, "%s: %s", name, why);
  return false;
}

static uint64_t dim(const Q4xDflashGeometry *g, Dim d)
{
  switch (d)
  {
  case D_EMBD: return (uint64_t)g->n_embd;
  case D_TAPS: return (uint64_t)g->n_taps * (uint64_t)g->n_embd;
  case D_Q: return (uint64_t)g->n_heads * (uint64_t)g->head_dim;
  case D_KV: return (uint64_t)g->n_kv_heads * (uint64_t)g->head_dim;
  case D_HEAD: return (uint64_t)g->head_dim;
  case D_FF: return (uint64_t)g->ff;
  case D_CONV: return Q4X_DFLASH_CONV;
  case D_CONV_PROJ: return 2ull * Q4X_DFLASH_CONV * (uint64_t)(g->n_embd / g->conv_group); /* prepare's, finish's */
  case D_RANK: return (uint64_t)g->sel_rank;
  case D_VOCAB: return (uint64_t)g->n_vocab;
  case D_TWO: return 2; /* before and after its block (prepare, finish) */
  default: return 0;
  }
}

static const GgufTensor *claim(const Gguf *g, bool *taken, const Q4xDflashGeometry *geo, const Role *r, const char *name,
                               char *err)
{
  const GgufTensor *t = gguf_tensor(g, name);
  if (!t)
    return fail(err, name, "missing"), NULL;
  int n = 0;
  while (n < 3 && r->dims[n] != D_NONE)
    n++;
  if (t->n_dims != (uint32_t)n)
    return fail(err, name, "of another number of dims"), NULL;
  uint64_t values = 1;
  for (int i = 0; i < n; i++)
  {
    if (t->dims[i] != dim(geo, r->dims[i]))
      return fail(err, name, "of another shape"), NULL;
    values *= t->dims[i];
  }
  if (r->path == F32 ? t->type != T_F32 : (qtype_paths(t->type) & r->path) != r->path)
    return fail(err, name, "a type its reader does not take"), NULL;
  if (t->bytes != gguf_bytes(t->type, values))
    return fail(err, name, "its bytes not its type's for its shape"), NULL;
  taken[t - g->tensors] = true;
  return t;
}

/* an integer key in lo .. hi, else `why` */
static bool in(GgufMeta *m, const char *key, int lo, int hi, const char *why, int *out)
{
  int v;
  if (!meta_uint_in(m, key, 0, INT_MAX, &v))
    return false;
  if (v < lo || v > hi)
    return meta_fail(m, key, why);
  *out = v;
  return true;
}

static bool geometry(const Gguf *gg, const Q4xWeights *w, Q4xDflashGeometry *g, char *err)
{
  GgufMeta m = {gg, err, NULL, 0};
  const Q4xGeometry *t = &w->geo;
  int embd = 0, vocab = 0, target_layers = 0, kernel = 0;
  if (!meta_text_is(&m, "general.architecture", "dflash2"))
    return false;
  if (!in(&m, KEY("block_count"), 1, Q4X_DFLASH_MAX_LAYERS, "its layers past Q4X_DFLASH_MAX_LAYERS", &g->n_layers) ||
      !in(&m, KEY("embedding_length"), t->n_embd, t->n_embd, "not the target's width", &embd) ||
      !in(&m, KEY("vocab_size"), t->n_vocab, t->n_vocab, "not the target's vocabulary", &vocab) ||
      !in(&m, KEY("target_block_count"), t->n_layers, t->n_layers, "not the target's layers", &target_layers) ||
      !in(&m, KEY("feed_forward_length"), 1, INT_MAX, "no width", &g->ff) ||
      !in(&m, KEY("attention.head_count"), 1, INT_MAX, "no heads", &g->n_heads) ||
      !in(&m, KEY("attention.head_count_kv"), 1, INT_MAX, "no heads", &g->n_kv_heads) ||
      !in(&m, KEY("attention.key_length"), Q4X_DFLASH_HEAD_DIM, Q4X_DFLASH_HEAD_DIM, "not the kernel's head dim (128)",
          &g->head_dim) ||
      !meta_uint_is(&m, KEY("attention.value_length"), Q4X_DFLASH_HEAD_DIM) ||
      !in(&m, KEY("attention.sliding_window"), 1, Q4X_DFLASH_WINDOW_MAX, "a window past Q4X_DFLASH_WINDOW_MAX",
          &g->window) ||
      !in(&m, KEY("block_size"), 2, Q4X_DFLASH_BLOCK_MAX, "a block past Q4X_DFLASH_BLOCK_MAX", &g->block) ||
      !in(&m, KEY("mask_token_id"), 0, t->n_vocab - 1, "not a token of the vocabulary", &g->mask_token) ||
      !in(&m, KEY("conv.kernel"), Q4X_DFLASH_CONV, Q4X_DFLASH_CONV, "not the kernel's taps (2)", &kernel) ||
      !in(&m, KEY("conv.group"), 1, t->n_embd, "no group", &g->conv_group) ||
      !in(&m, KEY("selector.rank"), 1, INT_MAX, "no rank", &g->sel_rank) ||
      !in(&m, KEY("selector.top_k"), 1, Q4X_DFLASH_TOPK_MAX, "more candidates than Q4X_DFLASH_TOPK_MAX", &g->sel_top_k) ||
      !meta_real(&m, KEY("attention.layer_norm_rms_epsilon"), &g->eps) || !meta_real(&m, KEY("rope.freq_base"), &g->rope_base))
    return false;
  if (g->n_heads % g->n_kv_heads)
    return meta_fail(&m, KEY("attention.head_count_kv"), "the heads not a multiple of it");
  if (embd % g->conv_group)
    return meta_fail(&m, KEY("conv.group"), "does not divide the width");
  int64_t taps[Q4X_MAX_TAPS];
  size_t n = 0;
  if (!meta_ints(&m, KEY("target_layers"), 1, Q4X_MAX_TAPS, taps, &n))
    return false;
  for (size_t i = 0; i < n; i++)
  {
    if (taps[i] < 0 || taps[i] >= t->n_layers || (i && taps[i] <= taps[i - 1]))
      return meta_fail(&m, KEY("target_layers"), "a tap past the target's layers, or out of order");
    g->taps[i] = (int32_t)taps[i];
  }
  g->n_taps = (int)n, g->n_embd = embd, g->n_vocab = vocab;
  return true;
}

Q4xDflash *q4x_dflash_create(const Gguf *g, const Q4xWeights *target, char err[Q4X_ERR])
{
  if (!g || !target)
    return fail(err, "q4x_dflash_create", "no drafter or no target"), NULL;
  Q4xDflash *d = calloc(1, sizeof *d);
  bool *taken = calloc(g->n_tensors ? g->n_tensors : 1, sizeof *taken);
  if (!d || !taken)
    return free(d), free(taken), fail(err, "q4x_dflash_create", "out of memory"), NULL;
  d->model = g;
  bool ok = geometry(g, target, &d->geo, err);
  for (size_t i = 0; ok && i < sizeof tops / sizeof *tops; i++)
    ok = (*(const GgufTensor **)((char *)d + tops[i].field) = claim(g, taken, &d->geo, &tops[i], tops[i].name, err));
  char name[NAME_MAX_LEN];
  for (int l = 0; ok && l < d->geo.n_layers; l++)
    for (size_t i = 0; ok && i < sizeof layers / sizeof *layers; i++)
    {
      snprintf(name, sizeof name, "blk.%d.%s", l, layers[i].name);
      ok = (*(const GgufTensor **)((char *)&d->layer[l] + layers[i].field) = claim(g, taken, &d->geo, &layers[i], name, err));
    }
  for (size_t i = 0; ok && i < g->n_tensors; i++)
    if (!taken[i])
      ok = fail(err, g->tensors[i].name, "a tensor no role of the drafter takes");
  free(taken);
  if (!ok)
    return free(d), NULL;
  return d;
}

void q4x_dflash_destroy(Q4xDflash *d) { free(d); }
