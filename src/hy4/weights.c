#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/hy4/weights.h"
#include "osmos/model/qtype.h"

/* a role's types: a mask of the table's ids (qtype.h), or any type its path reads */
#define TY(name) (1u << QTYPE_##name)
#define MAT QTYPE_MV         /* a matrix a token reads: the matvec */
#define EXPERTS QTYPE_MV_TOK /* the routed experts, by their tokens */
#define EMBED QTYPE_EMBED    /* a token's row (the embedding; the Engram table's rows) */
#define RAW 0                /* read as its bytes: `types` */
#define F32 TY(F32)          /* a norm, a bias, a sink, the router, iHC's bases and scales: llama.cpp keeps them F32 */
#define NAME_MAX_LEN 96

typedef enum Dim
{
  D_NONE, D_ONE, D_TWO, D_EMBD, D_VOCAB, D_FF, D_HC, D_HC2, D_HC_FLAT, D_QLORA, D_KVLORA, D_KV_A, D_NOPE, D_VDIM,
  D_HEADS, D_Q, D_V, D_IXD, D_IXH, D_IXQ, D_POOL, D_EXPERTS, D_FFEXP, D_LATENT, D_FFSH, D_EG_IN, D_EG_KV, D_EG_KV1,
  D_EG_DIM, D_EG_COLS, D_EG_ROWS,
} Dim;

/* which layers have a role */
typedef enum When { ALL, SLIDING, SPARSE, DENSE, MOE, ENGRAM } When;

typedef struct Role
{
  const char *name;
  size_t field;
  When when;
  uint32_t path;
  uint32_t types;
  Dim dims[GGUF_MAX_DIMS];
} Role;

#define TOP(f) offsetof(Hy4Weights, f)
#define LAY(f) offsetof(Hy4Layer, f)

/* dims as GGUF writes them: the row's values first (a checkpoint's [rows, cols] is {cols, rows}) */
static const Role tops[] = {
  {"token_embd.weight", TOP(embed), ALL, EMBED, 0, {D_EMBD, D_VOCAB}},
  {"output_norm.weight", TOP(output_norm), ALL, RAW, F32, {D_EMBD}},
  {"output.weight", TOP(output), ALL, MAT, 0, {D_EMBD, D_VOCAB}},
  {"output_hc_fn.weight", TOP(out_hc_fn), ALL, MAT, 0, {D_HC_FLAT, D_HC}},
  {"output_hc_base.weight", TOP(out_hc_base), ALL, RAW, F32, {D_HC}},
  {"output_hc_scale.weight", TOP(out_hc_scale), ALL, RAW, F32, {D_ONE}},
  {"per_layer_token_embd.weight", TOP(engram_table), ALL, EMBED, 0, {D_EG_DIM, D_EG_ROWS}},
};

static const Role roles[] = {
  {"hc_attn_fn.weight", LAY(hc_attn_fn), ALL, MAT, 0, {D_HC_FLAT, D_HC2}},
  {"hc_attn_base.weight", LAY(hc_attn_base), ALL, RAW, F32, {D_HC2}},
  {"hc_attn_scale.weight", LAY(hc_attn_scale), ALL, RAW, F32, {D_TWO}},
  {"hc_ffn_fn.weight", LAY(hc_ffn_fn), ALL, MAT, 0, {D_HC_FLAT, D_HC2}},
  {"hc_ffn_base.weight", LAY(hc_ffn_base), ALL, RAW, F32, {D_HC2}},
  {"hc_ffn_scale.weight", LAY(hc_ffn_scale), ALL, RAW, F32, {D_TWO}},
  {"attn_norm.weight", LAY(attn_norm), ALL, RAW, F32, {D_EMBD}},
  {"attn_q_a.weight", LAY(q_a), ALL, MAT, 0, {D_EMBD, D_QLORA}},
  {"attn_q_a_norm.weight", LAY(q_a_norm), ALL, RAW, F32, {D_QLORA}},
  {"attn_q_b.weight", LAY(q_b), ALL, MAT, 0, {D_QLORA, D_Q}},
  {"attn_kv_a_mqa.weight", LAY(kv_a), ALL, MAT, 0, {D_EMBD, D_KV_A}},
  {"attn_kv_a_norm.weight", LAY(kv_a_norm), ALL, RAW, F32, {D_KVLORA}},
  {"attn_k_b.weight", LAY(k_b), ALL, MAT, 0, {D_NOPE, D_KVLORA, D_HEADS}},
  {"attn_v_b.weight", LAY(v_b), ALL, MAT, 0, {D_KVLORA, D_VDIM, D_HEADS}},
  {"attn_gate.weight", LAY(gate), ALL, MAT, 0, {D_EMBD, D_V}},
  {"attn_output.weight", LAY(out), ALL, MAT, 0, {D_V, D_EMBD}},
  {"attn_sinks.weight", LAY(sinks), ALL, RAW, F32, {D_HEADS}},
  {"post_attention_norm.weight", LAY(post_attn_norm), ALL, RAW, F32, {D_EMBD}},
  {"indexer.attn_q_b.weight", LAY(ix_q_b), SPARSE, MAT, 0, {D_QLORA, D_IXQ}},
  {"indexer.attn_k.weight", LAY(ix_k), SPARSE, MAT, 0, {D_EMBD, D_IXD}},
  {"indexer.k_norm.weight", LAY(ix_k_norm), SPARSE, RAW, F32, {D_IXD}},
  {"indexer.proj.weight", LAY(ix_proj), SPARSE, MAT, 0, {D_EMBD, D_IXH}},
  {"indexer_compressor_value.weight", LAY(ix_pool_value), SPARSE, MAT, 0, {D_IXD, D_IXD}},
  {"indexer_compressor_gate.weight", LAY(ix_pool_gate), SPARSE, MAT, 0, {D_IXD, D_IXD}},
  {"indexer_compressor_ape.weight", LAY(ix_pool_ape), SPARSE, RAW, F32, {D_IXD, D_POOL}},
  {"indexer_compressor_norm.weight", LAY(ix_pool_norm), SPARSE, RAW, F32, {D_IXD}},
  {"ffn_norm.weight", LAY(ffn_norm), ALL, RAW, F32, {D_EMBD}},
  {"post_ffw_norm.weight", LAY(post_ffn_norm), ALL, RAW, F32, {D_EMBD}},
  {"ffn_gate.weight", LAY(ffn_gate), DENSE, MAT, 0, {D_EMBD, D_FF}},
  {"ffn_up.weight", LAY(ffn_up), DENSE, MAT, 0, {D_EMBD, D_FF}},
  {"ffn_down.weight", LAY(ffn_down), DENSE, MAT, 0, {D_FF, D_EMBD}},
  {"ffn_gate_inp.weight", LAY(router), MOE, RAW, F32, {D_EMBD, D_EXPERTS}},
  {"exp_probs_b.bias", LAY(router_bias), MOE, RAW, F32, {D_EXPERTS}},
  {"ffn_gate_exps.weight", LAY(exp_gate), MOE, EXPERTS, 0, {D_LATENT, D_FFEXP, D_EXPERTS}},
  {"ffn_up_exps.weight", LAY(exp_up), MOE, EXPERTS, 0, {D_LATENT, D_FFEXP, D_EXPERTS}},
  {"ffn_down_exps.weight", LAY(exp_down), MOE, EXPERTS, 0, {D_FFEXP, D_LATENT, D_EXPERTS}},
  {"ffn_routed_down.weight", LAY(routed_down), MOE, MAT, 0, {D_EMBD, D_LATENT}},
  {"ffn_routed_up.weight", LAY(routed_up), MOE, MAT, 0, {D_LATENT, D_EMBD}},
  {"ffn_gate_shexp.weight", LAY(sh_gate), MOE, MAT, 0, {D_EMBD, D_FFSH}},
  {"ffn_up_shexp.weight", LAY(sh_up), MOE, MAT, 0, {D_EMBD, D_FFSH}},
  {"ffn_down_shexp.weight", LAY(sh_down), MOE, MAT, 0, {D_FFSH, D_EMBD}},
  {"engram_kv.weight", LAY(eg_kv), ENGRAM, MAT, 0, {D_EG_IN, D_EG_KV}},
  {"engram_kv_b.weight", LAY(eg_kv_b), ENGRAM, RAW, F32, {D_EMBD, D_EG_KV1}},
  {"engram_gate.weight", LAY(eg_gate), ENGRAM, RAW, F32, {D_EMBD, D_HC}},
  {"engram_lookup_norm.weight", LAY(eg_lookup_norm), ENGRAM, RAW, F32, {D_EG_DIM, D_EG_COLS}},
  {"engram_pre_gamma.weight", LAY(eg_pre_gamma), ENGRAM, RAW, F32, {D_EMBD}},
  {"engram_hc_scale.weight", LAY(eg_hc_scale), ENGRAM, RAW, F32, {D_EMBD, D_HC}},
};

#define N_TOPS (sizeof tops / sizeof *tops)
#define N_ROLES (sizeof roles / sizeof *roles)

/* what a role's dims are measured against: the geometry, the layer (its heads), the Engram table's rows */
typedef struct Ctx
{
  const Gguf *model;
  const Hy4Geometry *g;
  bool *taken;
  uint64_t table_rows;
  int layer;
} Ctx;

static bool fail(char *err, const char *name, const char *why)
{
  snprintf(err, HY4_ERR, "%s: %s", name, why);
  return false;
}

static uint64_t dim(const Ctx *c, Dim d)
{
  const Hy4Geometry *g = c->g;
  const uint64_t heads = c->layer >= 0 ? (uint64_t)g->n_heads[c->layer] : 0;
  const uint64_t cols = (uint64_t)(g->engram_ngram - 1) * (uint64_t)g->engram_heads;
  switch (d)
  {
  case D_ONE: return 1;
  case D_TWO: return 2;
  case D_EMBD: return (uint64_t)g->n_embd;
  case D_VOCAB: return (uint64_t)g->n_vocab;
  case D_FF: return (uint64_t)g->n_ff;
  case D_HC: return (uint64_t)g->hc;
  case D_HC2: return 2 * (uint64_t)g->hc;
  case D_HC_FLAT: return (uint64_t)g->hc * (uint64_t)g->n_embd;
  case D_QLORA: return (uint64_t)g->q_lora;
  case D_KVLORA: return (uint64_t)g->kv_lora;
  case D_KV_A: return (uint64_t)g->kv_lora + (uint64_t)g->rope_dim;
  case D_NOPE: return (uint64_t)g->nope_dim;
  case D_VDIM: return (uint64_t)g->v_dim;
  case D_HEADS: return heads;
  case D_Q: return heads * (uint64_t)(g->nope_dim + g->rope_dim);
  case D_V: return heads * (uint64_t)g->v_dim;
  case D_IXD: return (uint64_t)g->ix_dim;
  case D_IXH: return (uint64_t)g->ix_heads;
  case D_IXQ: return (uint64_t)g->ix_heads * (uint64_t)g->ix_dim;
  case D_POOL: return (uint64_t)g->ix_pool;
  case D_EXPERTS: return (uint64_t)g->n_experts;
  case D_FFEXP: return (uint64_t)g->ff_exp;
  case D_LATENT: return (uint64_t)g->latent;
  case D_FFSH: return (uint64_t)g->ff_exp * (uint64_t)g->n_shexp;
  case D_EG_IN: return cols * (uint64_t)g->engram_dim;
  case D_EG_KV: return (uint64_t)g->n_embd * ((uint64_t)g->hc + 1);
  case D_EG_KV1: return (uint64_t)g->hc + 1;
  case D_EG_DIM: return (uint64_t)g->engram_dim;
  case D_EG_COLS: return cols;
  case D_EG_ROWS: return c->table_rows;
  default: return 0;
  }
}

static bool applies(const Hy4Geometry *g, When w, int l)
{
  switch (w)
  {
  case ALL: return true;
  case SLIDING: return g->sliding[l];
  case SPARSE: return !g->sliding[l];
  case DENSE: return l < g->n_dense;
  case MOE: return l >= g->n_dense;
  case ENGRAM: return g->engram[l];
  }
  return false;
}

/* the tensor of that name checked against its role; taken[] marks it */
static const GgufTensor *claim(const Ctx *c, const Role *r, const char *name, char *err)
{
  const GgufTensor *t = gguf_tensor(c->model, name);
  if (!t)
    return fail(err, name, "missing"), NULL;
  int n = 0;
  while (n < GGUF_MAX_DIMS && r->dims[n] != D_NONE)
    n++;
  if (t->n_dims != (uint32_t)n)
    return fail(err, name, "of another number of dims"), NULL;
  uint64_t values = 1;
  for (int i = 0; i < n; i++)
  {
    if (t->dims[i] != dim(c, r->dims[i]))
      return fail(err, name, "of another shape"), NULL;
    values *= t->dims[i];
  }
  const uint32_t paths = qtype_paths(t->type);
  if (r->path == RAW ? !(t->type < 32 && (r->types >> t->type & 1)) : (paths & r->path) != r->path)
    return fail(err, name, "a type its reader does not take"), NULL;
  if (t->bytes != gguf_bytes(t->type, values))
    return fail(err, name, "its bytes not its type's for its shape"), NULL;
  c->taken[t - c->model->tensors] = true;
  return t;
}

/* a list of the Engram layers' columns: n_engram x cols integers */
static bool engram_list(const Gguf *model, const char *key, size_t n, int64_t *out, char *err)
{
  const GgufKv *k = gguf_kv(model, key);
  if (!k || k->type != GGUF_ARRAY || k->n != n)
    return fail(err, key, "missing, or not a column an Engram layer");
  for (size_t i = 0; i < n; i++)
    if (!gguf_int(k, i, &out[i]) || out[i] < 0)
      return fail(err, key, "a value not a non-negative integer");
  return true;
}

#define MAX_COLS (HY4_MAX_ENGRAM * 64)

/* each Engram layer's rows (its columns' span from its first offset) and the table's (the last column's end:
   llama.cpp's offsets count across the layers' tables) */
static bool engram_spans(const Gguf *model, const Hy4Geometry *g, uint64_t *rows, uint64_t *table, char *err)
{
  const int cols = (g->engram_ngram - 1) * g->engram_heads;
  const size_t n = (size_t)g->n_engram * (size_t)cols;
  int64_t off[MAX_COLS], size[MAX_COLS];
  if (n > MAX_COLS)
    return fail(err, "hy_v4_flash.ple.heads_per_ngram", "more columns than this build keeps");
  if (!engram_list(model, "hy_v4_flash.ple.head_offsets", n, off, err) ||
      !engram_list(model, "hy_v4_flash.ple.head_vocab_sizes", n, size, err))
    return false;
  *table = 0;
  for (int e = 0; e < g->n_engram; e++)
  {
    int64_t end = off[e * cols];
    for (int k = 0; k < cols; k++)
      end = off[e * cols + k] + size[e * cols + k] > end ? off[e * cols + k] + size[e * cols + k] : end;
    rows[e] = (uint64_t)(end - off[e * cols]);
    *table = (uint64_t)end > *table ? (uint64_t)end : *table;
  }
  return true;
}

/* blk.N. with N a nextn layer's (the drafter's) */
static bool drafter_tensor(const char *name, const Hy4Geometry *g)
{
  int l;
  return sscanf(name, "blk.%d.", &l) == 1 && l >= g->n_layers && l < g->n_layers + g->n_nextn;
}

void hy4_weights_destroy(Hy4Weights *w) { free(w); }

Hy4Weights *hy4_weights_create(const Gguf *model, char err[HY4_ERR])
{
  if (!model)
    return fail(err, "hy4_weights_create", "no model"), NULL;
  Hy4Weights *w = calloc(1, sizeof *w);
  bool *taken = calloc(model->n_tensors ? model->n_tensors : 1, sizeof *taken);
  if (!w || !taken)
    return free(w), free(taken), fail(err, "hy4_weights_create", "out of memory"), NULL;
  w->model = model;
  const Hy4Geometry *g = &w->geo;
  Ctx c = {model, g, taken, 0, -1};
  bool ok = hy4_geometry(model, &w->geo, err) && engram_spans(model, g, w->engram_rows, &c.table_rows, err);
  for (size_t i = 0; ok && i < N_TOPS; i++)
    ok = (*(const GgufTensor **)((char *)w + tops[i].field) = claim(&c, &tops[i], tops[i].name, err));
  char name[NAME_MAX_LEN];
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    c.layer = l;
    for (size_t i = 0; ok && i < N_ROLES; i++)
      if (applies(g, roles[i].when, l))
      {
        snprintf(name, sizeof name, "blk.%d.%s", l, roles[i].name);
        ok = (*(const GgufTensor **)((char *)&w->layer[l] + roles[i].field) = claim(&c, &roles[i], name, err));
      }
  }
  for (size_t i = 0; ok && i < model->n_tensors; i++)
    if (!taken[i] && !drafter_tensor(model->tensors[i].name, g))
      ok = fail(err, model->tensors[i].name, "a tensor no role of this layer's kind takes");
  free(taken);
  if (!ok)
    return free(w), NULL;
  return w;
}

/* the map's covered length (the reference's lookup table): llama.cpp's converter extends the checkpoint's table by
   identity to the vocabulary - the map's trailing run of i -> i, each above every entry before the run. a map that
   is identity from its start has no extension: all of it covered */
static uint64_t covered_length(const GgufKv *map)
{
  int64_t v;
  uint64_t run = map->n;
  while (run > 0 && gguf_int(map, run - 1, &v) && v == (int64_t)(run - 1))
    run--;
  if (run == 0)
    return map->n;
  int64_t below = -1;
  for (uint64_t i = 0; i < run; i++)
    if (gguf_int(map, i, &v) && v > below)
      below = v;
  while (run < map->n && (int64_t)run <= below) /* a run entry not past the covered ones: covered */
    below = (int64_t)run > below ? (int64_t)run : below, run++;
  return run;
}

bool hy4_engram_shape(const Hy4Weights *w, EngramShape *s, char err[HY4_ERR])
{
  if (!w || !s)
    return fail(err, "hy4_engram_shape", "no weights");
  const Hy4Geometry *g = &w->geo;
  const GgufKv *map = gguf_kv(w->model, "hy_v4_flash.engram.token_map");
  if (!map || map->type != GGUF_ARRAY || map->n != (uint64_t)g->n_vocab)
    return fail(err, "hy_v4_flash.engram.token_map", "missing, or not an entry a token");
  const uint64_t covered = covered_length(map);
  int64_t compressed = 0, v;
  for (uint64_t i = 0; i < covered; i++)
  {
    if (!gguf_int(map, i, &v) || v < 0)
      return fail(err, "hy_v4_flash.engram.token_map", "an entry not a non-negative integer");
    compressed = v + 1 > compressed ? v + 1 : compressed;
  }
  *s = (EngramShape){w->model, "hy_v4_flash", g->n_engram, g->engram_ngram, g->engram_heads, map->n, covered,
                     g->engram_pad, compressed, w->engram_rows};
  return true;
}
