#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "substrate/model/qtype.h"
#include "weights_geo.h"

/* a role's types, a mask of the table's ids (qtype.h): TY by name, TY_ID by a tensor's id (below 32: a mask's bits) */
#define TY_ID(id) (1u << (id))
#define TY(name) TY_ID(QTYPE_##name)
#define TY_BITS 32
#define EXPERT_TYPES (TY(IQ1_M) | TY(IQ2_XXS) | TY(IQ2_XS))
#define HC_SCALES 3 /* the hyper-connection mix's pre, post and comb scales */
#define NAME_MAX_LEN 96
#define EXPERT_PARTS 3 /* an expert's gate, up and down */
_Static_assert(EXPERT_PARTS <= POOL_PARTS, "a pool unit holds an expert's three matrices");

/* a dimension, named by what it is in the geometry */
typedef enum Dim
{
  D_NONE, D_EMBD, D_VOCAB, D_HC, D_HC_EMBD, D_HC_MIX, D_HC_SCALES, D_QLORA, D_Q, D_HEAD, D_HEADS, D_OA_IN, D_OLORA,
  D_OGROUPS, D_OB_IN, D_IX_DIM, D_IX_Q, D_IX_HEADS, D_EG_IN, D_EG_OUT, D_EG_HEAD, D_EG_ROWS, D_EG_COLS, D_EG_LAYERS,
  D_EG_NGRAM, D_EXPERTS, D_FF,
} Dim;

/* which layers have a role */
typedef enum When { ALL, KV_SRC, GATED, IX_SRC, ENGRAM } When;

typedef struct Role
{
  const char *name;
  size_t field;
  When when;
  uint32_t types;
  Dim dims[GGUF_MAX_DIMS];
  bool in_engram;
} Role;

#define TOP(f) offsetof(Weights, f)
#define LAY(f) offsetof(LayerWeights, f)

static const Role tops[] = {
  {"token_embd.weight", TOP(embed), ALL, TY(Q5_K), {D_EMBD, D_VOCAB}, false},
  {"output.weight", TOP(output), ALL, TY(Q5_K), {D_EMBD, D_VOCAB}, false},
  {"output_norm.weight", TOP(output_norm), ALL, TY(F32), {D_EMBD}, false},
  {"engram_token_map.weight", TOP(eg_token_map), ALL, TY(I32), {D_VOCAB}, false},
  {"engram_primes.weight", TOP(eg_primes), ALL, TY(I64), {D_EG_COLS, D_EG_LAYERS}, false},
  {"engram_offsets.weight", TOP(eg_offsets), ALL, TY(I64), {D_EG_COLS, D_EG_LAYERS}, false},
  {"engram_multipliers.weight", TOP(eg_multipliers), ALL, TY(I64), {D_EG_NGRAM, D_EG_LAYERS}, false},
};

static const Role roles[] = {
  {"attn_norm.weight", LAY(attn_norm), ALL, TY(F32), {D_EMBD}, false},
  {"attn_q_a.weight", LAY(q_a), ALL, TY(Q8_0), {D_EMBD, D_QLORA}, false},
  {"attn_q_a_norm.weight", LAY(q_a_norm), ALL, TY(F32), {D_QLORA}, false},
  {"attn_q_b.weight", LAY(q_b), ALL, TY(Q5_K), {D_QLORA, D_Q}, false},
  {"attn_kv.weight", LAY(kv), ALL, TY(Q8_0), {D_EMBD, D_HEAD}, false},
  {"attn_kv_a_norm.weight", LAY(kv_norm), ALL, TY(F32), {D_HEAD}, false},
  {"attn_output_a.weight", LAY(out_a), ALL, TY(Q5_K), {D_OA_IN, D_OLORA, D_OGROUPS}, false},
  {"attn_output_b.weight", LAY(out_b), ALL, TY(Q6_K), {D_OB_IN, D_EMBD}, false},
  {"attn_sinks.weight", LAY(sinks), ALL, TY(F32), {D_HEADS}, false},
  {"attn_compressor_kv.weight", LAY(comp_kv), KV_SRC, TY(BF16), {D_EMBD, D_HEAD}, false},
  {"attn_compressor_gate.weight", LAY(comp_gate), GATED, TY(BF16), {D_EMBD, D_HEAD}, false},
  {"attn_compressor_norm.weight", LAY(comp_norm), KV_SRC, TY(F32), {D_HEAD}, false},
  {"indexer.attn_k.weight", LAY(ix_k), KV_SRC, TY(BF16), {D_HEAD, D_IX_DIM}, false},
  {"indexer.k_norm.weight", LAY(ix_k_norm), KV_SRC, TY(F32), {D_IX_DIM}, false},
  {"indexer.attn_q_b.weight", LAY(ix_q_b), IX_SRC, TY(BF16), {D_QLORA, D_IX_Q}, false},
  {"indexer.proj.weight", LAY(ix_proj), IX_SRC, TY(BF16), {D_EMBD, D_IX_HEADS}, false},
  {"engram_q.weight", LAY(eg_q), ENGRAM, TY(BF16), {D_EMBD, D_HC}, false},
  {"engram_k.weight", LAY(eg_k), ENGRAM, TY(BF16), {D_EMBD, D_HC}, false},
  {"engram_wkv.weight", LAY(eg_wkv), ENGRAM, TY(Q5_K), {D_EG_IN, D_EG_OUT}, false},
  {"engram_embd.weight", LAY(eg_embd), ENGRAM, TY(IQ4_NL), {D_EG_HEAD, D_EG_ROWS}, true},
  {"hc_attn_fn.weight", LAY(hc_attn_fn), ALL, TY(F32), {D_HC_EMBD, D_HC_MIX}, false},
  {"hc_attn_scale.weight", LAY(hc_attn_scale), ALL, TY(F32), {D_HC_SCALES}, false},
  {"hc_attn_base.weight", LAY(hc_attn_base), ALL, TY(F32), {D_HC_MIX}, false},
  {"hc_ffn_fn.weight", LAY(hc_ffn_fn), ALL, TY(F32), {D_HC_EMBD, D_HC_MIX}, false},
  {"hc_ffn_scale.weight", LAY(hc_ffn_scale), ALL, TY(F32), {D_HC_SCALES}, false},
  {"hc_ffn_base.weight", LAY(hc_ffn_base), ALL, TY(F32), {D_HC_MIX}, false},
  {"ffn_norm.weight", LAY(ffn_norm), ALL, TY(F32), {D_EMBD}, false},
  {"ffn_gate_inp.weight", LAY(router), ALL, TY(BF16), {D_EMBD, D_EXPERTS}, false},
  {"exp_probs_b.bias", LAY(router_bias), ALL, TY(F32), {D_EXPERTS}, false},
  {"ffn_gate_shexp.weight", LAY(sh_gate), ALL, TY(Q5_K), {D_EMBD, D_FF}, false},
  {"ffn_up_shexp.weight", LAY(sh_up), ALL, TY(Q5_K), {D_EMBD, D_FF}, false},
  {"ffn_down_shexp.weight", LAY(sh_down), ALL, TY(Q8_0) | TY(Q5_K), {D_FF, D_EMBD}, false},
  {"ffn_gate_exps.weight", LAY(exp_gate), ALL, EXPERT_TYPES, {D_EMBD, D_FF, D_EXPERTS}, false},
  {"ffn_up_exps.weight", LAY(exp_up), ALL, EXPERT_TYPES, {D_EMBD, D_FF, D_EXPERTS}, false},
  {"ffn_down_exps.weight", LAY(exp_down), ALL, EXPERT_TYPES, {D_FF, D_EMBD, D_EXPERTS}, false},
};

#define N_TOPS (sizeof tops / sizeof *tops)
#define N_ROLES (sizeof roles / sizeof *roles)

/* the check's state: the files, which of their tensors a role took, the error */
typedef struct Check
{
  Weights *w;
  bool *taken[2];
  char *err;
} Check;

static uint64_t dim(const Geometry *g, Dim d, int layer)
{
  const uint64_t hc = (uint64_t)g->hc, q = (uint64_t)g->n_heads * (uint64_t)g->head_dim;
  int n_eg = 0;
  for (int l = 0; l < g->n_layers; l++)
    n_eg += g->engram[l];
  switch (d)
  {
  case D_EMBD: return (uint64_t)g->n_embd;
  case D_VOCAB: return (uint64_t)g->n_vocab;
  case D_HC: return hc;
  case D_HC_EMBD: return hc * (uint64_t)g->n_embd;
  case D_HC_MIX: return (2 + hc) * hc;
  case D_HC_SCALES: return HC_SCALES;
  case D_QLORA: return (uint64_t)g->q_lora;
  case D_Q: return q;
  case D_HEAD: return (uint64_t)g->head_dim;
  case D_HEADS: return (uint64_t)g->n_heads;
  case D_OA_IN: return q / (uint64_t)g->o_groups;
  case D_OLORA: return (uint64_t)g->o_lora;
  case D_OGROUPS: return (uint64_t)g->o_groups;
  case D_OB_IN: return (uint64_t)g->o_groups * (uint64_t)g->o_lora;
  case D_IX_DIM: return (uint64_t)g->ix_head_dim;
  case D_IX_Q: return (uint64_t)g->ix_heads * (uint64_t)g->ix_head_dim;
  case D_IX_HEADS: return (uint64_t)g->ix_heads;
  case D_EG_IN: return (uint64_t)g->engram_cols * (uint64_t)g->engram_head_dim;
  case D_EG_OUT: return (hc + 1) * (uint64_t)g->n_embd;
  case D_EG_HEAD: return (uint64_t)g->engram_head_dim;
  case D_EG_ROWS: return layer >= 0 ? g->engram_rows[layer] : 0;
  case D_EG_COLS: return (uint64_t)g->engram_cols;
  case D_EG_LAYERS: return (uint64_t)n_eg;
  case D_EG_NGRAM: return (uint64_t)g->engram_max_ngram;
  case D_EXPERTS: return (uint64_t)g->n_experts;
  case D_FF: return (uint64_t)g->expert_ff;
  default: return 0;
  }
}

static bool has(const Geometry *g, When when, int l)
{
  switch (when)
  {
  case KV_SRC: return g->kv_source[l];
  case GATED: return g->kv_source[l] && g->compress[l] > 1;
  case IX_SRC: return g->ix_source[l];
  case ENGRAM: return g->engram[l];
  default: return true;
  }
}

static bool fail(Check *c, const char *name, const char *why)
{
  snprintf(c->err, WEIGHTS_ERR, "%s: %s", name, why);
  return false;
}

/* a role's tensor of layer l (-1 for the model's own), into its field: present iff it must be, of its dims, a type the
   role allows and the bytes that type's values take */
static bool take(Check *c, const Role *r, int l, const GgufTensor **field)
{
  char name[NAME_MAX_LEN];
  if (l >= 0)
    snprintf(name, sizeof name, "blk.%d.%s", l, r->name);
  else
    snprintf(name, sizeof name, "%s", r->name);
  const Gguf *g = r->in_engram ? c->w->engram : c->w->model;
  const GgufTensor *t = gguf_tensor(g, name);
  const bool want = l < 0 || has(&c->w->geo, r->when, l);
  if (!want)
    return !t || fail(c, name, "present on a layer that has none");
  if (!t)
    return fail(c, name, "missing");
  uint64_t values = 1;
  uint32_t n = 0;
  while (n < GGUF_MAX_DIMS && r->dims[n] != D_NONE)
    n++;
  if (t->n_dims != n)
    return fail(c, name, "of another number of dims");
  for (uint32_t i = 0; i < n; i++)
  {
    if (t->dims[i] != dim(&c->w->geo, r->dims[i], l))
      return fail(c, name, "of another shape");
    values *= t->dims[i];
  }
  if (t->type >= TY_BITS || !(r->types & TY_ID(t->type)))
    return fail(c, name, "of a type this role does not take");
  if (t->bytes != gguf_bytes(t->type, values))
    return fail(c, name, "not the bytes of its type");
  if (r->types == EXPERT_TYPES && t->bytes % t->dims[2])
    return fail(c, name, "experts of unequal bytes");
  c->taken[r->in_engram][t - g->tensors] = true;
  *field = t;
  return true;
}

static bool all_taken(Check *c)
{
  const Gguf *g[2] = {c->w->model, c->w->engram};
  for (int f = 0; f < 2; f++)
    for (size_t i = 0; i < g[f]->n_tensors; i++)
      if (!c->taken[f][i])
        return fail(c, g[f]->tensors[i].name, "a tensor this build does not know");
  return true;
}

static bool check(Check *c)
{
  Weights *w = c->w;
  const GgufTensor *embed = gguf_tensor(w->model, "token_embd.weight");
  if (!embed || embed->n_dims != 2 || embed->dims[1] == 0 || embed->dims[1] > INT32_MAX)
    return fail(c, "token_embd.weight", "missing, or no vocabulary");
  w->geo.n_vocab = (int)embed->dims[1];
  for (size_t i = 0; i < N_TOPS; i++)
    if (!take(c, &tops[i], -1, (const GgufTensor **)((char *)w + tops[i].field)))
      return false;
  for (int l = 0; l < w->geo.n_layers; l++)
    for (size_t i = 0; i < N_ROLES; i++)
      if (!take(c, &roles[i], l, (const GgufTensor **)((char *)&w->layer[l] + roles[i].field)))
        return false;
  return all_taken(c);
}

Weights *weights_create(const Gguf *model, const Gguf *engram, char err[WEIGHTS_ERR])
{
  if (!model)
  {
    snprintf(err, WEIGHTS_ERR, "no model");
    return NULL;
  }
  Weights *w = calloc(1, sizeof *w);
  Check c = {w, {NULL, NULL}, err};
  bool ok = w && weights_geometry(model, engram, &w->geo, err);
  if (ok)
  {
    w->model = model;
    w->engram = engram;
    c.taken[0] = calloc(model->n_tensors + 1, sizeof(bool));
    c.taken[1] = calloc(engram->n_tensors + 1, sizeof(bool));
    ok = (c.taken[0] && c.taken[1]) ? check(&c) : (snprintf(err, WEIGHTS_ERR, "out of memory"), false);
  }
  else if (!w)
    snprintf(err, WEIGHTS_ERR, "out of memory");
  free(c.taken[0]);
  free(c.taken[1]);
  if (!ok)
  {
    weights_destroy(w);
    return NULL;
  }
  return w;
}

void weights_destroy(Weights *w) { free(w); }

size_t weights_experts(const Weights *w, const int *model_fds, PoolUnit *units, size_t cap)
{
  if (!w)
    return 0;
  const size_t n_exp = (size_t)w->geo.n_experts, n = (size_t)w->geo.n_layers * n_exp;
  for (size_t i = 0; units && model_fds && i < n && i < cap; i++)
  {
    const LayerWeights *l = &w->layer[i / n_exp];
    const GgufTensor *t[EXPERT_PARTS] = {l->exp_gate, l->exp_up, l->exp_down};
    units[i] = (PoolUnit){.n_parts = EXPERT_PARTS};
    for (int k = 0; k < EXPERT_PARTS; k++)
    {
      const uint64_t slice = t[k]->bytes / n_exp;
      units[i].part[k] = (PoolRange){model_fds[t[k]->file], t[k]->offset + (i % n_exp) * slice, slice};
    }
  }
  return n;
}
