#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/ink2/weights.h"
#include "osmos/model/qtype.h"

/* a role's types: a mask of the table's ids (qtype.h), or any type its path reads */
#define TY(name) (1u << QTYPE_##name)
#define MAT QTYPE_MV         /* a matrix a token reads: the matvec */
#define EXPERTS QTYPE_MV_TOK /* the experts, by their tokens */
#define EMBED QTYPE_EMBED    /* a token's row */
#define RAW 0                /* read as its bytes: `types` */
#define SMALL (TY(BF16) | TY(F32)) /* a norm's, a clip's, a conv's, a query's: as the checkpoint stores them */
#define NAME_MAX_LEN 96
#define DRAFTER "mtp."      /* the DSpark drafter's tensors (mtp.safetensors): its own roles, not these */

typedef enum Dim
{
  D_NONE, D_EMBD, D_VROWS, D_Q, D_KV, D_REL, D_HEADS, D_HEAD, D_DREL, D_WINDOW, D_EXTENT, D_CONV, D_KVCONV,
  D_EXPERTS, D_FF, D_RES_QUERIES,
} Dim;

/* which layers have a role: every one, a sliding or a full one, and those past the first (an attention residual's
   query: layer 0's attention reads the embedding itself) */
typedef enum When { ALL, SLIDING, FULL, SLIDING_PAST_0, FULL_PAST_0 } When;

typedef struct Role
{
  const char *name;
  size_t field;
  When when;
  uint32_t path;
  uint32_t types;
  Dim dims[GGUF_MAX_DIMS];
} Role;

#define TOP(f) offsetof(Ink2Weights, f)
#define LAY(f) offsetof(Ink2Layer, f)

/* dims as GGUF writes them: the row's values first (a checkpoint's [rows, cols] is {cols, rows}) */
static const Role tops[] = {
  {"token_embd.weight", TOP(embed), ALL, EMBED, 0, {D_EMBD, D_VROWS}},
  {"token_embd_norm.weight", TOP(embed_norm), ALL, RAW, SMALL, {D_EMBD}},
  {"output_res_query", TOP(output_res_query), ALL, RAW, SMALL, {D_EMBD}},
  {"output_norm.weight", TOP(output_norm), ALL, RAW, SMALL, {D_EMBD}},
  {"output.weight", TOP(output), ALL, MAT, 0, {D_EMBD, D_VROWS}},
};

static const Role roles[] = {
  {"attn_res_query", LAY(attn_res_query), SLIDING_PAST_0, RAW, SMALL, {D_EMBD}},
  {"attn_res_query", LAY(attn_res_query), FULL_PAST_0, RAW, SMALL, {D_EMBD, D_RES_QUERIES}},
  {"ffn_res_query", LAY(ffn_res_query), ALL, RAW, SMALL, {D_EMBD}},
  {"attn_norm.weight", LAY(attn_norm), SLIDING, RAW, SMALL, {D_EMBD}},
  {"attn_q_norm.weight", LAY(q_norm), FULL, RAW, SMALL, {D_EMBD}},
  {"attn_k_norm.weight", LAY(k_norm), FULL, RAW, SMALL, {D_EMBD}},
  {"attn_v_norm.weight", LAY(v_norm), FULL, RAW, SMALL, {D_EMBD}},
  {"attn_q.weight", LAY(q), ALL, MAT, 0, {D_EMBD, D_Q}},
  {"attn_k.weight", LAY(k), ALL, MAT, 0, {D_EMBD, D_KV}},
  {"attn_v.weight", LAY(v), ALL, MAT, 0, {D_EMBD, D_KV}},
  {"attn_rel.weight", LAY(rel), ALL, MAT, 0, {D_EMBD, D_REL}},
  {"attn_output.weight", LAY(out), ALL, MAT, 0, {D_Q, D_EMBD}},
  {"attn_gate.weight", LAY(gate), ALL, MAT, 0, {D_EMBD, D_Q}},
  {"attn_sink.weight", LAY(sink), ALL, MAT, 0, {D_EMBD, D_HEADS}},
  {"attn_q_clip", LAY(q_clip), ALL, RAW, SMALL, {D_HEAD, D_HEADS}},
  {"attn_rel_clip", LAY(rel_clip), ALL, RAW, SMALL, {D_DREL, D_HEADS}},
  {"attn_rel_proj", LAY(rel_proj), SLIDING, RAW, SMALL, {D_WINDOW, D_DREL}},
  {"attn_rel_proj", LAY(rel_proj), FULL, RAW, SMALL, {D_EXTENT, D_DREL}},
  {"attn_k_conv", LAY(k_conv), SLIDING, RAW, SMALL, {D_CONV, D_KV}},
  {"attn_v_conv", LAY(v_conv), SLIDING, RAW, SMALL, {D_CONV, D_KV}},
  {"attn_k_conv_static", LAY(k_conv_static), FULL, RAW, SMALL, {D_KV, D_CONV}},
  {"attn_v_conv_static", LAY(v_conv_static), FULL, RAW, SMALL, {D_KV, D_CONV}},
  {"attn_k_conv_proj.weight", LAY(k_conv_proj), FULL, MAT, 0, {D_EMBD, D_KVCONV}},
  {"attn_v_conv_proj.weight", LAY(v_conv_proj), FULL, MAT, 0, {D_EMBD, D_KVCONV}},
  {"attn_out_conv", LAY(out_conv), ALL, RAW, SMALL, {D_CONV, D_EMBD}},
  {"ffn_out_conv", LAY(ffn_out_conv), ALL, RAW, SMALL, {D_CONV, D_EMBD}},
  {"ffn_norm.weight", LAY(ffn_norm), ALL, RAW, SMALL, {D_EMBD}},
  {"ffn_gate_inp.weight", LAY(router), ALL, MAT, 0, {D_EMBD, D_EXPERTS}},
  {"exp_probs_b.bias", LAY(router_bias), ALL, RAW, TY(F32), {D_EXPERTS}},
  {"ffn_gate_exps.weight", LAY(exp_gate), ALL, EXPERTS, 0, {D_EMBD, D_FF, D_EXPERTS}},
  {"ffn_up_exps.weight", LAY(exp_up), ALL, EXPERTS, 0, {D_EMBD, D_FF, D_EXPERTS}},
  {"ffn_down_exps.weight", LAY(exp_down), ALL, EXPERTS, 0, {D_FF, D_EMBD, D_EXPERTS}},
};

/* an expert tensor's own scale (NVFP4's: one an expert), beside it as <name>.scale - claimed after its expert, by type */
static const Role scale_role = {"", 0, ALL, RAW, TY(F32), {D_EXPERTS}};
static const struct
{
  size_t exp, scale;
  const char *name;
} scales[] = {
  {LAY(exp_gate), LAY(exp_gate_scale), "ffn_gate_exps.scale"},
  {LAY(exp_up), LAY(exp_up_scale), "ffn_up_exps.scale"},
  {LAY(exp_down), LAY(exp_down_scale), "ffn_down_exps.scale"},
};

#define N_TOPS (sizeof tops / sizeof *tops)
#define N_ROLES (sizeof roles / sizeof *roles)
#define N_SCALES (sizeof scales / sizeof *scales)

static bool fail(char *err, const char *name, const char *why)
{
  snprintf(err, INK2_ERR, "%s: %s", name, why);
  return false;
}

static uint64_t dim(const Ink2Geometry *g, Dim d)
{
  switch (d)
  {
  case D_EMBD: return (uint64_t)g->n_embd;
  case D_VROWS: return (uint64_t)g->n_vocab_rows;
  case D_Q: return (uint64_t)g->n_heads * (uint64_t)g->head_dim;
  case D_KV: return (uint64_t)g->n_kv_heads * (uint64_t)g->head_dim;
  case D_REL: return (uint64_t)g->n_heads * (uint64_t)g->d_rel;
  case D_HEADS: return (uint64_t)g->n_heads;
  case D_HEAD: return (uint64_t)g->head_dim;
  case D_DREL: return (uint64_t)g->d_rel;
  case D_WINDOW: return (uint64_t)g->window;
  case D_EXTENT: return (uint64_t)g->rel_extent;
  case D_CONV: return (uint64_t)g->conv;
  case D_KVCONV: return (uint64_t)g->n_kv_heads * (uint64_t)g->conv;
  case D_EXPERTS: return (uint64_t)g->n_experts;
  case D_FF: return (uint64_t)g->ff_exp;
  case D_RES_QUERIES: return INK2_RES_QUERIES;
  default: return 0;
  }
}

static bool applies(const Ink2Geometry *g, When w, int l)
{
  switch (w)
  {
  case ALL: return true;
  case SLIDING: return g->sliding[l];
  case FULL: return !g->sliding[l];
  case SLIDING_PAST_0: return l > 0 && g->sliding[l];
  case FULL_PAST_0: return l > 0 && !g->sliding[l];
  }
  return false;
}

/* the tensor of that name checked against its role; taken[] marks it */
static const GgufTensor *claim(const Gguf *model, bool *taken, const Ink2Geometry *g, const Role *r, const char *name,
                               char *err)
{
  const GgufTensor *t = gguf_tensor(model, name);
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
    if (t->dims[i] != dim(g, r->dims[i]))
      return fail(err, name, "of another shape"), NULL;
    values *= t->dims[i];
  }
  const uint32_t paths = qtype_paths(t->type);
  if (r->path == RAW ? !(t->type < 32 && (r->types >> t->type & 1)) : (paths & r->path) != r->path)
    return fail(err, name, "a type its reader does not take"), NULL;
  if (t->bytes != gguf_bytes(t->type, values))
    return fail(err, name, "its bytes not its type's for its shape"), NULL;
  taken[t - model->tensors] = true;
  return t;
}

/* an expert tensor's scale: NVFP4's blocks are not its whole values (their scale one an expert, beside it), every
   other expert type's are - a scale beside one of those is refused, not ignored */
static bool expert_scales(const Gguf *model, bool *taken, const Ink2Geometry *g, int l, Ink2Layer *L, char *err)
{
  char name[NAME_MAX_LEN];
  for (size_t i = 0; i < N_SCALES; i++)
  {
    const GgufTensor *e = *(const GgufTensor **)((char *)L + scales[i].exp);
    snprintf(name, sizeof name, "blk.%d.%s", l, scales[i].name);
    if (e->type != QTYPE_NVFP4)
    {
      if (gguf_tensor(model, name))
        return fail(err, name, "a scale beside an expert type whose blocks are its values");
      continue;
    }
    if (!(*(const GgufTensor **)((char *)L + scales[i].scale) = claim(model, taken, g, &scale_role, name, err)))
      return false;
  }
  return true;
}

void ink2_weights_destroy(Ink2Weights *w) { free(w); }

Ink2Weights *ink2_weights_create(const Gguf *model, char err[INK2_ERR])
{
  if (!model)
    return fail(err, "ink2_weights_create", "no model"), NULL;
  Ink2Weights *w = calloc(1, sizeof *w);
  bool *taken = calloc(model->n_tensors ? model->n_tensors : 1, sizeof *taken);
  if (!w || !taken)
    return free(w), free(taken), fail(err, "ink2_weights_create", "out of memory"), NULL;
  w->model = model;
  Ink2Geometry *g = &w->geo;
  const GgufTensor *embed = gguf_tensor(model, "token_embd.weight");
  bool ok = ink2_geometry(model, g, err) &&
            (embed && embed->n_dims == 2 ? (g->n_vocab_rows = (int)embed->dims[1]) > 0
                                         : fail(err, "token_embd.weight", "missing")) &&
            (g->n_vocab <= g->n_vocab_rows || fail(err, "inkling2.vocab_size", "more tokens than the embedding's rows"));
  for (size_t i = 0; ok && i < N_TOPS; i++)
    ok = (*(const GgufTensor **)((char *)w + tops[i].field) = claim(model, taken, g, &tops[i], tops[i].name, err));
  char name[NAME_MAX_LEN];
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    for (size_t i = 0; ok && i < N_ROLES; i++)
      if (applies(g, roles[i].when, l))
      {
        snprintf(name, sizeof name, "blk.%d.%s", l, roles[i].name);
        ok = (*(const GgufTensor **)((char *)&w->layer[l] + roles[i].field) =
                  claim(model, taken, g, &roles[i], name, err));
      }
    ok = ok && expert_scales(model, taken, g, l, &w->layer[l], err);
  }
  for (size_t i = 0; ok && i < model->n_tensors; i++)
    if (!taken[i] && strncmp(model->tensors[i].name, DRAFTER, strlen(DRAFTER)) != 0)
      ok = fail(err, model->tensors[i].name, "a tensor no role of this layer's kind takes");
  free(taken);
  if (!ok)
    return free(w), NULL;
  return w;
}

size_t ink2_weights_experts(const Ink2Weights *w, const int *model_fds, PoolUnit *units, size_t cap)
{
  if (!w)
    return 0;
  const size_t E = (size_t)w->geo.n_experts, n = (size_t)w->geo.n_layers * E;
  for (int l = 0; l < w->geo.n_layers; l++) /* every layer's three first: nothing written for a model refused */
  {
    const GgufTensor *t[POOL_PARTS] = {w->layer[l].exp_gate, w->layer[l].exp_up, w->layer[l].exp_down};
    for (int k = 0; k < POOL_PARTS; k++)
      if (!t[k] || t[k]->bytes % E)
        return 0;
  }
  for (size_t i = 0; units && model_fds && i < n && i < cap; i++)
  {
    const Ink2Layer *l = &w->layer[i / E];
    const GgufTensor *t[POOL_PARTS] = {l->exp_gate, l->exp_up, l->exp_down};
    units[i] = (PoolUnit){.n_parts = POOL_PARTS};
    for (int k = 0; k < POOL_PARTS; k++)
    {
      const uint64_t slice = t[k]->bytes / E;
      units[i].part[k] = (PoolRange){model_fds[t[k]->file], t[k]->offset + (i % E) * slice, slice};
    }
  }
  return n;
}
