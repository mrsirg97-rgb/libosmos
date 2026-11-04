#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/qtype.h"
#include "weights_geo.h"

/* a role's types, a mask of the table's ids (qtype.h): TY by name, TY_ID by a tensor's id (below 32: a mask's bits) */
#define TY_ID(id) (1u << (id))
#define TY(name) TY_ID(QTYPE_##name)
#define TY_BITS 32
/* the paths a role is read by (qtype.h): it takes any type the table gives them all */
#define MAT QTYPE_MV                          /* fwd_mat: the matvec, the matmul where the type has one */
#define GROUPED QTYPE_MM                      /* mm_group alone (wo_a): no matvec to fall back on */
#define EXPERTS (QTYPE_MV_TOK | QTYPE_MM)     /* the decode's experts by their tokens, the prompt's grouped */
#define EMBED QTYPE_EMBED                     /* ops_embed: a token's row */
#define ROWS QTYPE_DECODE                     /* the Engram's rows, decoded a row at a time */
#define RAW 0                                 /* read as its bytes (a norm, a bias, the gate's halves): `types` */
#define NOT_STORED SIZE_MAX                   /* checked, never read: no field */
#define HC_SCALES 3 /* the hyper-connection mix's pre, post and comb scales */
#define NAME_MAX_LEN 96
#define EXPERT_PARTS 3 /* an expert's gate, up and down */
_Static_assert(EXPERT_PARTS <= POOL_PARTS, "a pool unit holds an expert's three matrices");

/* a dimension, named by what it is in the geometry */
typedef enum Dim
{
  D_NONE, D_EMBD, D_VOCAB, D_HC, D_HC_EMBD, D_HC_MIX, D_HC_SCALES, D_QLORA, D_Q, D_HEAD, D_HEADS, D_OA_IN, D_OLORA,
  D_OGROUPS, D_OB_IN, D_IX_DIM, D_IX_Q, D_IX_HEADS, D_EG_IN, D_EG_OUT, D_EG_HEAD, D_EG_ROWS, D_EG_COLS, D_EG_LAYERS,
  D_EG_NGRAM, D_EXPERTS, D_FF, D_EG_ALL_ROWS, D_OUT_ROWS,
} Dim;

/* which layers have a role; the Engram's two forms (geo.engram_own: weights.h) each their own roles */
typedef enum When { ALL, KV_SRC, GATED, IX_SRC, ENGRAM_FILE, ENGRAM_OWN, FILE_FORM, OWN_FORM } When;

typedef struct Role
{
  const char *name;
  size_t field;
  When when;
  uint32_t path;  /* the paths it is read by; RAW: one of `types` */
  uint32_t types;
  Dim dims[GGUF_MAX_DIMS];
  Dim alt[GGUF_MAX_DIMS]; /* another spelling of the same bytes (wo_a's groups as one dim), or none */
  bool in_engram;
  bool optional; /* present or not; present, checked */
} Role;

#define TOP(f) offsetof(Weights, f)
#define LAY(f) offsetof(LayerWeights, f)

static const Role tops[] = {
  {"token_embd.weight", TOP(embed), ALL, EMBED, 0, {D_EMBD, D_VOCAB}, {D_NONE}, false, false},
  {"output.weight", TOP(output), ALL, MAT, 0, {D_EMBD, D_VOCAB}, {D_NONE}, false, false},
  {"output_norm.weight", TOP(output_norm), ALL, RAW, TY(F32), {D_EMBD}, {D_NONE}, false, false},
  {"engram_token_map.weight", TOP(eg_token_map), FILE_FORM, RAW, TY(I32), {D_VOCAB}, {D_NONE}, false, false},
  {"engram_primes.weight", TOP(eg_primes), FILE_FORM, RAW, TY(I64), {D_EG_COLS, D_EG_LAYERS}, {D_NONE}, false, false},
  {"engram_offsets.weight", TOP(eg_offsets), FILE_FORM, RAW, TY(I64), {D_EG_COLS, D_EG_LAYERS}, {D_NONE}, false,
   false},
  {"engram_multipliers.weight", TOP(eg_multipliers), FILE_FORM, RAW, TY(I64), {D_EG_NGRAM, D_EG_LAYERS}, {D_NONE},
   false, false},
  /* the layers' tables, one after another: each layer's eg_embd a slice of it (slices()) */
  {"per_layer_token_embd.weight", NOT_STORED, OWN_FORM, ROWS, 0, {D_EG_HEAD, D_EG_ALL_ROWS}, {D_NONE}, false, false},
};

static const Role roles[] = {
  {"attn_norm.weight", LAY(attn_norm), ALL, RAW, TY(F32), {D_EMBD}, {D_NONE}, false, false},
  {"attn_q_a.weight", LAY(q_a), ALL, MAT, 0, {D_EMBD, D_QLORA}, {D_NONE}, false, false},
  {"attn_q_a_norm.weight", LAY(q_a_norm), ALL, RAW, TY(F32), {D_QLORA}, {D_NONE}, false, false},
  {"attn_q_b.weight", LAY(q_b), ALL, MAT, 0, {D_QLORA, D_Q}, {D_NONE}, false, false},
  {"attn_kv.weight", LAY(kv), ALL, MAT, 0, {D_EMBD, D_HEAD}, {D_NONE}, false, false},
  {"attn_kv_a_norm.weight", LAY(kv_norm), ALL, RAW, TY(F32), {D_HEAD}, {D_NONE}, false, false},
  {"attn_output_a.weight", LAY(out_a), ALL, GROUPED, 0, {D_OA_IN, D_OLORA, D_OGROUPS}, {D_OA_IN, D_OUT_ROWS}, false,
   false},
  {"attn_output_b.weight", LAY(out_b), ALL, MAT, 0, {D_OB_IN, D_EMBD}, {D_NONE}, false, false},
  {"attn_sinks.weight", LAY(sinks), ALL, RAW, TY(F32), {D_HEADS}, {D_NONE}, false, false},
  {"attn_compressor_kv.weight", LAY(comp_kv), KV_SRC, MAT, 0, {D_EMBD, D_HEAD}, {D_NONE}, false, false},
  {"attn_compressor_gate.weight", LAY(comp_gate), GATED, MAT, 0, {D_EMBD, D_HEAD}, {D_NONE}, false, false},
  {"attn_compressor_norm.weight", LAY(comp_norm), KV_SRC, RAW, TY(F32), {D_HEAD}, {D_NONE}, false, false},
  {"indexer.attn_k.weight", LAY(ix_k), KV_SRC, MAT, 0, {D_HEAD, D_IX_DIM}, {D_NONE}, false, false},
  {"indexer.k_norm.weight", LAY(ix_k_norm), KV_SRC, RAW, TY(F32), {D_IX_DIM}, {D_NONE}, false, false},
  {"indexer.attn_q_b.weight", LAY(ix_q_b), IX_SRC, MAT, 0, {D_QLORA, D_IX_Q}, {D_NONE}, false, false},
  {"indexer.proj.weight", LAY(ix_proj), IX_SRC, MAT, 0, {D_EMBD, D_IX_HEADS}, {D_NONE}, false, false},
  {"engram_q.weight", LAY(eg_q), ENGRAM_FILE, RAW, TY(BF16), {D_EMBD, D_HC}, {D_NONE}, false, false},
  {"engram_k.weight", LAY(eg_k), ENGRAM_FILE, RAW, TY(BF16), {D_EMBD, D_HC}, {D_NONE}, false, false},
  {"engram_gate.weight", LAY(eg_gate), ENGRAM_OWN, RAW, TY(F32), {D_EMBD, D_HC}, {D_NONE}, false, false},
  {"engram_wkv.weight", LAY(eg_wkv), ENGRAM_FILE, MAT, 0, {D_EG_IN, D_EG_OUT}, {D_NONE}, false, false},
  {"engram_kv.weight", LAY(eg_wkv), ENGRAM_OWN, MAT, 0, {D_EG_IN, D_EG_OUT}, {D_NONE}, false, false},
  {"engram_embd.weight", LAY(eg_embd), ENGRAM_FILE, ROWS, 0, {D_EG_HEAD, D_EG_ROWS}, {D_NONE}, true, false},
  {"hc_attn_fn.weight", LAY(hc_attn_fn), ALL, RAW, TY(F32), {D_HC_EMBD, D_HC_MIX}, {D_NONE}, false, false},
  {"hc_attn_scale.weight", LAY(hc_attn_scale), ALL, RAW, TY(F32), {D_HC_SCALES}, {D_NONE}, false, false},
  {"hc_attn_base.weight", LAY(hc_attn_base), ALL, RAW, TY(F32), {D_HC_MIX}, {D_NONE}, false, false},
  {"hc_ffn_fn.weight", LAY(hc_ffn_fn), ALL, RAW, TY(F32), {D_HC_EMBD, D_HC_MIX}, {D_NONE}, false, false},
  {"hc_ffn_scale.weight", LAY(hc_ffn_scale), ALL, RAW, TY(F32), {D_HC_SCALES}, {D_NONE}, false, false},
  {"hc_ffn_base.weight", LAY(hc_ffn_base), ALL, RAW, TY(F32), {D_HC_MIX}, {D_NONE}, false, false},
  {"ffn_norm.weight", LAY(ffn_norm), ALL, RAW, TY(F32), {D_EMBD}, {D_NONE}, false, false},
  {"ffn_gate_inp.weight", LAY(router), ALL, MAT, 0, {D_EMBD, D_EXPERTS}, {D_NONE}, false, false},
  {"exp_probs_b.bias", LAY(router_bias), ALL, RAW, TY(F32), {D_EXPERTS}, {D_NONE}, false, false},
  {"exp_probs_b_vl.bias", NOT_STORED, ALL, RAW, TY(F32), {D_EXPERTS}, {D_NONE}, false, true}, /* images' routing */
  {"ffn_gate_shexp.weight", LAY(sh_gate), ALL, MAT, 0, {D_EMBD, D_FF}, {D_NONE}, false, false},
  {"ffn_up_shexp.weight", LAY(sh_up), ALL, MAT, 0, {D_EMBD, D_FF}, {D_NONE}, false, false},
  {"ffn_down_shexp.weight", LAY(sh_down), ALL, MAT, 0, {D_FF, D_EMBD}, {D_NONE}, false, false},
  {"ffn_gate_exps.weight", LAY(exp_gate), ALL, EXPERTS, 0, {D_EMBD, D_FF, D_EXPERTS}, {D_NONE}, false, false},
  {"ffn_up_exps.weight", LAY(exp_up), ALL, EXPERTS, 0, {D_EMBD, D_FF, D_EXPERTS}, {D_NONE}, false, false},
  {"ffn_down_exps.weight", LAY(exp_down), ALL, EXPERTS, 0, {D_FF, D_EMBD, D_EXPERTS}, {D_NONE}, false, false},
};

#define N_TOPS (sizeof tops / sizeof *tops)
#define N_ROLES (sizeof roles / sizeof *roles)

/* the check's state: the files, which of their tensors a role took, the error */
typedef struct Check
{
  Weights *w;
  bool *taken[3]; /* the model's tensors, the Engram file's, the overlay's */
  char *err;
} Check;

/* the overlay's copy of the tensor of that name, or NULL */
static const GgufTensor *overlaid(const Weights *w, const char *name)
{
  for (size_t i = 0; w->overlay && i < w->overlay->n_tensors; i++)
    if (strcmp(w->ov[i].name, name) == 0)
      return &w->ov[i];
  return NULL;
}

static uint64_t dim(const Geometry *g, Dim d, int layer)
{
  const uint64_t hc = (uint64_t)g->hc, q = (uint64_t)g->n_heads * (uint64_t)g->head_dim;
  int n_eg = 0;
  uint64_t all_rows = 0;
  for (int l = 0; l < g->n_layers; l++)
    n_eg += g->engram[l], all_rows += g->engram[l] ? g->engram_rows[l] : 0;
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
  case D_EG_ALL_ROWS: return all_rows;
  case D_OUT_ROWS: return (uint64_t)g->o_lora * (uint64_t)g->o_groups;
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
  case ENGRAM_FILE: return g->engram[l] && !g->engram_own;
  case ENGRAM_OWN: return g->engram[l] && g->engram_own;
  case FILE_FORM: return !g->engram_own;
  case OWN_FORM: return g->engram_own;
  default: return true;
  }
}

static bool fail(Check *c, const char *name, const char *why)
{
  snprintf(c->err, WEIGHTS_ERR, "%s: %s", name, why);
  return false;
}

/* the role's dims (or their other spelling), the tensor's values counted into *values */
static bool shaped(const Check *c, const GgufTensor *t, const Dim *dims, int l, uint64_t *values)
{
  uint32_t n = 0;
  while (n < GGUF_MAX_DIMS && dims[n] != D_NONE)
    n++;
  if (!n || t->n_dims != n)
    return false;
  *values = 1;
  for (uint32_t i = 0; i < n; i++)
  {
    if (t->dims[i] != dim(&c->w->geo, dims[i], l))
      return false;
    *values *= t->dims[i];
  }
  return true;
}

static bool takes(const Role *r, uint32_t type)
{
  if (r->path == RAW)
    return type < TY_BITS && (r->types & TY_ID(type));
  return (qtype_paths(type) & r->path) == r->path;
}

/* a role's tensor of layer l (-1 for the model's own), into its field: present iff it must be (an optional one
   present or not), of its dims, a type the role takes and the bytes that type's values take */
static bool take(Check *c, const Role *r, int l, const GgufTensor **field)
{
  char name[NAME_MAX_LEN];
  if (l >= 0)
    snprintf(name, sizeof name, "blk.%d.%s", l, r->name);
  else
    snprintf(name, sizeof name, "%s", r->name);
  const Gguf *g = r->in_engram ? c->w->engram : c->w->model;
  const GgufTensor *t = g ? gguf_tensor(g, name) : NULL;
  const GgufTensor *o = r->in_engram ? NULL : overlaid(c->w, name); /* in its place: the model's checked, then it */
  if (o && t)
  {
    c->taken[0][t - g->tensors] = true;
    c->taken[2][o - c->w->ov] = true;
    t = o;
  }
  const bool want = has(&c->w->geo, r->when, l < 0 ? 0 : l);
  if (!want)
    return !t || fail(c, name, l < 0 ? "a tensor of the other Engram form" : "present on a layer that has none");
  if (!t)
    return r->optional || fail(c, name, "missing");
  uint64_t values = 0;
  if (!shaped(c, t, r->dims, l, &values) && !shaped(c, t, r->alt, l, &values))
    return fail(c, name, "of another shape");
  if (!takes(r, t->type))
    return fail(c, name, "of a type this role does not take");
  if (t->bytes != gguf_bytes(t->type, values))
    return fail(c, name, "not the bytes of its type");
  if (r->path == EXPERTS && t->bytes % t->dims[2])
    return fail(c, name, "experts of unequal bytes");
  if (t != o)
    c->taken[r->in_engram][t - g->tensors] = true;
  if (field)
    *field = t;
  return true;
}

/* the Engram's own form: each layer's table a slice of per_layer_token_embd, the layers in order (the tensor's rows
   their sum: its role's shape) */
static void slices(Weights *w)
{
  const GgufTensor *ple = gguf_tensor(w->model, "per_layer_token_embd.weight");
  const uint64_t row = gguf_bytes(ple->type, ple->dims[0]);
  uint64_t at = 0;
  for (int l = 0; l < w->geo.n_layers; l++)
  {
    if (!w->geo.engram[l])
      continue;
    GgufTensor *s = &w->eg_slice[l];
    *s = *ple;
    snprintf(w->eg_slice_name[l], WEIGHTS_SLICE_NAME, "per_layer_token_embd.weight[blk.%d]", l);
    s->name = w->eg_slice_name[l];
    s->dims[1] = w->geo.engram_rows[l];
    s->offset = ple->offset + at * row;
    s->bytes = s->dims[1] * row;
    at += s->dims[1];
    w->layer[l].eg_embd = s;
  }
}

static bool all_taken(Check *c)
{
  const Gguf *g[3] = {c->w->model, c->w->engram, c->w->overlay};
  for (int f = 0; f < 3; f++)
    for (size_t i = 0; g[f] && i < g[f]->n_tensors; i++)
      if (!c->taken[f][i])
        return fail(c, g[f]->tensors[i].name,
                    f == 2 ? "an overlay tensor no role reads" : "a tensor this build does not know");
  return true;
}

/* the files a tensor's index finds (the model's, then the overlay's) and the overlay's tensors as copies indexing
   them: each the model's name, of the model's tensor's dims */
static bool files(Check *c)
{
  Weights *w = c->w;
  const Gguf *m = w->model, *o = w->overlay;
  if (m->n_files + (o ? o->n_files : 0) > WEIGHTS_MAX_FILES)
    return fail(c, o ? o->paths[0] : m->paths[0], "more files than a model takes");
  for (int f = 0; f < m->n_files; f++)
    w->paths[w->n_files++] = m->paths[f];
  if (!o)
    return true;
  for (int f = 0; f < o->n_files; f++)
    w->paths[w->n_files++] = o->paths[f];
  if (!(w->ov = calloc(o->n_tensors + 1, sizeof *w->ov)))
    return fail(c, o->paths[0], "out of memory");
  for (size_t i = 0; i < o->n_tensors; i++)
  {
    const GgufTensor *t = &o->tensors[i], *mt = gguf_tensor(m, t->name);
    if (!mt)
      return fail(c, t->name, "an overlay tensor the model does not have");
    if (mt->n_dims != t->n_dims || memcmp(mt->dims, t->dims, t->n_dims * sizeof *t->dims) != 0)
      return fail(c, t->name, "an overlay tensor of other dims than the model's");
    w->ov[i] = *t;
    w->ov[i].file = m->n_files + t->file;
  }
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
    if (!take(c, &tops[i], -1,
              tops[i].field == NOT_STORED ? NULL : (const GgufTensor **)((char *)w + tops[i].field)))
      return false;
  for (int l = 0; l < w->geo.n_layers; l++)
    for (size_t i = 0; i < N_ROLES; i++)
      if (!take(c, &roles[i], l,
                roles[i].field == NOT_STORED ? NULL : (const GgufTensor **)((char *)&w->layer[l] + roles[i].field)))
        return false;
  if (!all_taken(c))
    return false;
  if (w->geo.engram_own)
    slices(w);
  return true;
}

Weights *weights_create(const Gguf *model, const Gguf *engram, char err[WEIGHTS_ERR])
{
  return weights_create_with(&(WeightsOptions){model, engram, NULL}, err);
}

Weights *weights_create_with(const WeightsOptions *o, char err[WEIGHTS_ERR])
{
  if (!o || !o->model)
  {
    snprintf(err, WEIGHTS_ERR, "no model");
    return NULL;
  }
  Weights *w = calloc(1, sizeof *w);
  Check c = {w, {NULL, NULL, NULL}, err};
  bool ok = w && weights_geometry(o->model, o->engram, &w->geo, err);
  if (ok)
  {
    w->model = o->model, w->engram = o->engram, w->overlay = o->overlay;
    c.taken[0] = calloc(o->model->n_tensors + 1, sizeof(bool));
    c.taken[1] = calloc((o->engram ? o->engram->n_tensors : 0) + 1, sizeof(bool));
    c.taken[2] = calloc((o->overlay ? o->overlay->n_tensors : 0) + 1, sizeof(bool));
    ok = (c.taken[0] && c.taken[1] && c.taken[2]) ? files(&c) && check(&c)
                                                    : (snprintf(err, WEIGHTS_ERR, "out of memory"), false);
  }
  else if (!w)
    snprintf(err, WEIGHTS_ERR, "out of memory");
  free(c.taken[0]);
  free(c.taken[1]);
  free(c.taken[2]);
  if (!ok)
  {
    weights_destroy(w);
    return NULL;
  }
  return w;
}

size_t weights_tensors(const Weights *w)
{
  return w->model->n_tensors + (w->overlay ? w->overlay->n_tensors : 0);
}

size_t weights_tensor_index(const Weights *w, const GgufTensor *t)
{
  const bool own = t >= w->model->tensors && t < w->model->tensors + w->model->n_tensors;
  return own ? (size_t)(t - w->model->tensors) : w->model->n_tensors + (size_t)(t - w->ov);
}

void weights_destroy(Weights *w)
{
  if (w)
    free(w->ov);
  free(w);
}

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
