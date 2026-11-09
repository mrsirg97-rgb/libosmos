#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/gpu/q4x_ops.h"
#include "osmos/model/qtype.h"
#include "../model/gguf_meta.h"
#include "q4x_geo.h"

/* a role's types: a mask of the table's ids (qtype.h), or any type its path reads */
#define TY(name) (1u << QTYPE_##name)
#define MAT QTYPE_MV        /* a matrix a token reads: the matvec */
#define EXPERTS QTYPE_MV_TOK /* the experts, by their tokens */
#define EMBED QTYPE_EMBED   /* a token's row */
#define ROWS QTYPE_DECODE   /* the n-gram table's rows, decoded a row at a time */
#define RAW 0               /* read as its bytes: `types` */
#define NAME_MAX_LEN 96

typedef enum Dim
{
  D_NONE, D_EMBD, D_VOCAB, D_HC, D_HC_EMBD, D_HC_RANK, D_EXPERTS, D_FF, D_SH_FF, D_QKV, D_INNER, D_CONV, D_VHEADS,
  D_STATE, D_Q2, D_Q, D_KV, D_HEAD, D_IX_Q, D_IX_DIM, D_PLE_CONV, D_PLE_DIM, D_PLE_ROWS, D_2EMBD,
} Dim;

typedef enum When { ALL, DELTANET, ATTENTION, PLE } When;

typedef struct Role
{
  const char *name;
  size_t field;
  When when;
  uint32_t path;
  uint32_t types;
  Dim dims[GGUF_MAX_DIMS];
} Role;

#define TOP(f) offsetof(Q4xWeights, f)
#define LAY(f) offsetof(Q4xLayer, f)

static const Role tops[] = {
  {"token_embd.weight", TOP(embed), ALL, EMBED, 0, {D_EMBD, D_VOCAB}},
  {"output.weight", TOP(output), ALL, MAT, 0, {D_EMBD, D_VOCAB}},
  {"output_hc_norm.weight", TOP(out_hc_norm), ALL, RAW, TY(F32), {D_HC_EMBD}},
  {"output_hc_down.weight", TOP(out_hc_down), ALL, MAT, 0, {D_HC_EMBD, D_HC_RANK}},
  {"output_hc_up.weight", TOP(out_hc_up), ALL, MAT, 0, {D_HC_RANK, D_HC_EMBD}},
  {"per_layer_token_embd.weight", TOP(ple_table), ALL, RAW, Q4X_ROWS_TYPES, {D_PLE_DIM, D_PLE_ROWS}}, /* q4x_rows' types */
};

static const Role roles[] = {
  {"hc_attn_norm.weight", LAY(hc_attn_norm), ALL, RAW, TY(F32), {D_HC_EMBD}},
  {"hc_attn_down.weight", LAY(hc_attn_down), ALL, MAT, 0, {D_HC_EMBD, D_HC_RANK}},
  {"hc_attn_up.weight", LAY(hc_attn_up), ALL, MAT, 0, {D_HC_RANK, D_HC_EMBD}},
  {"hc_attn_inject.weight", LAY(hc_attn_inject), ALL, MAT, 0, {D_HC_EMBD, D_HC}},
  {"hc_ffn_norm.weight", LAY(hc_ffn_norm), ALL, RAW, TY(F32), {D_HC_EMBD}},
  {"hc_ffn_down.weight", LAY(hc_ffn_down), ALL, MAT, 0, {D_HC_EMBD, D_HC_RANK}},
  {"hc_ffn_up.weight", LAY(hc_ffn_up), ALL, MAT, 0, {D_HC_RANK, D_HC_EMBD}},
  {"hc_ffn_inject.weight", LAY(hc_ffn_inject), ALL, MAT, 0, {D_HC_EMBD, D_HC}},
  {"attn_q.weight", LAY(q), ATTENTION, MAT, 0, {D_EMBD, D_Q2}},
  {"attn_k.weight", LAY(k), ATTENTION, MAT, 0, {D_EMBD, D_KV}},
  {"attn_v.weight", LAY(v), ATTENTION, MAT, 0, {D_EMBD, D_KV}},
  {"attn_output.weight", LAY(out), ATTENTION, MAT, 0, {D_Q, D_EMBD}},
  {"attn_q_norm.weight", LAY(q_norm), ATTENTION, RAW, TY(F32), {D_HEAD}},
  {"attn_k_norm.weight", LAY(k_norm), ATTENTION, RAW, TY(F32), {D_HEAD}},
  {"indexer.q_proj.weight", LAY(ix_q), ATTENTION, MAT, 0, {D_EMBD, D_IX_Q}},
  {"indexer.k_proj.weight", LAY(ix_k), ATTENTION, MAT, 0, {D_EMBD, D_IX_DIM}},
  {"indexer.q_norm.weight", LAY(ix_q_norm), ATTENTION, RAW, TY(F32), {D_IX_DIM}},
  {"indexer.k_norm.weight", LAY(ix_k_norm), ATTENTION, RAW, TY(F32), {D_IX_DIM}},
  {"attn_qkv.weight", LAY(qkv), DELTANET, MAT, 0, {D_EMBD, D_QKV}},
  {"attn_gate.weight", LAY(gate), DELTANET, MAT, 0, {D_EMBD, D_INNER}},
  {"ssm_conv1d.weight", LAY(conv), DELTANET, RAW, TY(F32), {D_CONV, D_QKV}},
  {"ssm_dt.bias", LAY(dt), DELTANET, RAW, TY(F32), {D_VHEADS}},
  {"ssm_a", LAY(a), DELTANET, RAW, TY(F32), {D_VHEADS}},
  {"ssm_alpha.weight", LAY(alpha), DELTANET, MAT, 0, {D_EMBD, D_VHEADS}},
  {"ssm_beta.weight", LAY(beta), DELTANET, MAT, 0, {D_EMBD, D_VHEADS}},
  {"ssm_norm.weight", LAY(ssm_norm), DELTANET, RAW, TY(F32), {D_STATE}},
  {"ssm_out.weight", LAY(ssm_out), DELTANET, MAT, 0, {D_INNER, D_EMBD}},
  {"ple_key.weight", LAY(ple_key), PLE, MAT, 0, {D_EMBD, D_HC_EMBD}},
  {"ple_value.weight", LAY(ple_value), PLE, MAT, 0, {D_EMBD, D_EMBD}},
  {"ple_norm_key.weight", LAY(ple_norm_key), PLE, RAW, TY(F32), {D_HC_EMBD}},
  {"ple_norm_query.weight", LAY(ple_norm_query), PLE, RAW, TY(F32), {D_HC_EMBD}},
  {"ple_norm_conv.weight", LAY(ple_norm_conv), PLE, RAW, TY(F32), {D_HC_EMBD}},
  {"ple_conv1d.weight", LAY(ple_conv), PLE, RAW, TY(F32), {D_PLE_CONV, D_HC_EMBD}},
  {"ffn_gate_inp.weight", LAY(router), ALL, MAT, 0, {D_EMBD, D_EXPERTS}},
  {"ffn_gate_exps.weight", LAY(exp_gate), ALL, EXPERTS, 0, {D_EMBD, D_FF, D_EXPERTS}},
  {"ffn_up_exps.weight", LAY(exp_up), ALL, EXPERTS, 0, {D_EMBD, D_FF, D_EXPERTS}},
  {"ffn_down_exps.weight", LAY(exp_down), ALL, EXPERTS, 0, {D_FF, D_EMBD, D_EXPERTS}},
  {"ffn_gate_inp_shexp.weight", LAY(sh_gate_inp), ALL, RAW, TY(F32), {D_EMBD}},
  {"ffn_gate_shexp.weight", LAY(sh_gate), ALL, MAT, 0, {D_EMBD, D_SH_FF}},
  {"ffn_up_shexp.weight", LAY(sh_up), ALL, MAT, 0, {D_EMBD, D_SH_FF}},
  {"ffn_down_shexp.weight", LAY(sh_down), ALL, MAT, 0, {D_SH_FF, D_EMBD}},
};

#define NEXTN(f) offsetof(Q4xMtp, f)

/* the MTP head's own, after its layer's (blk.<n_layers>.nextn.*) */
static const Role nextn[] = {
  {"nextn.eh_proj.weight", NEXTN(eh_proj), ALL, MAT, 0, {D_2EMBD, D_EMBD}},
  {"nextn.enorm.weight", NEXTN(enorm), ALL, RAW, TY(F32), {D_EMBD}},
  {"nextn.hnorm.weight", NEXTN(hnorm), ALL, RAW, TY(F32), {D_HC_EMBD}},
  {"nextn.hc_head_norm.weight", NEXTN(head_norm), ALL, RAW, TY(F32), {D_HC_EMBD}},
  {"nextn.hc_head_down.weight", NEXTN(head_down), ALL, MAT, 0, {D_HC_EMBD, D_HC_RANK}},
  {"nextn.hc_head_up.weight", NEXTN(head_up), ALL, MAT, 0, {D_HC_RANK, D_HC_EMBD}},
};

#define N_TOPS (sizeof tops / sizeof *tops)
#define N_NEXTN (sizeof nextn / sizeof *nextn)
#define N_ROLES (sizeof roles / sizeof *roles)

static bool fail(char *err, const char *name, const char *why)
{
  snprintf(err, Q4X_ERR, "%s: %s", name, why);
  return false;
}

/* the most rows the n-gram heads reach: the table holds at least these (the converter pads it) */
static uint64_t ple_rows(const Q4xGeometry *g)
{
  return g->ple_offset[g->ple_heads - 1] + g->ple_rows[g->ple_heads - 1];
}

static uint64_t dim(const Q4xGeometry *g, Dim d)
{
  const uint64_t hc_embd = (uint64_t)g->hc * (uint64_t)g->n_embd;
  switch (d)
  {
  case D_EMBD: return (uint64_t)g->n_embd;
  case D_VOCAB: return (uint64_t)g->n_vocab;
  case D_HC: return (uint64_t)g->hc;
  case D_HC_EMBD: return hc_embd;
  case D_HC_RANK: return (uint64_t)g->hc_rank;
  case D_EXPERTS: return (uint64_t)g->n_experts;
  case D_FF: return (uint64_t)g->ff_exp;
  case D_SH_FF: return (uint64_t)g->ff_shexp;
  case D_QKV: return 2ull * (uint64_t)g->ssm_groups * (uint64_t)g->ssm_state + (uint64_t)g->ssm_inner;
  case D_INNER: return (uint64_t)g->ssm_inner;
  case D_CONV: return (uint64_t)g->ssm_conv;
  case D_VHEADS: return (uint64_t)g->ssm_v_heads;
  case D_STATE: return (uint64_t)g->ssm_state;
  case D_Q2: return 2ull * (uint64_t)g->n_heads * (uint64_t)g->head_dim; /* each head's q, then its gate */
  case D_Q: return (uint64_t)g->n_heads * (uint64_t)g->head_dim;
  case D_KV: return (uint64_t)g->n_kv_heads * (uint64_t)g->head_dim;
  case D_HEAD: return (uint64_t)g->head_dim;
  case D_IX_Q: return (uint64_t)g->ix_heads * (uint64_t)g->ix_dim;
  case D_IX_DIM: return (uint64_t)g->ix_dim;
  case D_PLE_CONV: return (uint64_t)g->ple_conv;
  case D_PLE_DIM: return (uint64_t)g->ple_dim;
  case D_PLE_ROWS: return ple_rows(g);
  case D_2EMBD: return 2ull * (uint64_t)g->n_embd;
  default: return 0;
  }
}

static bool applies(const Q4xGeometry *g, When w, int l)
{
  switch (w)
  {
  case ALL: return true;
  case DELTANET: return g->recurrent[l];
  case ATTENTION: return !g->recurrent[l];
  case PLE: return l == g->ple_layer;
  }
  return false;
}

/* the tensor of that name checked against its role - an overlay's where it has one (standing in for the model's of
   its name, which must be there), else the model's; taken[] marks the model's (the overlay's: taken_ov[]) */
typedef struct Claim
{
  const Gguf *model, *overlay;
  bool *taken, *taken_ov;
} Claim;

static const GgufTensor *claim(const Claim *c, const Q4xGeometry *g, const Role *r, const char *name, char *err)
{
  const GgufTensor *mt = gguf_tensor(c->model, name), *ot = c->overlay ? gguf_tensor(c->overlay, name) : NULL;
  if (!mt)
    return fail(err, name, ot ? "an overlay tensor the model has none of" : "missing"), NULL;
  if (ot)
    c->taken[mt - c->model->tensors] = true; /* shadowed: the overlay's is read */
  const GgufTensor *t = ot ? ot : mt;
  int n = 0;
  while (n < GGUF_MAX_DIMS && r->dims[n] != D_NONE)
    n++;
  if (t->n_dims != (uint32_t)n)
    return fail(err, name, "of another number of dims"), NULL;
  uint64_t values = 1;
  for (int i = 0; i < n; i++)
  {
    const uint64_t want = dim(g, r->dims[i]);
    if (r->dims[i] == D_PLE_ROWS ? t->dims[i] < want : t->dims[i] != want)
      return fail(err, name, r->dims[i] == D_PLE_ROWS ? "fewer rows than the n-gram heads reach" : "of another shape"),
             NULL;
    values *= t->dims[i];
  }
  const uint32_t paths = qtype_paths(t->type);
  if (r->path == RAW ? !(t->type < 32 && (r->types >> t->type & 1)) : (paths & r->path) != r->path)
    return fail(err, name, "a type its reader does not take"), NULL;
  if (t->bytes != gguf_bytes(t->type, values))
    return fail(err, name, "its bytes not its type's for its shape"), NULL;
  if (ot)
    c->taken_ov[ot - c->overlay->tensors] = true;
  else
    c->taken[mt - c->model->tensors] = true;
  return t;
}

void q4x_weights_destroy(Q4xWeights *w) { free(w); }

Q4xWeights *q4x_weights_create(const Gguf *model, const Gguf *overlay, char err[Q4X_ERR])
{
  if (!model)
    return fail(err, "q4x_weights_create", "no model"), NULL;
  Q4xWeights *w = calloc(1, sizeof *w);
  bool *taken = calloc(model->n_tensors ? model->n_tensors : 1, sizeof *taken);
  bool *taken_ov = calloc(overlay && overlay->n_tensors ? overlay->n_tensors : 1, sizeof *taken_ov);
  if (!w || !taken || !taken_ov)
    return free(w), free(taken), free(taken_ov), fail(err, "q4x_weights_create", "out of memory"), NULL;
  w->model = model, w->overlay = overlay;
  const Claim c = {model, overlay, taken, taken_ov};
  Q4xGeometry *g = &w->geo;
  const GgufTensor *embed = gguf_tensor(model, "token_embd.weight");
  bool ok = q4x_geometry(model, g, err) &&
            (embed && embed->n_dims == 2 ? (g->n_vocab = (int)embed->dims[1]) > 0 : fail(err, "token_embd.weight", "missing"));
  for (size_t i = 0; ok && i < N_TOPS; i++)
    ok = (*(const GgufTensor **)((char *)w + tops[i].field) = claim(&c, g, &tops[i], tops[i].name, err));
  char name[NAME_MAX_LEN];
  for (int l = 0; ok && l < g->n_layers; l++)
    for (size_t i = 0; ok && i < N_ROLES; i++)
      if (applies(g, roles[i].when, l))
      {
        snprintf(name, sizeof name, "blk.%d.%s", l, roles[i].name);
        ok = (*(const GgufTensor **)((char *)&w->layer[l] + roles[i].field) = claim(&c, g, &roles[i], name, err));
      }
  for (size_t i = 0; ok && i < model->n_tensors; i++)
    if (!taken[i])
      ok = fail(err, model->tensors[i].name, "a tensor no role of this layer's kind takes");
  for (size_t i = 0; ok && overlay && i < overlay->n_tensors; i++)
    if (!taken_ov[i])
      ok = fail(err, overlay->tensors[i].name, "an overlay tensor no role reads");
  free(taken), free(taken_ov);
  if (!ok)
    return free(w), NULL;
  return w;
}

void q4x_mtp_destroy(Q4xMtp *m) { free(m); }

/* the head's metadata against the model's geometry: its keys the model's widths, its one layer after the model's */
static bool mtp_meta(const Gguf *mtp, const Q4xGeometry *g, char *err)
{
  GgufMeta m = {mtp, err, NULL, 0};
  const struct
  {
    const char *key;
    int want;
  } same[] = {
      {"qwen4exp.block_count", g->n_layers + 1},
      {"qwen4exp.nextn_predict_layers", 1},
      {"qwen4exp.embedding_length", g->n_embd},
      {"qwen4exp.hyper_connection.count", g->hc},
      {"qwen4exp.hyper_connection.low_rank", g->hc_rank},
      {"qwen4exp.attention.head_count", g->n_heads},
      {"qwen4exp.attention.head_count_kv", g->n_kv_heads},
      {"qwen4exp.attention.key_length", g->head_dim},
      {"qwen4exp.rope.dimension_count", g->rope_dim},
      {"qwen4exp.expert_count", g->n_experts},
      {"qwen4exp.expert_used_count", g->n_used},
      {"qwen4exp.expert_feed_forward_length", g->ff_exp},
      {"qwen4exp.expert_shared_feed_forward_length", g->ff_shexp},
  };
  if (!meta_text_is(&m, "general.architecture", "qwen4exp"))
    return false;
  if (!meta_present(&m, "qwen4exp.nextn_predict_layers"))
    return meta_fail(&m, "qwen4exp.nextn_predict_layers", "missing: not an MTP (nextn) head");
  for (size_t i = 0; i < sizeof same / sizeof *same; i++)
    if (!meta_uint_is(&m, same[i].key, (uint64_t)same[i].want))
      return false;
  return meta_flag_set(&m, "qwen4exp.nextn_shared_target_tensors") ||
         meta_fail(&m, "qwen4exp.nextn_shared_target_tensors",
                   "not set: a head carrying its own embedding and output (the shared variant borrows the model's)");
}

Q4xMtp *q4x_mtp_create(const Gguf *mtp, const Q4xWeights *w, char err[Q4X_ERR])
{
  if (!mtp || !w)
    return fail(err, "q4x_mtp_create", "no head or no model"), NULL;
  Q4xMtp *m = calloc(1, sizeof *m);
  bool *taken = calloc(mtp->n_tensors ? mtp->n_tensors : 1, sizeof *taken);
  if (!m || !taken)
    return free(m), free(taken), fail(err, "q4x_mtp_create", "out of memory"), NULL;
  m->model = mtp;
  const Claim c = {mtp, NULL, taken, NULL}; /* the head's own file: no overlay */
  const Q4xGeometry *g = &w->geo;
  const int l = g->n_layers;
  char name[NAME_MAX_LEN];
  bool ok = mtp_meta(mtp, g, err);
  for (size_t i = 0; ok && i < N_ROLES; i++)
    if (roles[i].when == ALL || roles[i].when == ATTENTION)
    {
      snprintf(name, sizeof name, "blk.%d.%s", l, roles[i].name);
      ok = (*(const GgufTensor **)((char *)&m->layer + roles[i].field) = claim(&c, g, &roles[i], name, err));
    }
  for (size_t i = 0; ok && i < N_NEXTN; i++)
  {
    snprintf(name, sizeof name, "blk.%d.%s", l, nextn[i].name);
    ok = (*(const GgufTensor **)((char *)m + nextn[i].field) = claim(&c, g, &nextn[i], name, err));
  }
  for (size_t i = 0; ok && i < mtp->n_tensors; i++)
    if (!taken[i])
      ok = fail(err, mtp->tensors[i].name, "a tensor no role of the MTP head takes");
  free(taken);
  if (!ok)
    return free(m), NULL;
  return m;
}
