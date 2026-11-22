#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/ink2/weights.h"
#include "osmos/model/expert_units.h"
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
  D_NONE, D_EMBD, D_EMBD2, D_VROWS, D_Q, D_KV, D_REL, D_HEADS, D_HEAD, D_DREL, D_WINDOW, D_EXTENT, D_CONV, D_KVCONV,
  D_EXPERTS, D_FF, D_FF2, D_RES_QUERIES, D_RANK, D_CORE2,
} Dim;

/* which layers have a role: every one, a sliding or a full one, those past the first (an attention residual's query:
   layer 0's attention reads the embedding itself), the loop's core; the top's: always, or with a loop */
typedef enum When { ALL, SLIDING, FULL, SLIDING_PAST_0, FULL_PAST_0, CORE, LOOP } When;

/* a role's tensor in one of the files' dialects (Ink2Keys): its name and dims; NULL, none there */
typedef struct Spelling
{
  const char *name;
  Dim dims[GGUF_MAX_DIMS];
} Spelling;

typedef struct Role
{
  size_t field;
  When when;
  uint32_t path;
  uint32_t types;
  Spelling as[INK2_KEYS_N]; /* ours, unsloth's Inkling-2-Small's, upstream's */
} Role;

#define TOP(f) offsetof(Ink2Weights, f)
#define LAY(f) offsetof(Ink2Layer, f)
#define SP(n, ...) {n, {__VA_ARGS__}}
#define SAME(n, ...) {SP(n, __VA_ARGS__), SP(n, __VA_ARGS__), SP(n, __VA_ARGS__)}
#define LOOPED(n, ...) {SP(n, __VA_ARGS__), {NULL, {D_NONE}}, SP(n, __VA_ARGS__)} /* the loop's: ours, unsloth's Mini */

/* dims as GGUF writes them: the row's values first (a checkpoint's [rows, cols] is {cols, rows}) */
static const Role tops[] = {
  {TOP(embed), ALL, EMBED, 0, SAME("token_embd.weight", D_EMBD, D_VROWS)},
  {TOP(embed_norm), ALL, RAW, SMALL, SAME("token_embd_norm.weight", D_EMBD)},
  {TOP(output_res_query), ALL, RAW, SMALL,
   {SP("output_res_query", D_EMBD), SP("output_res_score.weight", D_EMBD), SP("output_res_query.weight", D_EMBD)}},
  {TOP(output_norm), ALL, RAW, SMALL, SAME("output_norm.weight", D_EMBD)},
  {TOP(output), ALL, MAT, 0, SAME("output.weight", D_EMBD, D_VROWS)},
  {TOP(loop_query), LOOP, RAW, SMALL, LOOPED("loop_res_query.weight", D_EMBD, D_CORE2)},
  {TOP(loop_kv_query), LOOP, RAW, SMALL, LOOPED("loop_kv_res_query.weight", D_EMBD, D_CORE2)},
};

#define CTRL(c, n)                                                                                                     \
  {LAY(ctrl_down[c]), CORE, MAT, 0, LOOPED("loop_" n "_down.weight", D_EMBD2, D_RANK)},                         \
      {LAY(ctrl_gain[c]), CORE, MAT, 0, LOOPED("loop_" n "_gain.weight", D_RANK, D_EMBD)},                      \
      {LAY(ctrl_forget[c]), CORE, MAT, 0, LOOPED("loop_" n "_forget.weight", D_RANK, D_EMBD)}

static const Role roles[] = {
  {LAY(attn_res_query), SLIDING_PAST_0, RAW, SMALL,
   {SP("attn_res_query", D_EMBD), SP("attn_res_score.weight", D_EMBD), SP("attn_res_query.weight", D_EMBD)}},
  {LAY(attn_res_query), FULL_PAST_0, RAW, SMALL,
   {SP("attn_res_query", D_EMBD, D_RES_QUERIES), SP("attn_res_score.weight", D_EMBD, D_RES_QUERIES),
    SP("attn_res_query.weight", D_EMBD, D_RES_QUERIES)}},
  {LAY(ffn_res_query), ALL, RAW, SMALL,
   {SP("ffn_res_query", D_EMBD), SP("ffn_res_score.weight", D_EMBD), SP("ffn_res_query.weight", D_EMBD)}},
  {LAY(attn_norm), SLIDING, RAW, SMALL, SAME("attn_norm.weight", D_EMBD)},
  {LAY(q_norm), FULL, RAW, SMALL,
   {SP("attn_q_norm.weight", D_EMBD), SP("attn_q_in_norm.weight", D_EMBD), SP("attn_q_inp_norm.weight", D_EMBD)}},
  {LAY(k_norm), FULL, RAW, SMALL,
   {SP("attn_k_norm.weight", D_EMBD), SP("attn_k_in_norm.weight", D_EMBD), SP("attn_k_inp_norm.weight", D_EMBD)}},
  {LAY(v_norm), FULL, RAW, SMALL,
   {SP("attn_v_norm.weight", D_EMBD), SP("attn_v_in_norm.weight", D_EMBD), SP("attn_v_inp_norm.weight", D_EMBD)}},
  {LAY(q), ALL, MAT, 0, SAME("attn_q.weight", D_EMBD, D_Q)},
  {LAY(k), ALL, MAT, 0, SAME("attn_k.weight", D_EMBD, D_KV)},
  {LAY(v), ALL, MAT, 0, SAME("attn_v.weight", D_EMBD, D_KV)},
  {LAY(rel), ALL, MAT, 0,
   {SP("attn_rel.weight", D_EMBD, D_REL), SP("attn_r.weight", D_EMBD, D_REL), SP("attn_rel.weight", D_EMBD, D_REL)}},
  {LAY(out), ALL, MAT, 0, SAME("attn_output.weight", D_Q, D_EMBD)},
  {LAY(gate), ALL, MAT, 0, SAME("attn_gate.weight", D_EMBD, D_Q)},
  {LAY(sink), ALL, MAT, 0,
   {SP("attn_sink.weight", D_EMBD, D_HEADS), SP("attn_sink_proj.weight", D_EMBD, D_HEADS),
    SP("attn_sink_proj.weight", D_EMBD, D_HEADS)}},
  {LAY(q_clip), ALL, RAW, SMALL,
   {SP("attn_q_clip", D_HEAD, D_HEADS), SP("attn_q_clip.weight", D_HEAD, D_HEADS),
    SP("attn_q_clip.weight", D_HEAD, D_HEADS)}},
  {LAY(rel_clip), ALL, RAW, SMALL,
   {SP("attn_rel_clip", D_DREL, D_HEADS), SP("attn_r_clip.weight", D_DREL, D_HEADS),
    SP("attn_rel_clip.weight", D_DREL, D_HEADS)}},
  {LAY(rel_proj), SLIDING, RAW, SMALL,
   {SP("attn_rel_proj", D_WINDOW, D_DREL), SP("attn_rel_proj.weight", D_WINDOW, D_DREL),
    SP("attn_rel_bias.weight", D_DREL, D_WINDOW)}},
  {LAY(rel_proj), FULL, RAW, SMALL,
   {SP("attn_rel_proj", D_EXTENT, D_DREL), SP("attn_rel_proj.weight", D_EXTENT, D_DREL),
    SP("attn_rel_bias.weight", D_DREL, D_EXTENT)}},
  {LAY(k_conv), SLIDING, RAW, SMALL,
   {SP("attn_k_conv", D_CONV, D_KV), SP("shortconv_k.weight", D_CONV, D_KV), SP("attn_k_conv.weight", D_CONV, D_KV)}},
  {LAY(v_conv), SLIDING, RAW, SMALL,
   {SP("attn_v_conv", D_CONV, D_KV), SP("shortconv_v.weight", D_CONV, D_KV), SP("attn_v_conv.weight", D_CONV, D_KV)}},
  {LAY(k_conv_static), FULL, RAW, SMALL,
   {SP("attn_k_conv_static", D_KV, D_CONV), SP("shortconv_k.weight", D_CONV, D_KV),
    SP("attn_k_conv.weight", D_CONV, D_KV)}},
  {LAY(v_conv_static), FULL, RAW, SMALL,
   {SP("attn_v_conv_static", D_KV, D_CONV), SP("shortconv_v.weight", D_CONV, D_KV),
    SP("attn_v_conv.weight", D_CONV, D_KV)}},
  {LAY(k_conv_proj), FULL, MAT, 0,
   {SP("attn_k_conv_proj.weight", D_EMBD, D_KVCONV), SP("shortconv_k_dyn.weight", D_EMBD, D_KVCONV),
    SP("attn_k_conv_proj.weight", D_EMBD, D_KVCONV)}},
  {LAY(v_conv_proj), FULL, MAT, 0,
   {SP("attn_v_conv_proj.weight", D_EMBD, D_KVCONV), SP("shortconv_v_dyn.weight", D_EMBD, D_KVCONV),
    SP("attn_v_conv_proj.weight", D_EMBD, D_KVCONV)}},
  {LAY(out_conv), ALL, RAW, SMALL,
   {SP("attn_out_conv", D_CONV, D_EMBD), SP("shortconv_attn.weight", D_CONV, D_EMBD),
    SP("attn_post_conv.weight", D_CONV, D_EMBD)}},
  {LAY(ffn_out_conv), ALL, RAW, SMALL,
   {SP("ffn_out_conv", D_CONV, D_EMBD), SP("shortconv_mlp.weight", D_CONV, D_EMBD),
    SP("ffn_post_conv.weight", D_CONV, D_EMBD)}},
  {LAY(ffn_norm), ALL, RAW, SMALL, SAME("ffn_norm.weight", D_EMBD)},
  {LAY(router), ALL, MAT, 0, SAME("ffn_gate_inp.weight", D_EMBD, D_EXPERTS)},
  {LAY(router_bias), ALL, RAW, TY(F32), SAME("exp_probs_b.bias", D_EXPERTS)},
  {LAY(exp_gate), ALL, EXPERTS, 0,
   {SP("ffn_gate_exps.weight", D_EMBD, D_FF, D_EXPERTS), SP("ffn_gate_exps.weight", D_EMBD, D_FF, D_EXPERTS),
    SP("ffn_gate_up_exps.weight", D_EMBD, D_FF2, D_EXPERTS)}},
  {LAY(exp_up), ALL, EXPERTS, 0,
   {SP("ffn_up_exps.weight", D_EMBD, D_FF, D_EXPERTS), SP("ffn_up_exps.weight", D_EMBD, D_FF, D_EXPERTS),
    SP("ffn_gate_up_exps.weight", D_EMBD, D_FF2, D_EXPERTS)}},
  {LAY(exp_down), ALL, EXPERTS, 0, SAME("ffn_down_exps.weight", D_FF, D_EMBD, D_EXPERTS)},
  CTRL(INK2_CTRL_Q, "attn"),
  CTRL(INK2_CTRL_K, "k"),
  CTRL(INK2_CTRL_V, "v"),
  CTRL(INK2_CTRL_FFN, "mlp"),
};

/* an expert tensor's own scale (NVFP4's: one an expert), beside it as <name>.scale - claimed after its expert, by type;
   a fused gate and up's one scale both's */
static const Role scale_role = {0, ALL, RAW, TY(F32), SAME("", D_EXPERTS)};
static const struct
{
  size_t exp, scale;
  const char *name[INK2_KEYS_N];
} scales[] = {
  {LAY(exp_gate), LAY(exp_gate_scale), {"ffn_gate_exps.scale", "ffn_gate_exps.scale", "ffn_gate_up_exps.scale"}},
  {LAY(exp_up), LAY(exp_up_scale), {"ffn_up_exps.scale", "ffn_up_exps.scale", "ffn_gate_up_exps.scale"}},
  {LAY(exp_down), LAY(exp_down_scale), {"ffn_down_exps.scale", "ffn_down_exps.scale", "ffn_down_exps.scale"}},
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
  case D_EMBD2: return 2u * (uint64_t)g->n_embd;
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
  case D_FF2: return 2u * (uint64_t)g->ff_exp;
  case D_RES_QUERIES: return INK2_RES_QUERIES;
  case D_RANK: return (uint64_t)g->loop.gate_rank;
  case D_CORE2: return 2u * (uint64_t)(g->loop.end - g->loop.start);
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
  case CORE: return g->loop.passes && l >= g->loop.start && l < g->loop.end;
  case LOOP: return g->loop.passes > 0;
  }
  return false;
}

/* the tensor of that name checked against its role's spelling in the file's dialect; taken[] marks it */
static const GgufTensor *claim(const Gguf *model, bool *taken, const Ink2Geometry *g, const Role *r, const char *name,
                               char *err)
{
  const GgufTensor *t = gguf_tensor(model, name);
  if (!t)
    return fail(err, name, "missing"), NULL;
  const Dim *dims = r->as[g->keys].dims;
  int n = 0;
  while (n < GGUF_MAX_DIMS && dims[n] != D_NONE)
    n++;
  if (t->n_dims != (uint32_t)n)
    return fail(err, name, "of another number of dims"), NULL;
  uint64_t values = 1;
  for (int i = 0; i < n; i++)
  {
    if (t->dims[i] != dim(g, dims[i]))
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

/* a role's tensor in its slot of `base`: `prefix` before its name (a layer's blk.N.) */
static bool take(const Gguf *model, bool *taken, const Ink2Geometry *g, const Role *r, const char *prefix, void *base,
                 char *err)
{
  const char *n = r->as[g->keys].name;
  char name[NAME_MAX_LEN];
  snprintf(name, sizeof name, "%s%s", prefix, n ? n : "?");
  if (!n)
    return fail(err, name, "a role this file's keys have no tensor for");
  return (*(const GgufTensor **)((char *)base + r->field) = claim(model, taken, g, r, name, err)) != NULL;
}

/* an expert tensor's scale: NVFP4's blocks are not its whole values (their scale one an expert, beside it), every
   other expert type's are - a scale beside one of those is refused, not ignored. unsloth's Inkling-2-Small keeps the
   checkpoint's NVFP4 outer scale beside its requantized experts too (layers 0-39: their values quantized without it):
   there a scale is taken wherever it is, any type - applied after the expert's matrix, as NVFP4's */
static bool expert_scales(const Gguf *model, bool *taken, const Ink2Geometry *g, int l, Ink2Layer *L, char *err)
{
  char name[NAME_MAX_LEN];
  for (size_t i = 0; i < N_SCALES; i++)
  {
    const GgufTensor *e = *(const GgufTensor **)((char *)L + scales[i].exp);
    const GgufTensor **slot = (const GgufTensor **)((char *)L + scales[i].scale);
    snprintf(name, sizeof name, "blk.%d.%s", l, scales[i].name[g->keys]);
    if (g->keys == INK2_KEYS_UNSLOTH_SMALL && e->type != QTYPE_NVFP4)
    {
      if (gguf_tensor(model, name) && !(*slot = claim(model, taken, g, &scale_role, name, err)))
        return false;
      continue;
    }
    if (e->type != QTYPE_NVFP4)
    {
      if (gguf_tensor(model, name))
        return fail(err, name, "a scale beside an expert type whose blocks are its values");
      continue;
    }
    if (!(*slot = claim(model, taken, g, &scale_role, name, err)))
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
    if (applies(g, tops[i].when, 0))
      ok = take(model, taken, g, &tops[i], "", w, err);
  char prefix[NAME_MAX_LEN];
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    Ink2Layer *L = &w->layer[l];
    snprintf(prefix, sizeof prefix, "blk.%d.", l);
    for (size_t i = 0; ok && i < N_ROLES; i++)
      if (applies(g, roles[i].when, l))
        ok = take(model, taken, g, &roles[i], prefix, L, err);
    L->exp_fused = ok && L->exp_gate == L->exp_up;
    ok = ok && expert_scales(model, taken, g, l, L, err);
  }
  for (size_t i = 0; ok && i < model->n_tensors; i++)
    if (!taken[i] && strncmp(model->tensors[i].name, DRAFTER, strlen(DRAFTER)) != 0)
      ok = fail(err, model->tensors[i].name, "a tensor no role of this layer's kind takes");
  free(taken);
  if (!ok)
    return free(w), NULL;
  return w;
}

/* an expert's three parts: its slice of gate, up and down - a fused gate and up's slice halved, its gate rows first */
size_t ink2_weights_experts(const Ink2Weights *w, const int *model_fds, PoolUnit *units, size_t cap)
{
  if (!w || w->geo.n_layers > INK2_MAX_LAYERS)
    return 0;
  ExpertTensors t[INK2_MAX_LAYERS];
  for (int l = 0; l < w->geo.n_layers; l++)
  {
    const Ink2Layer *L = &w->layer[l];
    t[l] = (ExpertTensors){L->exp_gate, L->exp_up, L->exp_down, L->exp_fused};
  }
  return expert_units(t, (uint32_t)w->geo.n_layers, (uint32_t)w->geo.n_experts, model_fds, units, cap);
}
