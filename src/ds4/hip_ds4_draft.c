#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4_draft_impl.h"
#include "osmos/mem/ledger_hip.h"
#include "osmos/model/qtype.h"
#include "osmos/mem/hip_slots.h"
#include "osmos/run/draft.h"
#include "osmos/err.h"

/* the drafter: its weights in GPU memory of their own, its rings from the trunk's taps, a round's head (hip_ds4_draft.c)
   behind its blocks (hip_ds4_draft_round.c) */

#define DRAFTER_READERS 4 /* the load's readers: the fetcher's (DESIGN.md) */


/* a tensor's device bytes, by name (b >= 0: "mtp.b." + name); NULL if it has none */
static const void *dev_of(Drafter *d, int b, const char *name, uint32_t *type, uint64_t *values);

static const void *dev(Drafter *d, int b, const char *name, uint32_t *type) { return dev_of(d, b, name, type, NULL); }

static const void *dev_of(Drafter *d, int b, const char *name, uint32_t *type, uint64_t *values)
{
  char full[DSPARK_NAME_MAX];
  if (b >= 0)
    snprintf(full, sizeof full, "mtp.%d.%s", b, name);
  else
    snprintf(full, sizeof full, "%s", name);
  const DsparkTensor *t = dspark_tensor(d->ix, full);
  const void *p[POOL_PARTS];
  if (!t || !resident_dev(d->res, (size_t)(t - d->ix->t), p))
    return NULL;
  if (type)
    *type = t->type;
  if (values)
  {
    *values = 1;
    for (uint32_t k = 0; k < t->n_dims; k++)
      *values *= t->dims[k]; /* (dspark_index: their product checked against its bytes, not past 64 bits) */
  }
  return p[0];
}

/* a BF16 vector widened to F32 (an F32 one copied), into memory of its own */
static bool widen(Drafter *d, int b, const char *name, uint64_t n, float **out)
{
  uint32_t type = 0;
  uint64_t values = 0;
  const void *src = dev_of(d, b, name, &type, &values);
  if (!src || values != n || !fwd_alloc(&d->bufs, out, n * sizeof(float))) /* its n values, no fewer */
    return false;
  return type == 0 ? fwd_copy(*out, src, n * sizeof(float)) : quant_dequant(type, src, n, *out, NULL);
}

static bool block(Drafter *d, int b)
{
  DBlock *k = &d->b[b];
  const uint32_t D = (uint32_t)d->f->g->n_embd, hd = (uint32_t)d->f->g->head_dim, ql = (uint32_t)d->f->g->q_lora;
  k->wq_a = dev(d, b, "attn.wq_a.weight", NULL), k->wq_b = dev(d, b, "attn.wq_b.weight", NULL);
  k->wkv = dev(d, b, "attn.wkv.weight", NULL), k->wo_b = dev(d, b, "attn.wo_b.weight", NULL);
  k->sh1 = dev(d, b, "ffn.shared_experts.w1.weight", NULL), k->sh3 = dev(d, b, "ffn.shared_experts.w3.weight", NULL);
  k->sh2 = dev(d, b, "ffn.shared_experts.w2.weight", NULL), k->gate = dev(d, b, "ffn.gate.weight", NULL);
  k->sink = dev(d, b, "attn.attn_sink", NULL), k->gate_bias = dev(d, b, "ffn.gate.bias", NULL);
  k->hca_fn = dev(d, b, "hc_attn_fn", NULL), k->hca_base = dev(d, b, "hc_attn_base", NULL);
  k->hca_scale = dev(d, b, "hc_attn_scale", NULL), k->hcf_fn = dev(d, b, "hc_ffn_fn", NULL);
  k->hcf_base = dev(d, b, "hc_ffn_base", NULL), k->hcf_scale = dev(d, b, "hc_ffn_scale", NULL);
  bool wide = widen(d, b, "attn.q_norm.weight", ql, &k->q_norm) && widen(d, b, "attn.kv_norm.weight", hd, &k->kv_norm) &&
              widen(d, b, "attn_norm.weight", D, &k->attn_norm) && widen(d, b, "ffn_norm.weight", D, &k->ffn_norm);
  const void *wo_a = dev(d, b, "attn.wo_a.weight", NULL);
  bool ok = k->wq_a && k->wq_b && k->wkv && k->wo_b && k->sh1 && k->sh3 && k->sh2 && k->gate && k->sink &&
            k->gate_bias && k->hca_fn && k->hca_base && k->hca_scale && k->hcf_fn && k->hcf_base && k->hcf_scale &&
            wide && wo_a; /* (a widen that failed partway: its buffer made, its values not - refused) */
  /* the experts' tables, and wo_a's groups a row (group g's rows g * o_lora on, its 4,096 columns Q8_0) */
  const void *host[DRAFTER_EXPERTS];
  static const char *role[3] = {"w1", "w3", "w2"};
  for (int r = 0; ok && r < 3; r++)
  {
    for (int e = 0; ok && e < DRAFTER_EXPERTS; e++)
    {
      char n[DSPARK_NAME_MAX];
      uint32_t type = 0;
      snprintf(n, sizeof n, "ffn.experts.%d.%s.weight", e, role[r]);
      ok = (host[e] = dev(d, b, n, &type)) != NULL;
      if (ok && !d->ex_type) /* the first expert's type: every one's (one matvec a role) */
        d->ex_type = type;
      ok = ok && type == d->ex_type && (qtype_paths(type) & QTYPE_MV);
    }
    ok = ok && fwd_alloc(&d->bufs, &k->ex[r], sizeof host) && fwd_put(k->ex[r], host, sizeof host);
  }
  const uint32_t G = (uint32_t)d->f->g->o_groups, OL = (uint32_t)d->f->g->o_lora;
  const uint32_t oin = (uint32_t)d->f->g->n_heads * hd / G;
  const void *oa[DRAFTER_BLOCK * FWD_MAX_GROUPS];
  for (uint32_t t = 0; t < DRAFTER_BLOCK; t++)
    for (uint32_t g = 0; g < G; g++)
      oa[t * G + g] = (const uint8_t *)wo_a + (uint64_t)g * OL * gguf_bytes(DRAFTER_Q8_0, oin);
  return ok && fwd_alloc(&d->bufs, &k->oa_tab, DRAFTER_BLOCK * G * sizeof *oa) &&
         fwd_put(k->oa_tab, oa, DRAFTER_BLOCK * G * sizeof *oa);
}

static bool scratch(Drafter *d)
{
  const Geometry *g = d->f->g;
  const uint64_t T = DRAFTER_BLOCK, D = g->n_embd, hc = g->hc, F = sizeof(float);
  const uint64_t HQ = (uint64_t)g->n_heads * g->head_dim;
  const uint64_t FF = g->expert_ff, P = DRAFTER_PAIRS, V = g->n_vocab, hd = g->head_dim;
  struct
  {
    void *p;
    uint64_t bytes;
  } want[] = {
      {&d->h, T * hc * D * F}, {&d->mid, T * hc * D * F}, {&d->x, T * D * F}, {&d->qa, T * g->q_lora * F},
      {&d->q, T * HQ * F}, {&d->kv, T * hd * F}, {&d->o, T * HQ * F}, {&d->oa, T * g->o_groups * g->o_lora * F},
      {&d->y, T * D * F}, {&d->xf, T * D * F}, {&d->logits, T * DRAFTER_EXPERTS * F}, {&d->rw, P * F},
      {&d->ga, P * FF * F}, {&d->up, P * FF * F}, {&d->hh, P * FF * F}, {&d->ye, P * D * F}, {&d->moe, T * D * F},
      {&d->sga, T * FF * F}, {&d->sup, T * FF * F}, {&d->shh, T * FF * F}, {&d->sye, T * D * F},
      {&d->hcp, T * ((2 + hc) * hc + 1) * OPS_HC_PARTS * F}, {&d->a_pre, T * hc * F}, {&d->a_post, T * hc * F},
      {&d->a_comb, T * hc * hc * F}, {&d->f_pre, T * hc * F}, {&d->f_post, T * hc * F},
      {&d->f_comb, T * hc * hc * F}, {&d->pre0, T * hc * F},
      {&d->att, OPS_ATTN_SCRATCH(T, g->n_heads, (uint64_t)g->window + T) * F}, {&d->xc, T * D * F},
      {&d->xn, T * D * F}, {&d->vocab, T * V * F}, {&d->mk, V * F}, {&d->cat, (D + d->rank) * F},
      {&d->cval, F}, {&d->main_x, (uint64_t)FWD_MAX_BATCH * D * F}, {&d->kvs, (uint64_t)FWD_MAX_BATCH * hd * F},
      {&d->kvs_p, (uint64_t)FWD_MAX_BATCH * kv_row_bytes(KV_FP8, hd)},
      {&d->ridx, P * 4}, {&d->sel, T * OPS_MAX_SELECT * 4}, {&d->n_sel, T * 4}, {&d->pos_q, T * 4},
      {&d->pos_a, T * 4}, {&d->ids, T * 4}, {&d->pair, P * 4}, {&d->tap_pos, FWD_MAX_BATCH * 4}, {&d->best, 4},
      {&d->best_part, ops_argmax_scratch(1)},
      {&d->tab, 3 * P * sizeof(void *)},
  };
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!fwd_alloc(&d->bufs, want[i].p, want[i].bytes))
      return false;
  /* constant: the block's own rows every row's selected keys (no causal mask in a block), each pair its own row
     in the routed sum, the first block's pre one-hot on stream 0 */
  int32_t sel[DRAFTER_BLOCK * OPS_MAX_SELECT] = {0}, n_sel[DRAFTER_BLOCK], pair[DRAFTER_PAIRS];
  float pre0[DRAFTER_BLOCK * OPS_MAX_HC] = {0};
  for (uint32_t t = 0; t < T; t++)
  {
    n_sel[t] = (int32_t)T, pre0[t * hc] = 1.0f;
    for (uint32_t j = 0; j < T; j++)
      sel[t * OPS_MAX_SELECT + j] = (int32_t)j;
  }
  for (uint32_t m = 0; m < P; m++)
    pair[m] = (int32_t)m;
  const uint32_t W = (uint32_t)g->window;
  const AttnCoreDesc ad = {.heads = g->n_heads, .kv_heads = 1, .dk = g->head_dim, .dv = g->head_dim,
                           .k_rows = ATTN_ROWS_FP8, .v_rows = ATTN_ROWS_FP8, .s_rows = ATTN_ROWS_F32, .max_sel = (uint32_t)T,
                           .window = W, .slots = W, .scaling = (float)(1.0 / sqrt((double)hd)), .term = ATTN_TERM_NONE,
                           .sink = ATTN_SINK_HEAD, .gate = ATTN_GATE_NONE, .block = ATTN_BLOCK_HEADS,
                           .max_tokens = (uint32_t)T, .max_keys = W + (uint32_t)T};
  char aerr[OSMOS_ERR];
  if (!(d->latent = attn_core_create(&ad, d->att, OPS_ATTN_SCRATCH(T, g->n_heads, (uint64_t)W + T) * F, aerr)))
    return false;
  return fwd_put(d->sel, sel, sizeof sel) && fwd_put(d->n_sel, n_sel, sizeof n_sel) &&
         fwd_put(d->pair, pair, sizeof pair) && fwd_put(d->pre0, pre0, T * hc * F) &&
         fwd_pinned(&d->bufs, &d->pin, sizeof *d->pin);
}

/* the trunk's layers the drafter taps: the model's GGUF's where it names them, else dspark.json's; both named, equal */
static bool targets(const Forward *f, const DsparkIndex *ix, bool *at, char *err)
{
  bool any = false, theirs[WEIGHTS_MAX_LAYERS] = {false};
  for (int l = 0; l < f->n_layers; l++)
    any |= (at[l] = f->g->dspark_target[l]);
  for (uint32_t i = 0; i < ix->n_targets; i++)
  {
    if (ix->targets[i] >= (uint32_t)f->n_layers)
      return osmos_fail(err, "dspark.json: a target layer past the model's");
    theirs[ix->targets[i]] = true;
  }
  if (any && ix->n_targets && memcmp(at, theirs, (size_t)f->n_layers * sizeof *at) != 0)
    return osmos_fail(err, "the model's DSpark target layers and dspark.json's differ");
  if (!any && !ix->n_targets)
    return osmos_fail(err, "no DSpark target layers: neither the model's GGUF nor dspark.json names them");
  if (!any)
    memcpy(at, theirs, (size_t)f->n_layers * sizeof *at);
  return true;
}

bool fwd_drafter_plan(const Forward *f, const char *dir, DsparkIndex **ix, bool *at, char *err)
{
  char xerr[OSMOS_ERR] = "";
  if (!(*ix = dspark_index(dir, xerr)))
    return osmos_fail(err, "%s", xerr);
  const DsparkTensor *mk = dspark_tensor(*ix, "mtp.2.markov_head.embed.weight");
  const DsparkTensor *mp = dspark_tensor(*ix, "mtp.0.main_proj.weight");
  bool ok = targets(f, *ix, at, err);
  uint64_t n_tap = 0;
  for (int l = 0; ok && l < f->n_layers; l++)
    n_tap += at[l];
  ok = ok && ((mk && mp && n_tap && mp->dims[1] == n_tap * f->g->n_embd) ||
              osmos_fail(err, "a drafter whose taps are not the model's target layers'"));
  if (!ok)
    dspark_index_destroy(*ix), *ix = NULL;
  return ok;
}

Drafter *fwd_drafter_make(Forward *f, DsparkIndex *ix, char *err)
{
  Drafter *d = calloc(1, sizeof *d);
  if (!d)
    return dspark_index_destroy(ix), osmos_fail(err, "out of memory (the drafter)"), NULL;
  d->f = f, d->fd = -1, d->bufs.ledger = f->ledger, d->ix = ix; /* (its plan's: the taps f's already) */
  char perr[OSMOS_ERR] = "";
  bool ok = (d->fd = open(d->ix->bin, O_RDONLY)) >= 0 || osmos_fail(err, "dspark.bin could not be opened");
  PoolUnit *u = ok ? calloc(d->ix->n, sizeof *u) : NULL;
  ok = ok && (u || osmos_fail(err, "out of memory"));
  for (uint32_t i = 0; ok && i < d->ix->n; i++)
    u[i] = (PoolUnit){.part = {{d->fd, d->ix->t[i].offset, d->ix->t[i].bytes}}, .n_parts = 1};
  ok = ok && (((d->slots = hip_slots_create(f->ledger, perr)) &&
               (d->res = resident_create(u, d->ix->n, hip_slots(d->slots), DRAFTER_READERS, perr))) ||
              osmos_fail(err, "%s", perr));
  free(u);
  const DsparkTensor *mk = ok ? dspark_tensor(d->ix, "mtp.2.markov_head.embed.weight") : NULL;
  const DsparkTensor *mp = ok ? dspark_tensor(d->ix, "mtp.0.main_proj.weight") : NULL;
  ok = ok && ((mk && mp && f->taps) || osmos_fail(err, "the drafter's taps not planned (fwd_drafter_plan)"));
  if (ok)
  {
    d->rank = (uint32_t)mk->dims[1];
    d->main_proj = dev(d, 0, "main_proj.weight", &d->main_type);
    d->mk_embed = dev(d, 2, "markov_head.embed.weight", NULL), d->mk_head = dev(d, 2, "markov_head.head.weight", NULL);
    const bool wide = widen(d, 0, "main_norm.weight", f->g->n_embd, &d->main_norm) &&
                      widen(d, 2, "norm.weight", f->g->n_embd, &d->norm) &&
                      widen(d, 2, "confidence_head.proj.weight", (uint64_t)f->g->n_embd + d->rank, &d->conf_w);
    ok = (d->main_proj && d->mk_embed && d->mk_head && wide) ||
         osmos_fail(err, "a drafter without its heads");
  }
  for (int b = 0; ok && b < DRAFTER_BLOCKS; b++)
    ok = block(d, b) || osmos_fail(err, "a drafter block's weights missing, its experts of mixed types or of a type the matvec "
                              "does not take, or GPU memory refused");
  ok = ok && (scratch(d) || osmos_fail(err, "GPU memory refused"));
  if (!ok)
  {
    fwd_drafter_destroy(d);
    return NULL;
  }
  return d;
}

uint64_t fwd_drafter_bytes(const Drafter *d) { return d ? d->bufs.bytes + resident_bytes(d->res) : 0; }

void fwd_drafter_destroy(Drafter *d)
{
  if (!d)
    return;
  attn_core_destroy(d->latent);
  gpu_bufs_free(&d->bufs); /* every GPU and pinned buffer the drafter made: from its record */
  resident_destroy(d->res);
  hip_slots_destroy(d->slots); /* after the resident set: its memory released through it */
  if (d->fd >= 0)
    close(d->fd);
  dspark_index_destroy(d->ix);
  free(d);
}

/* each block's ring row for a run of tapped positions: main_x = rms(main_proj taps) main_norm, then a block's
   rms(wkv main_x) kv_norm, roped at its position, FP8-rounded - at ring row position % window */
static bool seed_run(Drafter *d, Seq *s, uint32_t i0, uint32_t n)
{
  Forward *f = d->f;
  const Geometry *g = f->g;
  const uint32_t D = g->n_embd, hd = g->head_dim, W = (uint32_t)g->window, nt = (uint32_t)f->n_tap * D;
  const uint32_t rb = (uint32_t)kv_row_bytes(KV_FP8, hd); /* its rings' rows as the trunk's (kvfmt.h) */
  memcpy(d->pin->tap_pos, f->tap_pos + i0, n * sizeof *f->tap_pos);
  bool ok = fwd_put_async(d->tap_pos, d->pin->tap_pos, n * 4) &&
            mv_strided(d->main_type, d->main_proj, D, nt, f->taps + (uint64_t)i0 * nt, nt, n, d->main_x, D, NULL) &&
            ops_rms(d->main_x, D, D, d->main_norm, g->rms_eps, n, d->main_x, D, NULL);
  for (int b = 0; ok && b < DRAFTER_BLOCKS; b++)
    ok = mv_strided(DRAFTER_Q8_0, d->b[b].wkv, hd, D, d->main_x, D, n, d->kvs, hd, NULL) &&
         ops_rms(d->kvs, hd, hd, d->b[b].kv_norm, g->rms_eps, n, d->kvs, hd, NULL) &&
         ops_rope(d->kvs, 1, hd, g->rope_dim, hd, f->freqs_win, d->tap_pos, false, n, NULL) &&
         ops_fp8_round(d->kvs, (uint64_t)n * hd, d->kvs_p, hd, NULL) &&
         ops_put_rows(s->dring + (uint64_t)b * W * rb, W, d->kvs_p, d->tap_pos, 1, KV_FP8, hd, n, NULL);
  /* the pinned positions are read by the copy before the next seed writes them: the stream in order, and the next
     seed's memcpy is the host's - wait for this one's copy */
  return ok && fwd_settle(d->f);
}

bool drafter_seed(Drafter *d, Seq *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  Forward *f = d ? d->f : NULL;
  if (!f || !s || s->f != f)
    return osmos_fail(err, "no drafter, or a sequence of another forward");
  const Geometry *g = f->g;
  const uint32_t W = (uint32_t)g->window, hd = g->head_dim;
  (void)hd; /* (the rings the sequence's from its making) */
  if (to - from > (int32_t)W) /* only the window's last positions stay in the ring */
    from = to - (int32_t)W;
  int32_t want = from;
  for (uint32_t i = 0; i < f->tap_n;)
  {
    if (f->tap_bs[i] != s || f->tap_pos[i] < from || f->tap_pos[i] >= to)
    {
      i++;
      continue;
    }
    uint32_t e = i + 1; /* a run of the sequence's positions one after another */
    while (e < f->tap_n && f->tap_bs[e] == s && f->tap_pos[e] == f->tap_pos[e - 1] + 1 && f->tap_pos[e] < to)
      e++;
    if (f->tap_pos[i] != want)
      return osmos_fail(err, "a position the last step did not tap");
    if (!seed_run(d, s, i, e - i))
      return osmos_fail(err, "a GPU call failed (the drafter's seed)");
    want = f->tap_pos[e - 1] + 1, i = e;
  }
  if (want != to)
    return osmos_fail(err, "a position the last step did not tap");
  s->dseeded = true; /* its rings this sequence's context from here */
  return true;
}

/* the confidence of row t: sigmoid(conf_w . [x_t ; E[prev]]), E[prev] already in cat's tail */
static bool confidence(Drafter *d, uint32_t t, float *out)
{
  const uint32_t D = d->f->g->n_embd, n = D + d->rank;
  if (!fwd_copy(d->cat, d->xc + (uint64_t)t * D, (uint64_t)D * sizeof(float)) ||
      !mv_strided(DRAFTER_F32, d->conf_w, 1, n, d->cat, n, 1, d->cval, 1, NULL) ||
      hipMemcpyAsync(&d->pin->conf, d->cval, 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      !fwd_settle(d->f))
    return false;
  *out = 1.0f / (1.0f + expf(-d->pin->conf));
  return true;
}

int drafter_draft(Drafter *d, Seq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[DRAFTER_BLOCK],
                  float conf[DRAFTER_BLOCK], char err[OSMOS_ERR])
{
  Forward *f = d ? d->f : NULL;
  if (!f || !s || s->f != f || !s->dring || !s->dseeded || y < 0 || y >= f->g->n_vocab)
    return osmos_fail(err, "no drafter, an unseeded sequence or a token outside the vocabulary"), -1;
  const Geometry *g = f->g;
  const uint32_t D = g->n_embd, V = (uint32_t)g->n_vocab, T = DRAFTER_BLOCK;
  const int32_t p0 = s->pos;
  for (uint32_t t = 0; t < T; t++)
    d->pin->ids[t] = t ? DRAFTER_NOISE : y, d->pin->pos_q[t] = p0 + (int32_t)t, d->pin->pos_a[t] = p0 - 1;
  if (!fwd_put_async(d->ids, d->pin->ids, T * 4) || !fwd_put_async(d->pos_q, d->pin->pos_q, T * 4) ||
      !fwd_put_async(d->pos_a, d->pin->pos_a, T * 4) || !drafter_blocks(d, s, p0, err))
    return err[0] ? -1 : (osmos_fail(err, "a GPU call failed (the drafter's blocks)"), -1);
  int n = 0;
  float cum = 1.0f;
  bool head = false;
  for (int32_t prev = y; n < (int)T; prev = drafts[n++])
  {
    /* E[prev] into cat's tail (the confidence's and the markov step's), then the confidence: the cut before the
       row's head and markov step */
    const uint8_t *e_row = (const uint8_t *)d->mk_embed + (uint64_t)prev * d->rank * sizeof(uint16_t);
    if (!quant_dequant(DRAFTER_BF16, e_row, d->rank, d->cat + D, NULL) ||
        !confidence(d, (uint32_t)n, &conf[n]))
      return osmos_fail(err, "a GPU call failed (the drafter's confidence)"), -1;
    if (!draft_accept(conf[n], conf_min, cum_min, &cum))
      break;
    if (!head) /* the head for every row at once, the first time a row passes: the head read once */
    {
      head = ops_rms(d->xc, D, D, d->norm, g->rms_eps, T, d->xn, D, NULL) &&
             mv_strided(fwd_type(f, f->w->output), fwd_dev(f, f->w->output), V, D, d->xn, D, T, d->vocab, V, NULL);
      if (!head)
        return osmos_fail(err, "a GPU call failed (the drafter's head)"), -1;
    }
    float *row = d->vocab + (uint64_t)n * V;
    if (!mv_strided(DRAFTER_BF16, d->mk_head, V, d->rank, d->cat + D, d->rank, 1, d->mk, V, NULL) ||
        !ops_add(row, d->mk, V, NULL) || !ops_argmax(row, V, 1, V, d->best, d->best_part, NULL) ||
        hipMemcpyAsync(&d->pin->best, d->best, 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
        !fwd_settle(d->f))
      return osmos_fail(err, "a GPU call failed (the drafter's markov step)"), -1;
    if (d->pin->best < 0) /* a NaN in the row: no draft from it, nor past it */
      break;
    drafts[n] = d->pin->best; /* greedy: the most, ties to the lowest token (ops_argmax's) */
  }
  return n;
}
