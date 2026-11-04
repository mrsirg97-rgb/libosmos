#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drafter_impl.h"
#include "osmos/model/qtype.h"
#include "osmos/mem/hip_slots.h"

/* the drafter: its weights in GPU memory of their own, its rings from the trunk's taps, a round's head (hip_drafter.c)
   behind its blocks (hip_drafter_round.c) */

#define DRAFTER_READERS 4 /* the load's readers: the fetcher's (DESIGN.md) */

static bool fail(char *err, const char *why) { return fwd_fail(err, why); }

/* a tensor's device bytes, by name (b >= 0: "mtp.b." + name); NULL if it has none */
static const void *dev(Drafter *d, int b, const char *name, uint32_t *type)
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
  return p[0];
}

/* a BF16 vector widened to F32 (an F32 one copied), into memory of its own */
static float *widen(Drafter *d, int b, const char *name, uint64_t n)
{
  uint32_t type = 0;
  const void *src = dev(d, b, name, &type);
  float *out = NULL;
  if (!src || !fwd_alloc(&out, n * sizeof(float)))
    return NULL;
  const bool ok = type == 0 ? fwd_copy(out, src, n * sizeof(float)) : quant_dequant(type, src, n, out, NULL);
  if (!ok)
    return hipFree(out), NULL;
  return out;
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
  k->q_norm = widen(d, b, "attn.q_norm.weight", ql), k->kv_norm = widen(d, b, "attn.kv_norm.weight", hd);
  k->attn_norm = widen(d, b, "attn_norm.weight", D), k->ffn_norm = widen(d, b, "ffn_norm.weight", D);
  const void *wo_a = dev(d, b, "attn.wo_a.weight", NULL);
  bool ok = k->wq_a && k->wq_b && k->wkv && k->wo_b && k->sh1 && k->sh3 && k->sh2 && k->gate && k->sink &&
            k->gate_bias && k->hca_fn && k->hca_base && k->hca_scale && k->hcf_fn && k->hcf_base && k->hcf_scale &&
            k->q_norm && k->kv_norm && k->attn_norm && k->ffn_norm && wo_a;
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
    ok = ok && fwd_alloc(&k->ex[r], sizeof host) && fwd_put(k->ex[r], host, sizeof host);
  }
  const uint32_t G = (uint32_t)d->f->g->o_groups, OL = (uint32_t)d->f->g->o_lora;
  const uint32_t oin = (uint32_t)d->f->g->n_heads * hd / G;
  const void *oa[DRAFTER_BLOCK * FWD_MAX_GROUPS];
  for (uint32_t t = 0; t < DRAFTER_BLOCK; t++)
    for (uint32_t g = 0; g < G; g++)
      oa[t * G + g] = (const uint8_t *)wo_a + (uint64_t)g * OL * gguf_bytes(DRAFTER_Q8_0, oin);
  return ok && fwd_alloc(&k->oa_tab, DRAFTER_BLOCK * G * sizeof *oa) &&
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
      {&d->pos_a, T * 4}, {&d->ids, T * 4}, {&d->pair, P * 4}, {&d->tap_pos, FWD_MAX_BATCH * 4},
      {&d->tab, 3 * P * sizeof(void *)},
  };
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!fwd_alloc(want[i].p, want[i].bytes))
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
  return fwd_put(d->sel, sel, sizeof sel) && fwd_put(d->n_sel, n_sel, sizeof n_sel) &&
         fwd_put(d->pair, pair, sizeof pair) && fwd_put(d->pre0, pre0, T * hc * F) &&
         hipHostMalloc((void **)&d->pin, sizeof *d->pin + V * F, 0) == hipSuccess;
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
      return fail(err, "dspark.json: a target layer past the model's");
    theirs[ix->targets[i]] = true;
  }
  if (any && ix->n_targets && memcmp(at, theirs, (size_t)f->n_layers * sizeof *at) != 0)
    return fail(err, "the model's DSpark target layers and dspark.json's differ");
  if (!any && !ix->n_targets)
    return fail(err, "no DSpark target layers: neither the model's GGUF nor dspark.json names them");
  if (!any)
    memcpy(at, theirs, (size_t)f->n_layers * sizeof *at);
  return true;
}

Drafter *drafter_create(Forward *f, const char *dir, char err[FORWARD_ERR])
{
  Drafter *d = f ? calloc(1, sizeof *d) : NULL;
  if (!d)
    return fail(err, "no forward, or out of memory"), NULL;
  d->f = f, d->fd = -1;
  char xerr[DSPARK_ERR] = "", perr[POOL_ERR] = "";
  bool ok = (d->ix = dspark_index(dir, xerr)) != NULL || fail(err, xerr);
  ok = ok && ((d->fd = open(d->ix->bin, O_RDONLY)) >= 0 || fail(err, "dspark.bin could not be opened"));
  PoolUnit *u = ok ? calloc(d->ix->n, sizeof *u) : NULL;
  ok = ok && (u || fail(err, "out of memory"));
  for (uint32_t i = 0; ok && i < d->ix->n; i++)
    u[i] = (PoolUnit){.part = {{d->fd, d->ix->t[i].offset, d->ix->t[i].bytes}}, .n_parts = 1};
  ok = ok && ((d->res = resident_create(u, d->ix->n, hip_slots(), DRAFTER_READERS, perr)) || fail(err, perr));
  free(u);
  const DsparkTensor *mk = ok ? dspark_tensor(d->ix, "mtp.2.markov_head.embed.weight") : NULL;
  const DsparkTensor *mp = ok ? dspark_tensor(d->ix, "mtp.0.main_proj.weight") : NULL;
  bool at[WEIGHTS_MAX_LAYERS] = {false};
  ok = ok && targets(f, d->ix, at, err);
  ok = ok && ((mk && mp && fwd_taps_on(f, at) && mp->dims[1] == (uint64_t)f->n_tap * f->g->n_embd) ||
              fail(err, "a drafter whose taps are not the model's target layers'"));
  if (ok)
  {
    d->rank = (uint32_t)mk->dims[1];
    d->main_proj = dev(d, 0, "main_proj.weight", &d->main_type);
    d->mk_embed = dev(d, 2, "markov_head.embed.weight", NULL), d->mk_head = dev(d, 2, "markov_head.head.weight", NULL);
    d->main_norm = widen(d, 0, "main_norm.weight", f->g->n_embd), d->norm = widen(d, 2, "norm.weight", f->g->n_embd);
    d->conf_w = widen(d, 2, "confidence_head.proj.weight", (uint64_t)f->g->n_embd + d->rank);
    ok = (d->main_proj && d->mk_embed && d->mk_head && d->main_norm && d->norm && d->conf_w) ||
         fail(err, "a drafter without its heads");
  }
  for (int b = 0; ok && b < DRAFTER_BLOCKS; b++)
    ok = block(d, b) || fail(err, "a drafter block's weights missing, its experts of mixed types or of a type the matvec "
                              "does not take, or GPU memory refused");
  ok = ok && (scratch(d) || fail(err, "GPU memory refused")) && fwd_room_left(f, err);
  if (!ok)
  {
    drafter_destroy(d);
    return NULL;
  }
  return d;
}

void drafter_destroy(Drafter *d)
{
  if (!d)
    return;
  for (int b = 0; b < DRAFTER_BLOCKS; b++)
  {
    DBlock *k = &d->b[b];
    hipFree(k->q_norm), hipFree(k->kv_norm), hipFree(k->attn_norm), hipFree(k->ffn_norm), hipFree(k->oa_tab);
    for (int r = 0; r < 3; r++)
      hipFree(k->ex[r]);
  }
  hipFree(d->main_norm), hipFree(d->norm), hipFree(d->conf_w);
  float **bufs[] = {&d->h, &d->mid, &d->x, &d->qa, &d->q, &d->kv, &d->o, &d->oa, &d->y, &d->xf, &d->logits, &d->rw,
                    &d->ga, &d->up, &d->hh, &d->ye, &d->moe, &d->sga, &d->sup, &d->shh, &d->sye, &d->hcp, &d->a_pre,
                    &d->a_post, &d->a_comb, &d->f_pre, &d->f_post, &d->f_comb, &d->pre0, &d->att, &d->xc, &d->xn,
                    &d->vocab, &d->mk, &d->cat, &d->cval, &d->main_x, &d->kvs};
  for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++)
    hipFree(*bufs[i]);
  hipFree(d->kvs_p);
  int32_t **ibufs[] = {&d->ridx, &d->sel, &d->n_sel, &d->pos_q, &d->pos_a, &d->ids, &d->pair, &d->tap_pos};
  for (size_t i = 0; i < sizeof ibufs / sizeof *ibufs; i++)
    hipFree(*ibufs[i]);
  hipFree(d->tab);
  hipHostFree(d->pin);
  resident_destroy(d->res);
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
  return ok && hipStreamSynchronize(NULL) == hipSuccess;
}

bool drafter_seed(Drafter *d, Seq *s, int32_t from, int32_t to, char err[FORWARD_ERR])
{
  Forward *f = d ? d->f : NULL;
  if (!f || !s || s->f != f)
    return fail(err, "no drafter, or a sequence of another forward");
  const Geometry *g = f->g;
  const uint32_t W = (uint32_t)g->window, hd = g->head_dim;
  if (!s->dring && !fwd_alloc(&s->dring, (uint64_t)DRAFTER_BLOCKS * W * kv_row_bytes(KV_FP8, hd)))
    return fail(err, "GPU memory refused (the drafter's rings)");
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
      return fail(err, "a position the last step did not tap");
    if (!seed_run(d, s, i, e - i))
      return fail(err, "a GPU call failed (the drafter's seed)");
    want = f->tap_pos[e - 1] + 1, i = e;
  }
  return want == to || fail(err, "a position the last step did not tap");
}

/* the confidence of row t: sigmoid(conf_w . [x_t ; E[prev]]), E[prev] already in cat's tail */
static bool confidence(Drafter *d, uint32_t t, float *out)
{
  const uint32_t D = d->f->g->n_embd, n = D + d->rank;
  if (!fwd_copy(d->cat, d->xc + (uint64_t)t * D, (uint64_t)D * sizeof(float)) ||
      !mv_strided(DRAFTER_F32, d->conf_w, 1, n, d->cat, n, 1, d->cval, 1, NULL) ||
      hipMemcpyAsync(&d->pin->conf, d->cval, 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      hipStreamSynchronize(NULL) != hipSuccess)
    return false;
  *out = 1.0f / (1.0f + expf(-d->pin->conf));
  return true;
}

int drafter_draft(Drafter *d, Seq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[DRAFTER_BLOCK],
                  float conf[DRAFTER_BLOCK], char err[FORWARD_ERR])
{
  Forward *f = d ? d->f : NULL;
  if (!f || !s || s->f != f || !s->dring || y < 0 || y >= f->g->n_vocab)
    return fail(err, "no drafter, an unseeded sequence or a token outside the vocabulary"), -1;
  const Geometry *g = f->g;
  const uint32_t D = g->n_embd, V = (uint32_t)g->n_vocab, T = DRAFTER_BLOCK;
  const int32_t p0 = s->pos;
  for (uint32_t t = 0; t < T; t++)
    d->pin->ids[t] = t ? DRAFTER_NOISE : y, d->pin->pos_q[t] = p0 + (int32_t)t, d->pin->pos_a[t] = p0 - 1;
  if (!fwd_put_async(d->ids, d->pin->ids, T * 4) || !fwd_put_async(d->pos_q, d->pin->pos_q, T * 4) ||
      !fwd_put_async(d->pos_a, d->pin->pos_a, T * 4) || !drafter_blocks(d, s, p0, err))
    return err[0] ? -1 : (fail(err, "a GPU call failed (the drafter's blocks)"), -1);
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
      return fail(err, "a GPU call failed (the drafter's confidence)"), -1;
    cum *= conf[n];
    if (conf[n] < conf_min || cum < cum_min)
      break;
    if (!head) /* the head for every row at once, the first time a row passes: the head read once */
    {
      head = ops_rms(d->xc, D, D, d->norm, g->rms_eps, T, d->xn, D, NULL) &&
             mv_strided(fwd_type(f, f->w->output), fwd_dev(f, f->w->output), V, D, d->xn, D, T, d->vocab, V, NULL);
      if (!head)
        return fail(err, "a GPU call failed (the drafter's head)"), -1;
    }
    float *row = d->vocab + (uint64_t)n * V;
    if (!mv_strided(DRAFTER_BF16, d->mk_head, V, d->rank, d->cat + D, d->rank, 1, d->mk, V, NULL) ||
        !ops_add(row, d->mk, V, NULL) ||
        hipMemcpyAsync(d->pin->row, row, (uint64_t)V * sizeof(float), hipMemcpyDeviceToHost, NULL) != hipSuccess ||
        hipStreamSynchronize(NULL) != hipSuccess)
      return fail(err, "a GPU call failed (the drafter's markov step)"), -1;
    uint32_t best = 0; /* greedy: the most, ties to the lowest token */
    for (uint32_t v = 1; v < V; v++)
      if (d->pin->row[v] > d->pin->row[best])
        best = v;
    drafts[n] = (int32_t)best;
  }
  return n;
}
