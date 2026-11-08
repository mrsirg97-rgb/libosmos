#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/quant.h"

/* every Engram row the step reads - each sub-batch's tokens' rows for each of its Engram layers (the layers ascending:
   fwd_engram's nth) - begun at the step's start on Engram's readers, into the stage one after another, so they read
   while the GPU embeds and runs the layers before them (the rows are the tokens' alone). the layers past dec would
   read compacted tokens; DeepSeek-V4.1's Engram layers are before it (refused otherwise) */
bool fwd_engram_issue(Forward *f, char *err)
{
  const Geometry *g = f->g;
  const uint32_t cols = (uint32_t)g->engram_cols, hd = (uint32_t)g->engram_head_dim;
  if (f->eg_pending) /* a step that failed, or stopped before its Engram layers: its rows done with first */
  {
    f->eg_pending = false;
    (void)rowread_wait(f->eg_read);
  }
  uint64_t at = 0, rb = 0;
  for (uint32_t k = 0; k < f->n_sub; k++)
    for (int l = 0, nth = 0; l < f->n_layers; l++)
    {
      if (!g->engram[l])
        continue;
      if (f->dec >= 0 && l > f->dec)
        return fwd_fail(err, "an Engram layer past the decoder's first: its tokens compacted");
      const GgufTensor *t = f->w->layer[l].eg_embd;
      const uint64_t b = gguf_bytes(t->type, hd);
      if (rb && b != rb)
        return fwd_fail(err, "Engram layers of different row sizes");
      rb = b, f->eg_at[k][nth] = at;
      const Sub *s = &f->sub[k];
      for (uint32_t i = 0; i < s->n; i++)
      {
        const int64_t *rows = s->rows + (uint64_t)i * WEIGHTS_MAX_LAYERS * cols + (uint64_t)nth * cols;
        for (uint32_t c = 0; c < cols; c++)
          f->eg_off[at++] = t->offset + (uint64_t)rows[c] * rb;
      }
      nth++;
    }
  if (!at)
    return true;
  if (!rowread_start(f->eg_read, f->eg_fd, f->eg_off, (uint32_t)at, (uint32_t)rb, f->stage))
    return fwd_fail(err, "Engram's readers refused the step's rows");
  f->eg_pending = true;
  return true;
}

/* the nth Engram layer's rows for each token (fwd_engram_issue's, waited for the first time a layer needs them),
   decoded (BF16, as ref.py's), projected and gated into the streams */
bool fwd_engram(Forward *f, int l, int nth, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const GgufTensor *t = L->eg_embd;
  const uint32_t cols = (uint32_t)g->engram_cols, hd = (uint32_t)g->engram_head_dim, D = g->n_embd, hc = g->hc;
  const uint32_t n = f->n;
  const uint64_t rb = gguf_bytes(t->type, hd);
  if (f->eg_pending)
  {
    f->eg_pending = false;
    if (!rowread_wait(f->eg_read))
      return fwd_fail(err, "an Engram row could not be read");
  }
  const uint8_t *stage = f->stage + f->eg_at[f->cur][nth] * rb;
  const uint64_t vals = (uint64_t)n * cols * hd;
  const bool ok = fwd_put_async(b->erow, stage, (uint64_t)n * cols * rb) &&
                  quant_dequant(t->type, b->erow, vals, b->erow_f, NULL) && ops_round_bf16(b->erow_f, vals, NULL) &&
                  fwd_mat(f, L->eg_wkv, fwd_dev(f, L->eg_wkv), (hc + 1) * D, cols * hd, b->erow_f, cols * hd, n,
                          b->ekv, (hc + 1) * D) &&
                  ops_engram_gate(b->h, b->ekv, f->eg_gate[l], hc, D, g->rms_eps, n, NULL);
  return ok || fwd_fail(err, "a GPU call failed (Engram)");
}

/* a bump allocator over the pinned staging region and its device copy: a table's host and device addresses */
typedef struct Carve
{
  uint8_t *host, *dev;
  uint64_t used;
} Carve;

static void *carve(Carve *c, uint64_t bytes, void **dev)
{
  const uint64_t at = (c->used + 15) & ~(uint64_t)15;
  c->used = at + bytes;
  *dev = c->dev + at;
  return c->host + at;
}

static int cmp_expert(const void *a, const void *b)
{
  const int32_t *x = a, *y = b;
  return (x[0] > y[0]) - (x[0] < y[0]);
}

static int cmp_u64(const void *a, const void *b)
{
  const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

/* one role's experts (their tables in ascending count): a launch a run of the same column tiles (mm_col_tiles), so
   a one-tile expert is not run in a hot one's many-tile kernel. each expert's rows its own: the order of the
   launches changes no bit */
static bool group(uint32_t type, const void *const *dtab, const int32_t *cnt, const int32_t *dcnt, uint32_t nd,
                  uint32_t rows, uint32_t cols, const void *x16, const uint32_t *dxo, float *y, const uint32_t *dyo,
                  uint32_t ys)
{
  bool ok = true;
  for (uint32_t a = 0, z = 0; ok && a < nd; a = z)
  {
    const uint32_t ct = mm_col_tiles(type, (uint32_t)cnt[a]);
    for (z = a + 1; z < nd && mm_col_tiles(type, (uint32_t)cnt[z]) == ct;)
      z++;
    ok = mm_group(type, dtab + a, dcnt + a, z - a, (uint32_t)cnt[z - 1], rows, cols, x16, dxo + a, cols, y, dyo + a,
                  ys, NULL);
  }
  return ok;
}

/* the shared expert, batched, into its own buffers: queued before the router's result is read, so the GPU has it to
   do while the host pins the routed ones */
static bool shared(Forward *f, int l)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, n = f->n;
  const bool fits = n <= MM_MAX_TOKENS; /* the swiglu writes the down's x16 (ops_swiglu_x16) */
  /* the input's x16 the FFN's mix wrote (f->xf_x16; the router's and the guess's matrices, between, read F32 - or
     write this same x16 of xf) */
  return fwd_mat2(f, L->sh_gate, fwd_dev(f, L->sh_gate), FF, b->sga, FF, L->sh_up, fwd_dev(f, L->sh_up), FF, b->sup,
                  FF, D, b->xf, D, n, f->xf_x16) &&
         (fits ? ops_swiglu_x16(b->sga, b->sup, FF, g->clamp_shexp[l], NULL, n, b->shh, b->x16, b->overflow, NULL)
               : ops_swiglu(b->sga, b->sup, FF, g->clamp_shexp[l], NULL, n, b->shh, NULL)) &&
         fwd_mat_ready(f, L->sh_down, fwd_dev(f, L->sh_down), D, FF, b->shh, FF, n, b->sye, D, fits);
}

/* the next layer's streamed experts read while this one runs, when a step picks at least as many as a layer has (a
   prefill's: most of a layer's are picked; a decode's few would make the reads waste). this layer's unpicked ones
   are wanted a round on, not now: the first to go */
static void prefetch(Forward *f, int l, const int32_t *ids, uint32_t picks)
{
  const uint32_t E = (uint32_t)f->g->n_experts;
  if (picks < E)
    return;
  bool picked[OPS_MAX_EXPERTS] = {false};
  for (uint32_t m = 0; m < picks; m++)
    picked[ids[m]] = true;
  const uint32_t here = (uint32_t)l * E, next = (uint32_t)((l + 1) % f->n_layers) * E;
  for (uint32_t e = 0; e < E; e++)
    if (f->rix[here + e] < 0 && !picked[e] && pool_state(f->pool, here + e) != POOL_COLD)
      pool_want(f->pool, here + e, f->visits - 1 + (uint64_t)f->n_layers);
  for (uint32_t e = 0; e < E; e++)
    if (f->rix[next + e] < 0)
      pool_want(f->pool, next + e, f->visits + 1); /* the next layer's visit */
}

/* the routed experts: each distinct one resident or held in the pool (their reads at once) and read once for
   all the tokens routed to it - gate, up, swiglu (each pair its weight, before the down projection, as ref.py) and
   down, a launch each (mm_group); each token's summed in ascending expert id (ref.py's index_add order), the shared
   expert last. nothing here waits for the GPU but the routing event; the experts are held for the layer's every
   sub-batch and released at the next layer's routing event (or the step's end), when the stream is past them */
bool fwd_moe(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used, n = f->n, P = n * K;
  bool ok = fwd_mat(f, L->router, fwd_dev(f, L->router), E, D, b->xf, D, n, b->logits, E);
  if (ok)
    fwd_probe(f, "router_logits", l, f->at, b->logits, (uint64_t)n * E, false);
  /* the picks to the host and marked, then the shared expert queued: the GPU runs it while the host waits. the router
     writes them into the host's mapped r_ids (the routing event covers the kernel); a probe's into ridx, copied */
  ok = ok && ops_router(b->logits, (const float *)fwd_dev(f, L->router_bias), E, K, g->expert_scale, n,
                        f->probe ? b->ridx : f->r_ids_dev, b->rw, NULL) &&
       (!f->probe ||
        hipMemcpyAsync(f->r_ids, b->ridx, P * sizeof(int32_t), hipMemcpyDeviceToHost, NULL) == hipSuccess) &&
       (!f->cpu || !f->lane || P >= E || /* the CPU lane's: the step's FFN inputs and weights, with the picks */
        (hipMemcpyAsync(f->cpu_xf, b->xf, (uint64_t)n * D * sizeof(float), hipMemcpyDeviceToHost, NULL) ==
             hipSuccess &&
         hipMemcpyAsync(f->cpu_rw, b->rw, P * sizeof(float), hipMemcpyDeviceToHost, NULL) == hipSuccess)) &&
       fwd_guess_queue(f, l) && hipEventRecord(f->routed, NULL) == hipSuccess && shared(f, l) &&
       fwd_wait(f, f->routed);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the router, the shared expert)");
  if (!fwd_drain(f, err))
    return false;
  if (f->held_layer != l) /* the stream is past the last layer's experts: the router came after them */
  {
    fwd_release_held(f);
    f->visits++, f->held_layer = l, pool_now(f->pool, f->visits);
  }
  const int32_t *ids = f->r_ids;
  fwd_guess_score(f, l, ids, P);
  fwd_guess_want(f, l);
  fwd_probe(f, "route_idx", l, f->at, b->ridx, P, true);
  fwd_probe(f, "route_w", l, f->at, b->rw, P, false);
  /* the distinct experts in ascending id, each with its pairs (in token order: at most one a token): an expert's
     rows base .. base + cnt, its slots those tokens; one launch of mm_group reads each once for all of its tokens */
  uint32_t nd = 0, *by = f->m_by, at[OPS_MAX_EXPERTS + 1] = {0};
  for (uint32_t m = 0; m < P; m++)
  {
    if (ids[m] < 0 || (uint32_t)ids[m] >= E)
      return fwd_fail(err, "the router picked an expert the model does not have");
    at[ids[m] + 1]++;
    f->picks[(uint32_t)l * E + (uint32_t)ids[m]]++;
  }
  for (uint32_t e = 0; e < E; e++) /* by expert, then pair: a counting sort, each expert's pairs in pair order */
    at[e + 1] += at[e];
  for (uint32_t m = 0; m < P; m++)
    by[at[ids[m]]++] = m;
  uint32_t ne = 0; /* the tables' entries: an expert a part of at most MM_MAX_TOKENS of its pairs (mm_group's) */
  for (uint32_t r = 0, run = 0; r < P; r++)
  {
    const bool first = r == 0 || ids[by[r]] != ids[by[r - 1]];
    nd += first, run = first ? 1 : run + 1;
    ne += run % MM_MAX_TOKENS == 1 || MM_MAX_TOKENS == 1;
  }
  /* the layer's tables, carved from one pinned region and sent in one copy */
  Carve c = {f->r_up, (uint8_t *)b->up, 0};
  const void **tab, **dtab;
  int32_t *cnt, *order, *gidx, *hidx, *pidx, *dcnt, *dorder, *dgidx, *dhidx, *dpidx;
  uint32_t *xg, *yg, *xd, *yd, *dxg, *dyg, *dxd, *dyd;
  tab = carve(&c, 3 * ne * sizeof *tab, (void **)&dtab), cnt = carve(&c, ne * 4, (void **)&dcnt);
  order = carve(&c, P * 4, (void **)&dorder), pidx = carve(&c, P * 4, (void **)&dpidx);
  /* an expert's slots: its tokens, padded to whole tiles - at most P + (MM_TILE - 1) an expert */
  uint32_t slots = 0;
  for (uint32_t r = 0, c_ = 0; r < P; r++)
  {
    c_ = (r == 0 || ids[by[r]] != ids[by[r - 1]]) ? 1 : c_ + 1;
    const bool last = r + 1 == P || ids[by[r + 1]] != ids[by[r]];
    if (last)
      slots += (c_ + MM_TILE - 1) / MM_TILE * MM_TILE;
  }
  gidx = carve(&c, slots * 4, (void **)&dgidx), hidx = carve(&c, slots * 4, (void **)&dhidx);
  xg = carve(&c, ne * 4, (void **)&dxg), yg = carve(&c, ne * 4, (void **)&dyg);
  xd = carve(&c, ne * 4, (void **)&dxd), yd = carve(&c, ne * 4, (void **)&dyd);
  for (uint32_t s = 0; s < slots; s++)
    gidx[s] = -1, hidx[s] = -1;
  uint32_t *row_of = f->m_row, *base = f->m_base, *slot0 = f->m_slot0, unit_of[OPS_MAX_EXPERTS];
  for (uint32_t r = 0, j = 0, next_slot = 0; r < P; r++)
  {
    const uint32_t m = by[r];
    if (r == 0 || ids[m] != ids[by[r - 1]])
    {
      if (r)
        j++, next_slot += (uint32_t)(cnt[j - 1] + MM_TILE - 1) / MM_TILE * MM_TILE;
      base[j] = r, cnt[j] = 0, slot0[j] = next_slot;
      const uint32_t unit = (uint32_t)l * E + (uint32_t)ids[m];
      unit_of[j] = unit;
      if (fwd_cpu_takes(f, unit, P)) /* the lane's CPU half computes it from the files; its read into the pool begun
                                        and held, not waited for (a swap's, at a later visit) */
      {
        f->on_cpu[unit] = f->visits;
        if (!f->is_held[unit] && pool_pin(f->pool, unit, (const void *[POOL_PARTS]){0}) == POOL_WAIT)
        {
          if (f->n_pend < FWD_MAX_PAIRS)
            f->cpu_pend[f->n_pend++] = unit;
          else
            f->held[f->n_held++] = unit;
          f->is_held[unit] = true;
        }
      }
      else if (f->rix[unit] < 0 && !f->is_held[unit]) /* held once for all the layer's sub-batches */
      {
        /* full while reads are under way (a prompt's next layer, read ahead): a reading unit is made room of once
           its read ends - wait for one, pin again; full with none reading is the pool's last word */
        PoolPin pin;
        while ((pin = pool_pin(f->pool, unit, (const void *[POOL_PARTS]){0})) == POOL_FULL && pool_reading(f->pool))
          if (!fwd_wait_read(f, err))
            return false;
        if (pin == POOL_FULL)
          return fwd_fail(err, "an expert the pool cannot hold (its room, or GPU memory refused)");
        f->held[f->n_held++] = unit, f->is_held[unit] = true;
      }
    }
    const uint32_t slot = slot0[j] + (uint32_t)cnt[j]++;
    gidx[slot] = (int32_t)(m / K), hidx[slot] = (int32_t)r;
    pidx[r] = (int32_t)m, row_of[m] = r;
  }
  prefetch(f, l, ids, P);
  if (f->lane && P < E) /* the lane's small step: every expert in the matvec's order (hip_lane.c) */
  {
    if (!fwd_lane_experts(f, l, ids, P, err))
      return false;
    if (!fwd_swap_layer(f, l, err) || !fwd_cpu_settle(f, err))
      return false;
    fwd_probe(f, "moe_out", l, f->at, b->moe, (uint64_t)n * D, false);
    return true;
  }
  /* the entries (an expert's parts of MM_MAX_TOKENS: its slots and rows from part q's first) in ascending count,
     then expert, then part: group()'s runs of the same column tiles contiguous */
  int32_t cnt_j[OPS_MAX_EXPERTS];
  memcpy(cnt_j, cnt, nd * sizeof *cnt);
  uint64_t key[OPS_MAX_EXPERTS + FWD_MAX_STEP * OPS_MAX_TOPK / MM_MAX_TOKENS]; /* part's count, expert, part */
  for (uint32_t j = 0, e = 0; j < nd; j++)
    for (uint32_t q = 0; q * MM_MAX_TOKENS < (uint32_t)cnt_j[j]; q++, e++)
    {
      const uint32_t left = (uint32_t)cnt_j[j] - q * MM_MAX_TOKENS;
      key[e] = (uint64_t)(left < MM_MAX_TOKENS ? left : MM_MAX_TOKENS) << 40 | (uint64_t)j << 16 | q;
    }
  qsort(key, ne, sizeof *key, cmp_u64);
  for (uint32_t p = 0; ok && p < ne; p++)
  {
    const uint32_t j = (uint32_t)(key[p] >> 16 & 0xFFFFFF), q = (uint32_t)(key[p] & 0xFFFF);
    const uint32_t s0 = slot0[j] + q * MM_MAX_TOKENS, r0 = base[j] + q * MM_MAX_TOKENS;
    const void *ex[POOL_PARTS];
    const int32_t at_res = f->rix[unit_of[j]];
    ok = at_res >= 0 ? resident_dev(f->res, (size_t)at_res, ex) : fwd_pinned(f, unit_of[j], ex, err);
    tab[p] = ex[0], tab[ne + p] = ex[1], tab[2 * ne + p] = ex[2], cnt[p] = (int32_t)(key[p] >> 40);
    xg[p] = s0 * D, yg[p] = r0 * FF, xd[p] = s0 * FF, yd[p] = r0 * D;
  }
  for (uint32_t i = 0; i < n; i++) /* each token's rows in ascending expert id */
  {
    int32_t ord[OPS_MAX_TOPK][2];
    for (uint32_t k = 0; k < K; k++)
      ord[k][0] = ids[i * K + k], ord[k][1] = (int32_t)row_of[i * K + k];
    qsort(ord, K, sizeof *ord, cmp_expert);
    for (uint32_t k = 0; k < K; k++)
      order[i * K + k] = ord[k][1];
  }
  ok = ok && fwd_put_async(b->up, f->r_up, c.used) && ops_gather_rows(b->rwp, b->rw, dpidx, P, 1, NULL) &&
       mm_x16_rows(b->xf, D, dgidx, slots, D, b->xe16, b->overflow, NULL) &&
       group(L->exp_gate->type, dtab, cnt, dcnt, ne, FF, D, b->xe16, dxg, b->ga, dyg, FF) &&
       group(L->exp_up->type, dtab + ne, cnt, dcnt, ne, FF, D, b->xe16, dxg, b->upr, dyg, FF) &&
       ops_swiglu(b->ga, b->upr, FF, g->clamp_exp[l], b->rwp, P, b->hh, NULL) &&
       mm_x16_rows(b->hh, FF, dhidx, slots, FF, b->xe16, b->overflow, NULL) &&
       group(L->exp_down->type, dtab + 2 * ne, cnt, dcnt, ne, D, FF, b->xe16, dxd, b->ye, dyd, D) &&
       ops_gather_sum_add(b->moe, b->ye, dorder, K, n, D, b->sye, NULL);
  if (!ok)
    return err[0] ? false : fwd_fail(err, "a GPU call failed (the experts)");
  if (!fwd_swap_layer(f, l, err))
    return false;
  fwd_probe(f, "moe_out", l, f->at, b->moe, (uint64_t)n * D, false);
  return true;
}
