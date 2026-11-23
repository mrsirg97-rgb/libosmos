#include <string.h>

#include "dflash_core.h"
#include "q4x_impl.h"
#include "osmos/err.h"

/* DFlash2's drafter on Qwen (osmos/q4x/forward.h, q4x_dflash_seed / q4x_dflash_draft): the core's math
   (dflash_core.h) through Qwen's calls - its matrices' path (q4x_mat, q4x_mat2: fused), its embedding's rows, its
   output head, its probe and its f16 overflow check, the default stream */

static const void *h_dev(void *c, const GgufTensor *t) { return q4x_dev((const Q4xForward *)c, t); }

static bool h_mat(void *c, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
                  float *y, uint32_t ys)
{
  return q4x_mat((Q4xForward *)c, t, rows, cols, x, xs, n, y, ys, false);
}

static bool h_mat2(void *c, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
                   uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n)
{
  return q4x_mat2((Q4xForward *)c, t0, rows0, y0, ys0, t1, rows1, y1, ys1, cols, x, xs, n, false);
}

static bool h_embed(void *c, const int32_t *tok, uint32_t n, float *h)
{
  const Q4xForward *f = c;
  const GgufTensor *e = f->w->embed;
  return q4x_rows(e->type, q4x_dev(f, e), e->bytes / e->dims[1], (uint32_t)f->df->geo.n_embd, tok, n, h, NULL);
}

static bool h_head(void *c, const float *x, uint32_t xs, uint32_t n, float *logits)
{
  Q4xForward *f = c;
  const uint32_t V = (uint32_t)f->df->geo.n_vocab;
  return q4x_mat(f, f->w->output, V, (uint32_t)f->df->geo.n_embd, x, xs, n, logits, V, false);
}

static void h_seen(void *c, const float *hn, uint64_t n) { q4x_seen((const Q4xForward *)c, "df_out", -1, (float *)hn, n); }
static bool h_before(void *c) { return q4x_overflow_read((Q4xForward *)c); }
static bool h_after(void *c, char *err) { return q4x_overflow_ok((Q4xForward *)c, err); }

static DfHost host(Q4xForward *f)
{
  return (DfHost){f, NULL, h_dev, h_mat, h_mat2, h_embed, h_head, h_seen, h_before, h_after};
}

static DfRings rings(const Q4xForward *f, Q4xSeq *s) { return (DfRings){s->dk, s->dv, f->df_ring}; }

bool q4x_dflash_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  if (!f || !s || s->f != f || !f->df)
    return osmos_fail(err, "no forward with a drafter, or a sequence of another forward");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its states part written): load it, or make another");
  if (s->step_n && s->step_id != f->steps) /* 0.9.2: b.taps a step's alone, from its row 0 - a batch of more than
                                              Q4X_MAX_RUNS runs is several steps, the later ones' taps over its */
    return osmos_fail(err, "the drafter's taps another step's: seed right after the sequence's step (a batch of at most "
                         "64 sequences)");
  if (from >= to || !s->step_n || from < s->step_pos0 || to > s->step_pos0 + (int32_t)s->step_n || to > s->pos)
    return osmos_fail(err, "positions not of the sequence's last step, or past its position");
  const uint32_t T = (uint32_t)f->df->geo.n_taps * (uint32_t)f->df->geo.n_embd;
  const float *taps = f->b.taps + (uint64_t)(s->step_t0 + (uint32_t)(from - s->step_pos0)) * T;
  const DfHost h = host(f);
  const DfRings r = rings(f, s);
  if (!df_seed(&h, f->df, &f->d, &r, taps, from, (uint32_t)(to - from)))
    return osmos_fail(err, "a GPU call failed (the drafter's context)");
  s->df_lo = from > s->df_pos ? from : from < s->df_lo ? from : s->df_lo; /* past a gap: the context starts over */
  s->df_pos = to;
  return true;
}

int q4x_dflash_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                     float *conf, char err[OSMOS_ERR])
{
  if (!f || !s || s->f != f || !f->df || (max && (!drafts || !conf)))
    return osmos_fail(err, "no forward with a drafter, a sequence of another forward, or no room"), -1;
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its states part written): load it, or make another"), -1;
  const Q4xDflashGeometry *g = &f->df->geo;
  if (y < 0 || y >= g->n_vocab)
    return osmos_fail(err, "a token not of the vocabulary"), -1;
  if (s->df_pos != s->pos)
    return osmos_fail(err, "the drafter's context not seeded to the sequence's position"), -1;
  const int32_t pos0 = s->pos, room = (int32_t)f->max_ctx - 1 - pos0; /* a draft's position below max_ctx */
  uint32_t n = (uint32_t)g->block - 1 < max ? (uint32_t)g->block - 1 : max;
  n = room <= 0 ? 0 : (uint32_t)room < n ? (uint32_t)room : n;
  if (!n)
    return 0;
  const int32_t lo = pos0 - (int32_t)g->window, ctx_from = lo > s->df_lo ? lo : s->df_lo; /* (the ring past it) */
  const DfHost h = host(f);
  const DfRings r = rings(f, s);
  return df_draft(&h, f->df, &f->d, &r, y, pos0, ctx_from, n, conf_min, cum_min, drafts, conf, err, OSMOS_ERR);
}
