/* DFlash2 on LFM3 (osmos/lfm3/forward.h: lfm3_dflash_seed, lfm3_dflash_draft): the DFlash2 core's math
   (src/q4x/dflash_core.h, Qwen's too) through LFM3's calls - its taps, its embedding and tied head, every matrix the
   matvec's (a draft never sets a read's f16 overflow: the verify step's check its own), the step's stream. a round's
   drafts capped where its verify would leave the sequence's reach */
#include <math.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "lfm3_impl.h"

#include "../q4x/dflash_core.h"
#include "osmos/err.h"

static const void *ddev(const Lfm3Forward *f, const GgufTensor *t) { return f->df_dev[t - f->df->model->tensors]; }

/* the core's calls on LFM3: the drafter's tensors from its own table, every matrix the matvec's, the step's stream */
static const void *h_dev(void *c, const GgufTensor *t) { return ddev((const Lfm3Forward *)c, t); }

static bool h_mat(void *c, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
                  float *y, uint32_t ys)
{
  const Lfm3Forward *f = c;
  return mv_strided(t->type, ddev(f, t), rows, cols, x, xs, n, y, ys, f->st);
}

static bool h_embed(void *c, const int32_t *tok, uint32_t n, float *h)
{
  const Lfm3Forward *f = c;
  return ops_embed(f->w->embd->type, lfm3_dev(f, f->w->embd), (uint32_t)f->w->geo.n_embd, tok, n, 1, h, f->st);
}

static bool h_head(void *c, const float *x, uint32_t xs, uint32_t n, float *logits)
{
  const Lfm3Forward *f = c; /* the step's: LFM3's tied head, Q8_0 */
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  return mv_strided(f->head_type, f->head, V, (uint32_t)f->w->geo.n_embd, x, xs, n, logits, V, f->st);
}

static DfHost host(Lfm3Forward *f) { return (DfHost){f, f->st, h_dev, h_mat, NULL, h_embed, h_head, NULL, NULL, NULL}; }

static DfRings rings(const Lfm3Forward *f, Lfm3Seq *s) { return (DfRings){s->dk, s->dv, f->df_ring}; }

bool lfm3_dflash_seed(Lfm3Forward *f, Lfm3Seq *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  if (!f || !s || s->f != f || !f->df)
    return osmos_fail(err, "no forward with a drafter, or a sequence of another forward");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step: load it, or make another");
  if (s->step_n && s->step_id != f->steps)
    return osmos_fail(err, "the drafter's taps another step's: seed right after the sequence's step");
  if (from >= to || !s->step_n || from < s->step_pos0 || to > s->step_pos0 + (int32_t)s->step_n || to > s->pos)
    return osmos_fail(err, "positions not of the sequence's last step, or past its position");
  const uint32_t T = (uint32_t)f->df->geo.n_taps * (uint32_t)f->df->geo.n_embd;
  const float *taps = f->taps + (uint64_t)(s->step_t0 + (uint32_t)(from - s->step_pos0)) * T;
  const DfHost h = host(f);
  const DfRings r = rings(f, s);
  if (!df_seed(&h, f->df, &f->d, &r, taps, from, (uint32_t)(to - from)))
    return osmos_fail(err, "a GPU call failed (the drafter's context)");
  s->df_lo = from > s->df_pos ? from : from < s->df_lo ? from : s->df_lo; /* past a gap: the context starts over */
  s->df_pos = to;
  return true;
}

/* a round's drafts at most: its verify (y and the drafts, at pos ..) never closing an LSA block or crossing a YaRN
   stage - a rollback reaches neither (lfm3_seq_reach), so every draft the verify drops stays within it */
static uint32_t round_room(const Lfm3Forward *f, const Lfm3Seq *s)
{
  int64_t room = INT32_MAX;
  if (f->lsa) /* the verify's end below the next close: pos + drafts + 1 < its block's end */
  {
    const int64_t end = ((int64_t)s->pos / LSA_WINDOW + 1) * LSA_WINDOW;
    room = end - s->pos - 2;
  }
  if (!f->yarn_static_ctx && s->stage + 1 < f->stages) /* the verify's end at the next stage's from at most */
  {
    const int64_t r = (int64_t)f->stage[s->stage + 1].from - s->pos - 1;
    room = r < room ? r : room;
  }
  return room > 0 ? (uint32_t)room : 0;
}

int lfm3_dflash_draft(Lfm3Forward *f, Lfm3Seq *s, int32_t y, uint32_t max, float conf_min, float cum_min,
                      int32_t *drafts, float *conf, char err[OSMOS_ERR])
{
  if (!f || !s || s->f != f || !f->df || (max && (!drafts || !conf)))
    return osmos_fail(err, "no forward with a drafter, a sequence of another forward, or no room"), -1;
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step: load it, or make another"), -1;
  const Q4xDflash *df = f->df;
  const Q4xDflashGeometry *g = &df->geo;
  if (y < 0 || y >= g->n_vocab)
    return osmos_fail(err, "a token not of the vocabulary"), -1;
  if (s->df_pos != s->pos)
    return osmos_fail(err, "the drafter's context not seeded to the sequence's position"), -1;
  const int32_t pos0 = s->pos, room = (int32_t)f->max_ctx - 1 - pos0; /* a draft's position below max_ctx */
  uint32_t n = (uint32_t)g->block - 1 < max ? (uint32_t)g->block - 1 : max;
  n = room <= 0 ? 0 : (uint32_t)room < n ? (uint32_t)room : n;
  const uint32_t reach = round_room(f, s);
  n = reach < n ? reach : n;
  if (!n)
    return 0;
  const int32_t lo = pos0 - (int32_t)g->window, ctx_from = lo > s->df_lo ? lo : s->df_lo; /* (the ring past it) */
  const DfHost h = host(f);
  const DfRings r = rings(f, s);
  return df_draft(&h, f->df, &f->d, &r, y, pos0, ctx_from, n, conf_min, cum_min, drafts, conf, err);
}
