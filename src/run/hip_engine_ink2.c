/* Inkling-2-Small: its model opened and every tensor checked (osmos/ink2/weights.h: the reference's names or
   llama.cpp's, unsloth's packs), its engine osmos/ink2/forward.h's - the dense weights resident, the experts streamed
   through the membrane at the budget's room, a sequence's caches and tails, a step a run at a time; the drafter
   (--dspark: the checkpoint's mtp.safetensors as ink2_mtp_gguf.py's GGUF, DSpark with the RNN Markov head) its seed
   and draft, block - 1 drafts a round */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <hip/hip_runtime_api.h>

#include "engine_impl.h"
#include "osmos/ink2/forward.h"
#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"
#include "osmos/run/fit.h"

typedef struct Model
{
  Gguf *g;
  Ink2Weights *w;
} Model;

static void m_close(void *p)
{
  Model *m = p;
  if (!m)
    return;
  ink2_weights_destroy(m->w);
  gguf_destroy(m->g);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->lsa)
    return osmos_fail(err, "this family reads no LSA maps (--lsa: LFM3's)"), NULL;
  if (files->table)
    return osmos_fail(err, "no Engram table is read for this family"), NULL;
  if (files->overlay)
    return osmos_fail(err, "no overlay is taken yet (a pack's patch is assembled into its shards)"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return osmos_fail(err, "out of memory"), NULL;
  char werr[OSMOS_ERR > OSMOS_ERR ? OSMOS_ERR : OSMOS_ERR] = "";
  if (!(m->g = gguf_create(files->model, werr)) || !(m->w = ink2_weights_create(m->g, werr)))
  {
    if (err)
      osmos_fail(err, "%s", werr);
    return m_close(m), NULL;
  }
  const Ink2Geometry *geo = &m->w->geo;
  /* a decode round's most tokens below every expert's read (a step past it reads them all, as a prompt's) */
  *facts = (OsmosFacts){"inkling2", (uint32_t)geo->n_vocab, geo->native_ctx,
                        (uint32_t)((geo->n_experts - 1) / geo->n_used), geo->native_ctx, true};
  /* (seed_window: a drafter's context every position - its full layer's attention reads them all) */
  return m;
}

_Static_assert(INK2_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");

/* a step's most tokens when the server names none: a prompt's chunk. a prompt step reads every expert not in memory
   once, so its count is the prompt's speed - against the forward's memory, the logits' [max_tokens][n_vocab] with it */
#define INK2_STEP_TOKENS 8192

/* the logits' rows a step keeps at most: a prompt's last row, a decode's or a drafted round's every row, each
   --parallel lane's */
#define INK2_LOGIT_ROWS (OSMOS_MAX_LANES * (OSMOS_MAX_BLOCK + 1)) /* a round's most kept rows: each lane's drafts and its token */
/* a streamed build's (Inkling-2-Small's) floor: swept at temperature 0, every floor the same tokens */
#define INK2_DRAFT_CUM_MIN 0.5f
/* every expert in memory (ink2_forward_whole): a verified token's experts read from memory, not streamed - the round
   longer; swept, every floor the same tokens */
#define INK2_DRAFT_CUM_MIN_WHOLE 0.5f

/* the profile's round accounting (OsmosOptions' profile, a measurement's): the steps' time by their tokens (1 .. 16,
   then more), the drafter's seeds and drafts and the drafts they gave, printed at e_destroy - the engine's own */
#define PROF_N 17
typedef struct Rounds
{
  bool on;
  double step_s[PROF_N], seed_s, draft_s;
  uint64_t steps[PROF_N], seeds, drafts, drafted;
} Rounds;

static double now_s(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void rp_print(const Rounds *rp)
{
  if (!rp->on)
    return;
  fprintf(stderr, "ink2 rounds: steps by tokens (count, ms each):");
  for (int k = 1; k < PROF_N; k++)
    if (rp->steps[k])
      fprintf(stderr, " %d%s:%llu/%.1f", k, k == PROF_N - 1 ? "+" : "", (unsigned long long)rp->steps[k],
              1e3 * rp->step_s[k] / (double)rp->steps[k]);
  fprintf(stderr, "\nink2 rounds: %llu seeds %.2f ms each, %llu drafts %.2f ms each (%llu tokens drafted)\n",
          (unsigned long long)rp->seeds, rp->seeds ? 1e3 * rp->seed_s / (double)rp->seeds : 0.0,
          (unsigned long long)rp->drafts, rp->drafts ? 1e3 * rp->draft_s / (double)rp->drafts : 0.0,
          (unsigned long long)rp->drafted);
}

typedef struct Eng
{
  OsmosLedger *ledger; /* the engine's (borrowed: osmos_create's, destroyed after this) */
  Rounds rp;           /* the profile's rounds (OsmosOptions' profile) */
  Ink2Forward *f;
  Gguf *mg;        /* the drafter's file, or NULL */
  Ink2Mtp *mtp;    /* the drafter, or NULL */
  uint64_t *picks; /* the picks file's counts, or NULL */
} Eng;

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  rp_print(&e->rp);
  ink2_forward_destroy(e->f);
  ink2_mtp_destroy(e->mtp);
  gguf_destroy(e->mg);
  free(e->picks);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, OsmosLedger *ledger, char err[OSMOS_ERR])
{
  const Model *m = pm;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return osmos_fail(err, "out of memory"), NULL;
  e->ledger = ledger;
  e->rp.on = o->profile;
  if (o->picks) /* a picks file: the router's picks [layer][expert] as uint64, nothing after (DeepSeek's layout) */
  {
    const size_t n = (size_t)m->w->geo.n_layers * m->w->geo.n_experts;
    FILE *pf = fopen(o->picks, "rb");
    const bool ok = pf && (e->picks = malloc(n * sizeof *e->picks)) && fread(e->picks, sizeof *e->picks, n, pf) == n &&
                    fgetc(pf) == EOF;
    if (pf)
      fclose(pf);
    if (!ok)
      return osmos_fail(err, "the picks file could not be read, or is not this model's"), e_destroy(e), NULL;
  }
  /* the drafter: its GGUF (tools/convert/ink2_mtp_gguf.py's) - a directory holding one, or the file */
  if (o->drafter)
  {
    char path[4096], ierr[OSMOS_ERR] = "";
    snprintf(path, sizeof path, "%s/ink2-mtp.gguf", o->drafter);
    FILE *probe = fopen(path, "rb");
    if (probe)
      fclose(probe);
    else
      snprintf(path, sizeof path, "%s", o->drafter);
    if (!(e->mg = gguf_create(path, err)))
      return e_destroy(e), NULL;
    if (!(e->mtp = ink2_mtp_create(e->mg, m->w, ierr)))
      return osmos_fail(err, "%s", ierr), e_destroy(e), NULL;
  }
  const Ink2Options io = {o->max_ctx, osmos_step_tokens(o, INK2_STEP_TOKENS), 0, e->picks, e->mtp, o->budget,
                          INK2_LOGIT_ROWS, o->kv_q8, ledger, o->profile};
  if (!(e->f = ink2_forward_create(m->w, &io, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return ink2_identity(((const Eng *)p)->f); }

static void e_stats(const void *p, OsmosStats *s) { s->budget = ink2_forward_bytes(((const Eng *)p)->f); }

/* the batch's sequences one step (each one's consecutive tokens a run, INK2_MAX_RUNS of them at most); a sequence
   met again after another's refused */
static bool e_step(void *p, void *const *run, const uint32_t *len, uint32_t runs, const int32_t *tokens, bool read,
                   char err[OSMOS_ERR])
{
  _Static_assert(INK2_MAX_RUNS == OSMOS_FAMILY_RUNS, "a step's runs the engine's");
  (void)read; /* (one math for every step) */
  Eng *e = p;
  uint32_t n = 0;
  for (uint32_t r = 0; r < runs; r++)
    n += len[r];
  const double t0 = e->rp.on ? now_s() : 0;
  if (!ink2_step(e->f, (Ink2Seq *const *)run, len, runs, tokens, err))
    return false;
  if (e->rp.on) /* (ink2_step synchronizes: its time the step's) */
  {
    const int k = n < PROF_N - 1 ? (int)n : PROF_N - 1;
    e->rp.step_s[k] += now_s() - t0, e->rp.steps[k]++;
  }
  return true;
}

static void e_kept(const void *p, KeptView *k) { ink2_kept_view(((const Eng *)p)->f, k); }

static uint32_t e_block(const void *p)
{
  const Eng *e = p;
  return e->mtp ? (uint32_t)e->mtp->block - 1 : 0;
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  const double t0 = ((Eng *)p)->rp.on ? now_s() : 0;
  const bool ok = ink2_seq_seed(((Eng *)p)->f, s, from, to, err);
  if (((Eng *)p)->rp.on)
    (void)hipDeviceSynchronize(), ((Eng *)p)->rp.seed_s += now_s() - t0, ((Eng *)p)->rp.seeds++;
  return ok;
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  /* a verified token reads its own experts, not the engine's ~free one: a round ends at a higher product of
     confidences than the engine's floor */
  const float floor_ = ink2_forward_whole(((Eng *)p)->f) ? INK2_DRAFT_CUM_MIN_WHOLE : INK2_DRAFT_CUM_MIN;
  if (cum_min < floor_)
    cum_min = floor_;
  const uint32_t most = e_block(p);
  const double t0 = ((Eng *)p)->rp.on ? now_s() : 0;
  const int got = ink2_seq_draft(((Eng *)p)->f, s, y, most, conf_min, cum_min, drafts, conf, err);
  if (((Eng *)p)->rp.on)
    ((Eng *)p)->rp.draft_s += now_s() - t0, ((Eng *)p)->rp.drafts++, ((Eng *)p)->rp.drafted += got > 0 ? (uint64_t)got : 0;
  return got;
}

static void *s_create(void *p, char err[OSMOS_ERR]) { return ink2_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { ink2_seq_destroy(s); }
static bool s_reset(void *s, char err[OSMOS_ERR]) { return ink2_seq_reset(s, err); }
static bool s_prompt(void *s, uint32_t end) { return ink2_seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return ink2_seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return ink2_seq_reach(s); }
static int32_t s_position(const void *s) { return ink2_seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return ink2_seq_state_bytes(s); }

static bool s_save_to(const void *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  return ink2_seq_save_to(s, sink, ctx, err);
}

static bool s_load_from(void *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return ink2_seq_load_from(s, bytes, src, ctx, err);
}

static uint64_t s_mark_bytes(const void *s) { return ink2_seq_mark_bytes(s); }
static bool s_mark_save(const void *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  return ink2_seq_mark_save(s, out, bytes, err);
}
static bool s_mark_load(void *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  return ink2_seq_mark_load(s, in, bytes, err);
}

const OsmosFamily osmos_ink2 = {
    "inkling2",
    OSMOS_TAKES_BUDGET | OSMOS_TAKES_PICKS | OSMOS_TAKES_DRAFTER | OSMOS_TAKES_KV_Q8 | OSMOS_TAKES_PROFILE, /* its options (engine_impl.h) */
    m_open,     m_close,       e_create,  e_destroy,   e_identity, e_stats,
    e_step,     e_kept,      e_block,   s_create,    s_destroy, s_reset,  s_prompt, s_rollback,
    s_reach,    s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    false, /* a prompt's read its batch: one math for every step */
    s_mark_bytes, s_mark_save, s_mark_load, /* a mark the conv tails and windows (ink2_seq_mark_save) */
};
