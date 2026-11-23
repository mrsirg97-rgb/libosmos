/* Inkling-2-Small, the third family (0.8.0): its model opened and every tensor checked (osmos/ink2/weights.h: the
   reference's names or llama.cpp's, unsloth's packs), its engine osmos/ink2/forward.h's - the dense weights resident,
   the experts streamed through the membrane at the budget's room, a sequence's caches and tails, a step a run at a
   time; the drafter (--dspark: the checkpoint's mtp.safetensors as ink2_mtp_gguf.py's GGUF, DSpark with the RNN
   Markov head) its seed and draft, block - 1 drafts a round */
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
   once, so its count is the prompt's speed: an 8K prompt (ours, a 40 GiB room) 115 / 238 / 321-333 tok/s at 2,048 /
   4,096 / 8,192, the forward 54.8 / 58.0 / 64.5 GiB with the room (the logits' [max_tokens][n_vocab] 6.6 GB of
   8,192's) */
#define INK2_STEP_TOKENS 8192

/* the logits' rows a step keeps at most: a prompt's last row, a decode's or a drafted round's every row, each
   --parallel lane's - 1,024 x the vocabulary's 201,024 F32 0.77 GiB (a step's every row at 8,192: 6.1 GiB) */
#define INK2_LOGIT_ROWS (OSMOS_MAX_LANES * (OSMOS_MAX_BLOCK + 1)) /* a round's most kept rows: each lane's drafts and its token */
/* a streamed build's (Inkling-2-Small's): measured again (0.15.2, low-power, osmos-api, code / prose at temperature 0,
   500 tokens, every floor the same tokens): 0.95 11.5 / 11.1 tok/s, 0.9 (0.8.x's) 11.8 / 11.3 and 11.8 / 11.5, 0.7
   12.5 / 12.0, 0.5 12.7 / 12.4, 0.3 12.6 / 12.1 - the verify cheaper than when 0.9 won, the rounds shorter now pay */
#define INK2_DRAFT_CUM_MIN 0.5f
/* every expert in memory (ink2_forward_whole): a verified token's experts read from memory, not streamed - the round
   longer. measured (Inkling-2-Mini, its BF16 drafter, low-power, 2026-11-20; a 2,048-token code prompt / a 64-token
   one, 256 tokens): 0.9 43.4 / 29.2 tok/s, 0.7 47.6 / 32.7, 0.5 48.6 / 32.8, 0.3 48.7 / 32.1, 0.1 48.7 / 29.9 (plain
   24.2 / 25.6), every floor the same tokens */
#define INK2_DRAFT_CUM_MIN_WHOLE 0.5f

/* INK2_PROFILE's round accounting (a measurement's): the steps' time by their tokens (1 .. 16, then more), the
   drafter's seeds and drafts and the drafts they gave, printed at e_destroy */
#define PROF_N 17
static struct
{
  int on;
  double step_s[PROF_N], seed_s, draft_s;
  uint64_t steps[PROF_N], seeds, drafts, drafted;
} g_rp;

static double now_s(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int rp_on(void)
{
  if (!g_rp.on)
    g_rp.on = getenv("INK2_PROFILE") && atoi(getenv("INK2_PROFILE")) ? 1 : -1;
  return g_rp.on > 0;
}

static void rp_print(void)
{
  if (!rp_on())
    return;
  fprintf(stderr, "ink2 rounds: steps by tokens (count, ms each):");
  for (int k = 1; k < PROF_N; k++)
    if (g_rp.steps[k])
      fprintf(stderr, " %d%s:%llu/%.1f", k, k == PROF_N - 1 ? "+" : "", (unsigned long long)g_rp.steps[k],
              1e3 * g_rp.step_s[k] / (double)g_rp.steps[k]);
  fprintf(stderr, "\nink2 rounds: %llu seeds %.2f ms each, %llu drafts %.2f ms each (%llu tokens drafted)\n",
          (unsigned long long)g_rp.seeds, g_rp.seeds ? 1e3 * g_rp.seed_s / (double)g_rp.seeds : 0.0,
          (unsigned long long)g_rp.drafts, g_rp.drafts ? 1e3 * g_rp.draft_s / (double)g_rp.drafts : 0.0,
          (unsigned long long)g_rp.drafted);
}

typedef struct Eng
{
  OsmosLedger *ledger; /* the engine's (borrowed: osmos_create's, destroyed after this) */
  Ink2Forward *f;
  Gguf *mg;        /* the drafter's file, or NULL */
  Ink2Mtp *mtp;    /* the drafter, or NULL */
  uint64_t *picks; /* the picks file's counts, or NULL */
  uint32_t vocab, n;
} Eng;

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  rp_print();
  ink2_forward_destroy(e->f);
  ink2_mtp_destroy(e->mtp);
  gguf_destroy(e->mg);
  free(e->picks);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, OsmosLedger *ledger, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->npu)
    return osmos_fail(err, "no NPU kernels for this family"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return osmos_fail(err, "out of memory"), NULL;
  e->ledger = ledger;
  e->vocab = (uint32_t)m->w->geo.n_vocab;
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
                          INK2_LOGIT_ROWS, o->kv_q8};
  if (!(e->f = ink2_forward_create(m->w, &io, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return ink2_identity(((const Eng *)p)->f); }

static void e_stats(const void *p, OsmosStats *s) { s->budget = ink2_forward_bytes(((const Eng *)p)->f); }

/* the batch's sequences one step (each one's consecutive tokens a run, INK2_MAX_RUNS of them at most); a sequence
   met again after another's refused */
static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  _Static_assert(INK2_MAX_RUNS == OSMOS_FAMILY_RUNS, "a step's runs the engine's");
  Eng *e = p;
  e->n = 0;
  if (!osmos_runs_consecutive(seqs, n, err))
    return false;
  Ink2Seq *run[INK2_MAX_RUNS];
  uint32_t first[INK2_MAX_RUNS], len[INK2_MAX_RUNS], runs;
  if (osmos_runs(seqs, n, 0, first, len, &runs) < n) /* the runs counted before any step: a batch of more refused whole */
    return osmos_fail(err, "a batch of more than INK2_MAX_RUNS sequences");
  for (uint32_t r = 0; r < runs; r++)
    run[r] = oseq_impl(seqs[first[r]]);
  const double t0 = rp_on() ? now_s() : 0;
  if (!n || !ink2_step(e->f, run, len, runs, tokens, err))
    return n ? false : osmos_fail(err, "an empty batch");
  if (rp_on()) /* (ink2_step synchronizes: its time the step's) */
  {
    const int k = n < PROF_N - 1 ? (int)n : PROF_N - 1;
    g_rp.step_s[k] += now_s() - t0, g_rp.steps[k]++;
  }
  e->n = n;
  return true;
}

static const float *e_logits(const void *p, uint32_t row)
{
  const Eng *e = p;
  return row < e->n ? ink2_row_logits(e->f, row) : NULL;
}

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  Eng *e = p;
  if (!rows || (uint64_t)row0 + rows > e->n)
    return osmos_fail(err, "a row the last step kept no logits of");
  return ink2_greedy(e->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p)
{
  const Eng *e = p;
  return e->mtp ? (uint32_t)e->mtp->block - 1 : 0;
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  const double t0 = rp_on() ? now_s() : 0;
  const bool ok = ink2_seq_seed(((Eng *)p)->f, s, from, to, err);
  if (rp_on())
    (void)hipDeviceSynchronize(), g_rp.seed_s += now_s() - t0, g_rp.seeds++;
  return ok;
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  /* a verified token reads its own experts (~0.4-0.6 of a step's), not the engine's ~free one: a round ends at
     a higher product of confidences than the engine's floor - 0.9 measured best with a ~12 ms draft (0.8.x: p2 13.8
     -> 14.5 tok/s, plain 9.1; p1 13.5 -> 13.4, plain 11.5; p3 14.2 -> 16.9, plain 14.3; 0.5, 0.6 below 0.7,
     0.8 its tie with p1 4% under); 0.5 since 0.15.2 (INK2_DRAFT_CUM_MIN's measure) */
  const float floor_ = ink2_forward_whole(((Eng *)p)->f) ? INK2_DRAFT_CUM_MIN_WHOLE : INK2_DRAFT_CUM_MIN;
  if (cum_min < floor_)
    cum_min = floor_;
  const uint32_t most = e_block(p);
  const double t0 = rp_on() ? now_s() : 0;
  const int got = ink2_seq_draft(((Eng *)p)->f, s, y, most, conf_min, cum_min, drafts, conf, err);
  if (rp_on())
    g_rp.draft_s += now_s() - t0, g_rp.drafts++, g_rp.drafted += got > 0 ? (uint64_t)got : 0;
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
    "inkling2", m_open,     m_close,       e_create,  e_destroy,   e_identity, e_stats,
    e_batch,    e_logits,   e_greedy,      e_block,   s_create,    s_destroy, s_reset,  s_prompt, s_rollback,
    s_reach,    s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    NULL, /* a prompt's read its batch: one math for every step */
    s_mark_bytes, s_mark_save, s_mark_load, /* a mark the conv tails and windows (ink2_seq_mark_save) */
};
