#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "engine_impl.h"
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"

/* LiquidAI's LFM3 (lfm3: vibes1.7b) behind the engine: every weight resident, lfm3's sequence (DESIGN.md, "LFM3 step
   2"). one math (no read yet: step 3's) - a token's bits however its step is split or shared; the logits of the kept
   rows alone. no drafter yet: block 0 */
_Static_assert(LFM3_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");
_Static_assert(OSMOS_ERR >= LFM3_ERR && OSMOS_ERR >= GGUF_ERR, "one error buffer for the engine and the family's calls");

/* a step's most tokens when the server names none (a prompt's chunk), and a round's most kept rows */
#define LFM3_STEP_TOKENS 2048
#define LFM3_LOGIT_ROWS (OSMOS_MAX_LANES * (OSMOS_MAX_BLOCK + 1))

typedef struct Model
{
  Gguf *g;
  Lfm3Weights *w;
} Model;

typedef struct Eng
{
  Lfm3Forward *f;
} Eng;

static bool fail(char err[OSMOS_ERR], const char *what)
{
  snprintf(err, OSMOS_ERR, "%s", what);
  return false;
}

static void m_close(void *p)
{
  Model *m = p;
  if (!m)
    return;
  lfm3_weights_destroy(m->w);
  gguf_destroy(m->g);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->table || files->overlay)
    return fail(err, "lfm3: no table file or overlay is read"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  if (!(m->g = gguf_create(files->model, err)) || !(m->w = lfm3_weights_create(m->g, err)))
    return m_close(m), NULL;
  const Lfm3Geometry *geo = &m->w->geo;
  *facts = (OsmosFacts){"lfm3", (uint32_t)geo->n_vocab, geo->native_ctx, LFM3_LOGIT_ROWS, 0, false};
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  lfm3_forward_destroy(e->f);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->budget || o->picks || o->no_guess || o->npu)
    return fail(err, "lfm3: every weight resident - no budget, picks, no_guess or NPU"), NULL;
  if (o->drafter)
    return fail(err, "lfm3: no drafter yet"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  const uint32_t steps = o->max_tokens ? o->max_tokens : LFM3_STEP_TOKENS;
  const Lfm3Options ho = {steps < o->max_ctx ? steps : o->max_ctx, o->max_ctx, LFM3_LOGIT_ROWS};
  if (!(e->f = lfm3_forward_create(m->w, &ho, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return lfm3_identity(((const Eng *)p)->f); }

static bool e_room(const void *p, uint64_t *bytes)
{
  (void)p;
  size_t free_b = 0, total = 0;
  if (hipMemGetInfo(&free_b, &total) != hipSuccess)
    return false;
  *bytes = free_b;
  return true;
}

static void e_stats(const void *p, OsmosStats *s) { (void)p, (void)s; }

/* the batch's sequences one step (each one's consecutive tokens a run), LFM3_MAX_RUNS at a time */
static bool batch(Eng *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i] && seqs[i - 1] != seqs[i])
        return fail(err, "a sequence's tokens not consecutive in the batch");
  Lfm3Seq *run[LFM3_MAX_RUNS];
  uint32_t len[LFM3_MAX_RUNS];
  for (uint32_t at = 0; at < n;)
  {
    uint32_t runs = 0, k = 0;
    for (; at + k < n && runs < LFM3_MAX_RUNS; runs++)
    {
      uint32_t m = 1;
      while (at + k + m < n && seqs[at + k + m] == seqs[at + k])
        m++;
      run[runs] = oseq_impl(seqs[at + k]), len[runs] = m, k += m;
    }
    if (at + k < n)
      return fail(err, "lfm3: a batch past LFM3_MAX_RUNS sequences"); /* its logits one step's: never split */
    if (!lfm3_step_runs(e->f, run, len, runs, tokens + at, err))
      return false;
    at += k;
  }
  return true;
}

static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return batch(p, seqs, tokens, n, err);
}

static const float *e_logits(const void *p, uint32_t row) { return lfm3_row_logits(((const Eng *)p)->f, row); }

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  return lfm3_greedy(((Eng *)p)->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p)
{
  (void)p;
  return 0;
}

static void *s_create(void *p, char err[OSMOS_ERR]) { return lfm3_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { lfm3_seq_destroy(s); }
static bool s_prompt(void *s, uint32_t end) { return lfm3_seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return lfm3_seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return lfm3_seq_reach(s); }
static int32_t s_position(const void *s) { return lfm3_seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return lfm3_seq_state_bytes(s); }

static bool s_save_to(const void *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  return lfm3_seq_save_to(s, sink, ctx, err);
}

static bool s_load_from(void *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return lfm3_seq_load_from(s, bytes, src, ctx, err);
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  (void)p, (void)s, (void)from, (void)to, (void)err;
  return true; /* no drafter: nothing to seed */
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  (void)p, (void)s, (void)y, (void)conf_min, (void)cum_min, (void)drafts, (void)conf, (void)err;
  return 0; /* no drafter: no drafts */
}

const OsmosFamily osmos_lfm3 = {
    "lfm3", m_open,     m_close,       e_create,  e_destroy,   e_identity, e_room,  e_stats,
    e_batch,       e_logits,   e_greedy,      e_block,   s_create,    s_destroy,  s_prompt, s_rollback,
    s_reach,       s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    NULL, /* one math: a prompt's read its batch (the matrix cores' step 3's) */
    NULL, NULL, NULL, /* no marks: the whole state */
};
