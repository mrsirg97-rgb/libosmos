#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"
#include "osmos/q4x/dflash.h"

/* LiquidAI's LFM3 (lfm3: osmos1.7b) behind the engine: every weight resident, lfm3's sequence (DESIGN.md, "LFM3 step
   2"). a batch is a decode's math, a prompt's read the matrix cores' (two maths, as q35's and hy4's), each a token's
   bits however its step is split or shared; the logits of the kept rows alone. its drafter o->drafter's GGUF, a
   DFlash2 (lfm3_dflash_*: q4x's drafter on LFM3's taps), or none: block 0 */
_Static_assert(LFM3_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");
_Static_assert(Q4X_DFLASH_BLOCK_MAX - 1 <= OSMOS_MAX_BLOCK, "a DFlash block's drafts within the engine's");
_Static_assert(OSMOS_ERR >= Q4X_ERR, "the drafter's loader's errors in the engine's buffer");
_Static_assert(OSMOS_ERR >= LFM3_ERR && OSMOS_ERR >= GGUF_ERR, "one error buffer for the engine and the family's calls");

/* a step's most tokens when the server names none (a prompt's chunk), and a round's most kept rows */
#define LFM3_STEP_TOKENS 2048
#define LFM3_LOGIT_ROWS (OSMOS_MAX_LANES * (OSMOS_MAX_BLOCK + 1))

typedef struct Model
{
  Gguf *g;
  Lfm3Weights *w;
  char *lsa; /* the maps' path (its forwards' lsa), NULL dense */
} Model;

typedef struct Eng
{
  Lfm3Forward *f;
  Gguf *dg;      /* the drafter's file, or NULL */
  Q4xDflash *df; /* its DFlash2, or NULL */
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
  free(m->lsa);
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
  if (files->lsa && !(m->lsa = strdup(files->lsa)))
    return fail(err, "out of memory"), m_close(m), NULL;
  const Lfm3Geometry *geo = &m->w->geo;
  /* seed_window: a DFlash2 drafter's context the prompt's last Q4X_DFLASH_WINDOW_MAX positions (its sliding window at
     most, the loader's check) - 0 left every prompt unseeded, a drafted turn refused (0.11.2) */
  *facts = (OsmosFacts){"lfm3", (uint32_t)geo->n_vocab, geo->native_ctx, LFM3_LOGIT_ROWS, Q4X_DFLASH_WINDOW_MAX, false};
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  lfm3_forward_destroy(e->f);
  q4x_dflash_destroy(e->df);
  gguf_destroy(e->dg);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->budget || o->picks || o->no_guess || o->npu)
    return fail(err, "lfm3: every weight resident - no budget, picks, no_guess or NPU"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  if (o->drafter)
  {
    if (!(e->dg = gguf_create(o->drafter, err)))
      return e_destroy(e), NULL;
    const GgufKv *arch = gguf_kv(e->dg, "general.architecture");
    if (!arch || arch->type != GGUF_STRING || strcmp(arch->v.s, "dflash2"))
      return fail(err, "lfm3: a drafter not a DFlash2 (general.architecture dflash2)"), e_destroy(e), NULL;
    const Lfm3Geometry *geo = &m->w->geo;
    const Q4xDflashTarget t = {geo->n_embd, geo->n_vocab, geo->n_layers};
    if (!(e->df = q4x_dflash_create_for(e->dg, &t, err)))
      return e_destroy(e), NULL;
  }
  const uint32_t steps = o->max_tokens ? o->max_tokens : LFM3_STEP_TOKENS;
  const Lfm3Options ho = {steps < o->max_ctx ? steps : o->max_ctx, o->max_ctx, LFM3_LOGIT_ROWS, LFM3_HEAD_Q8_0, m->lsa, 0, NULL, 0, 0, 0, e->df};
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
static bool batch(Eng *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, bool read, char err[OSMOS_ERR])
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
    if (!(read ? lfm3_step_read : lfm3_step_runs)(e->f, run, len, runs, tokens + at, err))
      return false;
    at += k;
  }
  return true;
}

static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return batch(p, seqs, tokens, n, 0, err);
}

static bool e_read(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return batch(p, seqs, tokens, n, true, err);
}

static const float *e_logits(const void *p, uint32_t row) { return lfm3_row_logits(((const Eng *)p)->f, row); }

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  return lfm3_greedy(((Eng *)p)->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p)
{
  const Eng *e = p;
  return e->df ? (uint32_t)e->df->geo.block - 1 : 0;
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
  Eng *e = p;
  return !e->df || lfm3_dflash_seed(e->f, s, from, to, err); /* no drafter: nothing to seed */
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  Eng *e = p;
  return e->df ? lfm3_dflash_draft(e->f, s, y, e_block(e), conf_min, cum_min, drafts, conf, err) : 0;
}

const OsmosFamily osmos_lfm3 = {
    "lfm3", m_open,     m_close,       e_create,  e_destroy,   e_identity, e_room,  e_stats,
    e_batch,       e_logits,   e_greedy,      e_block,   s_create,    s_destroy,  s_prompt, s_rollback,
    s_reach,       s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    e_read, /* a prompt's read the matrix cores' math, a decode's the matvecs': two maths (lfm3_step_read) */
    NULL, NULL, NULL, /* no marks: the whole state */
};
