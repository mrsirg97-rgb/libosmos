#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "engine_impl.h"
#include "osmos/hy4/forward.h"
#include "osmos/model/gguf.h"

/* Tencent's Hy4-flash (hy_v4_flash) behind the engine: every weight resident but the Engram table (its rows read from
   the model's file a step at a time), hy4's sequence (DESIGN.md, "0.10.0 step 2"). a batch is a decode's math, a
   prompt's read the matrix cores' (two maths, as DeepSeek's and q35's), each a token's bits however its step is split
   or shared; the logits of the kept rows alone. the drafter, when named, DFly2 (the release's
   model-00148-of-00148.safetensors): HY4_DFLY_DRAFTS a round, verified in a decode's step */
_Static_assert(HY4_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");
_Static_assert(OSMOS_ERR >= HY4_ERR && OSMOS_ERR >= GGUF_ERR, "one error buffer for the engine and the family's calls");

/* a step's most tokens when the server names none (a prompt's chunk), and a round's most kept rows */
#define HY4_STEP_TOKENS 2048
#define HY4_LOGIT_ROWS (OSMOS_MAX_LANES * (OSMOS_MAX_BLOCK + 1))
#define HY4_MAX_FILES 16

typedef struct Model
{
  Gguf *g;
  Hy4Weights *w;
  int fds[HY4_MAX_FILES]; /* the model's files, open: the Engram rows' reads */
} Model;

typedef struct Eng
{
  Hy4Forward *f;
  uint32_t drafts; /* a round's drafts (DFly2's), 0 without a drafter */
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
  for (int i = 0; i < HY4_MAX_FILES; i++)
    if (m->fds[i] >= 0)
      close(m->fds[i]);
  hy4_weights_destroy(m->w);
  gguf_destroy(m->g);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->table || files->overlay)
    return fail(err, "hy_v4_flash: its Engram table is the model's own; no table file or overlay is read"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  for (int i = 0; i < HY4_MAX_FILES; i++)
    m->fds[i] = -1;
  if (!(m->g = gguf_create(files->model, err)) || !(m->w = hy4_weights_create(m->g, err)))
    return m_close(m), NULL;
  if (m->g->n_files > HY4_MAX_FILES)
    return fail(err, "hy_v4_flash: more files than this build opens"), m_close(m), NULL;
  for (int i = 0; i < m->g->n_files; i++)
    if ((m->fds[i] = open(m->g->paths[i], O_RDONLY)) < 0)
      return fail(err, "hy_v4_flash: a model file not opened (its Engram rows' reads)"), m_close(m), NULL;
  const Hy4Geometry *geo = &m->w->geo;
  *facts = (OsmosFacts){"hy_v4_flash", (uint32_t)geo->n_vocab, geo->native_ctx, HY4_LOGIT_ROWS, 0, false};
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  hy4_forward_destroy(e->f);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->budget || o->picks || o->no_guess || o->npu)
    return fail(err, "hy_v4_flash: every weight resident - no budget, picks, no_guess or NPU"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  const uint32_t steps = o->max_tokens ? o->max_tokens : HY4_STEP_TOKENS;
  const Hy4Options ho = {steps < o->max_ctx ? steps : o->max_ctx, o->max_ctx, HY4_LOGIT_ROWS, false};
  if (!(e->f = hy4_forward_create(m->w, m->fds, &ho, err)))
    return e_destroy(e), NULL;
  if (o->drafter && !hy4_dfly_load(e->f, o->drafter, true, err)) /* (Q8_0: half the draft's reads) */
    return e_destroy(e), NULL;
  e->drafts = o->drafter ? HY4_DFLY_DRAFTS : 0;
  return e;
}

static uint64_t e_identity(const void *p) { return hy4_identity(((const Eng *)p)->f); }

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

/* the batch's sequences one step (each one's consecutive tokens a run), HY4_MAX_RUNS at a time */
static bool batch(Eng *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, bool read, char err[OSMOS_ERR])
{
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i] && seqs[i - 1] != seqs[i])
        return fail(err, "a sequence's tokens not consecutive in the batch");
  Hy4Seq *run[HY4_MAX_RUNS];
  uint32_t len[HY4_MAX_RUNS];
  for (uint32_t at = 0; at < n;)
  {
    uint32_t runs = 0, k = 0;
    for (; at + k < n && runs < HY4_MAX_RUNS; runs++)
    {
      uint32_t m = 1;
      while (at + k + m < n && seqs[at + k + m] == seqs[at + k])
        m++;
      run[runs] = oseq_impl(seqs[at + k]), len[runs] = m, k += m;
    }
    if (at + k < n)
      return fail(err, "hy_v4_flash: a batch past HY4_MAX_RUNS sequences"); /* its logits one step's: never split */
    if (!(read ? hy4_step_read : hy4_step_runs)(e->f, run, len, runs, tokens + at, err))
      return false;
    at += k;
  }
  return true;
}

static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return batch(p, seqs, tokens, n, false, err);
}

static bool e_read(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return batch(p, seqs, tokens, n, true, err);
}

static const float *e_logits(const void *p, uint32_t row) { return hy4_row_logits(((const Eng *)p)->f, row); }

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  return hy4_greedy(((Eng *)p)->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p) { return ((const Eng *)p)->drafts; }

static void *s_create(void *p, char err[OSMOS_ERR]) { return hy4_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { hy4_seq_destroy(s); }
static bool s_prompt(void *s, uint32_t end) { return hy4_seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return hy4_seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return hy4_seq_reach(s); }
static int32_t s_position(const void *s) { return hy4_seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return hy4_seq_state_bytes(s); }

static bool s_save_to(const void *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  return hy4_seq_save_to(s, sink, ctx, err);
}

static bool s_load_from(void *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return hy4_seq_load_from(s, bytes, src, ctx, err);
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  (void)p, (void)s, (void)from, (void)to, (void)err;
  return true; /* DFly2 reads the target's own state (every step keeps its snapshot): nothing to seed */
}

/* a round's drafts while their probabilities' product is at least this: a verified token reads its own experts (~14
   ms of a step's 63), so an unlikely one costs more than it brings - 0.3 between code's best (0.4: 24.4 tok/s, plain
   15.8) and prose's (0.25: 22.5), 2026-11-19, Q4_K_M at 1K */
#define HY4_DRAFT_CUM_MIN 0.3f

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  const Eng *e = p;
  if (cum_min < HY4_DRAFT_CUM_MIN)
    cum_min = HY4_DRAFT_CUM_MIN;
  return e->drafts ? hy4_seq_draft(e->f, s, y, e->drafts, conf_min, cum_min, drafts, conf, err) : 0;
}

const OsmosFamily osmos_hy4 = {
    "hy_v4_flash", m_open,     m_close,       e_create,  e_destroy,   e_identity, e_room,  e_stats,
    e_batch,       e_logits,   e_greedy,      e_block,   s_create,    s_destroy,  s_prompt, s_rollback,
    s_reach,       s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    e_read, /* a prompt's read the matrix cores' math, a decode's the matvecs': two maths (hy4_step_read) */
    NULL, NULL, NULL, /* no marks: the whole state */
};
