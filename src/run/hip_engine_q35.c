#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "engine_impl.h"
#include "osmos/model/gguf.h"
#include "osmos/q35/forward.h"

/* Qwen3.5 / 3.8's dense models (qwen35: Qwen3.8-27B, vibes27b) behind the engine: every weight resident (no budget,
   picks or guess), q35's sequence (DESIGN.md, "0.9.0: q35 behind the engine"). a batch is a decode's math, a prompt's
   read the matmul's (two maths, as DeepSeek's: the engine's read), each a token's bits however its step is split or
   shared; the logits of the kept rows alone. no drafter yet: block 0 */
_Static_assert(Q35_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");
_Static_assert(OSMOS_ERR >= Q35_ERR && OSMOS_ERR >= GGUF_ERR, "one error buffer for the engine and the family's calls");

/* a step's most tokens when the server names none (a prompt's chunk), and a round's most kept rows: each lane's
   token and its drafts (OSMOS_MAX_LANES x 9 rows of 248,320 floats: 0.57 GB) */
#define Q35_STEP_TOKENS 4096
#define Q35_LOGIT_ROWS (OSMOS_MAX_LANES * (OSMOS_MAX_BLOCK + 1))

typedef struct Model
{
  Gguf *g;
  Q35Weights *w;
} Model;

typedef struct Eng
{
  Q35Forward *f;
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
  q35_weights_destroy(m->w);
  gguf_destroy(m->g);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->lsa)
    return snprintf(err, OSMOS_ERR, "%s", "this family reads no LSA maps (--lsa: LFM3's)"), NULL;
  if (files->table || files->overlay)
    return fail(err, "qwen35: no Engram table or overlay is read for this family"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  if (!(m->g = gguf_create(files->model, err)) || !(m->w = q35_weights_create(m->g, err)))
    return m_close(m), NULL;
  const Q35Geometry *geo = &m->w->geo;
  *facts = (OsmosFacts){"qwen35", (uint32_t)geo->n_vocab, geo->native_ctx, Q35_LOGIT_ROWS, 0, false};
  if (!facts->native_ctx)
    return fail(err, "qwen35.context_length: missing"), m_close(m), NULL;
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  q35_forward_destroy(e->f);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->budget || o->picks || o->no_guess || o->npu || o->kv_q8)
    return fail(err, "qwen35: every weight resident - no budget, picks, no_guess, NPU or kv_q8"), NULL;
  if (o->drafter)
    return fail(err, "qwen35: no drafter yet (its MTP head and DFlash: 0.9.0's next step)"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  const uint32_t steps = osmos_step_tokens(o, Q35_STEP_TOKENS);
  const Q35Options qo = {steps, o->max_ctx, Q35_LOGIT_ROWS};
  if (!(e->f = q35_forward_create_with(m->w, &qo, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return q35_identity(((const Eng *)p)->f); }

static void e_stats(const void *p, OsmosStats *s) { (void)p, (void)s; }

/* the batch's sequences one step (each one's consecutive tokens a run), Q35_MAX_RUNS at a time, in a decode's math
   or (read) a prompt's; a sequence met again after another's refused */
static bool batch(Eng *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, bool read, char err[OSMOS_ERR])
{
  _Static_assert(Q35_MAX_RUNS == OSMOS_FAMILY_RUNS, "a step's runs the engine's");
  if (!osmos_runs_consecutive(seqs, n, err))
    return false;
  Q35Seq *run[Q35_MAX_RUNS];
  uint32_t first[Q35_MAX_RUNS], len[Q35_MAX_RUNS], runs;
  for (uint32_t at = 0; at < n;)
  {
    const uint32_t k = osmos_runs(seqs, n, at, first, len, &runs);
    if (at + k < n)
      return fail(err, "qwen35: a batch past Q35_MAX_RUNS sequences"); /* its logits one step's: never split */
    for (uint32_t r = 0; r < runs; r++)
      run[r] = oseq_impl(seqs[first[r]]);
    if (!(read ? q35_step_read : q35_step_runs)(e->f, run, len, runs, tokens + at, err))
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

static const float *e_logits(const void *p, uint32_t row) { return q35_row_logits(((const Eng *)p)->f, row); }

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  return q35_greedy(((Eng *)p)->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p)
{
  (void)p;
  return 0;
}

static void *s_create(void *p, char err[OSMOS_ERR]) { return q35_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { q35_seq_destroy(s); }
static bool s_prompt(void *s, uint32_t end) { return q35_seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return q35_seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return q35_seq_reach(s); }
static int32_t s_position(const void *s) { return q35_seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return q35_seq_state_bytes(s); }

static bool s_save_to(const void *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  return q35_seq_save_to(s, sink, ctx, err);
}

static bool s_load_from(void *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return q35_seq_load_from(s, bytes, src, ctx, err);
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

static uint64_t s_mark_bytes(const void *s) { return q35_seq_mark_bytes(s); }
static bool s_mark_save(const void *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  return q35_seq_mark_save(s, out, bytes, err);
}
static bool s_mark_load(void *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  return q35_seq_mark_load(s, in, bytes, err);
}

const OsmosFamily osmos_q35 = {
    "qwen35",   m_open,     m_close,       e_create,  e_destroy,   e_identity, e_stats,
    e_batch,    e_logits,   e_greedy,      e_block,   s_create,    s_destroy,  s_prompt, s_rollback,
    s_reach,    s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    e_read, /* a prompt's read the matmul's math, a decode's mv_fp4's: two maths (q35_step_read) */
    s_mark_bytes, s_mark_save, s_mark_load,
};
