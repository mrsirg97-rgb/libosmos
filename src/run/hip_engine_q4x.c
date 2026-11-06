#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"

/* Qwen3.8-Flash-Next behind the engine: every weight resident (no budget, no picks, nothing streamed); its drafter
   the MTP head (o->drafter: the head's GGUF, q4x_mtp_*). a batch runs a sequence's consecutive tokens at a time (q4x_batch: one sequence's step), each run's logits at
   its own rows of the one table - every row kept. a rollback steps its states' rings back (q4x_seq_rollback) */

_Static_assert(Q4X_MTP_BLOCK <= OSMOS_MAX_BLOCK, "the head's drafts within the engine's");
_Static_assert(Q4X_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");
_Static_assert(OSMOS_ERR >= Q4X_FWD_ERR && OSMOS_ERR >= Q4X_ERR && OSMOS_ERR >= GGUF_ERR,
               "one error buffer for the engine and the family's calls");

typedef struct Model
{
  Gguf *g;
  Q4xWeights *w;
} Model;

typedef struct Eng
{
  Q4xForward *f;
  Gguf *mg;    /* the MTP head's file, or NULL */
  Q4xMtp *mtp; /* its weights, or NULL */
  uint32_t vocab, n; /* a logits row; the last batch's rows (every one kept) */
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
  q4x_weights_destroy(m->w);
  gguf_destroy(m->g);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->table || files->overlay)
    return fail(err, "qwen4exp: no Engram table and no overlay are read for this family"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  if (!(m->g = gguf_create(files->model, err)) || !(m->w = q4x_weights_create(m->g, err)))
    return m_close(m), NULL;
  const Q4xGeometry *geo = &m->w->geo;
  const GgufKv *ctx = gguf_kv(m->g, "qwen4exp.context_length");
  /* a decode round's most tokens: below every expert's read (a step past it reads them all, as a prompt's) */
  *facts = (OsmosFacts){"qwen4exp", (uint32_t)geo->n_vocab,
                        ctx && ctx->type == GGUF_UINT32 ? (uint32_t)ctx->v.u : 0,
                        (uint32_t)((geo->n_experts - 1) / geo->n_used), 0, false};
  facts->seed_window = facts->native_ctx; /* the MTP head's cache: every position seeded, as llama.cpp's process() */
  if (!facts->native_ctx)
    return fail(err, "qwen4exp.context_length: missing"), m_close(m), NULL;
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  q4x_forward_destroy(e->f);
  q4x_mtp_destroy(e->mtp);
  gguf_destroy(e->mg);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->budget || o->picks || o->no_guess)
    return fail(err, "qwen4exp: every weight resident - no budget, picks or no_guess"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  e->vocab = (uint32_t)m->w->geo.n_vocab;
  if (o->drafter && (!(e->mg = gguf_create(o->drafter, err)) || !(e->mtp = q4x_mtp_create(e->mg, m->w, err))))
    return e_destroy(e), NULL;
  if (!(e->f = q4x_forward_create(m->w, e->mtp, o->max_ctx, o->max_tokens ? o->max_tokens : o->max_ctx, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return q4x_identity(((const Eng *)p)->f); }

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

/* each sequence's consecutive tokens one step, at their rows; a sequence met again after another's refused */
static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  Eng *e = p;
  e->n = 0;
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i] && seqs[i - 1] != seqs[i])
        return fail(err, "a sequence's tokens not consecutive in the batch");
  for (uint32_t at = 0, k; at < n; at += k)
  {
    for (k = 1; at + k < n && seqs[at + k] == seqs[at];)
      k++;
    if (!q4x_batch(e->f, oseq_impl(seqs[at]), tokens + at, k, at, err))
      return false;
  }
  e->n = n;
  return true;
}

static const float *e_logits(const void *p, uint32_t row)
{
  const Eng *e = p;
  return row < e->n ? q4x_logits(e->f) + (uint64_t)row * e->vocab : NULL;
}

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  Eng *e = p;
  if (!rows || (uint64_t)row0 + rows > e->n)
    return fail(err, "a row the last step kept no logits of");
  return q4x_greedy(e->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p) { return ((const Eng *)p)->mtp ? Q4X_MTP_BLOCK : 0; }

static void *s_create(void *p, char err[OSMOS_ERR]) { return q4x_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { q4x_seq_destroy(s); }
static bool s_prompt(void *s, uint32_t end) { return (void)s, (void)end, true; }
static bool s_rollback(void *s, uint32_t n) { return q4x_seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return q4x_seq_reach(s); }
static int32_t s_position(const void *s) { return q4x_seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return q4x_seq_state_bytes(s); }

static bool s_save(const void *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  return q4x_seq_save(s, out, bytes, err);
}

static bool s_load(void *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  return q4x_seq_load(s, in, bytes, err);
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  return q4x_mtp_seed(((Eng *)p)->f, s, from, to, err);
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  return q4x_mtp_draft(((Eng *)p)->f, s, y, Q4X_MTP_BLOCK, conf_min, cum_min, drafts, conf, err);
}

const OsmosFamily osmos_q4x = {
    "qwen4exp", m_open,     m_close,       e_create, e_destroy, e_identity, e_room,     e_stats,
    e_batch,    e_logits,   e_greedy,      e_block,  s_create,  s_destroy,  s_prompt,   s_rollback,
    s_reach,    s_position, s_state_bytes, s_save,   s_load,    s_seed,     s_draft,
};
