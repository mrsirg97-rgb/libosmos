#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/drafter.h"
#include "osmos/run/forward.h"

/* DeepSeek-V4.1-Flash behind the engine: each call the forward's and the drafter's own, as osmos-serve made them
   before 0.4.0 - its decode in the lane's order (ForwardOptions' lane, cpu_threads 0: every expert on the GPU) */

_Static_assert(OSMOS_ERR == OSMOS_ERR && OSMOS_ERR >= OSMOS_ERR && OSMOS_ERR >= OSMOS_ERR,
               "one error buffer for the engine and the family's calls");
_Static_assert(DRAFTER_BLOCK <= OSMOS_MAX_BLOCK, "DSpark's block within the engine's");

typedef struct Model
{
  Gguf *g, *table, *overlay;
  Weights *w;
} Model;

typedef struct Eng
{
  OsmosLedger *ledger; /* the engine's (borrowed: osmos_create's, destroyed after this) */
  Forward *f;
  Drafter *d;      /* or NULL */
  uint64_t *picks; /* or NULL: held while the forward is */
  Seq **seqs;      /* a batch's sequences, FORWARD_MAX_TOKENS of them */
  uint32_t vocab;   /* a logits row */
  uint32_t n, kept; /* the last batch's rows, and its last rows the forward keeps logits of */
  uint32_t rows;    /* the forward's logit_rows: a decode step's most tokens (the facts' round_tokens) */
} Eng;

static void m_close(void *p)
{
  Model *m = p;
  if (!m)
    return;
  weights_destroy(m->w);
  gguf_destroy(m->g);
  gguf_destroy(m->table);
  gguf_destroy(m->overlay);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->lsa)
    return osmos_fail(err, "this family reads no LSA maps (--lsa: LFM3's)"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return osmos_fail(err, "out of memory"), NULL;
  char gerr[OSMOS_ERR] = "";
  if (!(m->g = gguf_create(files->model, gerr)) || (files->table && !(m->table = gguf_create(files->table, gerr))) ||
      (files->overlay && !(m->overlay = gguf_create(files->overlay, gerr))))
    return osmos_fail(err, "%s", gerr), m_close(m), NULL;
  const WeightsOptions wo = {m->g, m->table, m->overlay};
  if (!(m->w = weights_create_with(&wo, err)))
    return m_close(m), NULL;
  const Geometry *geo = &m->w->geo;
  const GgufKv *ctx = gguf_kv(m->g, "deepseek41.context_length");
  *facts = (OsmosFacts){"deepseek41", (uint32_t)geo->n_vocab,
                         ctx && ctx->type == GGUF_UINT32 ? (uint32_t)ctx->v.u : 0,
                         (uint32_t)((geo->n_experts - 1) / geo->n_used), (uint32_t)geo->window, true};
  if (!facts->native_ctx)
    return osmos_fail(err, "deepseek41.context_length: missing"), m_close(m), NULL;
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  forward_destroy(e->f);
  free(e->picks);
  free(e->seqs);
  free(e);
}

/* a picks file: the router's picks [layer][expert] as uint64, n_layers x n_experts of them, nothing after */
static uint64_t *read_picks(const char *path, const Geometry *geo, char err[OSMOS_ERR])
{
  const size_t n = (size_t)geo->n_layers * geo->n_experts;
  FILE *pf = fopen(path, "rb");
  uint64_t *p = malloc(n * sizeof *p);
  const bool ok = pf && p && fread(p, sizeof *p, n, pf) == n && fgetc(pf) == EOF;
  if (pf)
    fclose(pf);
  if (!ok)
    return free(p), osmos_fail(err, "the picks file could not be read, or is not this model's"), NULL;
  return p;
}

static void *e_create(const void *pm, const OsmosOptions *o, OsmosLedger *ledger, char err[OSMOS_ERR])
{
  const Model *m = pm;
  Eng *e = calloc(1, sizeof *e);
  if (!e || !(e->seqs = malloc(FORWARD_MAX_TOKENS * sizeof *e->seqs)))
    return free(e), osmos_fail(err, "out of memory"), NULL;
  e->ledger = ledger;
  e->vocab = (uint32_t)m->w->geo.n_vocab;
  if (o->picks && !(e->picks = read_picks(o->picks, &m->w->geo, err)))
    return e_destroy(e), NULL;
  e->rows = (uint32_t)((m->w->geo.n_experts - 1) / m->w->geo.n_used); /* m_open's round_tokens */
  if (o->kv_q8 || o->profile || o->rope_stages)
    return e_destroy(e), osmos_fail(err, "kv_q8 and profile are Inkling's, rope_stages osmos1.7b's"), NULL;
  /* whole: the lane (every engine's, all its experts on the GPU), the NPU, the drafter - the forward's own */
  const ForwardOptions fo = {m->w->geo.n_layers,
                             o->max_ctx,
                             o->budget,
                             e->picks,
                             osmos_step_tokens(o, FORWARD_MAX_TOKENS),
                             o->no_guess,
                             o->fill_set,
                             o->fill,
                             0,
                             e->rows,
                             ledger,
                             true,
                             0,
                             NULL,
                             o->npu,
                             o->npu_share > 0 ? o->npu_share : OSMOS_NPU_SHARE_DS,
                             o->drafter};
  if (!(e->f = forward_create_with(m->w, &fo, err)))
    return e_destroy(e), NULL;
  e->d = forward_drafter(e->f); /* (borrowed: the forward's) */
  return e;
}

static uint64_t e_identity(const void *p) { return forward_identity(((const Eng *)p)->f); }

static void e_stats(const void *p, OsmosStats *s)
{
  const Forward *f = ((const Eng *)p)->f;
  s->budget = forward_budget(f), s->cpu_picks = forward_cpu_picks(f);
  forward_guesses(f, &s->streamed, &s->guessed, &s->guess_read);
  forward_waits(f, s->waits, s->wait_ns);
}

static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  Eng *e = p;
  if (n > FORWARD_MAX_TOKENS)
    return osmos_fail(err, "a batch of more than the forward's max_tokens");
  for (uint32_t i = 0; i < n; i++)
    e->seqs[i] = oseq_impl(seqs[i]);
  e->n = e->kept = 0;
  if (!forward_batch(e->f, e->seqs, tokens, n, err))
    return false;
  e->n = n, e->kept = forward_logit_rows(e->f); /* the last sub-batch's last rows (ForwardOptions' logit_rows) */
  return true;
}

/* a prompt's read: osmos_batch with the forward reading (forward_reading) - its bits a prompt's, not a decode's, at
   any step size, so a turn continued after a reused prefix reads its rest as a fresh read reads it (0.7.1) */
static bool e_read(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  Eng *e = p;
  forward_reading(e->f, true);
  const bool ok = e_batch(p, seqs, tokens, n, err);
  forward_reading(e->f, false);
  return ok;
}

/* the batch's row r among the rows the forward keeps (the last kept of n), or -1 */
static void e_kept(const void *p, KeptView *k) { forward_kept_view(((const Eng *)p)->f, k); }

static uint32_t e_block(const void *p) { return ((const Eng *)p)->d ? DRAFTER_BLOCK : 0; }

static void *s_create(void *p, char err[OSMOS_ERR]) { return seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { seq_destroy(s); }
static bool s_reset(void *s, char err[OSMOS_ERR]) { return seq_reset(s, err); }
static bool s_prompt(void *s, uint32_t end) { return seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return seq_reach(s); }
static int32_t s_position(const void *s) { return seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return seq_state_bytes(s); }

static bool s_save_to(const void *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  return seq_save_to(s, sink, ctx, err);
}

static bool s_load_from(void *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return seq_load_from(s, bytes, src, ctx, err);
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  return drafter_seed(((Eng *)p)->d, s, from, to, err);
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  return drafter_draft(((Eng *)p)->d, s, y, conf_min, cum_min, drafts, conf, err);
}

const OsmosFamily osmos_ds4 = {
    "deepseek41", m_open,   m_close,   e_create, e_destroy,  e_identity,    e_stats,
    e_batch,      e_kept,  e_block,  s_create,   s_destroy, s_reset,     s_prompt, s_rollback,
    s_reach,      s_position, s_state_bytes, s_save_to, s_load_from, s_seed,        s_draft,  NULL, /* its drafts one sequence at a time */
    e_read, /* a prompt's read: the large steps' math at any size (forward_reading) */
    NULL,   NULL, NULL, /* a mark its whole state */
};
