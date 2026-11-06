#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/drafter.h"
#include "osmos/run/forward.h"

/* DeepSeek-V4.1-Flash behind the engine: each call the forward's and the drafter's own, as osmos-serve made them
   before 0.4.0 - its decode in the lane's order (forward_lane 0, every expert on the GPU) */

_Static_assert(OSMOS_ERR == FORWARD_ERR && OSMOS_ERR >= WEIGHTS_ERR && OSMOS_ERR >= GGUF_ERR,
               "one error buffer for the engine and the family's calls");
_Static_assert(DRAFTER_BLOCK <= OSMOS_MAX_BLOCK, "DSpark's block within the engine's");

typedef struct Model
{
  Gguf *g, *table, *overlay;
  Weights *w;
} Model;

typedef struct Eng
{
  Forward *f;
  Drafter *d;      /* or NULL */
  uint64_t *picks; /* or NULL: held while the forward is */
  Seq **seqs;      /* a batch's sequences, FORWARD_MAX_TOKENS of them */
  uint32_t vocab;   /* a logits row */
  uint32_t n, kept; /* the last batch's rows, and its last rows the forward keeps logits of */
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
  weights_destroy(m->w);
  gguf_destroy(m->g);
  gguf_destroy(m->table);
  gguf_destroy(m->overlay);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  char gerr[GGUF_ERR] = "";
  if (!(m->g = gguf_create(files->model, gerr)) || (files->table && !(m->table = gguf_create(files->table, gerr))) ||
      (files->overlay && !(m->overlay = gguf_create(files->overlay, gerr))))
    return fail(err, gerr), m_close(m), NULL;
  const WeightsOptions wo = {m->g, m->table, m->overlay};
  if (!(m->w = weights_create_with(&wo, err)))
    return m_close(m), NULL;
  const Geometry *geo = &m->w->geo;
  const GgufKv *ctx = gguf_kv(m->g, "deepseek41.context_length");
  *facts = (OsmosFacts){"deepseek41", (uint32_t)geo->n_vocab,
                         ctx && ctx->type == GGUF_UINT32 ? (uint32_t)ctx->v.u : 0,
                         (uint32_t)((geo->n_experts - 1) / geo->n_used), (uint32_t)geo->window, true};
  if (!facts->native_ctx)
    return fail(err, "deepseek41.context_length: missing"), m_close(m), NULL;
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  drafter_destroy(e->d);
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
    return free(p), fail(err, "the picks file could not be read, or is not this model's"), NULL;
  return p;
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  Eng *e = calloc(1, sizeof *e);
  if (!e || !(e->seqs = malloc(FORWARD_MAX_TOKENS * sizeof *e->seqs)))
    return free(e), fail(err, "out of memory"), NULL;
  e->vocab = (uint32_t)m->w->geo.n_vocab;
  if (o->picks && !(e->picks = read_picks(o->picks, &m->w->geo, err)))
    return e_destroy(e), NULL;
  const ForwardOptions fo = {m->w->geo.n_layers, o->max_ctx, o->budget, e->picks, o->max_tokens, o->no_guess};
  if (!(e->f = forward_create_with(m->w, &fo, err)) || !forward_lane(e->f, 0, err) ||
      (o->drafter && !(e->d = drafter_create(e->f, o->drafter, err))))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return forward_identity(((const Eng *)p)->f); }
static bool e_room(const void *p, uint64_t *bytes) { return forward_room(((const Eng *)p)->f, bytes); }

static void e_stats(const void *p, OsmosStats *s)
{
  const Forward *f = ((const Eng *)p)->f;
  s->budget = forward_budget(f), s->swaps = forward_swaps(f), s->cpu_picks = forward_cpu_picks(f);
  forward_guesses(f, &s->streamed, &s->guessed, &s->guess_read);
  forward_waits(f, s->waits, s->wait_ns);
}

static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  Eng *e = p;
  if (n > FORWARD_MAX_TOKENS)
    return fail(err, "a batch of more than the forward's max_tokens");
  for (uint32_t i = 0; i < n; i++)
    e->seqs[i] = oseq_impl(seqs[i]);
  e->n = e->kept = 0;
  if (!forward_batch(e->f, e->seqs, tokens, n, err))
    return false;
  e->n = n, e->kept = n - (n - 1) / MM_MAX_TOKENS * MM_MAX_TOKENS; /* the last sub-batch's (forward_batch) */
  return true;
}

/* the batch's row r among the rows the forward keeps (the last kept of n), or -1 */
static int64_t kept_row(const Eng *e, uint32_t r)
{
  return r < e->n && r >= e->n - e->kept ? (int64_t)r - (int64_t)(e->n - e->kept) : -1;
}

static const float *e_logits(const void *p, uint32_t row)
{
  const Eng *e = p;
  const int64_t k = kept_row(e, row);
  return k < 0 ? NULL : forward_logits(e->f) + (uint64_t)k * e->vocab;
}

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  Eng *e = p;
  const int64_t k = kept_row(e, row0);
  if (!rows || k < 0 || kept_row(e, row0 + rows - 1) < 0)
    return fail(err, "a row the last step kept no logits of");
  return forward_greedy(e->f, (uint32_t)k, rows, ids, err);
}

static uint32_t e_block(const void *p) { return ((const Eng *)p)->d ? DRAFTER_BLOCK : 0; }

static void *s_create(void *p, char err[OSMOS_ERR]) { return seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { seq_destroy(s); }
static bool s_prompt(void *s, uint32_t end) { return seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return seq_reach(s); }
static int32_t s_position(const void *s) { return seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return seq_state_bytes(s); }

static bool s_save(const void *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  return seq_save(s, out, bytes, err);
}

static bool s_load(void *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  return seq_load(s, in, bytes, err);
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
    "deepseek41", m_open,   m_close,   e_create, e_destroy,  e_identity,    e_room,  e_stats,
    e_batch,      e_logits, e_greedy,  e_block,  s_create,   s_destroy,     s_prompt, s_rollback,
    s_reach,      s_position, s_state_bytes, s_save, s_load, s_seed,        s_draft,
};
