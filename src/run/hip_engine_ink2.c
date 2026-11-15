/* Inkling-2-Small, the third family (0.8.0): its model opened and every tensor checked (osmos/ink2/weights.h: the
   reference's names or llama.cpp's, unsloth's packs), its engine osmos/ink2/forward.h's - the dense weights resident,
   the experts streamed through the membrane at the budget's room, a sequence's caches and tails, a step a run at a
   time; the drafter (--dspark: the checkpoint's mtp.safetensors as ink2_mtp_gguf.py's GGUF, DSpark with the RNN
   Markov head) its seed and draft, block - 1 drafts a round */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "engine_impl.h"
#include "osmos/ink2/forward.h"
#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"

typedef struct Model
{
  Gguf *g;
  Ink2Weights *w;
} Model;

static bool fail(char err[OSMOS_ERR], const char *why)
{
  if (err)
    snprintf(err, OSMOS_ERR, "inkling2: %s", why);
  return false;
}

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
  if (files->table)
    return fail(err, "no Engram table is read for this family"), NULL;
  if (files->overlay)
    return fail(err, "no overlay is taken yet (a pack's patch is assembled into its shards)"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  char werr[INK2_ERR > GGUF_ERR ? INK2_ERR : GGUF_ERR] = "";
  if (!(m->g = gguf_create(files->model, werr)) || !(m->w = ink2_weights_create(m->g, werr)))
  {
    if (err)
      snprintf(err, OSMOS_ERR, "inkling2: %s", werr);
    return m_close(m), NULL;
  }
  const Ink2Geometry *geo = &m->w->geo;
  /* a decode round's most tokens below every expert's read (a step past it reads them all, as a prompt's) */
  *facts = (OsmosFacts){"inkling2", (uint32_t)geo->n_vocab, geo->native_ctx,
                        (uint32_t)((geo->n_experts - 1) / geo->n_used), geo->native_ctx, true};
  /* (seed_window: a drafter's context every position - its full layer's attention reads them all) */
  return m;
}

_Static_assert(OSMOS_ERR >= INK2_FWD_ERR, "one error buffer for the engine and the family's calls");
_Static_assert(INK2_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");

#define INK2_STEP_TOKENS 512 /* a step's most tokens when the server names none: a prompt's chunk */

typedef struct Eng
{
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
  ink2_forward_destroy(e->f);
  ink2_mtp_destroy(e->mtp);
  gguf_destroy(e->mg);
  free(e->picks);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  const Model *m = pm;
  if (o->npu)
    return fail(err, "no NPU kernels for this family"), NULL;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
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
      return fail(err, "the picks file could not be read, or is not this model's"), e_destroy(e), NULL;
  }
  /* the drafter: its GGUF (tools/convert/ink2_mtp_gguf.py's) - a directory holding one, or the file */
  if (o->drafter)
  {
    char path[4096], ierr[INK2_ERR] = "";
    snprintf(path, sizeof path, "%s/ink2-mtp.gguf", o->drafter);
    FILE *probe = fopen(path, "rb");
    if (probe)
      fclose(probe);
    else
      snprintf(path, sizeof path, "%s", o->drafter);
    if (!(e->mg = gguf_create(path, err)))
      return e_destroy(e), NULL;
    if (!(e->mtp = ink2_mtp_create(e->mg, m->w, ierr)))
      return snprintf(err, OSMOS_ERR, "%s", ierr), e_destroy(e), NULL;
  }
  const uint32_t T = o->max_tokens ? o->max_tokens : INK2_STEP_TOKENS;
  const Ink2Options io = {o->max_ctx, T < o->max_ctx ? T : o->max_ctx, o->budget, e->picks, e->mtp};
  if (!(e->f = ink2_forward_create(m->w, &io, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return ink2_identity(((const Eng *)p)->f); }

static bool e_room(const void *p, uint64_t *bytes)
{
  (void)p;
  size_t free_b = 0, total = 0;
  if (hipMemGetInfo(&free_b, &total) != hipSuccess)
    return false;
  *bytes = free_b;
  return true;
}

static void e_stats(const void *p, OsmosStats *s) { s->budget = ink2_forward_bytes(((const Eng *)p)->f); }

/* the batch's sequences one step (each one's consecutive tokens a run, INK2_MAX_RUNS of them at most); a sequence
   met again after another's refused */
static bool e_batch(void *p, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  Eng *e = p;
  e->n = 0;
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i] && seqs[i - 1] != seqs[i])
        return fail(err, "a sequence's tokens not consecutive in the batch");
  Ink2Seq *run[INK2_MAX_RUNS];
  uint32_t len[INK2_MAX_RUNS], runs = 0;
  for (uint32_t at = 0; at < n; runs++) /* the runs counted before any step: a batch of more is refused whole */
  {
    if (runs == INK2_MAX_RUNS)
      return fail(err, "inkling2: a batch of more than INK2_MAX_RUNS sequences");
    uint32_t m = 1;
    while (at + m < n && seqs[at + m] == seqs[at])
      m++;
    run[runs] = oseq_impl(seqs[at]), len[runs] = m, at += m;
  }
  if (!n || !ink2_step(e->f, run, len, runs, tokens, err))
    return n ? false : fail(err, "an empty batch");
  e->n = n;
  return true;
}

static const float *e_logits(const void *p, uint32_t row)
{
  const Eng *e = p;
  return row < e->n && ink2_kept(e->f, row) ? ink2_logits(e->f) + (uint64_t)row * e->vocab : NULL;
}

static bool e_greedy(void *p, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  Eng *e = p;
  if (!rows || (uint64_t)row0 + rows > e->n)
    return fail(err, "a row the last step kept no logits of");
  return ink2_greedy(e->f, row0, rows, ids, err);
}

static uint32_t e_block(const void *p)
{
  const Eng *e = p;
  return e->mtp ? (uint32_t)e->mtp->block - 1 : 0;
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  return ink2_seq_seed(((Eng *)p)->f, s, from, to, err);
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  const char *cap = getenv("INK2_DRAFT_MAX"); /* (a measurement's: the drafts a round at most) */
  const uint32_t most = cap && atoi(cap) > 0 && (uint32_t)atoi(cap) < e_block(p) ? (uint32_t)atoi(cap) : e_block(p);
  return ink2_seq_draft(((Eng *)p)->f, s, y, most, conf_min, cum_min, drafts, conf, err);
}

static void *s_create(void *p, char err[OSMOS_ERR]) { return ink2_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { ink2_seq_destroy(s); }
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
    "inkling2", m_open,     m_close,       e_create,  e_destroy,   e_identity, e_room, e_stats,
    e_batch,    e_logits,   e_greedy,      e_block,   s_create,    s_destroy,  s_prompt, s_rollback,
    s_reach,    s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft, NULL,
    NULL, /* a prompt's read its batch: one math for every step */
    s_mark_bytes, s_mark_save, s_mark_load, /* a mark the conv tails and windows (ink2_seq_mark_save) */
};
