#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/run/fit.h"
#include "engine_impl.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/model/gguf.h"
#include "osmos/q4x/dflash.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"

#define Q4X_STEP_TOKENS 4096 /* a step's most tokens at max_tokens 0: the Qwen packs' own (packs.json) */

/* Qwen3.8-Flash-Next behind the engine: every weight resident (no budget, no picks, nothing streamed); its drafter
   o->drafter's GGUF - DFlash2 (general.architecture dflash2: q4x_dflash_*) or the MTP head (q4x_mtp_*). a batch one
   step for its sequences (q4x_step: every weight read once for all of them; Q4X_MAX_RUNS sequences a step), each
   run's logits at its own rows - every row kept but a told prompt's before its end - 1 (s_prompt: q4x_seq_prompt).
   a rollback steps its states' rings back (q4x_seq_rollback) */

_Static_assert(Q4X_MTP_BLOCK <= OSMOS_MAX_BLOCK, "the head's drafts within the engine's");
_Static_assert(Q4X_DFLASH_BLOCK_MAX - 1 <= OSMOS_MAX_BLOCK, "a DFlash block's drafts within the engine's");
_Static_assert(Q4X_MAX_BACK >= OSMOS_MAX_BLOCK, "a drafted round's tokens within a rollback's reach");

typedef struct Model
{
  Gguf *g, *ov; /* the model's files; an overlay's (an abliteration's), or NULL */
  Q4xWeights *w;
} Model;

typedef struct Eng
{
  OsmosLedger *ledger; /* the engine's (borrowed: osmos_create's, destroyed after this) */
  Q4xForward *f;
  Gguf *mg;      /* the drafter's file, or NULL */
  Q4xMtp *mtp;   /* the MTP head, or NULL */
  Q4xDflash *df; /* DFlash2, or NULL */
} Eng;

static void m_close(void *p)
{
  Model *m = p;
  if (!m)
    return;
  q4x_weights_destroy(m->w);
  gguf_destroy(m->g), gguf_destroy(m->ov);
  free(m);
}

static void *m_open(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR])
{
  if (files->lsa)
    return osmos_fail(err, "this family reads no LSA maps (--lsa: LFM3's)"), NULL;
  if (files->table)
    return osmos_fail(err, "qwen4exp: no Engram table is read for this family"), NULL;
  Model *m = calloc(1, sizeof *m);
  if (!m)
    return osmos_fail(err, "out of memory"), NULL;
  if (!(m->g = gguf_create(files->model, err)) || (files->overlay && !(m->ov = gguf_create(files->overlay, err))) ||
      !(m->w = q4x_weights_create(m->g, m->ov, err)))
    return m_close(m), NULL;
  const Q4xGeometry *geo = &m->w->geo;
  const GgufKv *ctx = gguf_kv(m->g, "qwen4exp.context_length");
  /* a decode round's most tokens: below every expert's read (a step past it reads them all, as a prompt's) */
  *facts = (OsmosFacts){"qwen4exp", (uint32_t)geo->n_vocab,
                        ctx && ctx->type == GGUF_UINT32 ? (uint32_t)ctx->v.u : 0,
                        (uint32_t)((geo->n_experts - 1) / geo->n_used), 0, true};
  facts->seed_window = facts->native_ctx; /* the MTP head's cache: every position seeded, as llama.cpp's process() */
  if (!facts->native_ctx)
    return osmos_fail(err, "qwen4exp.context_length: missing"), m_close(m), NULL;
  return m;
}

static void e_destroy(void *p)
{
  Eng *e = p;
  if (!e)
    return;
  q4x_forward_destroy(e->f);
  q4x_mtp_destroy(e->mtp);
  q4x_dflash_destroy(e->df);
  gguf_destroy(e->mg);
  free(e);
}

static void *e_create(const void *pm, const OsmosOptions *o, OsmosLedger *ledger, char err[OSMOS_ERR])
{
  const Model *m = pm;
  Eng *e = calloc(1, sizeof *e);
  if (!e)
    return osmos_fail(err, "out of memory"), NULL;
  e->ledger = ledger;
  if (o->drafter && !(e->mg = gguf_create(o->drafter, err)))
    return e_destroy(e), NULL;
  const bool dflash = e->mg && gguf_arch(e->mg) && !strcmp(gguf_arch(e->mg), "dflash2"); /* (else its MTP head) */
  if (e->mg && (dflash ? !(e->df = q4x_dflash_create(e->mg, m->w, err)) : !(e->mtp = q4x_mtp_create(e->mg, m->w, err))))
    return e_destroy(e), NULL;
  /* the experts sized to the budget (budget auto's builds: 0 the first, the smallest - a byte, not Q4xOptions' 0, every
     expert resident) */
  const Q4xOptions qo = {o->max_ctx,  osmos_step_tokens(o, Q4X_STEP_TOKENS),
                         o->stream_ngram, NULL, 0, e->df, o->budget, ledger,
                         o->npu, o->npu_share > 0 ? o->npu_share : OSMOS_NPU_SHARE_QWEN, o->no_int8};
  if (!(e->f = q4x_forward_create_with(m->w, e->mtp, &qo, err)))
    return e_destroy(e), NULL;
  return e;
}

static uint64_t e_identity(const void *p) { return q4x_identity(((const Eng *)p)->f); }

static void e_stats(const void *p, OsmosStats *s) { s->budget = q4x_forward_bytes(((const Eng *)p)->f); }

/* the batch's sequences one step (each one's consecutive tokens a run, at their rows), Q4X_MAX_RUNS at a time; a
   sequence met again after another's refused */
static bool e_step(void *p, void *const *run, const uint32_t *len, uint32_t runs, const int32_t *tokens, bool read,
                   char err[OSMOS_ERR])
{
  _Static_assert(Q4X_MAX_RUNS == OSMOS_FAMILY_RUNS, "a step's runs the engine's");
  q4x_forward_reading(((Eng *)p)->f, read); /* (apart's: a prompt's read never int8, at any size) */
  const bool ok = q4x_step(((Eng *)p)->f, (Q4xSeq *const *)run, len, runs, tokens, 0, err);
  q4x_forward_reading(((Eng *)p)->f, false);
  return ok;
}

static void e_kept(const void *p, KeptView *k) { q4x_kept_view(((const Eng *)p)->f, k); }

static bool e_apart(const void *p) { return q4x_forward_int8(((const Eng *)p)->f); }

static uint32_t e_block(const void *p)
{
  const Eng *e = p;
  return e->df ? (uint32_t)e->df->geo.block - 1 : e->mtp ? Q4X_MTP_BLOCK : 0;
}

static void *s_create(void *p, char err[OSMOS_ERR]) { return q4x_seq_create(((Eng *)p)->f, err); }
static void s_destroy(void *s) { q4x_seq_destroy(s); }
static bool s_reset(void *s, char err[OSMOS_ERR]) { return q4x_seq_reset_to(s, err); }
static bool s_prompt(void *s, uint32_t end) { return q4x_seq_prompt(s, end); }
static bool s_rollback(void *s, uint32_t n) { return q4x_seq_rollback(s, n); }
static uint32_t s_reach(const void *s) { return q4x_seq_reach(s); }
static int32_t s_position(const void *s) { return q4x_seq_position(s); }
static uint64_t s_state_bytes(const void *s) { return q4x_seq_state_bytes(s); }

static bool s_save_to(const void *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  return q4x_seq_save_to(s, sink, ctx, err);
}

static bool s_load_from(void *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return q4x_seq_load_from(s, bytes, src, ctx, err);
}

static bool s_seed(void *p, void *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  Eng *e = p;
  return e->df ? q4x_dflash_seed(e->f, s, from, to, err) : q4x_mtp_seed(e->f, s, from, to, err);
}

static int s_draft(void *p, void *s, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
                   char err[OSMOS_ERR])
{
  Eng *e = p;
  return e->df ? q4x_dflash_draft(e->f, s, y, e_block(e), conf_min, cum_min, drafts, conf, err)
               : q4x_mtp_draft(e->f, s, y, Q4X_MTP_BLOCK, conf_min, cum_min, drafts, conf, err);
}

/* the sequences' drafts together: DFlash2's a block a sequence, one after another; the MTP head's Q4X_MAX_RUNS at a
   time (q4x_mtp_draft_all: a depth one head step for them all) */
static bool s_draft_all(void *p, OsmosSeq *const *seqs, const int32_t *y, uint32_t n, float conf_min, float cum_min,
                        int32_t (*drafts)[OSMOS_MAX_BLOCK], float (*conf)[OSMOS_MAX_BLOCK], int *counts,
                        char err[OSMOS_ERR])
{
  Eng *e = p;
  for (uint32_t i = 0; e->df && i < n; i++)
    if ((counts[i] = s_draft(e, oseq_impl(seqs[i]), y[i], conf_min, cum_min, drafts[i], conf[i], err)) < 0)
      return false;
  if (e->df)
    return true;
  Q4xSeq *s[Q4X_MAX_RUNS];
  for (uint32_t at = 0; at < n; at += Q4X_MAX_RUNS)
  {
    const uint32_t k = n - at < Q4X_MAX_RUNS ? n - at : Q4X_MAX_RUNS;
    for (uint32_t i = 0; i < k; i++)
      s[i] = oseq_impl(seqs[at + i]);
    if (!q4x_mtp_draft_all(((Eng *)p)->f, s, y + at, k, Q4X_MTP_BLOCK, conf_min, cum_min, drafts[at], conf[at],
                           OSMOS_MAX_BLOCK, counts + at, err))
      return false;
  }
  return true;
}

static uint64_t s_mark_bytes(const void *s) { return q4x_seq_mark_bytes(s); }
static bool s_mark_save(const void *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  return q4x_seq_mark_save(s, out, bytes, err);
}
static bool s_mark_load(void *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  return q4x_seq_mark_load(s, in, bytes, err);
}

const OsmosFamily osmos_q4x = {
    "qwen4exp",
    OSMOS_TAKES_BUDGET | OSMOS_TAKES_DRAFTER | OSMOS_TAKES_STREAM_NGRAM | OSMOS_TAKES_NPU, /* its options (engine_impl.h) */
    m_open,     m_close,       e_create, e_destroy, e_identity, e_stats,
    e_step,     e_kept,      e_block,  s_create,  s_destroy, s_reset,  s_prompt,   s_rollback,
    s_reach,    s_position, s_state_bytes, s_save_to, s_load_from, s_seed,     s_draft,    s_draft_all,
    false, /* f16 alone: its small and large steps one math (the same bits on every path) */
    s_mark_bytes, s_mark_save, s_mark_load, /* a mark the states a later token changes (q4x_seq_mark_save) */
    e_apart,  /* the int8 decode on: a decode's experts int8, a prompt's read f16 at any size - apart */
};
