#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/gguf.h"
#include "osmos/text/answer.h"
#include "osmos/run/fit.h"
#include "osmos/mem/ledger_hip.h"
#include "osmos/err_hip.h"

/* the dispatch: a model's family by its file's general.architecture, then every call that family's */

static const OsmosFamily *const families[] = {&osmos_ds4, &osmos_q4x, &osmos_ink2, &osmos_q35, &osmos_lfm3,
                                                     &osmos_hy4};


struct OsmosModel
{
  const OsmosFamily *fam;
  void *impl;
  OsmosFacts facts;
};

struct Osmos
{
  uint32_t last_n; /* the last batch's tokens (0: none, or it failed): the kept view's rows at most */
  const OsmosFamily *fam;
  void *impl;
  uint32_t vocab; /* a logits row's floats (osmos_greedy_except) */
  float *row;     /* [vocab], a row copied out */
  float *drow;    /* [vocab] device: a row's copy, its excepted tokens -inf (osmos_sample_except) */
  int32_t *dpick; /* device: a drawn token */
  OsmosLedger *ledger; /* the engine's memory, one count (osmos/mem/ledger.h): its own, every family's build borrowing it */
  uint64_t ram_kept; /* the system's RAM a build leaves free: OsmosOptions' fill, else FIT_RAM_FREE */
};

struct OsmosSeq
{
  Osmos *e;
  void *impl;
};

void *oseq_impl(const OsmosSeq *s) { return s ? s->impl : NULL; }

/* a family's failure named once, by the family (osmos_context: its own cause kept, "<arch>: " before it) */
static bool named(const OsmosFamily *fam, char err[OSMOS_ERR], bool ok)
{
  if (!ok)
    osmos_context(err, fam->arch);
  return ok;
}

static int named_count(const OsmosFamily *fam, char err[OSMOS_ERR], int got)
{
  if (got < 0)
    osmos_context(err, fam->arch);
  return got;
}

OsmosModel *osmos_open(const OsmosFiles *files, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!files || !files->model)
    return osmos_fail(err, "no model file named"), NULL;
  char gerr[OSMOS_ERR] = "";
  Gguf *h = gguf_header(files->model, gerr);
  if (!h)
    return osmos_fail(err, "%s", gerr), NULL;
  char arch[64] = "";
  snprintf(arch, sizeof arch, "%s", gguf_arch(h) ? gguf_arch(h) : "");
  gguf_destroy(h);
  const OsmosFamily *fam = NULL;
  for (size_t i = 0; i < sizeof families / sizeof *families; i++)
    if (!strcmp(arch, families[i]->arch))
      fam = families[i];
  if (!fam)
  { /* the families named from the table itself: a list kept by hand went stale (three of six) */
    char known[OSMOS_ERR / 2] = "";
    for (size_t i = 0, at = 0; i < sizeof families / sizeof *families && at < sizeof known; i++)
      at += (size_t)snprintf(known + at, sizeof known - at, "%s%s", i ? ", " : "", families[i]->arch);
    return osmos_fail(err, "the architecture %s: no family here runs it (%s)", arch[0] ? arch : "(none)", known), NULL;
  }
  OsmosModel *m = calloc(1, sizeof *m);
  if (!m)
    return osmos_fail(err, "out of memory"), NULL;
  m->fam = fam;
  if (!named(fam, err, (m->impl = fam->open(files, &m->facts, err)) != NULL))
    return free(m), NULL;
  return m;
}

void osmos_close(OsmosModel *m)
{
  if (!m)
    return;
  m->fam->close(m->impl);
  free(m);
}

OsmosFacts osmos_facts(const OsmosModel *m)
{
  const OsmosFacts none = {0};
  return m ? m->facts : none;
}

Osmos *osmos_create(const OsmosModel *m, const OsmosOptions *o, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!m || !o)
    return osmos_fail(err, "no model or options"), NULL;
  Osmos *e = calloc(1, sizeof *e);
  if (!e)
    return osmos_fail(err, "out of memory"), NULL;
  e->fam = m->fam, e->vocab = (uint32_t)m->facts.vocab;
  if (!(e->row = malloc((size_t)e->vocab * sizeof *e->row)))
    return free(e), osmos_fail(err, "out of memory"), NULL;
  e->ram_kept = o->fill_set ? o->fill : FIT_RAM_FREE;
  /* the ledger first (the family's build borrows it), then the engine's own buffers - made whole, here: sampling's row
     and pick were made at their first use */
  if (!(e->ledger = ledger_create(e->ram_kept, err)) ||
      !ledger_check(e->ledger, ledger_malloc(e->ledger, (void **)&e->drow, (size_t)e->vocab * sizeof *e->drow), err,
                    "the engine's sampling row") ||
      !ledger_check(e->ledger, ledger_malloc(e->ledger, (void **)&e->dpick, sizeof *e->dpick), err,
                    "the engine's sampling pick") ||
      !named(m->fam, err, (e->impl = m->fam->create(m->impl, o, e->ledger, err)) != NULL))
    return osmos_destroy(e), NULL;
  return e;
}

void osmos_destroy(Osmos *e)
{
  if (!e)
    return;
  if (e->impl)
    e->fam->destroy(e->impl); /* (the family's build, its borrowed ledger still here) */
  ledger_free(e->ledger, e->drow), ledger_free(e->ledger, e->dpick);
  ledger_destroy(e->ledger); /* last: everything it counted returned */
  free(e->row), free(e);
}

uint64_t osmos_identity(const Osmos *e) { return e ? e->fam->identity(e->impl) : 0; }

/* the room a build left, every family's one rule: the GPU's free memory, or the system's RAM past what it keeps free,
   the less (fit_room) - the experts' pinned host slots in RAM, not the GPU's count (osmos1.7b and the 27B reported the
   GPU's free alone) */
bool osmos_room(const Osmos *e, uint64_t *bytes)
{
  uint64_t free_b = 0;
  if (!e || !bytes || !osmos_gpu_room(&free_b))
    return false;
  *bytes = fit_room(free_b, fit_ram_available(), e->ram_kept);
  return true;
}

void osmos_stats(const Osmos *e, OsmosStats *s)
{
  if (!s)
    return;
  memset(s, 0, sizeof *s);
  if (e)
    e->fam->stats(e->impl, s);
  if (e)
    s->held = ledger_taken(e->ledger);
}

/* a batch's step, every family's one path (0.16.0: each family built its runs - Qwen split past 64 sequences, the
   others refused - and osmos_read checked no token): the checks, the runs, the family's step; the rows the kept
   view reads its batch's */
static bool step(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, bool read, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!e || !seqs || !tokens || !n)
    return osmos_fail(err, "a batch of no tokens");
  e->last_n = 0; /* (no row kept until a step holds) */
  for (uint32_t i = 0; i < n; i++)
    if (!seqs[i] || seqs[i]->e != e)
      return osmos_fail(err, "a batch's sequence not this engine's");
  for (uint32_t i = 0; i < n; i++) /* every family's, before any reads one (Inkling, the 27B and Qwen embedded any) */
    if (tokens[i] < 0 || (uint32_t)tokens[i] >= e->vocab)
      return osmos_fail(err, "a token outside the vocabulary");
  uint32_t first[OSMOS_MAX_RUNS], len[OSMOS_MAX_RUNS], runs = 0;
  if (!osmos_runs_consecutive(seqs, n, err))
    return false;
  if (osmos_runs(seqs, n, 0, first, len, &runs) < n) /* counted before any step: refused whole */
    return osmos_fail(err, "a batch of more than %d sequences (a step's most: OSMOS_MAX_RUNS)", OSMOS_MAX_RUNS);
  void *run[OSMOS_MAX_RUNS];
  for (uint32_t r = 0; r < runs; r++)
    run[r] = seqs[first[r]]->impl;
  if (!named(e->fam, err, e->fam->step(e->impl, run, len, runs, tokens, read && e->fam->reads_apart, err)))
    return false;
  e->last_n = n;
  return true;
}

bool osmos_read(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return step(e, seqs, tokens, n, true, err);
}

bool osmos_reads_apart(const Osmos *e) { return e && e->fam->reads_apart; }

bool osmos_batch(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return step(e, seqs, tokens, n, false, err);
}

/* the last step's kept rows, the family's view (osmos/gpu/kept.h) */
static KeptView kept_of(const Osmos *e)
{
  KeptView k = {0};
  e->fam->kept(e->impl, &k);
  k.rows = e->last_n < k.rows ? e->last_n : k.rows; /* (the last batch's rows: a row past it another step's) */
  return k;
}

static const float *row_of(const Osmos *e, uint32_t row)
{
  const KeptView k = kept_of(e);
  return kept_row(&k, row);
}

const float *osmos_logits(const Osmos *e, uint32_t row) { return e ? row_of(e, row) : NULL; }

bool osmos_greedy(Osmos *e, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!e || !ids)
    return osmos_fail(err, "no engine");
  const KeptView k = kept_of(e);
  const char *why = kept_greedy(&k, row0, rows, ids);
  return !why || osmos_fail(err, "%s", why);
}

bool osmos_greedy_except(Osmos *e, uint32_t row, const int32_t *except, uint32_t n, int32_t *id, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!e || !id)
    return osmos_fail(err, "no engine");
  const float *dev = row_of(e, row);
  if (!dev)
    return osmos_fail(err, "the row's logits not kept");
  if (hipDeviceSynchronize() != hipSuccess ||
      hipMemcpy(e->row, dev, (size_t)e->vocab * sizeof *e->row, hipMemcpyDeviceToHost) != hipSuccess)
    return osmos_fail(err, "a GPU call failed (a row's logits)");
  *id = answer_best(e->row, e->vocab, except, n);
  return *id >= 0 || osmos_fail(err, "every token of the row excepted or NaN");
}

bool osmos_sampled(const OsmosSampling *s) { return s && s->temperature > 0; }

static bool draw(Osmos *e, const float *dev, const OsmosSampling *s, uint64_t position, int32_t *id,
                 char err[OSMOS_ERR])
{
  if (!(s->temperature < 1e30f) || !(s->top_p > 0 && s->top_p <= 1))
    return osmos_fail(err, "sampling: temperature finite and above 0, top_p in (0, 1]");
  if (!ops_sample(dev, e->vocab, s->temperature, s->top_p, s->top_k, ops_draw_uniform(s->seed, position), e->dpick,
                  NULL) ||
      hipMemcpy(id, e->dpick, sizeof *id, hipMemcpyDeviceToHost) != hipSuccess)
    return osmos_fail(err, "a GPU call failed (a sampled token)");
  return *id >= 0 || osmos_fail(err, "a sampled row with a NaN or +inf (or every token excepted)");
}

bool osmos_sample(Osmos *e, uint32_t row, const OsmosSampling *s, uint64_t position, int32_t *id, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!osmos_sampled(s))
    return osmos_greedy(e, row, 1, id, err);
  if (!e || !id)
    return osmos_fail(err, "no engine");
  const float *dev = row_of(e, row);
  if (!dev)
    return osmos_fail(err, "the row's logits not kept");
  if (hipDeviceSynchronize() != hipSuccess)
    return osmos_fail(err, "a GPU call failed (a row's logits)");
  return draw(e, dev, s, position, id, err);
}

bool osmos_sample_except(Osmos *e, uint32_t row, const OsmosSampling *s, uint64_t position, const int32_t *except,
                         uint32_t n, int32_t *id, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!osmos_sampled(s))
    return osmos_greedy_except(e, row, except, n, id, err);
  if (!e || !id)
    return osmos_fail(err, "no engine");
  const float *dev = row_of(e, row);
  if (!dev)
    return osmos_fail(err, "the row's logits not kept");
  const size_t bytes = (size_t)e->vocab * sizeof *e->drow;
  const float ninf = -__builtin_inff();
  if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(e->drow, dev, bytes, hipMemcpyDeviceToDevice) != hipSuccess)
    return osmos_fail(err, "a GPU call failed (a row's copy)");
  for (uint32_t i = 0; i < n; i++)
    if (except[i] >= 0 && (uint32_t)except[i] < e->vocab &&
        hipMemcpy(e->drow + except[i], &ninf, sizeof ninf, hipMemcpyHostToDevice) != hipSuccess)
      return osmos_fail(err, "a GPU call failed (a row's copy)");
  return draw(e, e->drow, s, position, id, err);
}

uint32_t osmos_block(const Osmos *e) { return e ? e->fam->block(e->impl) : 0; }

OsmosSeq *oseq_create(Osmos *e, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (e)
    ledger_read(e->ledger); /* (a build's start: what others took since counted) */
  if (!e)
    return osmos_fail(err, "no engine"), NULL;
  OsmosSeq *s = calloc(1, sizeof *s);
  if (!s)
    return osmos_fail(err, "out of memory"), NULL;
  s->e = e;
  if (!named(e->fam, err, (s->impl = e->fam->seq_create(e->impl, err)) != NULL))
    return free(s), NULL;
  return s;
}

bool oseq_reset(OsmosSeq *s, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  return s ? named(s->e->fam, err, s->e->fam->seq_reset(s->impl, err)) : osmos_fail(err, "no sequence");
}

void oseq_destroy(OsmosSeq *s)
{
  if (!s)
    return;
  s->e->fam->seq_destroy(s->impl);
  free(s);
}

bool oseq_prompt(OsmosSeq *s, uint32_t end) { return s && s->e->fam->prompt(s->impl, end); }
bool oseq_rollback(OsmosSeq *s, uint32_t n) { return s && s->e->fam->rollback(s->impl, n); }
uint32_t oseq_reach(const OsmosSeq *s) { return s ? s->e->fam->reach(s->impl) : 0; }
int32_t oseq_position(const OsmosSeq *s) { return s ? s->e->fam->position(s->impl) : -1; }
uint64_t oseq_state_bytes(const OsmosSeq *s) { return s ? s->e->fam->state_bytes(s->impl) : 0; }

bool oseq_save_to(const OsmosSeq *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  return s && sink ? named(s->e->fam, err, s->e->fam->save_to(s->impl, sink, ctx, err)) : osmos_fail(err, "no sequence or sink");
}

bool oseq_load_from(OsmosSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  return s && src ? named(s->e->fam, err, s->e->fam->load_from(s->impl, bytes, src, ctx, err)) : osmos_fail(err, "no sequence or source");
}

/* one buffer as the pieces' sink and source: its bytes in order, never past its end */
typedef struct
{
  uint8_t *p;
  uint64_t at, n;
} Mem;

static bool mem_sink(void *ctx, const void *bytes, uint64_t n)
{
  Mem *m = ctx;
  if (n > m->n - m->at)
    return false;
  memcpy(m->p + m->at, bytes, n), m->at += n;
  return true;
}

static bool mem_source(void *ctx, void *bytes, uint64_t n)
{
  Mem *m = ctx;
  if (n > m->n - m->at)
    return false;
  memcpy(bytes, m->p + m->at, n), m->at += n;
  return true;
}

bool oseq_save(const OsmosSeq *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !out)
    return osmos_fail(err, "no sequence or buffer");
  if (bytes != oseq_state_bytes(s))
    return osmos_fail(err, "a buffer not the state's size (oseq_state_bytes)");
  Mem m = {out, 0, bytes};
  return oseq_save_to(s, mem_sink, &m, err) && (m.at == bytes || osmos_fail(err, "a state shorter than its size"));
}

bool oseq_load(OsmosSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  Mem m = {(uint8_t *)in, 0, bytes};
  return s && in ? oseq_load_from(s, bytes, mem_source, &m, err) : osmos_fail(err, "no sequence or state");
}

uint64_t oseq_mark_bytes(const OsmosSeq *s)
{
  return !s ? 0 : s->e->fam->mark_bytes ? s->e->fam->mark_bytes(s->impl) : oseq_state_bytes(s);
}

bool oseq_mark_save(const OsmosSeq *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !out)
    return osmos_fail(err, "no sequence or buffer");
  return s->e->fam->mark_save ? named(s->e->fam, err, s->e->fam->mark_save(s->impl, out, bytes, err)) : oseq_save(s, out, bytes, err);
}

bool oseq_mark_load(OsmosSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !in)
    return osmos_fail(err, "no sequence or mark");
  return s->e->fam->mark_load ? named(s->e->fam, err, s->e->fam->mark_load(s->impl, in, bytes, err)) : oseq_load(s, in, bytes, err);
}

bool oseq_seed(OsmosSeq *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !osmos_block(s->e))
    return osmos_fail(err, "no drafter");
  return named(s->e->fam, err, s->e->fam->seed(s->e->impl, s->impl, from, to, err));
}

int oseq_draft(OsmosSeq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[OSMOS_MAX_BLOCK],
               float conf[OSMOS_MAX_BLOCK], char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !drafts || !conf || !osmos_block(s->e))
    return osmos_fail(err, "no drafter"), -1;
  if (y < 0 || (uint32_t)y >= s->e->vocab) /* every family's (Hy4's drafter returned 0, no drafts) */
    return osmos_fail(err, "a draft after a token outside the vocabulary"), -1;
  return named_count(s->e->fam, err, s->e->fam->draft(s->e->impl, s->impl, y, conf_min, cum_min, drafts, conf, err));
}

bool oseq_draft_all(OsmosSeq *const *s, const int32_t *y, uint32_t n, float conf_min, float cum_min,
                    int32_t (*drafts)[OSMOS_MAX_BLOCK], float (*conf)[OSMOS_MAX_BLOCK], int *counts,
                    char err[OSMOS_ERR])
{
  if (!s || !y || !n || !drafts || !conf || !counts || !s[0] || !osmos_block(s[0]->e))
    return osmos_fail(err, "no sequences, room or drafter");
  Osmos *e = s[0]->e;
  for (uint32_t i = 0; i < n; i++)
  {
    if (!s[i] || s[i]->e != e)
      return osmos_fail(err, "a sequence not this engine's");
    if (y[i] < 0 || (uint32_t)y[i] >= e->vocab)
      return osmos_fail(err, "a draft after a token outside the vocabulary");
    for (uint32_t j = 0; j < i; j++)
      if (s[j] == s[i])
        return osmos_fail(err, "a sequence twice");
  }
  if (e->fam->draft_all)
    return named(e->fam, err, e->fam->draft_all(e->impl, s, y, n, conf_min, cum_min, drafts, conf, counts, err));
  for (uint32_t i = 0; i < n; i++)
    if ((counts[i] = named_count(e->fam, err, e->fam->draft(e->impl, s[i]->impl, y[i], conf_min, cum_min, drafts[i], conf[i], err))) < 0)
      return false;
  return true;
}
