/* the engine (osmos/run/engine.h): a model opened by its family, an engine built on the ledger, a step's runs, its
   kept rows' logits, their picks and draws - each call its family's (engine_impl.h's table) */
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


void *oseq_impl(const OsmosSeq *s) { return s ? s->impl : NULL; }

/* a family's failure named once, by the family (osmos_context: its own cause kept, "<arch>: " before it) */
bool engine_named(const OsmosFamily *fam, char err[OSMOS_ERR], bool ok)
{
  if (!ok)
    osmos_context(err, fam->arch);
  return ok;
}

int engine_named_count(const OsmosFamily *fam, char err[OSMOS_ERR], int got)
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
  { /* the families named from the table itself, never a hand-kept list */
    char known[OSMOS_ERR / 2] = "";
    for (size_t i = 0, at = 0; i < sizeof families / sizeof *families && at < sizeof known; i++)
      at += (size_t)snprintf(known + at, sizeof known - at, "%s%s", i ? ", " : "", families[i]->arch);
    return osmos_fail(err, "the architecture %s: no family here runs it (%s)", arch[0] ? arch : "(none)", known), NULL;
  }
  OsmosModel *m = calloc(1, sizeof *m);
  if (!m)
    return osmos_fail(err, "out of memory"), NULL;
  m->fam = fam;
  if (!engine_named(fam, err, (m->impl = fam->open(files, &m->facts, err)) != NULL))
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

/* the first option set that the family does not take, or NULL (engine_impl.h's OSMOS_TAKES_*) */
static const char *untaken(const OsmosFamily *f, const OsmosOptions *o)
{
  const struct
  {
    bool set;
    uint32_t bit;
    const char *name;
  } opt[] = {
      {o->budget != 0, OSMOS_TAKES_BUDGET, "budget (its every weight resident)"},
      {o->picks != NULL, OSMOS_TAKES_PICKS, "picks"},
      {o->no_guess, OSMOS_TAKES_NO_GUESS, "no_guess"},
      {o->drafter != NULL, OSMOS_TAKES_DRAFTER, "drafter"},
      {o->stream_ngram, OSMOS_TAKES_STREAM_NGRAM, "stream_ngram (Qwen's n-gram table)"},
      {o->npu != NULL, OSMOS_TAKES_NPU, "npu (no NPU kernels for it)"},
      {o->kv_q8, OSMOS_TAKES_KV_Q8, "kv_q8 (Inkling's)"},
      {o->profile, OSMOS_TAKES_PROFILE, "profile (Inkling's)"},
      {o->rope_stages != NULL, OSMOS_TAKES_ROPE_STAGES, "rope_stages (osmos1.7b's)"},
  };
  for (size_t i = 0; i < sizeof opt / sizeof *opt; i++)
    if (opt[i].set && !(f->takes & opt[i].bit))
      return opt[i].name;
  return NULL;
}

Osmos *osmos_create(const OsmosModel *m, const OsmosOptions *o, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!m || !o)
    return osmos_fail(err, "no model or options"), NULL;
  const char *opt = untaken(m->fam, o);
  if (opt) /* (refused before anything is made, the family named) */
    return engine_named(m->fam, err, osmos_fail(err, "an option it does not take: %s", opt)), NULL;
  Osmos *e = calloc(1, sizeof *e);
  if (!e)
    return osmos_fail(err, "out of memory"), NULL;
  e->fam = m->fam, e->vocab = (uint32_t)m->facts.vocab;
  if (!(e->row = malloc((size_t)e->vocab * sizeof *e->row)))
    return free(e), osmos_fail(err, "out of memory"), NULL;
  e->ram_kept = o->fill_set ? o->fill : FIT_RAM_FREE;
  /* the ledger first (the family's build borrows it), then the engine's own buffers - made whole, here */
  if (!(e->ledger = ledger_create(e->ram_kept, err)) ||
      !ledger_check(e->ledger, ledger_malloc(e->ledger, (void **)&e->drow, (size_t)e->vocab * sizeof *e->drow), err,
                    "the engine's sampling row") ||
      !ledger_check(e->ledger, ledger_malloc(e->ledger, (void **)&e->dpick, sizeof *e->dpick), err,
                    "the engine's sampling pick") ||
      !engine_named(m->fam, err, (e->impl = m->fam->create(m->impl, o, e->ledger, err)) != NULL))
    return osmos_destroy(e), NULL;
  e->apart = m->fam->reads_apart || (m->fam->apart && m->fam->apart(e->impl));
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
   the less (fit_room) - the experts' pinned host slots in RAM, not the GPU's count */
bool osmos_room(const Osmos *e, uint64_t *bytes)
{
  uint64_t free_b = 0;
  if (!e || !bytes || !osmos_gpu_room(&free_b))
    return false;
  *bytes = fit_room(free_b, fit_ram_available(), e->ram_kept);
  return true;
}

bool osmos_machine_room(uint64_t ram_kept, uint64_t *bytes)
{
  uint64_t free_b = 0;
  if (!bytes || !osmos_gpu_room(&free_b))
    return false;
  *bytes = fit_room(free_b, fit_ram_available(), ram_kept);
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

/* a batch's step, every family's one path: the checks, the runs, the family's step; the rows the kept view reads its
   batch's */
static bool step(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, bool read, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!e || !seqs || !tokens || !n)
    return osmos_fail(err, "a batch of no tokens");
  e->last_n = 0; /* (no row kept until a step holds) */
  for (uint32_t i = 0; i < n; i++)
    if (!seqs[i] || seqs[i]->e != e)
      return osmos_fail(err, "a batch's sequence not this engine's");
  for (uint32_t i = 0; i < n; i++) /* every family's, before any reads one */
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
  if (!engine_named(e->fam, err, e->fam->step(e->impl, run, len, runs, tokens, read && e->apart, err)))
    return false;
  e->last_n = n;
  return true;
}

bool osmos_read(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  return step(e, seqs, tokens, n, true, err);
}

bool osmos_reads_apart(const Osmos *e) { return e && e->apart; }

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

static bool draw(Osmos *e, const float *dev, void *stream, const OsmosSampling *s, uint64_t position, int32_t *id,
                 char err[OSMOS_ERR])
{
  if (!(s->temperature < 1e30f) || !(s->top_p > 0 && s->top_p <= 1))
    return osmos_fail(err, "sampling: temperature finite and above 0, top_p in (0, 1]");
  if (!ops_sample(dev, e->vocab, s->temperature, s->top_p, s->top_k, ops_draw_uniform(s->seed, position), e->dpick,
                  stream) ||
      hipMemcpyAsync(id, e->dpick, sizeof *id, hipMemcpyDeviceToHost, (hipStream_t)stream) != hipSuccess ||
      hipStreamSynchronize((hipStream_t)stream) != hipSuccess)
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
  const KeptView k = kept_of(e);
  const float *dev = kept_row(&k, row);
  if (!dev)
    return osmos_fail(err, "the row's logits not kept");
  return draw(e, dev, k.stream, s, position, id, err);
}

bool osmos_sample_except(Osmos *e, uint32_t row, const OsmosSampling *s, uint64_t position, const int32_t *except,
                         uint32_t n, int32_t *id, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!osmos_sampled(s))
    return osmos_greedy_except(e, row, except, n, id, err);
  if (!e || !id)
    return osmos_fail(err, "no engine");
  const KeptView k = kept_of(e);
  const float *dev = kept_row(&k, row);
  if (!dev)
    return osmos_fail(err, "the row's logits not kept");
  const size_t bytes = (size_t)e->vocab * sizeof *e->drow;
  const float ninf = -__builtin_inff();
  const hipStream_t st = (hipStream_t)k.stream;
  if (hipMemcpyAsync(e->drow, dev, bytes, hipMemcpyDeviceToDevice, st) != hipSuccess)
    return osmos_fail(err, "a GPU call failed (a row's copy)");
  for (uint32_t i = 0; i < n; i++)
    if (except[i] >= 0 && (uint32_t)except[i] < e->vocab &&
        hipMemcpyAsync(e->drow + except[i], &ninf, sizeof ninf, hipMemcpyHostToDevice, st) != hipSuccess)
      return osmos_fail(err, "a GPU call failed (a row's copy)");
  return draw(e, e->drow, k.stream, s, position, id, err);
}

uint32_t osmos_block(const Osmos *e) { return e ? e->fam->block(e->impl) : 0; }
