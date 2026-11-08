#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
#include "osmos/model/gguf.h"

/* the dispatch: a model's family by its file's general.architecture, then every call that family's */

static const OsmosFamily *const families[] = {&osmos_ds4, &osmos_q4x};

struct OsmosModel
{
  const OsmosFamily *fam;
  void *impl;
  OsmosFacts facts;
};

struct Osmos
{
  const OsmosFamily *fam;
  void *impl;
};

struct OsmosSeq
{
  Osmos *e;
  void *impl;
};

static bool fail(char err[OSMOS_ERR], const char *what)
{
  if (err)
    snprintf(err, OSMOS_ERR, "%s", what);
  return false;
}

void *oseq_impl(const OsmosSeq *s) { return s ? s->impl : NULL; }

OsmosModel *osmos_open(const OsmosFiles *files, char err[OSMOS_ERR])
{
  if (!files || !files->model)
    return fail(err, "no model file named"), NULL;
  char gerr[GGUF_ERR] = "";
  Gguf *h = gguf_header(files->model, gerr);
  if (!h)
    return fail(err, gerr), NULL;
  const GgufKv *a = gguf_kv(h, "general.architecture");
  char arch[64] = "";
  if (a && a->type == GGUF_STRING)
    snprintf(arch, sizeof arch, "%s", a->v.s);
  gguf_destroy(h);
  const OsmosFamily *fam = NULL;
  for (size_t i = 0; i < sizeof families / sizeof *families; i++)
    if (!strcmp(arch, families[i]->arch))
      fam = families[i];
  if (!fam)
  {
    if (err)
      snprintf(err, OSMOS_ERR, "the architecture %s: no family here runs it (deepseek41, qwen4exp)", arch[0] ? arch : "(none)");
    return NULL;
  }
  OsmosModel *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  m->fam = fam;
  if (!(m->impl = fam->open(files, &m->facts, err)))
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
  if (!m || !o)
    return fail(err, "no model or options"), NULL;
  Osmos *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  e->fam = m->fam;
  if (!(e->impl = m->fam->create(m->impl, o, err)))
    return free(e), NULL;
  return e;
}

void osmos_destroy(Osmos *e)
{
  if (!e)
    return;
  e->fam->destroy(e->impl);
  free(e);
}

uint64_t osmos_identity(const Osmos *e) { return e ? e->fam->identity(e->impl) : 0; }

bool osmos_room(const Osmos *e, uint64_t *bytes) { return e && bytes && e->fam->room(e->impl, bytes); }

void osmos_stats(const Osmos *e, OsmosStats *s)
{
  if (!s)
    return;
  memset(s, 0, sizeof *s);
  if (e)
    e->fam->stats(e->impl, s);
}

bool osmos_batch(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR])
{
  if (!e || !seqs || !tokens || !n)
    return fail(err, "a batch of no tokens");
  for (uint32_t i = 0; i < n; i++)
    if (!seqs[i] || seqs[i]->e != e)
      return fail(err, "a batch's sequence not this engine's");
  return e->fam->batch(e->impl, seqs, tokens, n, err);
}

const float *osmos_logits(const Osmos *e, uint32_t row) { return e ? e->fam->logits(e->impl, row) : NULL; }

bool osmos_greedy(Osmos *e, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  return e && ids ? e->fam->greedy(e->impl, row0, rows, ids, err) : fail(err, "no engine");
}

uint32_t osmos_block(const Osmos *e) { return e ? e->fam->block(e->impl) : 0; }

OsmosSeq *oseq_create(Osmos *e, char err[OSMOS_ERR])
{
  if (!e)
    return fail(err, "no engine"), NULL;
  OsmosSeq *s = calloc(1, sizeof *s);
  if (!s)
    return fail(err, "out of memory"), NULL;
  s->e = e;
  if (!(s->impl = e->fam->seq_create(e->impl, err)))
    return free(s), NULL;
  return s;
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
  return s && sink ? s->e->fam->save_to(s->impl, sink, ctx, err) : fail(err, "no sequence or sink");
}

bool oseq_load_from(OsmosSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  return s && src ? s->e->fam->load_from(s->impl, bytes, src, ctx, err) : fail(err, "no sequence or source");
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
  if (!s || !out)
    return fail(err, "no sequence or buffer");
  if (bytes != oseq_state_bytes(s))
    return fail(err, "a buffer not the state's size (oseq_state_bytes)");
  Mem m = {out, 0, bytes};
  return oseq_save_to(s, mem_sink, &m, err) && (m.at == bytes || fail(err, "a state shorter than its size"));
}

bool oseq_load(OsmosSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  Mem m = {(uint8_t *)in, 0, bytes};
  return s && in ? oseq_load_from(s, bytes, mem_source, &m, err) : fail(err, "no sequence or state");
}

bool oseq_seed(OsmosSeq *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  if (!s || !osmos_block(s->e))
    return fail(err, "no drafter");
  return s->e->fam->seed(s->e->impl, s->impl, from, to, err);
}

int oseq_draft(OsmosSeq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[OSMOS_MAX_BLOCK],
               float conf[OSMOS_MAX_BLOCK], char err[OSMOS_ERR])
{
  if (!s || !drafts || !conf || !osmos_block(s->e))
    return fail(err, "no drafter"), -1;
  return s->e->fam->draft(s->e->impl, s->impl, y, conf_min, cum_min, drafts, conf, err);
}

bool oseq_draft_all(OsmosSeq *const *s, const int32_t *y, uint32_t n, float conf_min, float cum_min,
                    int32_t (*drafts)[OSMOS_MAX_BLOCK], float (*conf)[OSMOS_MAX_BLOCK], int *counts,
                    char err[OSMOS_ERR])
{
  if (!s || !y || !n || !drafts || !conf || !counts || !s[0] || !osmos_block(s[0]->e))
    return fail(err, "no sequences, room or drafter");
  Osmos *e = s[0]->e;
  for (uint32_t i = 0; i < n; i++)
  {
    if (!s[i] || s[i]->e != e)
      return fail(err, "a sequence not this engine's");
    for (uint32_t j = 0; j < i; j++)
      if (s[j] == s[i])
        return fail(err, "a sequence twice");
  }
  if (e->fam->draft_all)
    return e->fam->draft_all(e->impl, s, y, n, conf_min, cum_min, drafts, conf, counts, err);
  for (uint32_t i = 0; i < n; i++)
    if ((counts[i] = e->fam->draft(e->impl, s[i]->impl, y[i], conf_min, cum_min, drafts[i], conf[i], err)) < 0)
      return false;
  return true;
}
