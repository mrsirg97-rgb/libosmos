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

bool oseq_save(const OsmosSeq *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  return s && out ? s->e->fam->save(s->impl, out, bytes, err) : fail(err, "no sequence or buffer");
}

bool oseq_load(OsmosSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  return s && in ? s->e->fam->load(s->impl, in, bytes, err) : fail(err, "no sequence or state");
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
