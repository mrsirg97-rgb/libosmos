/* the engine's sequences (osmos/run/engine.h): made, reset and destroyed; a prompt's end, rollback and reach;
   saved and loaded whole or streamed, a mark's; a drafter's seed and drafts - each its family's */
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
  if (!engine_named(e->fam, err, (s->impl = e->fam->seq_create(e->impl, err)) != NULL))
    return free(s), NULL;
  return s;
}

bool oseq_reset(OsmosSeq *s, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  return s ? engine_named(s->e->fam, err, s->e->fam->seq_reset(s->impl, err)) : osmos_fail(err, "no sequence");
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
  return s && sink ? engine_named(s->e->fam, err, s->e->fam->save_to(s->impl, sink, ctx, err)) : osmos_fail(err, "no sequence or sink");
}

bool oseq_load_from(OsmosSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  return s && src ? engine_named(s->e->fam, err, s->e->fam->load_from(s->impl, bytes, src, ctx, err)) : osmos_fail(err, "no sequence or source");
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
  return s->e->fam->mark_save ? engine_named(s->e->fam, err, s->e->fam->mark_save(s->impl, out, bytes, err)) : oseq_save(s, out, bytes, err);
}

bool oseq_mark_load(OsmosSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !in)
    return osmos_fail(err, "no sequence or mark");
  return s->e->fam->mark_load ? engine_named(s->e->fam, err, s->e->fam->mark_load(s->impl, in, bytes, err)) : oseq_load(s, in, bytes, err);
}

bool oseq_seed(OsmosSeq *s, int32_t from, int32_t to, char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !osmos_block(s->e))
    return osmos_fail(err, "no drafter");
  return engine_named(s->e->fam, err, s->e->fam->seed(s->e->impl, s->impl, from, to, err));
}

int oseq_draft(OsmosSeq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[OSMOS_MAX_BLOCK],
               float conf[OSMOS_MAX_BLOCK], char err[OSMOS_ERR])
{
  osmos_clear(err); /* (osmos/err.h: an earlier failure's cause never reported as this one's) */
  if (!s || !drafts || !conf || !osmos_block(s->e))
    return osmos_fail(err, "no drafter"), -1;
  if (y < 0 || (uint32_t)y >= s->e->vocab) /* every family's (Hy4's drafter returned 0, no drafts) */
    return osmos_fail(err, "a draft after a token outside the vocabulary"), -1;
  return engine_named_count(s->e->fam, err, s->e->fam->draft(s->e->impl, s->impl, y, conf_min, cum_min, drafts, conf, err));
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
    return engine_named(e->fam, err, e->fam->draft_all(e->impl, s, y, n, conf_min, cum_min, drafts, conf, counts, err));
  for (uint32_t i = 0; i < n; i++)
    if ((counts[i] = engine_named_count(e->fam, err, e->fam->draft(e->impl, s[i]->impl, y[i], conf_min, cum_min, drafts[i], conf[i], err))) < 0)
      return false;
  return true;
}
