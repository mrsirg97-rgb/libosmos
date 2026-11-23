/* a hy4 sequence (osmos/hy4/forward.h): its sparse layers' MLA rows, token keys and pooled groups by position, its
   sliding layers' rings, its tokens (the n-grams' history); rollback; the prompt's end; its state saved and loaded */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/state_parts.h"
#include "osmos/gpu/hy4_ops.h"
#include "hy4_impl.h"
#include "osmos/err.h"

Hy4Seq *hy4_seq_create(Hy4Forward *f, char err[OSMOS_ERR])
{
  if (!f)
    return osmos_fail(err, "no forward"), NULL;
  Hy4Seq *s = calloc(1, sizeof *s);
  if (!s)
    return osmos_fail(err, "out of memory"), NULL;
  s->f = f, s->bufs.ledger = f->ledger;
  const Hy4Geometry *g = &f->w->geo;
  const uint64_t row = hy4_mla_row_bytes((uint32_t)g->kv_lora, (uint32_t)g->rope_dim, f->kv8), ctx = f->max_ctx;
  bool ok = (s->tokens = malloc(ctx * 4)) != NULL;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->sliding[l] ? hy4_alloc(&s->bufs, &s->cache[l], f->ring * row)
                       : hy4_alloc(&s->bufs, &s->cache[l], ctx * row) && hy4_alloc(&s->bufs, &s->keys[l], (uint64_t)f->kring * g->ix_dim * F) &&
                             hy4_alloc(&s->bufs, &s->groups[l], (ctx / g->ix_pool + 1) * g->ix_dim * F);
  const uint64_t top = (uint64_t)(g->ix_topk / g->ix_pool); /* DFly2's snapshot (small: always made) */
  ok = ok && hy4_alloc(&s->bufs, &s->dsel, (uint64_t)HY4_DFLY_LAYERS * HY4_DFLY_KEEP * top * 4) &&
       hy4_alloc(&s->bufs, &s->dn, (uint64_t)HY4_DFLY_LAYERS * HY4_DFLY_KEEP * 4) &&
       hy4_alloc(&s->bufs, &s->dhid, (uint64_t)HY4_DFLY_KEEP * g->n_embd * F);
  if (!ok)
    return hy4_seq_destroy(s), osmos_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

void hy4_seq_destroy(Hy4Seq *s)
{
  if (!s)
    return;
  gpu_bufs_free(&s->bufs); /* every GPU buffer it made: from its record */
  free(s->tokens);
  free(s);
}

int32_t hy4_seq_position(const Hy4Seq *s) { return s ? s->pos : -1; }

bool hy4_seq_prompt(Hy4Seq *s, uint32_t end)
{
  if (!s || s->broken || end > s->f->max_ctx) /* (an end past the context refused, as the other families') */
    return false;
  s->prompt_end = end;
  return true;
}

uint32_t hy4_seq_reach(const Hy4Seq *s) { return s && !s->broken ? s->back : 0; }

/* the rows past the position never read again (a later step writes them first); a group their keys completed pooled
   again when its keys land again; a sliding ring holds the window past the reach */
bool hy4_seq_rollback(Hy4Seq *s, uint32_t n)
{
  if (!s || s->broken || n > s->back)
    return false;
  s->pos -= (int32_t)n, s->back -= n;
  return true;
}

void hy4_seq_fail(Hy4Seq *s)
{
  if (s)
    s->broken = true;
}

/* a saved state: the header (its position and the two tokens before it - the next step's n-grams' history; the
   rest of it never read after a load, whose reach is 0), then the parts, then the sum (state_parts.h) */
#define STATE_MAGIC 0x53345948u /* "HY4S" */
#define STATE_VERSION 2u /* (2: the cache rows hy4_mla_row_bytes' - Q8_0 latents by default) */
#define HISTORY HY4_STATE_HISTORY

typedef struct Header
{
  uint32_t magic, version;
  uint64_t identity;
  int32_t pos;
  int32_t last[HISTORY];
} Header;

#define MAX_PARTS (3 * HY4_MAX_LAYERS)

/* the parts at position pos, in the string's order: a sparse layer's MLA rows below pos, its token keys' ring and its
   complete groups; a sliding layer's whole ring */
static int parts(const Hy4Seq *s, int32_t pos, StatePart *p)
{
  const Hy4Forward *f = s->f;
  const Hy4Geometry *g = &f->w->geo;
  const uint64_t row = hy4_mla_row_bytes((uint32_t)g->kv_lora, (uint32_t)g->rope_dim, f->kv8),
                 key = (uint64_t)g->ix_dim * F;
  int n = 0;
  for (int l = 0; l < g->n_layers; l++)
    if (g->sliding[l])
      p[n++] = (StatePart){s->cache[l], f->ring * row};
    else
      p[n++] = (StatePart){s->cache[l], (uint64_t)pos * row}, p[n++] = (StatePart){s->keys[l], (uint64_t)f->kring * key},
      p[n++] = (StatePart){s->groups[l], (uint64_t)(pos / g->ix_pool) * key};
  return n;
}

uint64_t hy4_seq_state_bytes(const Hy4Seq *s)
{
  if (!s)
    return 0;
  StatePart p[MAX_PARTS];
  return state_parts_bytes(sizeof(Header), p, parts(s, s->pos, p));
}

bool hy4_seq_save_to(const Hy4Seq *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  if (!s || !sink)
    return osmos_fail(err, "no sequence or sink");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its state part written): load it, or make another");
  Header h;
  memset(&h, 0, sizeof h);
  h.magic = STATE_MAGIC, h.version = STATE_VERSION, h.identity = hy4_identity(s->f), h.pos = s->pos;
  for (int j = 0; j < HISTORY; j++)
    h.last[j] = s->pos > j ? s->tokens[s->pos - 1 - j] : -1;
  StatePart p[MAX_PARTS];
  const char *why = state_parts_save(&h, sizeof h, p, parts(s, s->pos, p), sink, ctx);
  return !why || osmos_fail(err, "%s", why);
}

bool hy4_seq_reset(Hy4Seq *s, char err[OSMOS_ERR])
{
  if (!s)
    return osmos_fail(err, "no sequence");
  s->pos = 0, s->back = 0, s->prompt_end = 0, s->broken = false; /* empty: its rows never read again */
  s->dkeep = 0;                                                   /* (DFly2's snapshot: none until a step runs) */
  return true;
}

static bool refuse(Hy4Seq *s, char *err, const char *why)
{
  hy4_seq_reset(s, NULL);
  return osmos_fail(err, "%s", why);
}

bool hy4_seq_load_from(Hy4Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  if (!s || !src)
    return osmos_fail(err, "no sequence or source");
  s->dkeep = 0; /* (DFly2's snapshot: none until a step runs) */
  Header h;
  if (bytes < sizeof h + sizeof(uint64_t) || !src(ctx, &h, sizeof h))
    return refuse(s, err, "a state shorter than its header, or its source failed");
  const char *bad = state_head_check(h.magic, STATE_MAGIC, h.version, STATE_VERSION, h.identity, hy4_identity(s->f));
  bad = bad ? bad : state_pos_check(h.pos, s->f->max_ctx);
  bad = bad ? bad : state_tokens_check(h.last, h.pos < HISTORY ? h.pos : HISTORY, (uint32_t)s->f->w->geo.n_vocab);
  if (bad)
    return refuse(s, err, bad);
  StatePart p[MAX_PARTS];
  const int n = parts(s, h.pos, p);
  if (bytes != state_parts_bytes(sizeof h, p, n))
    return refuse(s, err, "a state not its position's size");
  const char *why = state_parts_load(&h, sizeof h, p, n, src, ctx);
  if (why)
    return refuse(s, err, why);
  for (int j = 0; j < HISTORY && h.pos > j; j++)
    s->tokens[h.pos - 1 - j] = h.last[j];
  s->pos = h.pos, s->back = 0, s->prompt_end = 0, s->broken = false;
  return true;
}
