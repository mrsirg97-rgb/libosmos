/* a hy4 sequence (osmos/hy4/forward.h): its sparse layers' MLA rows, token keys and pooled groups by position, its
   sliding layers' rings, its tokens (the n-grams' history); rollback; the prompt's end. its state's bytes are 2b's */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "hy4_impl.h"

Hy4Seq *hy4_seq_create(Hy4Forward *f, char err[HY4_ERR])
{
  if (!f)
    return hy4_fail(err, "no forward"), NULL;
  Hy4Seq *s = calloc(1, sizeof *s);
  if (!s)
    return hy4_fail(err, "out of memory"), NULL;
  s->f = f;
  const Hy4Geometry *g = &f->w->geo;
  const uint64_t row = (uint64_t)(g->kv_lora + g->rope_dim) * F, ctx = f->max_ctx;
  bool ok = (s->tokens = malloc(ctx * 4)) != NULL;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->sliding[l] ? hy4_alloc(&s->cache[l], f->ring * row)
                       : hy4_alloc(&s->cache[l], ctx * row) && hy4_alloc(&s->keys[l], ctx * g->ix_dim * F) &&
                             hy4_alloc(&s->groups[l], (ctx / g->ix_pool + 1) * g->ix_dim * F);
  if (!ok)
    return hy4_seq_destroy(s), hy4_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

void hy4_seq_destroy(Hy4Seq *s)
{
  if (!s)
    return;
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    hipFree(s->cache[l]), hipFree(s->keys[l]), hipFree(s->groups[l]);
  free(s->tokens);
  free(s);
}

int32_t hy4_seq_position(const Hy4Seq *s) { return s ? s->pos : -1; }

bool hy4_seq_prompt(Hy4Seq *s, uint32_t end)
{
  if (!s || s->broken)
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

uint64_t hy4_seq_state_bytes(const Hy4Seq *s) { return s ? 0 : 0; }

bool hy4_seq_save_to(const Hy4Seq *s, StateSink sink, void *ctx, char err[HY4_ERR])
{
  (void)s, (void)sink, (void)ctx;
  return hy4_fail(err, "hy4: a sequence's state not saved yet (step 2b)");
}

bool hy4_seq_load_from(Hy4Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[HY4_ERR])
{
  (void)s, (void)bytes, (void)src, (void)ctx;
  return hy4_fail(err, "hy4: a sequence's state not loaded yet (step 2b)");
}
