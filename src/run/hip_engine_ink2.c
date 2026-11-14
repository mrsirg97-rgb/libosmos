/* Inkling-2-Small, the third family (0.8.0): its model opened and every tensor checked (osmos/ink2/weights.h: the
   reference's names or llama.cpp's, unsloth's packs), its facts the server sizes by. the engine - decode with its
   sequence's state (the sliding layers' windows, the full layers' caches, the short convs' histories), its experts
   streamed through the pool, the drafter - is not built yet: create refuses, naming it, so a server given an Inkling
   pack says what it lacks instead of failing in it. tools/bench/ink2_run is the model's whole forward meanwhile (a
   layer at a time, held to transformers' oracle) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_impl.h"
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
                        (uint32_t)((geo->n_experts - 1) / geo->n_used), 0, true};
  return m;
}

static void *e_create(const void *pm, const OsmosOptions *o, char err[OSMOS_ERR])
{
  (void)pm, (void)o;
  return fail(err, "the model opened and its tensors checked, but this build has no engine for it yet (decode, its "
                   "sequence's state, expert streaming: 0.8.0 in progress) - tools/bench/ink2_run scores text with it"),
         NULL;
}

/* the engine's and the sequence's entries: none is reached while create refuses (OsmosFamily's callers hold an
   engine first) */
const OsmosFamily osmos_ink2 = {
    "inkling2", m_open, m_close, e_create, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL,       NULL,   NULL,    NULL,     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
};
