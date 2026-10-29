#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/model/engram.h"


struct Engram
{
  int n_layers;
  int ngram;
  int heads;
  int cols;
  int64_t pad;
  uint64_t vocab;
  int64_t compressed;
  int32_t *map;
  int64_t *mult;   /* [layer][ngram] */
  int64_t *primes; /* [layer][cols] */
  int64_t *off;    /* [layer][cols] */
};

static bool fail(char *err, const char *why)
{
  snprintf(err, ENGRAM_ERR, "engram: %s", why);
  return false;
}

/* a whole tensor into memory of its own */
static void *slurp(const GgufTensor *t, const int *fds)
{
  void *p = malloc(t->bytes ? t->bytes : 1);
  if (!p)
    return NULL;
  uint64_t got = 0;
  while (got < t->bytes)
  {
    const ssize_t n = pread(fds[t->file], (char *)p + got, t->bytes - got, (off_t)(t->offset + got));
    if (n <= 0)
    {
      free(p);
      return NULL;
    }
    got += (uint64_t)n;
  }
  return p;
}

static bool check(const Engram *e, const Geometry *g, char *err)
{
  for (uint64_t i = 0; i < e->vocab; i++)
    if (e->map[i] < 0 || e->map[i] >= e->compressed)
      return fail(err, "a token map entry not a compressed token");
  for (int l = 0, i = 0; l < g->n_layers; l++)
  {
    if (!g->engram[l])
      continue;
    for (int s = 0; s < e->ngram; s++)
      if (e->mult[i * e->ngram + s] <= 0 || !(e->mult[i * e->ngram + s] & 1) ||
          e->mult[i * e->ngram + s] > INT64_MAX / e->compressed)
        return fail(err, "a multiplier not odd, positive and small enough that no product overflows");
    for (int c = 0; c < e->cols; c++)
    {
      const int64_t p = e->primes[i * e->cols + c], o = e->off[i * e->cols + c];
      if (p <= 0 || o < 0 || (uint64_t)o + (uint64_t)p > g->engram_rows[l])
        return fail(err, "a column's rows past its table");
    }
    i++;
  }
  return true;
}

Engram *engram_create(const Weights *w, const int *fds, char err[ENGRAM_ERR])
{
  if (!w || !fds)
    return fail(err, "no weights"), NULL;
  const Geometry *g = &w->geo;
  Engram *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  for (int l = 0; l < g->n_layers; l++)
    e->n_layers += g->engram[l];
  e->ngram = g->engram_max_ngram;
  e->heads = g->engram_heads;
  e->cols = g->engram_cols;
  e->vocab = w->eg_token_map->dims[0];
  e->compressed = g->engram_vocab;
  bool ok = e->ngram <= ENGRAM_MAX_NGRAM || fail(err, "an n-gram longer than this build hashes");
  ok = ok && (e->map = slurp(w->eg_token_map, fds)) && (e->mult = slurp(w->eg_multipliers, fds)) &&
       (e->primes = slurp(w->eg_primes, fds)) && (e->off = slurp(w->eg_offsets, fds));
  if (ok && (uint64_t)g->engram_pad >= e->vocab)
    ok = fail(err, "the pad token past the map");
  if (ok)
  {
    e->pad = e->map[g->engram_pad];
    ok = check(e, g, err);
  }
  else if (!err[0])
    fail(err, "a table could not be read");
  if (!ok)
  {
    engram_destroy(e);
    return NULL;
  }
  return e;
}

void engram_destroy(Engram *e)
{
  if (!e)
    return;
  free(e->map);
  free(e->mult);
  free(e->primes);
  free(e->off);
  free(e);
}

int engram_cols(const Engram *e) { return e->cols; }
int engram_ngram(const Engram *e) { return e->ngram; }

bool engram_rows(const Engram *e, const int32_t *last, int n_last, int64_t *rows)
{
  int64_t tok[ENGRAM_MAX_NGRAM];
  for (int s = 0; s < e->ngram; s++)
  {
    if (s < n_last && (last[s] < 0 || (uint64_t)last[s] >= e->vocab))
      return false;
    tok[s] = s < n_last ? e->map[last[s]] : e->pad;
  }
  for (int l = 0; l < e->n_layers; l++)
  {
    const int64_t *m = e->mult + l * e->ngram;
    int64_t fold = tok[0] * m[0];
    for (int i = 1; i < e->ngram; i++)
    {
      fold ^= tok[i] * m[i];
      for (int h = 0; h < e->heads; h++)
      {
        const int c = (i - 1) * e->heads + h;
        rows[l * e->cols + c] = fold % e->primes[l * e->cols + c] + e->off[l * e->cols + c];
      }
    }
  }
  return true;
}
