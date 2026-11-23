#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/model/tensor.h"
#include "osmos/model/engram.h"
#include "osmos/err.h"


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

static bool fail(char *err, const char *why) { return osmos_fail(err, "the Engram table: %s", why); }

/* a whole tensor into memory of its own */
static void *slurp(const GgufTensor *t, const int *fds)
{
  void *p = malloc(t->bytes ? t->bytes : 1);
  if (!p)
    return NULL;
  if (!tensor_read_fd(fds[t->file], t, p))
    return free(p), NULL;
  return p;
}

/* a metadata list of n integers (any width) into memory of its own, as int64; `minus_first` each layer's `per`
   values less that layer's first (llama.cpp's offsets count across the layers' tables; a layer's rows its own) */
static int64_t *list(const Gguf *g, const char *arch, const char *key, size_t n, size_t per, bool minus_first,
                     char *err)
{
  char name[128];
  snprintf(name, sizeof name, "%s.%s", arch, key);
  const GgufKv *kv = gguf_kv(g, name);
  if (!kv || kv->type != GGUF_ARRAY || kv->n != n)
    return fail(err, minus_first ? "ple.head_offsets: missing or not a column a layer"
                                 : "a hash list (ple.*): missing or not its length"), NULL;
  int64_t *v = malloc(n ? n * sizeof *v : 1);
  if (!v)
    return fail(err, "out of memory"), NULL;
  for (size_t i = 0; i < n; i++)
    if (!gguf_int(kv, i, &v[i]))
    {
      free(v);
      return fail(err, "a hash list's value not an integer within int64"), NULL;
    }
  for (size_t i = n; minus_first && i-- > 0;)
    v[i] -= v[i - i % per];
  return v;
}

/* llama.cpp's form: the four tables the model's metadata (engram.token_map, ple.layer_multipliers,
   ple.head_vocab_sizes, ple.head_offsets) */
static bool from_meta(Engram *e, const EngramShape *s, char *err)
{
  char name[128];
  snprintf(name, sizeof name, "%s.engram.token_map", s->arch);
  const GgufKv *map = gguf_kv(s->model, name);
  if (!map || map->type != GGUF_ARRAY || map->n != s->n_vocab)
    return fail(err, "engram.token_map: missing, or not an entry a token");
  e->vocab = map->n;
  if (!(e->map = malloc(e->vocab * sizeof *e->map)))
    return fail(err, "out of memory");
  for (uint64_t i = 0; i < e->vocab; i++)
  {
    int64_t v;
    if (!gguf_int(map, i, &v) || v < INT32_MIN || v > INT32_MAX)
      return fail(err, "engram.token_map: an entry not an int32");
    e->map[i] = (int32_t)v;
  }
  const size_t L = (size_t)e->n_layers, cols = (size_t)e->cols;
  return (e->mult = list(s->model, s->arch, "ple.layer_multipliers", L * (size_t)e->ngram, 1, false, err)) &&
         (e->primes = list(s->model, s->arch, "ple.head_vocab_sizes", L * cols, 1, false, err)) &&
         (e->off = list(s->model, s->arch, "ple.head_offsets", L * cols, cols, true, err));
}

/* `rows[i]` the i-th Engram layer's table rows */
static bool check(const Engram *e, const uint64_t *rows, char *err)
{
  for (uint64_t i = 0; i < e->vocab; i++)
    if (e->map[i] < 0 || e->map[i] >= e->compressed)
      return fail(err, "a token map entry not a compressed token");
  for (int i = 0; i < e->n_layers; i++)
  {
    for (int s = 0; s < e->ngram; s++)
      if (e->mult[i * e->ngram + s] <= 0 || !(e->mult[i * e->ngram + s] & 1) ||
          e->mult[i * e->ngram + s] > INT64_MAX / e->compressed)
        return fail(err, "a multiplier not odd, positive and small enough that no product overflows");
    for (int c = 0; c < e->cols; c++)
    {
      const int64_t p = e->primes[i * e->cols + c], o = e->off[i * e->cols + c];
      if (p <= 0 || o < 0 || (uint64_t)o + (uint64_t)p > rows[i])
        return fail(err, "a column's rows past its table");
    }
  }
  return true;
}

/* the pad's compressed token; the entries past `covered` the pad's; the compressed count the covered entries'
   greatest + 1 where none is named; then the checks */
static Engram *finish(Engram *e, bool ok, uint64_t covered, int32_t pad, const uint64_t *rows, char *err)
{
  if (ok && (covered > e->vocab || pad < 0 || (uint64_t)pad >= covered))
    ok = fail(err, covered > e->vocab ? "the map covering past its entries" : "the pad token past the map");
  const bool derive = !e->compressed;
  for (uint64_t i = 0; ok && derive && i < covered; i++)
    e->compressed = e->map[i] >= e->compressed ? (int64_t)e->map[i] + 1 : e->compressed;
  if (ok)
  {
    e->pad = e->map[pad];
    for (uint64_t i = covered; i < e->vocab; i++)
      e->map[i] = (int32_t)e->pad;
    ok = check(e, rows, err);
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

Engram *engram_create_shape(const EngramShape *s, char err[OSMOS_ERR])
{
  if (!s || !s->model || !s->arch || !s->rows || s->n_layers <= 0 || s->ngram < 2 || s->heads <= 0)
    return fail(err, "no shape"), NULL;
  if (s->ngram > ENGRAM_MAX_NGRAM)
    return fail(err, "an n-gram longer than this build hashes"), NULL;
  Engram *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  e->n_layers = s->n_layers;
  e->ngram = s->ngram;
  e->heads = s->heads;
  e->cols = (s->ngram - 1) * s->heads;
  e->compressed = s->compressed;
  const bool ok = from_meta(e, s, err);
  return finish(e, ok, ok ? s->covered : 0, s->pad, s->rows, err);
}

Engram *engram_create(const Weights *w, const int *fds, char err[OSMOS_ERR])
{
  if (!w || !fds)
    return fail(err, "no weights"), NULL;
  const Geometry *g = &w->geo;
  uint64_t rows[WEIGHTS_MAX_LAYERS];
  int n = 0;
  for (int l = 0; l < g->n_layers; l++)
    if (g->engram[l])
      rows[n++] = g->engram_rows[l];
  if (g->engram_own)
  {
    const EngramShape s = {w->model, "deepseek41", n, g->engram_max_ngram, g->engram_heads, (uint64_t)g->n_vocab,
                           (uint64_t)g->n_vocab, g->engram_pad, g->engram_vocab, rows};
    return engram_create_shape(&s, err);
  }
  if (g->engram_max_ngram > ENGRAM_MAX_NGRAM)
    return fail(err, "an n-gram longer than this build hashes"), NULL;
  Engram *e = calloc(1, sizeof *e);
  if (!e)
    return fail(err, "out of memory"), NULL;
  e->n_layers = n;
  e->ngram = g->engram_max_ngram;
  e->heads = g->engram_heads;
  e->cols = g->engram_cols;
  e->compressed = g->engram_vocab;
  e->vocab = w->eg_token_map->dims[0];
  const bool ok = (e->map = slurp(w->eg_token_map, fds)) && (e->mult = slurp(w->eg_multipliers, fds)) &&
                  (e->primes = slurp(w->eg_primes, fds)) && (e->off = slurp(w->eg_offsets, fds));
  return finish(e, ok, e->vocab, g->engram_pad, rows, err);
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
