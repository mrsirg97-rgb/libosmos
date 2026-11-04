/* a conversation: its sequence, the tokens it holds, set aside and taken back (serve.h) */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "serve.h"

double now_ms(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

void out(const char *fmt, ...)
{
  va_list a;
  va_start(a, fmt);
  vprintf(fmt, a);
  va_end(a);
  fflush(stdout);
}

bool ids_push(Ids *a, int32_t x)
{
  if (a->n == a->cap)
  {
    const uint32_t cap = a->cap ? 2 * a->cap : 4096;
    int32_t *v = realloc(a->v, (size_t)cap * sizeof *v);
    if (!v)
      return false;
    a->v = v, a->cap = cap;
  }
  a->v[a->n++] = x;
  return true;
}

bool server_spec(const Server *sv, const Gen *g)
{
  return g->spec < 0 ? sv->d != NULL : g->spec != 0;
}

bool server_eos(const Server *sv, int32_t t)
{
  for (int i = 0; i < sv->n_eos; i++)
    if (sv->eos[i] == t)
      return true;
  return false;
}

bool parse_gen(char *line, Gen *g, char *why, size_t wn)
{
  memset(g, 0, sizeof *g);
  g->spec = -1;
  char *save = NULL, *tok = strtok_r(line + 4, " ", &save);
  if (!tok)
    return snprintf(why, wn, "expected: GEN <max_new> [keys] <id,id,...>"), false;
  g->max_new = atoll(tok);
  for (tok = strtok_r(NULL, " ", &save); tok; tok = strtok_r(NULL, " ", &save))
  {
    char *eq = strchr(tok, '=');
    if (!eq)
      break; /* the ids */
    *eq = 0;
    if (!strcmp(tok, "spec"))
      g->spec = atoi(eq + 1);
    else if (!strcmp(tok, "rid"))
      g->rid = atoll(eq + 1), g->has_rid = true;
    else if (strcmp(tok, "seed") != 0) /* (greedy: a seed changes nothing) */
      return snprintf(why, wn, "the key '%s' is not this engine's (greedy: rid, spec, seed)", tok), false;
  }
  if (!tok)
    return snprintf(why, wn, "a GEN with no ids"), false;
  uint32_t cap = 1;
  for (const char *c = tok; *c; c++)
    cap += *c == ',';
  if (!(g->ids = malloc((size_t)cap * sizeof *g->ids)))
    return snprintf(why, wn, "out of memory"), false;
  for (char *s2 = NULL, *x = strtok_r(tok, ",", &s2); x; x = strtok_r(NULL, ",", &s2))
    g->ids[g->n++] = (int32_t)atol(x);
  if (!g->n)
    return free(g->ids), snprintf(why, wn, "a GEN with no ids"), false;
  return true;
}

bool conv_create(Conv *c, const Server *sv)
{
  memset(c, 0, sizeof *c);
  return (c->s = seq_create(sv->f, c->err)) != NULL;
}

void conv_destroy(Conv *c)
{
  seq_destroy(c->s);
  free(c->held.v);
  memset(c, 0, sizeof *c);
}

bool conv_restart(Conv *c, const Server *sv)
{
  seq_destroy(c->s);
  c->held.n = 0;
  return (c->s = seq_create(sv->f, c->err)) != NULL;
}

void conv_set_aside(Conv *c, const Server *sv)
{
  if (!sv->slots || !c->held.n)
    return;
  const uint64_t bytes = seq_state_bytes(c->s);
  void *st = malloc(bytes);
  char serr[SLOTS_ERR] = "";
  bool kept = false;
  if (!st || !seq_save(c->s, st, bytes, c->err) ||
      !slots_put(sv->slots, c->held.v, c->held.n, st, bytes, &kept, serr))
    out("NOTE slots: a conversation of %u tokens not set aside (%s%s)\n", c->held.n, st ? c->err : "out of memory",
        serr);
  free(st);
}

static uint32_t common(const Ids *a, const int32_t *p, uint32_t n)
{
  uint32_t k = 0;
  while (k < a->n && k < n && a->v[k] == p[k])
    k++;
  return k;
}

uint32_t conv_reach(const Conv *c, const int32_t *p, uint32_t n)
{
  const uint32_t k = common(&c->held, p, n - 1); /* (one token left to read: its logits the answer's first) */
  return k == c->held.n || c->held.n - k <= SEQ_MAX_BACK ? k : 0;
}

bool conv_continue(Conv *c, const Server *sv, const int32_t *p, uint32_t n)
{
  const uint32_t k = common(&c->held, p, n - 1);
  if (k == c->held.n && k)
    return true;
  if (k && c->held.n - k <= SEQ_MAX_BACK && seq_rollback(c->s, c->held.n - k))
  {
    c->held.n = k;
    return true;
  }
  conv_set_aside(c, sv);
  uint32_t len = 0;
  const int64_t id = sv->slots ? slots_find(sv->slots, p, n - 1, 0, &len) : -1;
  if (!conv_restart(c, sv))
    return false;
  if (id < 0)
    return true;
  void *st = NULL;
  uint64_t bytes = 0;
  char serr[SLOTS_ERR] = "";
  if (!slots_take(sv->slots, id, &st, &bytes, serr) || !seq_load(c->s, st, bytes, c->err))
  {
    out("NOTE slots: a kept conversation of %u tokens not taken back (%s%s)\n", len, serr, st ? c->err : "");
    free(st);
    return conv_restart(c, sv);
  }
  free(st);
  for (uint32_t i = 0; i < len; i++)
    if (!ids_push(&c->held, p[i]))
      return false;
  return true;
}

bool conv_read(Conv *c, const Server *sv, const int32_t *p, uint32_t k)
{
  if (!k)
    return true;
  Seq **seqs = calloc(k, sizeof *seqs);
  if (!seqs)
    return snprintf(c->err, sizeof c->err, "out of memory"), false;
  for (uint32_t i = 0; i < k; i++)
    seqs[i] = c->s;
  const uint32_t at = c->held.n;
  const bool ok = forward_batch(sv->f, seqs, p + at, k, c->err);
  free(seqs);
  for (uint32_t i = 0; ok && i < k; i++)
    if (!ids_push(&c->held, p[at + i]))
      return snprintf(c->err, sizeof c->err, "out of memory"), false;
  return ok;
}
