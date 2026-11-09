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
  return g->spec < 0 ? osmos_block(sv->e) != 0 : g->spec != 0;
}

bool server_eos(const Server *sv, int32_t t)
{
  for (int i = 0; i < sv->n_eos; i++)
    if (sv->eos[i] == t)
      return true;
  return false;
}

/* row's pick, the row's tokens the held then extra[0 .. n): an end while the turn's answer has not begun (the reasoning
   closed, whitespace alone since: --answer-after's rule) is not taken - the row's best other token is. the same
   rule for a plain step's row and a drafted round's, so drafting keeps the bits */
bool server_answer(Server *sv, const Ids *held, const int32_t *extra, uint32_t n, uint32_t row, int32_t *pick,
                   char err[OSMOS_ERR])
{
  if (!sv->answer.blank || !server_eos(sv, *pick) || !answer_unbegun(&sv->answer, held->v, held->n, extra, n))
    return true;
  return osmos_greedy_except(sv->e, row, sv->eos, (uint32_t)sv->n_eos, pick, err);
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
  return (c->s = oseq_create(sv->e, c->err)) != NULL;
}

void conv_destroy(Conv *c)
{
  oseq_destroy(c->s);
  free(c->held.v);
  memset(c, 0, sizeof *c);
}

bool conv_restart(Conv *c, const Server *sv)
{
  oseq_destroy(c->s);
  c->held.n = 0;
  return (c->s = oseq_create(sv->e, c->err)) != NULL;
}

/* the conversation's state given to the store's sink, or taken from its source (slots.h's save / load) */
static bool conv_save(void *ctx, StateSink sink, void *sink_ctx)
{
  Conv *c = ctx;
  return oseq_save_to(c->s, sink, sink_ctx, c->err);
}

static bool conv_load(void *ctx, uint64_t bytes, StateSource src, void *src_ctx)
{
  Conv *c = ctx;
  return oseq_load_from(c->s, bytes, src, src_ctx, c->err);
}

/* set aside: into RAM where its room holds the state (slots_room), else straight to the folder in pieces - never
   whole in RAM past the room the operator's 8 GiB leaves */
void conv_set_aside(Conv *c, const Server *sv)
{
  if (!sv->slots || !c->held.n)
    return;
  const uint64_t bytes = oseq_state_bytes(c->s);
  char serr[SLOTS_ERR] = "";
  bool kept = false, ok;
  void *st = slots_room(sv->slots, bytes, serr) ? malloc(bytes) : NULL;
  c->err[0] = '\0';
  if (st && oseq_save(c->s, st, bytes, c->err))
    ok = slots_put(sv->slots, c->held.v, c->held.n, st, bytes, &kept, serr); /* (put: st the store's) */
  else
  {
    free(st), serr[0] = '\0';
    ok = slots_put_stream(sv->slots, c->held.v, c->held.n, bytes, conv_save, c, &kept, serr);
  }
  if (!ok)
    out("NOTE slots: a conversation of %u tokens not set aside (%s%s)\n", c->held.n, c->err, serr);
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
  return k == c->held.n || c->held.n - k <= oseq_reach(c->s) ? k : 0;
}

bool conv_continue(Conv *c, const Server *sv, const int32_t *p, uint32_t n)
{
  const uint32_t k = common(&c->held, p, n - 1);
  if (k == c->held.n && k)
    return true;
  if (k && c->held.n - k <= oseq_reach(c->s) && oseq_rollback(c->s, c->held.n - k))
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
  char serr[SLOTS_ERR] = "";
  c->err[0] = '\0';
  if (!slots_take_stream(sv->slots, id, conv_load, c, serr)) /* (a failed load leaves the sequence empty) */
  {
    out("NOTE slots: a kept conversation of %u tokens not taken back (%s%s)\n", len, serr, c->err);
    return conv_restart(c, sv);
  }
  for (uint32_t i = 0; i < len; i++)
    if (!ids_push(&c->held, p[i]))
      return false;
  return true;
}

bool conv_read(Conv *c, const Server *sv, const int32_t *p, uint32_t k, uint32_t n, bool seed)
{
  if (!k)
    return true;
  OsmosSeq **seqs = calloc(k, sizeof *seqs);
  if (!seqs)
    return snprintf(c->err, sizeof c->err, "out of memory"), false;
  for (uint32_t i = 0; i < k; i++)
    seqs[i] = c->s;
  const uint32_t at = c->held.n;
  const bool ok = osmos_batch(sv->e, seqs, p + at, k, c->err);
  free(seqs);
  for (uint32_t i = 0; ok && i < k; i++)
    if (!ids_push(&c->held, p[at + i]))
      return snprintf(c->err, sizeof c->err, "out of memory"), false;
  /* the piece's part of the prompt's last window, from its own step (serve.h) */
  const uint32_t lo = n > sv->window && n - sv->window > at ? n - sv->window : at;
  return ok && (!seed || lo >= at + k || oseq_seed(c->s, (int32_t)lo, (int32_t)(at + k), c->err));
}
