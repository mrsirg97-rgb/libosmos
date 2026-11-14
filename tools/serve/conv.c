/* a conversation: its sequence, the tokens it holds, set aside and taken back (serve.h) */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "osmos/run/gpu.h"
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
    else if (!strcmp(tok, "mark"))
      g->mark = (uint32_t)strtoul(eq + 1, NULL, 10);
    else if (strcmp(tok, "seed") != 0) /* (greedy: a seed changes nothing) */
      return snprintf(why, wn, "the key '%s' is not this engine's (greedy: rid, spec, seed, mark)", tok), false;
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
  free(c->held.v), free(c->prompt.buf), free(c->user.buf);
  memset(c, 0, sizeof *c);
}

bool conv_restart(Conv *c, const Server *sv)
{
  oseq_destroy(c->s);
  c->held.n = 0, c->prompt.n = 0, c->user.n = 0; /* (the points' memory kept for the next) */
  return (c->s = oseq_create(sv->e, c->err)) != NULL;
}

/* the sequence's mark kept as pt at its held tokens (serve.h's Point). none (no memory for it, or the mark refused):
   the turn goes on without it - a later turn reads more, never other bits */
static void take(Conv *c, Point *pt, const char *what)
{
  pt->n = 0;
  const uint64_t bytes = oseq_mark_bytes(c->s);
  if (bytes > pt->cap)
  {
    void *q = realloc(pt->buf, bytes);
    if (!q)
    {
      out("NOTE a conversation's %s point (%.2f GiB) not kept: no memory for it\n", what, bytes / (double)(1ull << 30));
      return;
    }
    pt->buf = q, pt->cap = bytes;
  }
  char err[OSMOS_ERR] = "";
  if (oseq_mark_save(c->s, pt->buf, bytes, err))
    pt->bytes = bytes, pt->n = c->held.n;
  else
    out("NOTE a conversation's %s point not kept (%s)\n", what, err);
}

/* the points past held's first n tokens no longer the sequence's (a rollback, a point gone back to) */
static void drop_past(Conv *c, uint32_t n)
{
  if (c->prompt.n > n)
    c->prompt.n = 0;
  if (c->user.n > n)
    c->user.n = 0;
}

/* back to pt: the sequence as it was there, held its first pt->n tokens. a load that fails: a fresh sequence (the
   state on the GPU no longer known) */
static bool go_back(Conv *c, const Server *sv, Point *pt)
{
  if (oseq_mark_load(c->s, pt->buf, pt->bytes, c->err))
  {
    c->held.n = pt->n;
    drop_past(c, pt->n);
    return true;
  }
  out("NOTE a conversation's point not loaded back (%s): read again from its start\n", c->err);
  return conv_restart(c, sv);
}

/* a conversation past its prompt's end on decoded tokens (a family that reads apart): back to its prompt's point - a
   turn's earlier answer (decoded, not a prompt's bits) read again from there, never continued on, so a turn's
   answer is the bits of its prompt read whole (0.7.1). with no point (its memory refused) the decoded state kept, as
   0.7.0 kept it: that turn's answer a continued one's bits, never the whole conversation read again for it */
static bool back_to_prompt(Conv *c, const Server *sv)
{
  if (!osmos_reads_apart(sv->e) || !c->prompt.n || c->held.n <= c->prompt.n)
    return true;
  return go_back(c, sv, &c->prompt);
}

/* after a request's failure (its ERR written): a new sequence for the conversation, or - the GPU's state lost (a
   kernel's fault: every call after it fails the same) or no sequence to be had - FATAL and exit, so osmos-api's
   "the engine exited" ends every request instead of each one answering ERR forever */
void conv_failed(Conv *c, const Server *sv)
{
  const bool alive = gpu_alive();
  if (alive && conv_restart(c, sv))
    return;
  out("FATAL %s\n", alive ? c->err : "the GPU's state lost after an error: the engine cannot continue");
  exit(1);
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
   whole in RAM past the room the operator's 8 GiB leaves. kept as its prompt's point left it (every family, 0.7.5):
   a decoded answer is what a client writes back otherwise (its reasoning dropped, its calls' arguments written
   again), so a kept answer would be no prefix of the turn that takes it back */
void conv_set_aside(Conv *c, const Server *sv)
{
  if (!sv->slots || !c->held.n || (c->prompt.n && c->held.n > c->prompt.n && !go_back(c, sv, &c->prompt)) ||
      !c->held.n || !slots_keeps(sv->slots, c->held.n))
    return; /* not kept: nothing saved, no room made (0.7.1: kept ones moved for it, then it was refused) */
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

/* the furthest of the conversation's places p continues: all it holds, a rollback's reach behind, a point */
static uint32_t furthest(const Conv *c, const Ids *held, uint32_t k, uint32_t reach)
{
  if (k == held->n || held->n - k <= reach)
    return k;
  const uint32_t a = c->prompt.n <= k ? c->prompt.n : 0, b = c->user.n <= k ? c->user.n : 0;
  return a > b ? a : b;
}

uint32_t conv_reach(const Conv *c, const Server *sv, const int32_t *p, uint32_t n)
{
  /* a family that reads apart reuses no decoded token: its reach the prompt's point at most */
  const Ids held = {c->held.v, osmos_reads_apart(sv->e) && c->prompt.n && c->held.n > c->prompt.n ? c->prompt.n
                                                                                                   : c->held.n, 0};
  const uint32_t k = common(&held, p, n - 1); /* (one token left to read: its logits the answer's first) */
  return furthest(c, &held, k, held.n == c->held.n ? oseq_reach(c->s) : 0);
}

bool conv_continue(Conv *c, const Server *sv, const int32_t *p, uint32_t n)
{
  if (!back_to_prompt(c, sv)) /* a decoded answer read again, never continued on (0.7.1) */
    return false;
  const uint32_t k = common(&c->held, p, n - 1);
  if (k == c->held.n && k)
    return true;
  if (k && c->held.n - k <= oseq_reach(c->s) && oseq_rollback(c->s, c->held.n - k))
  {
    c->held.n = k;
    drop_past(c, k);
    return true;
  }
  /* a point p continues (0.7.5): the prompt's, past a decoded answer the client wrote back otherwise; the last user
     turn's, before reasoning the template now drops */
  Point *pt = c->prompt.n && c->prompt.n <= k ? &c->prompt : c->user.n && c->user.n <= k ? &c->user : NULL;
  if (pt)
    return go_back(c, sv, pt);
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
      return snprintf(c->err, sizeof c->err, "out of memory (a kept conversation's tokens)"), false;
  return true;
}

/* k tokens of p (from held.n) through one osmos_read; the drafter's rings seeded with the piece's part of the last
   window of the prompt's n tokens (serve.h) */
static bool read_piece(Conv *c, const Server *sv, const int32_t *p, uint32_t k, uint32_t n, bool seed)
{
  if (!k)
    return true;
  OsmosSeq **seqs = calloc(k, sizeof *seqs);
  if (!seqs)
    return snprintf(c->err, sizeof c->err, "out of memory"), false;
  for (uint32_t i = 0; i < k; i++)
    seqs[i] = c->s;
  const uint32_t at = c->held.n;
  const bool ok = osmos_read(sv->e, seqs, p + at, k, c->err); /* a prompt's: its large steps' math at any size */
  free(seqs);
  for (uint32_t i = 0; ok && i < k; i++)
    if (!ids_push(&c->held, p[at + i]))
      return snprintf(c->err, sizeof c->err, "out of memory"), false;
  const uint32_t lo = n > sv->window && n - sv->window > at ? n - sv->window : at;
  return ok && (!seed || lo >= at + k || oseq_seed(c->s, (int32_t)lo, (int32_t)(at + k), c->err));
}

/* the points a prompt keeps (0.7.5): the last user turn's (mark, osmos-api's) and the prompt's one token short of
   its end - the generation prompt's last token (DeepSeek's <think>, Qwen's newline after it) is one a later turn
   renders otherwise. a piece stops at each, the point kept, then goes on: a prompt in pieces is the bits of one read
   whole (engine.h) */
/* the prompt's end as its first piece reads it: mark's when a mark lies ahead - the state there what a prompt ending
   at mark leaves (DeepSeek's decoder skip runs its last layers for the tokens within reach of the end: a point taken
   part-way through a longer prompt would lack them), the prompt's own past it (conv_read) */
uint32_t conv_prompt_end(const Conv *c, uint32_t n, uint32_t mark)
{
  return mark > c->held.n && mark < n - 1 ? mark : n;
}

bool conv_read(Conv *c, const Server *sv, const int32_t *p, uint32_t k, uint32_t n, uint32_t mark, bool seed,
               uint32_t *row)
{
  if (mark && mark == c->prompt.n && mark != c->user.n) /* the mark the last prompt's point (a tool loop's first
                                                           request ended there): kept as the user turn's */
  {
    const Point t = c->user;
    c->user = c->prompt, c->prompt = t, c->prompt.n = 0;
  }
  const uint32_t end = c->held.n + k, stops[2] = {mark < n - 1 ? mark : 0, n - 1};
  uint32_t len = 0; /* the last read's tokens */
  for (int i = 0; i < 2; i++)
    if (stops[i] > c->held.n && stops[i] <= end)
    {
      len = stops[i] - c->held.n;
      if (!read_piece(c, sv, p, len, n, seed))
        return false;
      take(c, i ? &c->prompt : &c->user, i ? "prompt's" : "last user turn's");
      if (!i && !oseq_prompt(c->s, n)) /* past the mark: the prompt's own end */
        return snprintf(c->err, sizeof c->err, "the prompt's end not past its mark"), false;
    }
  if (c->held.n < end)
  {
    len = end - c->held.n;
    if (!read_piece(c, sv, p, len, n, seed))
      return false;
  }
  *row = len - 1; /* the last read's last token: its logits the next token's */
  return true;
}
