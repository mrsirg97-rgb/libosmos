/* one conversation over stdin: a GEN at a time, a STOP or QUIT acted on between its steps and tokens (serve.h) */
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "serve.h"
#include "osmos/gpu/mm.h"

#define QUEUED 64 /* lines that came while a request ran, kept for after it */

typedef struct Lines
{
  char *buf;
  size_t len, cap;
  char *queue[QUEUED];
  int nq;
  bool eof;
} Lines;

typedef struct Ctl
{
  bool cancel, quit;
} Ctl;

static char *take_line(Lines *l)
{
  char *nl = l->len ? memchr(l->buf, '\n', l->len) : NULL;
  if (!nl)
    return NULL;
  const size_t n = (size_t)(nl - l->buf);
  char *line = malloc(n + 1);
  if (!line)
    return NULL;
  memcpy(line, l->buf, n), line[n] = 0;
  if (n && line[n - 1] == '\r')
    line[n - 1] = 0;
  memmove(l->buf, nl + 1, l->len - n - 1), l->len -= n + 1;
  return line;
}

/* stdin read once (waiting up to wait_ms; -1 forever); false at its end or out of memory */
static bool fill(Lines *l, int wait_ms)
{
  struct pollfd p = {0, POLLIN, 0};
  if (poll(&p, 1, wait_ms) <= 0)
    return true;
  if (l->len + 65536 > l->cap)
  {
    const size_t cap = l->cap ? 2 * l->cap : 1u << 20;
    char *b = realloc(l->buf, cap);
    if (!b)
      return l->eof = true, false;
    l->buf = b, l->cap = cap;
  }
  const ssize_t n = read(0, l->buf + l->len, l->cap - l->len);
  if (n <= 0)
    return l->eof = true, false;
  l->len += (size_t)n;
  return true;
}

static char *next_line(Lines *l)
{
  if (l->nq)
  {
    char *s = l->queue[0];
    memmove(l->queue, l->queue + 1, (size_t)--l->nq * sizeof *l->queue);
    return s;
  }
  for (;;)
  {
    char *s = take_line(l);
    if (s || l->eof || !fill(l, -1))
      return s ? s : take_line(l);
  }
}

/* the lines that came while a request runs: STOP / QUIT acted on, the others queued */
static void pump(Lines *l, Ctl *c)
{
  while (!l->eof && fill(l, 0))
  {
    bool any = false;
    for (char *s; (s = take_line(l)); any = true)
      if (!strcmp(s, "QUIT"))
        c->cancel = c->quit = true, free(s);
      else if (!strcmp(s, "STOP") || !strncmp(s, "STOP ", 5))
        c->cancel = true, free(s);
      else if (l->nq < QUEUED)
        l->queue[l->nq++] = s;
      else
        out("ERR busy: a line dropped (%d queued)\n", l->nq), free(s);
    struct pollfd p = {0, POLLIN, 0};
    if (!any || poll(&p, 1, 0) <= 0)
      break;
  }
  if (l->eof)
    c->cancel = c->quit = true;
}

typedef struct Answer
{
  long long produced, accepted, drafted, limit;
  const char *finish;
} Answer;

/* a drafted round from t: t and its drafts verified, the accepted printed (to an end token, the limit or a STOP), the
   state rolled back to all but the last printed token. *t the next; returns 1 to go on, 0 at the end, -1 failed */
static int round_drafted(Server *sv, Conv *c, int32_t *t, Answer *a, Lines *l, Ctl *ctl)
{
  int32_t dr[OSMOS_MAX_BLOCK], picks[OSMOS_MAX_BLOCK + 1], toks[OSMOS_MAX_BLOCK + 1];
  float cf[OSMOS_MAX_BLOCK];
  int k = oseq_draft(c->s, *t, OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, dr, cf, c->err);
  if (k < 0)
    return -1;
  const long long room = a->limit - a->produced;
  if (k > room - 1)
    k = room > 1 ? (int)(room - 1) : 0;
  const int32_t pos = oseq_position(c->s);
  if ((uint32_t)pos + (uint32_t)k + 2 >= sv->max_ctx)
    k = 0;
  toks[0] = *t;
  memcpy(toks + 1, dr, (size_t)k * sizeof *dr);
  OsmosSeq *ss[OSMOS_MAX_BLOCK + 1];
  for (int i = 0; i <= k; i++)
    ss[i] = c->s;
  if (!osmos_batch(sv->e, ss, toks, (uint32_t)k + 1, c->err) ||
      !osmos_greedy(sv->e, 0, (uint32_t)k + 1, picks, c->err))
    return -1;
  for (int i = 0; i <= k; i++) /* row i: the held, then toks[0 .. i] */
    if (!server_answer(sv, &c->held, toks, (uint32_t)i + 1, (uint32_t)i, &picks[i], c->err))
      return -1;
  int m = 0;
  while (m < k && picks[m] == dr[m])
    m++;
  a->drafted += k, a->accepted += m;
  int printed = 0;
  bool end = false;
  for (int i = 0; i <= m && !end; i++, printed++)
  {
    out("T %d\n", toks[i]), a->produced++;
    if (server_eos(sv, toks[i]))
      a->finish = "stop", end = true;
    else if (a->produced >= a->limit)
      end = true;
  }
  pump(l, ctl);
  if (ctl->cancel && !end)
    a->finish = "cancel", end = true;
  const int keep = end ? printed - 1 : m + 1;
  if (!oseq_rollback(c->s, (uint32_t)(k + 1 - keep)) || (keep && !oseq_seed(c->s, pos, pos + keep, c->err)))
    return -1;
  for (int i = 0; i < keep; i++)
    if (!ids_push(&c->held, toks[i]))
      return -1;
  *t = picks[m];
  return !end && (uint32_t)oseq_position(c->s) + 1 < sv->max_ctx;
}

/* a greedy token: printed, then fed (the last printed never is: the next turn's prompt holds it) */
static int step_greedy(Server *sv, Conv *c, int32_t *t, Answer *a, Lines *l, Ctl *ctl)
{
  out("T %d\n", *t), a->produced++;
  if (server_eos(sv, *t))
    return a->finish = "stop", 0;
  pump(l, ctl);
  if (ctl->cancel)
    return a->finish = "cancel", 0;
  if (a->produced >= a->limit || (uint32_t)oseq_position(c->s) + 1 >= sv->max_ctx)
    return 0;
  if (!osmos_batch(sv->e, &c->s, t, 1, c->err) || !ids_push(&c->held, *t) || !osmos_greedy(sv->e, 0, 1, t, c->err) ||
      !server_answer(sv, &c->held, NULL, 0, 0, t, c->err))
    return -1;
  return 1;
}

static void fail(Server *sv, Conv *c)
{
  out("ERR %s\n", c->err);
  conv_restart(c, sv);
}

/* one request: the prompt's rest a step at a time (PP), the answer (T), DONE */
static void generate(Server *sv, Conv *c, const Gen *g, Lines *l, Ctl *ctl)
{
  const double t0 = now_ms();
  const uint32_t n = g->n;
  if (n >= sv->max_ctx)
  {
    out("ERR a prompt of %u tokens: the context is %u\n", n, sv->max_ctx);
    return;
  }
  const bool spec = server_spec(sv, g);
  if (spec && !osmos_block(sv->e))
  {
    out("ERR spec=1 without a drafter (--dspark)\n");
    return;
  }
  if (!conv_continue(c, sv, g->ids, n))
  {
    out("ERR %s\n", c->err);
    return;
  }
  const uint32_t reused = c->held.n;
  if (!oseq_prompt(c->s, n))
  {
    snprintf(c->err, sizeof c->err, "the prompt's end not past the conversation's");
    fail(sv, c);
    return;
  }
  uint32_t row = 0;
  while (c->held.n < n)
  {
    const uint32_t k = n - c->held.n < sv->max_tokens ? n - c->held.n : sv->max_tokens;
    if (!conv_read(c, sv, g->ids, k, n, spec))
    {
      fail(sv, c);
      return;
    }
    row = k - 1;
    const double ms = now_ms() - t0;
    out("PP %u %u %.1f %.1f %u\n", c->held.n, n, ms, (c->held.n - reused) / (ms / 1e3), reused);
    pump(l, ctl);
    if (ctl->cancel && c->held.n < n) /* a prompt read part-way: its end past the state, the state not kept */
    {
      conv_restart(c, sv);
      out("DONE 0 %u %.1f 0.0 cancel 0 0 %u 0 0\n", n, now_ms() - t0, reused);
      return;
    }
  }
  int32_t t;
  if (!osmos_greedy(sv->e, row, 1, &t, c->err) || !server_answer(sv, &c->held, NULL, 0, row, &t, c->err))
  {
    fail(sv, c);
    return;
  }
  const double t1 = now_ms();
  Answer a = {0, 0, 0, g->max_new > 0 ? g->max_new : (long long)sv->max_ctx - n, "length"};
  int go;
  do
    go = spec ? round_drafted(sv, c, &t, &a, l, ctl) : step_greedy(sv, c, &t, &a, l, ctl);
  while (go > 0);
  if (go < 0)
  {
    fail(sv, c);
    return;
  }
  out("DONE %lld %u %.1f %.1f %s %lld %lld %u 0 0\n", a.produced, n, t1 - t0, now_ms() - t1, a.finish, a.accepted,
      a.drafted, reused);
}

int serve_single(Server *sv, Conv *c)
{
  out("READY %u stop\n", sv->max_ctx);
  Lines l = {0};
  Ctl ctl = {0};
  for (char *line; !ctl.quit && (line = next_line(&l)); free(line))
  {
    Gen g;
    char why[256];
    if (!strcmp(line, "QUIT"))
      ctl.quit = true;
    else if (!strncmp(line, "GEN ", 4) && !parse_gen(line, &g, why, sizeof why))
      out("ERR %s\n", why);
    else if (!strncmp(line, "GEN ", 4))
    {
      ctl.cancel = false;
      generate(sv, c, &g, &l, &ctl);
      free(g.ids);
    }
    else if (*line && strcmp(line, "STOP") && strncmp(line, "STOP ", 5))
      out("ERR unsupported command '%.32s' (GEN, STOP and QUIT)\n", line);
  }
  for (int i = 0; i < l.nq; i++)
    free(l.queue[i]);
  free(l.buf);
  return 0;
}
