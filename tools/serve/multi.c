/* several conversations at once (maya #587's, the operator's): N lanes, each a conversation; the loop libevt's - stdin's
   lines and the turns are its events, a line outranking a turn. a turn is a round (every answering lane's next token
   in one osmos_batch: a decode step, each token's bits its own) or a piece of one prompt (a step of max_tokens,
   alone: a prompt's step) - never both in one step, so neither changes the other's bits. pieces and rounds take turns;
   a prompt whole in one piece goes ahead of the longer reads (multi.h; the rounds: round.c) */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libevt/runtime/context.h"
#include "multi.h"
#include "osmos/gpu/mm.h"

typedef struct LineEv
{
  Multi *m;
  char *line;
} LineEv;

static void turn(void *scope);

static void queue_turn(Multi *m)
{
  if (m->turn_queued)
    return;
  Context *ctx = context_create(m, turn);
  m->turn_queued = ctx && engine_add(m->evt, ctx, PRIO_TURN) != 0;
  if (ctx && !m->turn_queued)
    context_destroy(ctx);
}

void lane_finish(Multi *m, Lane *ln, const char *why)
{
  out("DONE %lld %lld %u %.1f %.1f %s %lld %lld %u 0 0\n", ln->g.rid, ln->produced, ln->g.n, ln->t1 - ln->t0,
      ln->st == ANSWERING ? now_ms() - ln->t1 : 0.0, why, ln->accepted, ln->drafted, ln->reused);
  free(ln->g.ids);
  ln->g.ids = NULL, ln->st = IDLE;
  (void)m;
}

void lane_fail(Multi *m, Lane *ln)
{
  out("ERR %lld %s\n", ln->g.rid, ln->c.err);
  free(ln->g.ids);
  ln->g.ids = NULL, ln->st = IDLE;
  conv_failed(&ln->c, m->sv);
}

/* the idle lane a request takes: the one whose conversation it continues furthest, else the least recently used
   (its conversation set aside); -1 for none idle */
static int pick_lane(const Multi *m, const Gen *g)
{
  int best = -1, lru = -1;
  uint32_t reach = 0;
  for (int i = 0; i < m->n; i++)
  {
    const Lane *ln = &m->lane[i];
    if (ln->st != IDLE)
      continue;
    const uint32_t r = conv_reach(&ln->c, m->sv, g->ids, g->n);
    if (r > reach)
      reach = r, best = i;
    if (lru < 0 || ln->used < m->lane[lru].used)
      lru = i;
  }
  return best >= 0 ? best : lru;
}

/* a request into lane i: its conversation continued (or another taken back), its prompt read from there */
static void admit(Multi *m, int i, Gen *g)
{
  Lane *ln = &m->lane[i];
  ln->g = *g, ln->st = READING, ln->produced = ln->accepted = ln->drafted = 0, ln->t0 = now_ms(), ln->t1 = ln->t0, ln->used = ++m->clock;
  ln->limit = g->max_new > 0 ? g->max_new : (long long)m->sv->max_ctx - g->n;
  if (!conv_continue(&ln->c, m->sv, g->ids, g->n))
  {
    lane_fail(m, ln);
    return;
  }
  ln->reused = ln->c.held.n;
  if (!oseq_prompt(ln->c.s, conv_prompt_end(&ln->c, g->n, g->mark)))
  {
    snprintf(ln->c.err, sizeof ln->c.err, "the prompt's end not past the conversation's");
    lane_fail(m, ln);
  }
}

static void admit_waiting(Multi *m)
{
  while (m->nwait)
  {
    const int i = pick_lane(m, &m->wait[0]);
    if (i < 0)
      return;
    Gen g = m->wait[0];
    memmove(m->wait, m->wait + 1, (size_t)--m->nwait * sizeof *m->wait);
    admit(m, i, &g);
  }
}

static Lane *lane_of(Multi *m, long long rid)
{
  for (int i = 0; i < m->n; i++)
    if (m->lane[i].st != IDLE && m->lane[i].g.rid == rid)
      return &m->lane[i];
  return NULL;
}

/* STOP: a reading lane's state is not kept (its prompt's end past it), an answering one's is (to its last fed) */
static void stop(Multi *m, bool all, long long rid)
{
  for (int i = 0; i < m->n; i++)
  {
    Lane *ln = &m->lane[i];
    if (ln->st == IDLE || (!all && ln->g.rid != rid))
      continue;
    if (ln->st == READING)
      conv_restart(&ln->c, m->sv);
    lane_finish(m, ln, "cancel");
  }
  for (int k = 0; k < m->nwait;)
    if (all || m->wait[k].rid == rid)
    {
      out("DONE %lld 0 %u 0.0 0.0 cancel 0 0 0 0 0\n", m->wait[k].rid, m->wait[k].n);
      free(m->wait[k].ids);
      memmove(m->wait + k, m->wait + k + 1, (size_t)(--m->nwait - k) * sizeof *m->wait);
    }
    else
      k++;
}

static void gen(Multi *m, char *line)
{
  Gen g;
  char why[256];
  if (!parse_gen(line, &g, why, sizeof why))
  {
    out("ERR %s\n", why);
    return;
  }
  const char *bad = !g.has_rid ? "this engine runs several conversations: GEN needs rid=<id>"
                    : g.spec > 0 && !osmos_block(m->sv->e) ? "spec=1 without a drafter (--dspark)"
                    : g.n >= m->sv->max_ctx ? "a prompt past the context"
                    : lane_of(m, g.rid) ? "a rid already running"
                                        : NULL;
  if (bad)
  {
    out("ERR %lld %s\n", g.rid, bad);
    free(g.ids);
    return;
  }
  g.spec = server_spec(m->sv, &g); /* (resolved: the lane's rounds read it as 0 or 1) */
  const int i = pick_lane(m, &g);
  if (i >= 0)
    admit(m, i, &g);
  else if (m->nwait < m->capwait || (m->wait = realloc(m->wait, (size_t)(m->capwait = m->capwait ? 2 * m->capwait : 16) *
                                                                    sizeof *m->wait)))
    m->wait[m->nwait++] = g;
  else
    out("ERR %lld out of memory\n", g.rid), free(g.ids);
}

static void on_line(void *scope)
{
  LineEv *ev = scope;
  Multi *m = ev->m;
  char *line = ev->line;
  free(ev);
  if (!strcmp(line, "QUIT"))
  {
    stop(m, true, 0);
    engine_stop(m->evt);
  }
  else if (!strcmp(line, "STOP"))
    stop(m, true, 0);
  else if (!strncmp(line, "STOP ", 5))
    stop(m, false, atoll(line + 5));
  else if (!strncmp(line, "GEN ", 4))
    gen(m, line);
  else if (*line)
    out("ERR unsupported command '%.32s' (GEN, STOP and QUIT)\n", line);
  free(line);
  queue_turn(m);
}

/* a piece: the next step of one prompt - one whole in a piece first (a short question ahead of long reads), else in
   turn; its last piece picks the answer's first token */
static void piece(Multi *m)
{
  Server *sv = m->sv;
  int pick = -1;
  for (int i = 0; i < m->n && pick < 0; i++)
    if (m->lane[i].st == READING && m->lane[i].g.n - m->lane[i].c.held.n <= sv->max_tokens)
      pick = i;
  for (int s = 1; s <= m->n && pick < 0; s++)
    if (m->lane[(m->rr + s) % m->n].st == READING)
      pick = m->rr = (m->rr + s) % m->n;
  if (pick < 0)
    return;
  Lane *ln = &m->lane[pick];
  const uint32_t n = ln->g.n, k = n - ln->c.held.n < sv->max_tokens ? n - ln->c.held.n : sv->max_tokens;
  uint32_t row = 0;
  if (!conv_read(&ln->c, sv, ln->g.ids, k, n, ln->g.mark, ln->g.spec, &row))
  {
    lane_fail(m, ln);
    return;
  }
  const double ms = now_ms() - ln->t0;
  out("PP %lld %u %u %.1f %.1f %u\n", ln->g.rid, ln->c.held.n, n, ms, (ln->c.held.n - ln->reused) / (ms / 1e3),
      ln->reused);
  if (ln->c.held.n < n)
    return;
  if (!osmos_greedy(sv->e, row, 1, &ln->t, ln->c.err) || !server_answer(sv, &ln->c.held, NULL, 0, row, &ln->t,
                                                                          ln->c.err))
  {
    lane_fail(m, ln);
    return;
  }
  ln->st = ANSWERING, ln->t1 = now_ms();
}

static void turn(void *scope)
{
  Multi *m = scope;
  m->turn_queued = false;
  bool answering = false, reading = false;
  for (int i = 0; i < m->n; i++)
    answering |= m->lane[i].st == ANSWERING, reading |= m->lane[i].st == READING;
  if (answering && (!reading || m->last_piece))
    round_all(m), m->last_piece = false;
  else if (reading)
    piece(m), m->last_piece = true;
  admit_waiting(m);
  for (int i = 0; i < m->n; i++)
    if (m->lane[i].st != IDLE)
    {
      queue_turn(m);
      return;
    }
}

/* stdin's lines, each an event; its end a QUIT */
static void *reader(void *arg)
{
  Multi *m = arg;
  char *buf = NULL;
  size_t cap = 0;
  for (ssize_t len; (len = getline(&buf, &cap, stdin)) >= 0;)
  {
    while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
      buf[--len] = 0;
    LineEv *ev = malloc(sizeof *ev);
    Context *ctx = ev ? context_create(ev, on_line) : NULL;
    if (!ctx || !(ev->m = m, ev->line = strdup(buf)) || !engine_add(m->evt, ctx, PRIO_LINE))
    {
      free(ev ? ev->line : NULL), free(ev), context_destroy(ctx);
      break; /* (stopped: QUIT came) */
    }
    if (!strcmp(buf, "QUIT"))
      break;
  }
  if (feof(stdin))
  {
    LineEv *ev = malloc(sizeof *ev);
    Context *ctx = ev ? context_create(ev, on_line) : NULL;
    if (ctx && (ev->m = m, ev->line = strdup("QUIT")) && !engine_add(m->evt, ctx, PRIO_LINE))
      free(ev->line), free(ev), context_destroy(ctx);
  }
  free(buf);
  return NULL;
}

int serve_multi(Server *sv, Conv *lanes, int n)
{
  Multi m = {sv, calloc((size_t)n, sizeof(Lane)), n, engine_create(NULL, NULL, 0), NULL, 0, 0, false, false, 0, 0};
  if (!m.lane || !m.evt || (uint32_t)n > sv->round_tokens)
    return fprintf(stderr, "osmos-serve: the lanes could not be made (at most %u: a round's tokens)\n",
                   sv->round_tokens), 2;
  for (int i = 0; i < n; i++)
    m.lane[i].c = lanes[i];
  out("READY %u stop multi=%d\n", sv->max_ctx, n);
  pthread_t th;
  if (pthread_create(&th, NULL, reader, &m) != 0)
    return fprintf(stderr, "osmos-serve: the reader could not start\n"), 2;
  engine_run(m.evt);
  pthread_join(th, NULL); /* (it ends after the QUIT it read, or stdin's end it posted as one) */
  for (int i = 0; i < n; i++)
    lanes[i] = m.lane[i].c;
  engine_destroy(m.evt);
  free(m.lane);
  free(m.wait);
  return 0;
}
