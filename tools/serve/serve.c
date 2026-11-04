/* substrate-serve: the engine over stdin / stdout, maya's DeepSeek serve protocol (its src/ds4/ds4_main.cpp), so its
   harnesses (tools/ds4/slot_keep_test.py, parallel_test.py, agent_bench.py) drive this engine as they drive maya's.

     substrate-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] [--max-context N]
                     [--budget GiB] [--max-tokens N] [--eos id,id] [--serve]

   in:  GEN <max_new> [rid=<r>] [spec=<0|1>] [seed=<s>] <id,id,...>   STOP [<rid>]   QUIT
   out: READY <max_ctx> stop / T <token> / PP <done> <total> <ms> <tok/s> <reused> /
        DONE <produced> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <drafted> <reused> <hits> <lookups> /
        ERR <reason> / NOTE <text>

   one conversation in its sequence; a request that continues it reads only the rest of its prompt (a conversation
   that ran a few tokens past it - a drafted answer cut by max_new - rolled back to their common prefix). with
   SUBSTRATE_SLOT_KEEP=1 (DESIGN.md, "serving"; maya #356) a request that does not continue it sets it aside first and
   takes back the kept conversation its prompt continues: SUBSTRATE_SLOT_CAP (4) in RAM, past it SUBSTRATE_SLOT_DIR
   (<the model's folder>/slots) under SUBSTRATE_SLOT_GB (16), SUBSTRATE_SLOT_MIN (1024) tokens at least,
   SUBSTRATE_SLOT_DAYS (7). greedy: maya's sampling, penalty and thinking keys are refused, named */
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "substrate/gpu/mm.h"
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"
#include "substrate/run/drafter.h"
#include "substrate/run/forward.h"
#include "substrate/serve/slots.h"

#define MAX_EOS 16
#define CONF_MIN 0.5f /* the drafter's: maya's defaults (as dbench) */
#define CUM_MIN 0.5f
#define LINE_MAX_BYTES (64u << 20) /* a GEN of a 1M-token prompt: ~7 bytes a token */

static double now_ms(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void out(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void out(const char *fmt, ...)
{
  va_list a;
  va_start(a, fmt);
  vprintf(fmt, a);
  va_end(a);
  fflush(stdout);
}

/* ---- stdin's lines: read whole, a few queued while a request runs (a STOP or QUIT acted on at once) ---- */

typedef struct Lines
{
  char *buf;
  size_t len, cap;
  char *queue[64];
  int nq;
  bool eof;
} Lines;

/* the buffer's first whole line, taken (malloc'd), or NULL */
static char *take_line(Lines *l)
{
  char *nl = memchr(l->buf, '\n', l->len);
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

/* stdin read once (waiting up to wait_ms; -1 forever) into the buffer; false at its end */
static bool fill(Lines *l, int wait_ms)
{
  struct pollfd p = {0, POLLIN, 0};
  if (poll(&p, 1, wait_ms) <= 0)
    return true;
  if (l->len + 65536 > l->cap)
  {
    const size_t cap = l->cap ? 2 * l->cap : 1u << 20;
    if (cap > LINE_MAX_BYTES * 2 || !(l->buf = realloc(l->buf, cap)))
      return false;
    l->cap = cap;
  }
  const ssize_t n = read(0, l->buf + l->len, l->cap - l->len);
  if (n <= 0)
    return l->eof = true, false;
  l->len += (size_t)n;
  return true;
}

/* the next line: a queued one first, else read (blocking); NULL at stdin's end */
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
    if (s)
      return s;
    if (l->eof || !fill(l, -1))
      return take_line(l);
  }
}

typedef struct Ctl
{
  bool cancel, quit;
} Ctl;

/* while a request runs: the lines that came, a STOP (or QUIT) acted on, the others queued for after it */
static void pump(Lines *l, Ctl *c)
{
  while (fill(l, 0) && !l->eof)
  {
    char *s = take_line(l);
    if (!s)
      break;
    do
    {
      if (!strcmp(s, "QUIT"))
        c->cancel = c->quit = true, free(s);
      else if (!strcmp(s, "STOP") || !strncmp(s, "STOP ", 5))
        c->cancel = true, free(s);
      else if (l->nq < (int)(sizeof l->queue / sizeof *l->queue))
        l->queue[l->nq++] = s;
      else
        out("ERR busy: the line dropped (%d queued)\n", l->nq), free(s);
    } while ((s = take_line(l)));
    struct pollfd p = {0, POLLIN, 0};
    if (poll(&p, 1, 0) <= 0)
      break;
  }
  if (l->eof)
    c->cancel = c->quit = true;
}

/* ---- the engine and its one conversation ---- */

typedef struct Ids
{
  int32_t *v;
  uint32_t n, cap;
} Ids;

static bool ids_push(Ids *a, int32_t x)
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

typedef struct Engine
{
  Forward *f;
  Drafter *d;
  Slots *slots;
  Seq *s;
  Ids held; /* the tokens the sequence holds: seq_position(s) of them */
  uint32_t max_tokens, max_ctx;
  int32_t eos[MAX_EOS];
  int n_eos;
  char err[FORWARD_ERR];
} Engine;

static bool is_eos(const Engine *e, int32_t t)
{
  for (int i = 0; i < e->n_eos; i++)
    if (e->eos[i] == t)
      return true;
  return false;
}

/* the sequence emptied: a new one (its rows rewritten before they are read) */
static bool restart(Engine *e)
{
  seq_destroy(e->s);
  e->held.n = 0;
  return (e->s = seq_create(e->f, e->err)) != NULL;
}

/* the conversation in the sequence set aside (slots on, long enough) */
static void set_aside(Engine *e)
{
  if (!e->slots || !e->held.n)
    return;
  const uint64_t bytes = seq_state_bytes(e->s);
  void *st = malloc(bytes);
  char serr[SLOTS_ERR] = "";
  bool kept = false;
  if (!st || !seq_save(e->s, st, bytes, e->err) || !slots_put(e->slots, e->held.v, e->held.n, st, bytes, &kept, serr))
    out("NOTE slots: a conversation of %u tokens not set aside (%s%s)\n", e->held.n, st ? e->err : "out of memory",
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

/* the sequence made a prefix of prompt p (n tokens, at least one left to read): what it holds continued, rolled back
   to the common part (within seq_rollback's reach), or set aside and the longest kept prefix taken back; else
   emptied. the tokens it then holds: e->held.n (the reused) */
static bool continue_from(Engine *e, const int32_t *p, uint32_t n)
{
  const uint32_t c = common(&e->held, p, n - 1); /* (one token left to read: its logits the answer's first) */
  if (c == e->held.n && c)
    return true;
  const uint32_t back = e->held.n - c;
  if (c && back <= SEQ_MAX_BACK && seq_rollback(e->s, back))
  {
    e->held.n = c;
    return true;
  }
  set_aside(e);
  uint32_t len = 0;
  const int64_t id = e->slots ? slots_find(e->slots, p, n - 1, 0, &len) : -1;
  if (!restart(e))
    return false;
  if (id < 0)
    return true;
  void *st = NULL;
  uint64_t bytes = 0;
  char serr[SLOTS_ERR] = "";
  if (!slots_take(e->slots, id, &st, &bytes, serr) || !seq_load(e->s, st, bytes, e->err))
  {
    out("NOTE slots: a kept conversation of %u tokens not taken back (%s%s)\n", len, serr, st ? e->err : "");
    free(st);
    return restart(e);
  }
  free(st);
  for (uint32_t i = 0; i < len; i++)
    if (!ids_push(&e->held, p[i]))
      return false;
  return true;
}

/* a GEN's fields: max_new, its keys, its ids (malloc'd); false with the reason in why */
typedef struct Gen
{
  long long max_new;
  int spec;
  int32_t *ids;
  uint32_t n;
} Gen;

static bool parse_gen(char *line, Gen *g, char *why, size_t wn)
{
  memset(g, 0, sizeof *g);
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
    else if (!strcmp(tok, "rid") || !strcmp(tok, "seed"))
      continue; /* (one conversation: its rid not needed; greedy: no seed) */
    else
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

/* one request: the prompt's rest read a step at a time (PP), then the answer (T), then DONE */
static void generate(Engine *e, Gen *g, Lines *l, Ctl *c)
{
  const double t0 = now_ms();
  const uint32_t n = g->n;
  if (n >= e->max_ctx)
  {
    out("ERR a prompt of %u tokens: the context is %u\n", n, e->max_ctx);
    return;
  }
  if (g->spec && !e->d)
  {
    out("ERR spec=1 without a drafter (--dspark)\n");
    return;
  }
  if (!continue_from(e, g->ids, n))
  {
    out("ERR %s\n", e->err);
    return;
  }
  const uint32_t reused = e->held.n;
  if (!seq_prompt(e->s, n))
  {
    out("ERR the prompt's end not past the conversation's\n");
    restart(e);
    return;
  }
  /* the rest of the prompt, a step at a time */
  uint32_t row = 0;
  for (uint32_t at = reused; at < n;)
  {
    const uint32_t k = n - at < e->max_tokens ? n - at : e->max_tokens;
    Seq **seqs = malloc((size_t)k * sizeof *seqs);
    if (!seqs)
    {
      out("ERR out of memory\n");
      restart(e);
      return;
    }
    for (uint32_t i = 0; i < k; i++)
      seqs[i] = e->s;
    const bool ok = forward_batch(e->f, seqs, g->ids + at, k, e->err);
    free(seqs);
    if (!ok)
    {
      out("ERR %s\n", e->err);
      restart(e);
      return;
    }
    for (uint32_t i = 0; i < k; i++)
      ids_push(&e->held, g->ids[at + i]);
    at += k, row = (k - 1) % MM_MAX_TOKENS;
    const double ms = now_ms() - t0;
    out("PP %u %u %.1f %.1f %u\n", at, n, ms, (at - reused) / (ms / 1e3), reused);
    pump(l, c);
    if (c->cancel && at < n) /* a prompt read part-way: its end past the state, so the state is not kept */
    {
      restart(e);
      out("DONE 0 %u %.1f 0.0 cancel 0 0 %u 0 0\n", n, now_ms() - t0, reused);
      return;
    }
  }
  if (e->d && !drafter_seed(e->d, e->s, (int32_t)(n - (n - reused < 128 ? n - reused : 128)), (int32_t)n, e->err))
  {
    out("ERR %s\n", e->err);
    restart(e);
    return;
  }
  int32_t t;
  if (!forward_greedy(e->f, row, 1, &t, e->err))
  {
    out("ERR %s\n", e->err);
    restart(e);
    return;
  }
  const double t1 = now_ms();
  long long produced = 0, accepted = 0, drafted = 0;
  const long long limit = g->max_new > 0 ? g->max_new : (long long)e->max_ctx - n;
  const char *finish = "length";
  bool failed = false;
  while (!failed)
  {
    if (!g->spec) /* a token: printed, then fed (the last printed one never is: the next turn's prompt holds it) */
    {
      out("T %d\n", t), produced++;
      if (is_eos(e, t))
      {
        finish = "stop";
        break;
      }
      pump(l, c);
      if (c->cancel)
      {
        finish = "cancel";
        break;
      }
      if (produced >= limit || (uint32_t)seq_position(e->s) + 1 >= e->max_ctx)
        break;
      if (!forward_step(e->f, e->s, t, e->err) || !ids_push(&e->held, t) ||
          !forward_greedy(e->f, 0, 1, &t, e->err))
        failed = true;
      continue;
    }
    /* a round: t and its drafts verified, the accepted printed; the state then rolled back to all but the last
       printed token (a cut within the accepted ones: none of them past it) */
    int32_t dr[DRAFTER_BLOCK], picks[DRAFTER_BLOCK + 1], toks[DRAFTER_BLOCK + 1];
    float cf[DRAFTER_BLOCK];
    const long long room = limit - produced;
    int k = drafter_draft(e->d, e->s, t, CONF_MIN, CUM_MIN, dr, cf, e->err);
    if (k < 0)
    {
      failed = true;
      break;
    }
    if (k > room - 1)
      k = room - 1 > 0 ? (int)(room - 1) : 0;
    if ((uint32_t)seq_position(e->s) + (uint32_t)k + 2 >= e->max_ctx)
      k = 0;
    toks[0] = t;
    memcpy(toks + 1, dr, (size_t)k * sizeof *dr);
    Seq *ss[DRAFTER_BLOCK + 1];
    for (int i = 0; i <= k; i++)
      ss[i] = e->s;
    const int32_t pos = seq_position(e->s);
    if (!forward_batch(e->f, ss, toks, (uint32_t)k + 1, e->err) ||
        !forward_greedy(e->f, 0, (uint32_t)k + 1, picks, e->err))
    {
      failed = true;
      break;
    }
    int m = 0;
    while (m < k && picks[m] == dr[m])
      m++;
    drafted += k, accepted += m;
    /* printed: t, then the accepted drafts, stopping at an end token, the limit or a STOP */
    int printed = 0;
    bool end = false;
    for (int i = 0; i <= m && !end; i++)
    {
      const int32_t x = toks[i];
      out("T %d\n", x), produced++, printed++;
      if (is_eos(e, x))
        finish = "stop", end = true;
      else if (produced >= limit)
        end = true;
    }
    pump(l, c);
    if (c->cancel && !end)
      finish = "cancel", end = true;
    const int keep = end ? printed - 1 : m + 1; /* the round's fed tokens the state keeps */
    if (!seq_rollback(e->s, (uint32_t)(k + 1 - keep)) ||
        (keep && !drafter_seed(e->d, e->s, pos, pos + keep, e->err)))
    {
      failed = true;
      break;
    }
    for (int i = 0; i < keep; i++)
      ids_push(&e->held, toks[i]);
    if (end)
      break;
    t = picks[m];
    if ((uint32_t)seq_position(e->s) + 1 >= e->max_ctx)
      break;
  }
  if (failed)
  {
    out("ERR %s\n", e->err);
    restart(e);
    return;
  }
  out("DONE %lld %u %.1f %.1f %s %lld %lld %u 0 0\n", produced, n, t1 - t0, now_ms() - t1, finish, accepted, drafted,
      reused);
}

static const char *opt(int argc, char **argv, const char *name, const char *dflt)
{
  for (int i = 1; i + 1 < argc; i++)
    if (!strcmp(argv[i], name))
      return argv[i + 1];
  return dflt;
}

static double env_num(const char *name, double dflt)
{
  const char *v = getenv(name);
  return v && *v ? atof(v) : dflt;
}

int main(int argc, char **argv)
{
  const char *model = opt(argc, argv, "--model", NULL);
  if (!model)
  {
    fprintf(stderr, "usage: substrate-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] "
                    "[--max-context N] [--budget GiB] [--max-tokens N] [--eos id,id] [--serve]\n");
    return 2;
  }
  static Engine e;
  e.max_ctx = (uint32_t)atol(opt(argc, argv, "--max-context", "65536"));
  e.max_tokens = (uint32_t)atol(opt(argc, argv, "--max-tokens", "2048"));
  const double budget = atof(opt(argc, argv, "--budget", "106"));
  {
    char *list = strdup(opt(argc, argv, "--eos", "1")), *sv = NULL;
    for (char *x = strtok_r(list, ",", &sv); x && e.n_eos < MAX_EOS; x = strtok_r(NULL, ",", &sv))
      e.eos[e.n_eos++] = (int32_t)atol(x);
    free(list);
  }
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "";
  Gguf *g = gguf_create(model, gerr);
  const char *engram = opt(argc, argv, "--engram", NULL);
  Gguf *eg = engram ? gguf_create(engram, gerr) : NULL;
  if (!g || (engram && !eg))
    return fprintf(stderr, "substrate-serve: %s\n", gerr), 2;
  Weights *w = weights_create(g, eg, werr);
  if (!w)
    return fprintf(stderr, "substrate-serve: %s\n", werr), 2;
  uint64_t *picks = NULL;
  if (opt(argc, argv, "--picks", NULL))
  {
    const size_t np = (size_t)w->geo.n_layers * w->geo.n_experts;
    FILE *pf = fopen(opt(argc, argv, "--picks", NULL), "rb");
    picks = malloc(np * sizeof *picks);
    if (!pf || !picks || fread(picks, sizeof *picks, np, pf) != np)
      return fprintf(stderr, "substrate-serve: the picks file could not be read\n"), 2;
    fclose(pf);
  }
  const ForwardOptions fo = {(int)w->geo.n_layers, e.max_ctx, (uint64_t)(budget * (double)(1ull << 30)), picks,
                             e.max_tokens};
  if (!(e.f = forward_create_with(w, &fo, e.err)) || !forward_lane(e.f, 0, e.err))
    return fprintf(stderr, "substrate-serve: %s\n", e.err), 2;
  const char *dspark = opt(argc, argv, "--dspark", NULL);
  if (dspark && !(e.d = drafter_create(e.f, dspark, e.err)))
    return fprintf(stderr, "substrate-serve: %s\n", e.err), 2;
  if (env_num("SUBSTRATE_SLOT_KEEP", 0) != 0)
  {
    char dflt[4096];
    const char *slash = strrchr(model, '/');
    snprintf(dflt, sizeof dflt, "%.*s/slots", slash ? (int)(slash - model) : 1, slash ? model : ".");
    const char *dir = getenv("SUBSTRATE_SLOT_DIR") ? getenv("SUBSTRATE_SLOT_DIR") : dflt;
    const SlotsOptions so = {dir, (uint32_t)env_num("SUBSTRATE_SLOT_CAP", 4), env_num("SUBSTRATE_SLOT_GB", 16),
                             (uint32_t)env_num("SUBSTRATE_SLOT_MIN", 1024), env_num("SUBSTRATE_SLOT_DAYS", 7),
                             forward_identity(e.f)};
    char serr[SLOTS_ERR] = "";
    if (!(e.slots = slots_create(&so, serr)))
      return fprintf(stderr, "substrate-serve: %s\n", serr), 2;
    out("NOTE slots: %u conversations kept from an earlier run (%s)\n", slots_in_dir(e.slots), dir);
  }
  if (!(e.s = seq_create(e.f, e.err)))
    return fprintf(stderr, "substrate-serve: %s\n", e.err), 2;
  out("READY %u stop\n", e.max_ctx);
  Lines l = {0};
  Ctl c = {0};
  for (char *line; !c.quit && (line = next_line(&l));)
  {
    if (!strcmp(line, "QUIT"))
      c.quit = true;
    else if (!strncmp(line, "GEN ", 4))
    {
      Gen gen;
      char why[256];
      if (!parse_gen(line, &gen, why, sizeof why))
        out("ERR %s\n", why);
      else
      {
        c.cancel = false;
        generate(&e, &gen, &l, &c);
        free(gen.ids);
      }
    }
    else if (!strcmp(line, "STOP") || !strncmp(line, "STOP ", 5))
      ; /* nothing running */
    else if (*line)
      out("ERR unsupported command '%.32s' (GEN, STOP and QUIT)\n", line);
    free(line);
  }
  if (e.slots) /* the end: the conversation set aside, RAM's written to the folder */
  {
    char serr[SLOTS_ERR] = "";
    set_aside(&e);
    if (!slots_flush(e.slots, serr))
      out("NOTE slots: %s\n", serr);
    slots_destroy(e.slots);
  }
  seq_destroy(e.s);
  drafter_destroy(e.d);
  forward_destroy(e.f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  free(picks);
  free(e.held.v);
  free(l.buf);
  return 0;
}
