#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "api.h"

/* osmos-serve as the child: started with --parallel (every line carries its rid), its lines read by one thread and
   handed to the run they name; a run's GEN written under the lock. the child gone: every run ends with why */

static void end(Run *r, const char *finish, const char *why)
{
  pthread_mutex_lock(&r->mu);
  if (!r->done)
  {
    snprintf(r->finish, sizeof r->finish, "%s", finish);
    if (why)
      snprintf(r->err, sizeof r->err, "%s", why);
    r->done = true;
    pthread_cond_broadcast(&r->cv);
  }
  pthread_mutex_unlock(&r->mu);
}

static Run *find(Engine *e, long long rid) /* (under e->mu) */
{
  for (Run *r = e->runs; r; r = r->next)
    if (r->rid == rid)
      return r;
  return NULL;
}

static void token(Run *r, int tok)
{
  pthread_mutex_lock(&r->mu);
  if (r->n == r->cap)
  {
    int32_t *t = realloc(r->tok, (r->cap ? 2 * r->cap : 256) * sizeof *t);
    if (!t)
    {
      pthread_mutex_unlock(&r->mu);
      end(r, "error", "out of memory");
      return;
    }
    r->tok = t, r->cap = r->cap ? 2 * r->cap : 256;
  }
  r->tok[r->n++] = tok;
  pthread_cond_broadcast(&r->cv);
  pthread_mutex_unlock(&r->mu);
}

/* a run's line (under e->mu: its run held) */
static void run_line(Run *r, const char *what, const char *l)
{
  long long rid, produced;
  int tok;
  unsigned held, total, reused, prompt;
  double ms, rate, pms, dms;
  char finish[16] = "stop";
  if (!strcmp(what, "T") && sscanf(l, "T %lld %d", &rid, &tok) == 2)
    token(r, tok);
  else if (!strcmp(what, "PP") &&
           sscanf(l, "PP %lld %u %u %lf %lf %u", &rid, &held, &total, &ms, &rate, &reused) == 6)
  {
    pthread_mutex_lock(&r->mu);
    r->prompt_n = total, r->prompt_ms = ms, r->reused = reused;
    pthread_mutex_unlock(&r->mu);
  }
  else if (!strcmp(what, "DONE"))
  {
    if (sscanf(l, "DONE %lld %lld %u %lf %lf %15s", &rid, &produced, &prompt, &pms, &dms, finish) == 6)
    {
      pthread_mutex_lock(&r->mu);
      r->decode_ms = dms;
      if (!r->prompt_n)
        r->prompt_n = prompt;
      pthread_mutex_unlock(&r->mu);
    }
    end(r, finish, NULL);
  }
  else if (!strcmp(what, "ERR"))
  {
    const char *why = strchr(l + 4, ' ');
    end(r, "error", why ? why + 1 : "the engine refused it");
  }
}

static void line(Engine *e, char *l)
{
  unsigned ctx;
  if (sscanf(l, "READY %u", &ctx) == 1)
  {
    pthread_mutex_lock(&e->mu);
    e->ready = true, e->max_ctx = ctx;
    pthread_cond_broadcast(&e->ready_cv);
    pthread_mutex_unlock(&e->mu);
    fprintf(stderr, "osmos-api: the engine is ready (%u tokens of context)\n", ctx);
    return;
  }
  if (!strncmp(l, "NOTE ", 5))
  {
    fprintf(stderr, "osmos-api: engine: %s\n", l + 5);
    return;
  }
  long long rid;
  char what[8] = "";
  pthread_mutex_lock(&e->mu);
  Run *r = sscanf(l, "%7s %lld", what, &rid) == 2 ? find(e, rid) : NULL;
  if (r)
    run_line(r, what, l);
  else if (!strncmp(l, "ERR ", 4) || !strncmp(l, "FATAL ", 6)) /* FATAL: the engine exits next (its runs then ended) */
    fprintf(stderr, "osmos-api: engine: %s\n", l);
  pthread_mutex_unlock(&e->mu);
}

static void *reader(void *arg)
{
  Engine *e = arg;
  FILE *f = fdopen(e->out_fd, "r");
  char *l = NULL;
  size_t cap = 0;
  ssize_t n;
  while (f && (n = getline(&l, &cap, f)) > 0)
  {
    if (l[n - 1] == '\n')
      l[n - 1] = 0;
    line(e, l);
  }
  free(l);
  if (f)
    fclose(f);
  pthread_mutex_lock(&e->mu); /* the child is gone: every run ends, no new one starts */
  e->dead = true;
  pthread_cond_broadcast(&e->ready_cv);
  for (Run *r = e->runs; r; r = r->next)
    end(r, "error", "the engine exited");
  pthread_mutex_unlock(&e->mu);
  fprintf(stderr, "osmos-api: the engine exited\n");
  return NULL;
}

bool engine_start(Engine *e, char *const argv[], bool spec, char err[API_ERR])
{
  int in[2], out[2];
  if (pipe2(in, O_CLOEXEC) || pipe2(out, O_CLOEXEC))
    return snprintf(err, API_ERR, "pipes: %s", strerror(errno)), false;
  pthread_mutex_init(&e->mu, NULL);
  pthread_mutex_init(&e->wmu, NULL);
  pthread_cond_init(&e->ready_cv, NULL);
  e->spec = spec, e->next_rid = 1;
  e->pid = fork();
  if (e->pid < 0)
    return snprintf(err, API_ERR, "fork: %s", strerror(errno)), false;
  if (e->pid == 0)
  {
    prctl(PR_SET_PDEATHSIG, SIGTERM); /* this server gone (rig's stop): the engine with it */
    dup2(in[0], STDIN_FILENO), dup2(out[1], STDOUT_FILENO);
    execv(argv[0], argv);
    fprintf(stderr, "osmos-api: %s: %s\n", argv[0], strerror(errno));
    _exit(127);
  }
  close(in[0]), close(out[1]);
  e->in_fd = in[1], e->out_fd = out[0];
  signal(SIGPIPE, SIG_IGN); /* a write to a closed connection or a dead child: an error, not the end */
  if (pthread_create(&e->reader, NULL, reader, e))
    return snprintf(err, API_ERR, "the engine's reader did not start"), false;
  return true;
}

bool engine_ready(Engine *e)
{
  pthread_mutex_lock(&e->mu);
  const bool ok = e->ready && !e->dead;
  pthread_mutex_unlock(&e->mu);
  return ok;
}

Run *engine_run(Engine *e, const int32_t *ids, size_t n, unsigned max_new, char err[API_ERR])
{
  Run *r = calloc(1, sizeof *r);
  size_t cap = 64 + 12 * n;
  char *msg = malloc(cap);
  if (!r || !msg)
    return free(r), free(msg), snprintf(err, API_ERR, "out of memory"), NULL;
  pthread_mutex_init(&r->mu, NULL);
  pthread_cond_init(&r->cv, NULL);
  pthread_mutex_lock(&e->mu);
  if (!e->ready || e->dead)
  {
    pthread_mutex_unlock(&e->mu);
    return free(r), free(msg), snprintf(err, API_ERR, "the engine is not ready"), NULL;
  }
  r->rid = e->next_rid++;
  r->next = e->runs, e->runs = r; /* registered before its GEN: no line of it comes unclaimed */
  pthread_mutex_unlock(&e->mu);
  size_t w = (size_t)snprintf(msg, cap, "GEN %u rid=%lld spec=%d ", max_new, r->rid, e->spec ? 1 : 0);
  for (size_t i = 0; i < n; i++)
    w += (size_t)snprintf(msg + w, cap - w, i ? ",%d" : "%d", ids[i]);
  msg[w++] = '\n';
  bool ok = true;
  pthread_mutex_lock(&e->wmu);
  for (size_t at = 0; ok && at < w;)
  {
    const ssize_t k = write(e->in_fd, msg + at, w - at);
    ok = k > 0, at += k > 0 ? (size_t)k : 0;
  }
  pthread_mutex_unlock(&e->wmu);
  free(msg);
  if (!ok)
    end(r, "error", "the engine's input closed");
  return r;
}

void engine_stop(Engine *e, Run *r)
{
  char msg[64];
  const int n = snprintf(msg, sizeof msg, "STOP %lld\n", r->rid);
  pthread_mutex_lock(&e->wmu);
  if (!e->dead && write(e->in_fd, msg, (size_t)n) != n)
    fprintf(stderr, "osmos-api: STOP %lld not sent\n", r->rid);
  pthread_mutex_unlock(&e->wmu);
}

void engine_release(Engine *e, Run *r)
{
  pthread_mutex_lock(&e->mu);
  for (Run **p = &e->runs; *p; p = &(*p)->next)
    if (*p == r)
    {
      *p = r->next;
      break;
    }
  pthread_mutex_unlock(&e->mu);
  free(r->tok);
  pthread_mutex_destroy(&r->mu), pthread_cond_destroy(&r->cv);
  free(r);
}

/* the end: QUIT sent (osmos-serve sets every conversation aside and writes its RAM's to their folder) and the engine's
   exit awaited, wait_s at most - then killed. true when it exited by itself (0.7.1: a stop killed it, by PDEATHSIG) */
bool engine_quit(Engine *e, int wait_s)
{
  pthread_mutex_lock(&e->wmu);
  const bool sent = !e->dead && write(e->in_fd, "QUIT\n", 5) == 5;
  pthread_mutex_unlock(&e->wmu);
  close(e->in_fd); /* its stdin's end: a QUIT too, should the line not have gone */
  for (int i = 0; sent && i < wait_s * 10; i++)
  {
    int st;
    if (waitpid(e->pid, &st, WNOHANG) == e->pid)
      return true;
    usleep(100 * 1000);
  }
  kill(e->pid, SIGKILL);
  waitpid(e->pid, NULL, 0);
  return false;
}
