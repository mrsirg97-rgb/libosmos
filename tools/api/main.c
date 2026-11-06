/* osmos-api: the engine as an OpenAI-compatible server (DESIGN.md, "osmos-api").

     osmos-api --config <pack>/osmos.json [--port N] [--host A] [--engine <osmos-serve>]
     osmos-api --tokenize --model <first shard>     (stdin's text as the prompt's ids, one line: for tests)

   POST /v1/chat/completions, GET /v1/models, GET /health (503 while the engine loads). osmos-serve - beside this
   binary unless --engine names another - runs as the child with the pack's settings (osmos.json, the installer's);
   its exit ends this server (the launcher restarts it) */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "api.h"
#include "config.h"
#include "dialect.h"
#include "osmos/model/gguf.h"
#include "osmos/version.h"

#define API_PORT 8091  /* the installer's first (tools/install/rig.py PORT_BASE) */
#define API_CONNS 16   /* connections at once: past it a 503 (a local client's handful; the engine runs few) */
#define API_POLL_MS 1000

static const char *opt(int argc, char **argv, const char *name, const char *dflt)
{
  for (int i = 1; i + 1 < argc; i++)
    if (!strcmp(argv[i], name))
      return argv[i + 1];
  return dflt;
}

static bool has(int argc, char **argv, const char *name)
{
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], name))
      return true;
  return false;
}

static int die(const char *why)
{
  fprintf(stderr, "osmos-api: %s\n", why);
  return 1;
}

static Api api;
static pthread_mutex_t conns_mu = PTHREAD_MUTEX_INITIALIZER;
static int conns;

static void answer(Http *h)
{
  char err[API_ERR];
  const int status = http_read(h, err);
  if (status)
  {
    http_error(h->fd, status, err);
    return;
  }
  char *q = strchr(h->path, '?');
  if (q)
    *q = 0;
  const bool get = !strcmp(h->method, "GET"), post = !strcmp(h->method, "POST");
  if (!strcmp(h->path, "/health"))
  {
    const bool ready = engine_ready(&api.engine);
    const char *b = ready ? "{\"status\": \"ok\"}" : "{\"status\": \"loading\"}";
    http_send(h->fd, ready ? 200 : 503, "application/json", b, strlen(b));
  }
  else if (!strcmp(h->path, "/v1/models") && get)
  {
    JsonBuf b = {0};
    json_puts(&b, "{\"object\": \"list\", \"data\": [{\"id\": ");
    json_dump_string(&b, api.model, strlen(api.model));
    json_puts(&b, ", \"object\": \"model\", \"owned_by\": \"osmos\"}]}");
    if (!b.oom)
      http_send(h->fd, 200, "application/json", b.p, b.n);
    free(b.p);
  }
  else if (!strcmp(h->path, "/v1/chat/completions") && post)
    api_chat(&api, h);
  else if (!strcmp(h->path, "/v1/chat/completions") || !strcmp(h->path, "/v1/models"))
    http_error(h->fd, 405, "method not allowed");
  else
    http_error(h->fd, 404, "not found (POST /v1/chat/completions, GET /v1/models, GET /health)");
}

static void *connection(void *arg)
{
  Http h = {.fd = (int)(intptr_t)arg};
  answer(&h);
  free(h.body);
  close(h.fd);
  pthread_mutex_lock(&conns_mu);
  conns--;
  pthread_mutex_unlock(&conns_mu);
  return NULL;
}

static int listen_on(const char *host, int port)
{
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
  if (inet_pton(AF_INET, host, &a.sin_addr) != 1)
    return -1;
  const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0), one = 1;
  if (fd < 0 || setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) ||
      bind(fd, (struct sockaddr *)&a, sizeof a) || listen(fd, API_CONNS))
    return fd >= 0 ? close(fd), -1 : -1;
  return fd;
}

static int tokenize(const char *model)
{
  char err[TOKENIZER_ERR];
  Gguf *g = gguf_header(model, err); /* the tokenizer: the first shard's metadata */
  Tokenizer *t = g ? tokenizer_create(g, err) : NULL;
  gguf_destroy(g);
  if (!t)
    return die(err);
  JsonBuf in = {0};
  char buf[1 << 16];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, stdin)) > 0)
    json_put(&in, buf, k);
  int32_t *ids = malloc((in.n + 1) * sizeof *ids);
  const int64_t n = ids ? tokenizer_encode(t, in.p ? in.p : "", in.n, true, ids, in.n + 1) : -1;
  for (int64_t i = 0; i < n; i++)
    printf(i ? ",%d" : "%d", ids[i]);
  printf("\n");
  free(ids), free(in.p), tokenizer_destroy(t);
  return n < 0;
}

int main(int argc, char **argv)
{
  if (argc == 2 && !strcmp(argv[1], "--version"))
    return printf("osmos %s\n", OSMOS_VERSION), 0;
  if (has(argc, argv, "--tokenize"))
    return opt(argc, argv, "--model", NULL) ? tokenize(opt(argc, argv, "--model", NULL))
                                            : die("usage: osmos-api --tokenize --model <first shard>");
  const char *cfg_path = opt(argc, argv, "--config", NULL);
  if (!cfg_path)
    return die("usage: osmos-api --config <pack>/osmos.json [--port N] [--host A] [--engine <osmos-serve>]");
  char err[API_ERR], self[PATH_MAX], engine[PATH_MAX];
  const ssize_t sn = readlink("/proc/self/exe", self, sizeof self - 1);
  self[sn > 0 ? sn : 0] = 0;
  char *slash = strrchr(self, '/');
  snprintf(engine, sizeof engine, "%.*s/osmos-serve", slash ? (int)(slash - self) : 1, slash ? self : ".");
  Config c;
  if (!config_read(cfg_path, &c, err))
    return die(err);
  snprintf(api.model, sizeof api.model, "osmos-%s", c.pack);
  Gguf *g = gguf_header(c.model, err); /* the tokenizer and the family alone: osmos-serve opens the model */
  const GgufKv *arch = g ? gguf_kv(g, "general.architecture") : NULL;
  api.d = arch && arch->type == GGUF_STRING ? dialect_of(arch->v.s) : NULL;
  api.tok = g && api.d ? tokenizer_create(g, err) : NULL;
  if (g && !api.d)
    snprintf(err, sizeof err, "the model: no chat dialect for its architecture %.64s (deepseek41, qwen4exp)", arch && arch->type == GGUF_STRING ? arch->v.s : "(none)");
  gguf_destroy(g);
  if (!api.tok)
    return die(err);
  api.eos = tokenizer_find(api.tok, api.d->eos);
  if (api.eos < 0)
    return snprintf(err, sizeof err, "the tokenizer has no %s", api.d->eos), die(err);
  char *args[CONFIG_ARGS_MAX];
  if (!config_argv(&c, opt(argc, argv, "--engine", engine), api.eos, args, err))
    return die(err);
  const int port = atoi(opt(argc, argv, "--port", "8091"));
  const int lfd = listen_on(opt(argc, argv, "--host", "127.0.0.1"), port);
  if (lfd < 0)
    return die("could not listen there (--host, --port)");
  if (!engine_start(&api.engine, args, c.drafter[0] != 0, err))
    return die(err);
  fprintf(stderr, "osmos-api: %s on %s:%d (the engine loading)\n", api.model, opt(argc, argv, "--host", "127.0.0.1"),
          port);
  for (;;)
  {
    struct pollfd p = {.fd = lfd, .events = POLLIN};
    if (poll(&p, 1, API_POLL_MS) > 0)
    {
      const int fd = accept4(lfd, NULL, NULL, SOCK_CLOEXEC);
      if (fd < 0)
        continue;
      pthread_mutex_lock(&conns_mu);
      const bool room = conns < API_CONNS;
      conns += room;
      pthread_mutex_unlock(&conns_mu);
      pthread_t t;
      if (!room)
        http_error(fd, 503, "too many connections"), close(fd);
      else if (pthread_create(&t, NULL, connection, (void *)(intptr_t)fd))
        http_error(fd, 503, "no thread for it"), close(fd), pthread_mutex_lock(&conns_mu), conns--,
            pthread_mutex_unlock(&conns_mu);
      else
        pthread_detach(t);
    }
    pthread_mutex_lock(&api.engine.mu);
    const bool dead = api.engine.dead;
    pthread_mutex_unlock(&api.engine.mu);
    if (dead)
      return die("the engine exited: this server with it");
  }
}
