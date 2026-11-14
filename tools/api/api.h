/* osmos-api's shared shapes: the engine (osmos-serve, its child, over the protocol), a request's run on it, a
   connection's HTTP. design: DESIGN.md, "osmos-api" */
#pragma once

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"
#include "osmos/text/tokenizer.h"

#define API_LINGER_MS 2000 /* a refused request's bytes drained this long at most (http_linger) */
#define API_ERR 256
#define API_BODY_MAX (32u << 20) /* a request's body: a 64K-token context's text is ~0.3 MB; tools and history more */
#define API_HEAD_MAX (64u << 10) /* its request line and headers */

/* a GEN's run: the tokens as they come, then its DONE (or ERR); its owner waits on cv */
typedef struct Run
{
  long long rid;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int32_t *tok;
  size_t n, cap;
  bool done;
  char finish[16];     /* stop / length / cancel; "error" with err */
  char err[API_ERR];
  unsigned prompt_n, reused;
  double prompt_ms, decode_ms;
  struct Run *next;
} Run;

typedef struct Engine
{
  pid_t pid;
  int in_fd, out_fd;
  pthread_mutex_t mu;  /* the runs (the reader holds it a line through: a run is not freed under it) */
  pthread_mutex_t wmu; /* the child's stdin (apart: a write blocked on a full pipe never holds the reader up) */
  pthread_cond_t ready_cv;
  Run *runs;
  long long next_rid;
  bool ready, dead;
  unsigned max_ctx;
  bool spec;          /* a drafter: GEN spec=1 */
  pthread_t reader;
} Engine;

bool engine_start(Engine *e, char *const argv[], bool spec, char err[API_ERR]);
Run *engine_run(Engine *e, const int32_t *ids, size_t n, unsigned max_new, unsigned mark, char err[API_ERR]);
void engine_stop(Engine *e, Run *r);
void engine_release(Engine *e, Run *r);
bool engine_ready(Engine *e);
bool engine_quit(Engine *e, int wait_s);

/* a connection's request: its method, path, body */
typedef struct Http
{
  int fd;
  char method[8];
  char path[256];
  char *body;
  size_t body_n;
} Http;

int http_read(Http *h, char err[API_ERR]);
bool http_send(int fd, int status, const char *type, const char *body, size_t n);
bool http_send_h(int fd, int status, const char *type, const char *body, size_t n, const char *head);
bool http_stream_open(int fd);
bool http_stream_open_h(int fd, const char *head);
bool http_stream_event(int fd, const char *data, size_t n);
bool http_error(int fd, int status, const char *message);
void http_linger(int fd, bool wait);

/* the server: what a request needs */
typedef struct Api
{
  Engine engine;
  Tokenizer *tok;
  const struct Dialect *d; /* the model family's chat */
  char model[128];
  int32_t eos;
  bool fill_orphans; /* the pack's fill_orphans: a request's osmos.fill_orphans unset, this */
} Api;

void api_chat(Api *a, Http *h);
