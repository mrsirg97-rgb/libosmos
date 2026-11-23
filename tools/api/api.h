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
#define API_BODY_MAX (8u << 20) /* a request's body: twice the largest real one - 1M tokens (API_CTX_MAX) x 2.98 bytes a
                                   token x 1.2 for escaping, ~3.6 MB (DESIGN.md, "phase 5, the design"; 0.16.0: 32 MB) */
#define API_TREE_MAX (4 * API_BODY_MAX) /* a body's tree, or a call's arguments' parsed (one at a time, each freed): 2.3x
                                           the costliest shape measured - an agent's history with tool calls, 1.72x its
                                           body (a prompt's 1.00x); a body of tiny values ~65x: refused past it */
#define API_HEAD_MAX (64u << 10) /* its request line and headers */
/* a request whole within a deadline from its first byte (0.16.0: a byte every 9 s, under the 10 s idle, held a
   connection for hours): API_REQUEST_S for its headers, API_BODY_S_MB a MB of its body past them - a slow tunnel's
   1 MB/s (this box's loopback and LAN past 100 MB/s); past it 408, the connection closed */
#define API_REQUEST_S 10
#define API_ORIGINS_MAX 16 /* --allow-origin's at most */
#define API_BODY_S_MB 1

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
  char err[OSMOS_ERR];
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

bool engine_start(Engine *e, char *const argv[], bool spec, char err[OSMOS_ERR]);
/* a run's sampling (GEN temp= top_p= top_k= seed=): temperature 0 greedy, the keys not written; top_k 0 every token */
typedef struct Sampling
{
  double temperature, top_p;
  unsigned long long seed;
  unsigned top_k;
} Sampling;
Run *engine_run(Engine *e, const int32_t *ids, size_t n, unsigned max_new, unsigned mark, const Sampling *s,
                char err[OSMOS_ERR]);
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
  char origin[256]; /* the Origin header's value ("" none: a local client's; one past it refused - a browser's) */
  char host[256];   /* the Host header's value ("" none) */
  char cors[400];   /* an allowed origin's (--allow-origin) CORS header lines, every response's: "" none */
} Http;

int http_read(Http *h, char err[OSMOS_ERR]);
bool http_send(const Http *h, int status, const char *type, const char *body, size_t n);
bool http_send_h(const Http *h, int status, const char *type, const char *body, size_t n, const char *head);
/* a CORS preflight's answer (OPTIONS from an allowed origin): 204, the methods and headers a page may send */
bool http_preflight(const Http *h);
bool http_stream_open(const Http *h);
bool http_stream_open_h(const Http *h, const char *head);
bool http_stream_event(int fd, const char *data, size_t n);
/* an SSE comment (": keep-alive"), every client's to pass over: a long prompt's read not silent (0.11.2) */
bool http_stream_comment(int fd);
bool http_error(const Http *h, int status, const char *message);
bool http_error_code(const Http *h, int status, const char *message, const char *param, const char *code);
/* the same with header lines of its own ("Name: value\r\n" each, or NULL): a 503's Retry-After (0.11.2) */
bool http_error_head(const Http *h, int status, const char *message, const char *param, const char *code,
                     const char *head);
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
  bool preserve_thinking; /* the pack's preserve_thinking: a request's unset, this (LFM3's reader) */
  double temperature, top_p; /* the pack's sampling: a request's unset, these (0.14.0; temperature 0 greedy) */
  unsigned top_k;            /* likewise (0.15.2; 0 every token) */
  const char *allow[API_ORIGINS_MAX]; /* --allow-origin's: browsers' origins taken (CORS), any other refused */
  uint32_t keepalive_ms; /* a stream's keep-alive comment past this quiet (--keepalive-ms; 0.16.0 an env knob) */
  int n_allow;
} Api;

void api_chat(Api *a, Http *h);
