#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>

#include "api.h"

/* HTTP/1.1, as little as a local OpenAI client needs: one request a connection (Connection: close), its body by
   Content-Length (a chunked one refused), headers and body each capped; answers whole or as server-sent events */

static bool send_all(int fd, const char *p, size_t n)
{
  while (n)
  {
    const ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k <= 0)
      return false;
    p += k, n -= (size_t)k;
  }
  return true;
}

static const char *reason(int status)
{
  switch (status)
  {
  case 200: return "OK";
  case 400: return "Bad Request";
  case 403: return "Forbidden";
  case 404: return "Not Found";
  case 405: return "Method Not Allowed";
  case 411: return "Length Required";
  case 413: return "Payload Too Large";
  case 431: return "Request Header Fields Too Large";
  case 500: return "Internal Server Error";
  case 501: return "Not Implemented";
  case 503: return "Service Unavailable";
  default: return "Error";
  }
}

/* the request read: 0, or the status to answer with (err why) */
int http_read(Http *h, char err[API_ERR])
{
  char *head = malloc(API_HEAD_MAX + 1);
  size_t n = 0;
  char *end = NULL;
  if (!head)
    return snprintf(err, API_ERR, "out of memory"), 500;
  while (!end)
  {
    if (n == API_HEAD_MAX)
      return free(head), snprintf(err, API_ERR, "headers past %u bytes", API_HEAD_MAX), 431;
    const ssize_t k = recv(h->fd, head + n, API_HEAD_MAX - n, 0);
    if (k <= 0)
      return free(head), snprintf(err, API_ERR, "the connection closed in the headers"), 400;
    n += (size_t)k, head[n] = 0;
    end = strstr(head, "\r\n\r\n");
  }
  char version[16];
  if (sscanf(head, "%7s %255s %15s", h->method, h->path, version) != 3 || strncmp(version, "HTTP/1.", 7))
    return free(head), snprintf(err, API_ERR, "not an HTTP/1.x request line"), 400;
  size_t length = 0;
  bool has_length = false, cont = false;
  for (char *l = strstr(head, "\r\n") + 2; l < end; l = strstr(l, "\r\n") + 2)
  {
    if (!strncasecmp(l, "Content-Length:", 15))
    { /* digits alone, then the line's end (RFC 9110: 1*DIGIT; 0.7.1: "-1" read as huge - a 413 - and "12xyz" as 12);
         a second one that says otherwise refused, not the last taken */
      const char *d = l + 15;
      while (*d == ' ' || *d == '\t')
        d++;
      char *e;
      const unsigned long long v = *d >= '0' && *d <= '9' ? strtoull(d, &e, 10) : 0;
      if (*d < '0' || *d > '9' || (strspn(e, " \t") + e)[0] != '\r' || (has_length && v != length))
        return free(head), snprintf(err, API_ERR, "a Content-Length not one number of bytes"), 400;
      if (v > API_BODY_MAX)
        return free(head), snprintf(err, API_ERR, "a body past %u bytes", API_BODY_MAX), 413;
      length = (size_t)v, has_length = true;
    }
    else if (!strncasecmp(l, "Origin:", 7))
      h->origin = true;
    else if (!strncasecmp(l, "Host:", 5))
    { /* its value, spaces around it dropped; one longer than the buffer refused (no Host of ours is) */
      const char *v = l + 5, *e = strstr(l, "\r\n");
      while (v < e && (*v == ' ' || *v == '\t'))
        v++;
      while (e > v && (e[-1] == ' ' || e[-1] == '\t'))
        e--;
      if ((size_t)(e - v) >= sizeof h->host || h->host[0])
        return free(head), snprintf(err, API_ERR, "a Host header too long, or two"), 400;
      memcpy(h->host, v, (size_t)(e - v)), h->host[e - v] = 0;
    }
    else if (!strncasecmp(l, "Transfer-Encoding:", 18))
      return free(head), snprintf(err, API_ERR, "a chunked body (send its Content-Length)"), 411;
    else if (!strncasecmp(l, "Expect:", 7) && strcasestr(l, "100-continue") && strcasestr(l, "100-continue") < strstr(l, "\r\n"))
      cont = true;
  }
  if (strcmp(h->method, "POST") == 0 && !has_length)
    return free(head), snprintf(err, API_ERR, "a POST without Content-Length"), 411;
  h->body = malloc(length + 1);
  if (!h->body)
    return free(head), snprintf(err, API_ERR, "out of memory"), 500;
  const size_t have = n - (size_t)(end + 4 - head);
  h->body_n = have < length ? have : length;
  memcpy(h->body, end + 4, h->body_n);
  free(head);
  if (cont && h->body_n < length) /* the client waits for this before its body (curl's, past 1 KB: 0.11.2 - a second
                                     lost a request, its wait for one never sent) */
    send_all(h->fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
  while (h->body_n < length)
  {
    const ssize_t k = recv(h->fd, h->body + h->body_n, length - h->body_n, 0);
    if (k <= 0)
      return snprintf(err, API_ERR, "the connection closed in the body"), 400;
    h->body_n += (size_t)k;
  }
  h->body[h->body_n] = 0;
  return 0;
}

bool http_send(int fd, int status, const char *type, const char *body, size_t n)
{
  return http_send_h(fd, status, type, body, n, NULL);
}

/* with head: more header lines ("Name: value\r\n" each) before the blank line */
bool http_send_h(int fd, int status, const char *type, const char *body, size_t n, const char *head)
{
  char h[1024];
  const int k = snprintf(h, sizeof h,
                         "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n%s\r\n",
                         status, reason(status), type, n, head ? head : "");
  return k > 0 && (size_t)k < sizeof h && send_all(fd, h, (size_t)k) && send_all(fd, body, n);
}

bool http_stream_open(int fd)
{
  return http_stream_open_h(fd, NULL);
}

bool http_stream_open_h(int fd, const char *head)
{
  char h[1024];
  const int k = snprintf(h, sizeof h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                                      "Connection: close\r\n%s\r\n", head ? head : "");
  return k > 0 && (size_t)k < sizeof h && send_all(fd, h, (size_t)k);
}

bool http_stream_event(int fd, const char *data, size_t n)
{
  return send_all(fd, "data: ", 6) && send_all(fd, data, n) && send_all(fd, "\n\n", 2);
}

bool http_stream_comment(int fd) { return send_all(fd, ": keep-alive\n\n", 14); }

bool http_error(int fd, int status, const char *message) { return http_error_code(fd, status, message, NULL, NULL); }

/* with OpenAI's param and code (or NULL): a client's own handling keyed on them - context_length_exceeded the one
   harnesses compact a conversation on (0.9.2) */
bool http_error_code(int fd, int status, const char *message, const char *param, const char *code)
{
  return http_error_head(fd, status, message, param, code, NULL);
}

bool http_error_head(int fd, int status, const char *message, const char *param, const char *code, const char *head)
{
  JsonBuf b = {0};
  json_puts(&b, "{\"error\": {\"message\": ");
  json_dump_string(&b, message, strlen(message));
  json_puts(&b, status >= 500 ? ", \"type\": \"server_error\"" : ", \"type\": \"invalid_request_error\"");
  if (param)
    json_puts(&b, ", \"param\": "), json_dump_string(&b, param, strlen(param));
  if (code)
    json_puts(&b, ", \"code\": "), json_dump_string(&b, code, strlen(code));
  json_puts(&b, "}}");
  const bool ok = !b.oom && http_send_h(fd, status, "application/json", b.p, b.n, head);
  free(b.p);
  return ok;
}

/* an answer sent before the request was read whole (a refusal: 413, 400, 503): this end's writing shut, the rest of
   the client's bytes read and dropped - API_LINGER_MS at most - then the close. a close with bytes unread is a reset,
   and a client still sending sees that, never the status (0.7.1) */
void http_linger(int fd, bool wait)
{
  (void)shutdown(fd, SHUT_WR);
  if (!wait) /* the accept loop's own refusal (503): what came already drained, nothing waited for - it holds no thread
                of its own to wait in */
  {
    char sink[16384];
    while (recv(fd, sink, sizeof sink, MSG_DONTWAIT) > 0)
      ;
    return;
  }
  const struct timeval t = {0, API_LINGER_MS * 1000};
  (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof t);
  char sink[16384];
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (;;)
  {
    if (recv(fd, sink, sizeof sink, 0) <= 0)
      return;
    clock_gettime(CLOCK_MONOTONIC, &b);
    if ((b.tv_sec - a.tv_sec) * 1000 + (b.tv_nsec - a.tv_nsec) / 1000000 > API_LINGER_MS)
      return;
  }
}
