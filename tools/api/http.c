#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>

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
  bool has_length = false;
  for (char *l = strstr(head, "\r\n") + 2; l < end; l = strstr(l, "\r\n") + 2)
  {
    if (!strncasecmp(l, "Content-Length:", 15))
    {
      char *e;
      const unsigned long long v = strtoull(l + 15, &e, 10);
      if (e == l + 15 || v > API_BODY_MAX)
        return free(head), snprintf(err, API_ERR, "a body past %u bytes, or no length", API_BODY_MAX), 413;
      length = (size_t)v, has_length = true;
    }
    else if (!strncasecmp(l, "Transfer-Encoding:", 18))
      return free(head), snprintf(err, API_ERR, "a chunked body (send its Content-Length)"), 411;
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
  char head[256];
  const int k = snprintf(head, sizeof head,
                         "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                         status, reason(status), type, n);
  return send_all(fd, head, (size_t)k) && send_all(fd, body, n);
}

bool http_stream_open(int fd)
{
  static const char head[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                             "Connection: close\r\n\r\n";
  return send_all(fd, head, sizeof head - 1);
}

bool http_stream_event(int fd, const char *data, size_t n)
{
  return send_all(fd, "data: ", 6) && send_all(fd, data, n) && send_all(fd, "\n\n", 2);
}

bool http_error(int fd, int status, const char *message)
{
  JsonBuf b = {0};
  json_puts(&b, "{\"error\": {\"message\": ");
  json_dump_string(&b, message, strlen(message));
  json_puts(&b, status >= 500 ? ", \"type\": \"server_error\"}}" : ", \"type\": \"invalid_request_error\"}}");
  const bool ok = !b.oom && http_send(fd, status, "application/json", b.p, b.n);
  free(b.p);
  return ok;
}
