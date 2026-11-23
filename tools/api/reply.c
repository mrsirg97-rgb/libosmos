/* POST /v1/chat/completions, the answer (reply.h): the engine's tokens streamed as they land (answer.c) or
   answered whole; at DONE the completion parsed for its tool calls, then the finish, usage and timings (DESIGN.md,
   "osmos-api: the answer") */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include "answer.h"
#include "osmos/text/chat_qwen.h"
#include "osmos/err.h"
#include "api.h"
#include "reply.h"


/* this run's tag in its ids (its start time, base 36): a run's request numbers start at 1 again after a restart, and
   a conversation across one held two calls of one id (a harness matches a result to its call by it - 0.11.2) */
static char boot[16];
static void boot_make(void)
{
  unsigned long long t = (unsigned long long)time(NULL);
  char rev[16];
  int n = 0;
  do
    rev[n++] = "0123456789abcdefghijklmnopqrstuvwxyz"[t % 36], t /= 36;
  while (t && n < 12);
  for (int k = 0; k < n; k++)
    boot[k] = rev[n - 1 - k];
  boot[n] = '-', boot[n + 1] = 0;
}
const char *reply_boot_tag(void)
{
  static pthread_once_t once = PTHREAD_ONCE_INIT; /* (the request threads': made once) */
  pthread_once(&once, boot_make);
  return boot;
}



/* the client closed its end (a stream's reader gone: its run STOPped) */
static bool closed(int fd)
{
  char c;
  const ssize_t k = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  return k == 0 || (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

/* the run's tokens since `seen` as text; false when the run is done (and all of it read) */
bool reply_more(Api *a, Run *r, size_t *seen, Answer *s)
{
  bool ping = false;
  pthread_mutex_lock(&r->mu);
  while (*seen == r->n && !r->done)
  {
    if (s->stream && !s->gone && ++s->idle * WAIT_MS > s->keepalive_ms)
    {
      ping = true; /* written below, the lock let go: a slow client never holds the engine's appends */
      break;
    }
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    t.tv_nsec += WAIT_MS * 1000000L;
    t.tv_sec += t.tv_nsec / 1000000000L, t.tv_nsec %= 1000000000L;
    pthread_cond_timedwait(&r->cv, &r->mu, &t);
    if (*seen == r->n && !r->done && !s->gone && closed(s->fd))
    {
      s->gone = true; /* found gone: back to STOP it (once gone, the wait is for its DONE) */
      break;
    }
  }
  const size_t from = *seen, to = r->n;
  int32_t *tok = to > from ? malloc((to - from) * sizeof *tok) : NULL;
  if (tok)
    memcpy(tok, r->tok + from, (to - from) * sizeof *tok);
  const bool done = r->done && (tok || to == from);
  pthread_mutex_unlock(&r->mu);
  if (to > from)
    s->idle = 0;
  if (ping)
    s->idle = 0, s->gone = s->gone || !http_stream_comment(s->fd);
  if (to > from && !tok)
    return s->gone = true, false; /* out of memory: given up as a client gone */
  for (size_t i = 0; i < to - from; i++)
    if (!answer_take(s, a->tok, tok[i]))
      return free(tok), s->gone = true, false; /* out of memory: given up as a client gone */
  free(tok);
  *seen = to;
  if (!done && !s->stream && !s->gone && closed(s->fd))
    s->gone = true; /* a whole answer writes nothing until its end: its client's leaving looked for as tokens land too,
                       not only in a pause (0.7.1) */
  return !done;
}

/* the message parsed - tool calls given their ids - or, out of the format, its text as it streamed (*parsed false,
   err the parse's why) */
Json *reply_message(const Answer *s, const Request *rq, long long rid, bool *parsed, char err[OSMOS_ERR])
{
  char jerr[OSMOS_ERR] = "";
  JsonBuf b = {0};
  const ChatMode mode = rq->mode;
  if ((*parsed = s->d->parse(rq, s->text.p, s->shape.p, s->text.n, &b, err)))
  {
    Json *m = json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr);
    free(b.p);
    const Json *calls = json_get(m, "tool_calls");
    for (size_t i = 0; calls && i < calls->n; i++)
    {
      Json *c = calls->item[i];
      Json **item = realloc(c->item, (c->n + 1) * sizeof *item);
      char **key = realloc(c->key, (c->n + 1) * sizeof *key);
      size_t *klen = realloc(c->klen, (c->n + 1) * sizeof *klen);
      Json *id = calloc(1, sizeof *id);
      if (item)
        c->item = item;
      if (key)
        c->key = key;
      if (klen)
        c->klen = klen;
      if (!item || !key || !klen || !id || !(id->s = malloc(48)) || !(c->key[c->n] = strdup("id")))
        return free(id ? id->s : NULL), free(id), json_free(m), NULL;
      id->type = JSON_STRING, id->len = (size_t)snprintf(id->s, 48, "call_%s%lld_%zu", reply_boot_tag(), rid, i);
      c->item[c->n] = id, c->klen[c->n] = 2, c->n++;
    }
    return m;
  }
  free(b.p);
  if (s->d->loose) /* a block-shaped turn out of the format: its open blocks' text (the stream's) */
  {
    JsonBuf l = {0};
    s->d->loose(s->text.p ? s->text.p : "", s->shape.p ? s->shape.p : "", s->text.n, &l);
    Json *m = l.oom ? NULL : json_parse_within(l.p, l.n, API_TREE_MAX, NULL, jerr);
    free(l.p);
    return m;
  }
  /* out of the format: reasoning before </think>, the content after (the EOS dropped) */
  const char *p = s->text.p ? s->text.p : "", *sh = s->shape.p ? s->shape.p : "";
  size_t n = s->text.n;
  const char *eos_text = s->d->eos, *think_end = s->d->think_end;
  const size_t eos = strlen(eos_text);
  if (n >= eos && !memcmp(p + n - eos, eos_text, eos))
    n -= eos;
  const char *end = mode == CHAT_THINKING ? memmem(sh, n, think_end, strlen(think_end)) : NULL;
  size_t rn = mode != CHAT_THINKING ? 0 : end ? (size_t)(end - sh) : n;
  size_t c0 = mode != CHAT_THINKING ? 0 : end ? rn + strlen(think_end) : n, r0 = 0;
  if (s->d->trims) /* as the stream passed it over: a phase's leading whitespace, and a closed reasoning's trailing
                      (held before </think>, never sent) */
  {
    r0 = qchat_space_lead(sh, rn), c0 += qchat_space_lead(sh + c0, n - c0);
    if (end)
      rn -= qchat_space_trail(sh + r0, rn - r0);
  }
  json_puts(&b, "{\"role\": \"assistant\", \"content\": "), json_dump_string(&b, p + c0, n - c0);
  json_puts(&b, ", \"reasoning_content\": "), json_dump_string(&b, p + r0, rn - r0);
  json_puts(&b, ", \"tool_calls\": []}");
  Json *m = b.oom ? NULL : json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr);
  free(b.p);
  return m;
}

/* a run that gives no answer: a streamed one's error event (no [DONE]), or a 500. a completion its parse refused
   goes with it as its text (the EOS dropped): greedy, the same messages again are the same completion, so a client
   can only get past one by showing the model what it wrote (#34) - sent back, a client's text is defused as any */
void reply_fail_at(const Http *h, int status, bool stream, const char *why, const JsonBuf *completion, const char *eos)
{
  JsonBuf e = {0};
  json_puts(&e, "{\"error\": {\"message\": "), json_dump_string(&e, why, strlen(why));
  json_puts(&e, ", \"type\": \"server_error\"");
  if (completion)
  {
    size_t n = completion->n;
    if (n >= strlen(eos) && !memcmp(completion->p + n - strlen(eos), eos, strlen(eos)))
      n -= strlen(eos);
    json_puts(&e, ", \"completion\": "), json_dump_string(&e, n ? completion->p : "", n);
  }
  json_puts(&e, "}}");
  if (e.oom)
  {
    if (!stream) /* an open stream takes no status line: its end without [DONE] the error */
      http_error(h, 500, why);
  }
  else if (stream)
    http_stream_event(h->fd, e.p, e.n);
  else /* the model's answer attached (it did not parse): the same request the same answer - an OpenAI client told
          not to retry it (x-should-retry: theirs - 0.11.2, a whole answer generated three times for one error) */
    http_send_h(h, status, "application/json", e.p, e.n, completion ? "x-should-retry: false\r\n" : NULL);
  free(e.p);
}

void reply_fail(const Http *h, bool stream, const char *why, const JsonBuf *completion, const char *eos)
{
  reply_fail_at(h, 500, stream, why, completion, eos);
}

/* the x-osmos-loop header line (loop.h's value, "\r\n" ended) */
void reply_loop_head(char *out, size_t cap, const Answer *s)
{
  char v[400];
  loop_header(v, sizeof v, s->loop);
  const int w = snprintf(out, cap, "x-osmos-loop: %s\r\n", v);
  if (s->sampled && w > 0 && (size_t)w < cap)
    snprintf(out + w, cap - (size_t)w, "x-osmos-seed: %llu\r\n", s->seed);
}

/* usage and timings, as fields after a response's choices (", ...") */
static void usage(JsonBuf *b, const Run *r, size_t prompt_n)
{
  char t[512];
  const double pps = r->prompt_ms > 0 ? 1e3 * (double)(prompt_n - r->reused) / r->prompt_ms : 0;
  const double dps = r->decode_ms > 0 ? 1e3 * (double)r->n / r->decode_ms : 0;
  snprintf(t, sizeof t,
           ", \"usage\": {\"prompt_tokens\": %zu, \"completion_tokens\": %zu, \"total_tokens\": %zu, "
           "\"prompt_tokens_details\": {\"cached_tokens\": %u}}, \"timings\": {\"prompt_n\": %zu, \"prompt_ms\": "
           "%.3f, \"prompt_per_second\": %.3f, \"predicted_n\": %zu, \"predicted_ms\": %.3f, \"predicted_per_second\": "
           "%.3f}",
           prompt_n, r->n, prompt_n + r->n, r->reused, prompt_n, r->prompt_ms, pps, r->n, r->decode_ms, dps);
  json_puts(b, t);
}

static const char *finish_of(const Run *r, const Json *m)
{
  const Json *calls = json_get(m, "tool_calls");
  if (calls && calls->n)
    return "tool_calls";
  return !strcmp(r->finish, "length") ? "length" : "stop";
}

/* the end of a streamed answer: its tool calls, its finish (usage beside it when asked), [DONE] */
void reply_stream_end(Answer *s, const Request *rq, const Run *r, const Json *m, size_t prompt_n)
{
  const Json *calls = json_get(m, "tool_calls");
  if (calls && calls->n)
  {
    JsonBuf d = {0};
    json_puts(&d, "{\"tool_calls\": [");
    for (size_t i = 0; i < calls->n; i++)
    {
      char ix[48];
      snprintf(ix, sizeof ix, "%s{\"index\": %zu", i ? ", " : "", i);
      json_puts(&d, ix);
      for (size_t k = 0; k < calls->item[i]->n; k++)
        json_puts(&d, ", "), json_dump_string(&d, calls->item[i]->key[k], calls->item[i]->klen[k]),
            json_puts(&d, ": "), json_dump(&d, calls->item[i]->item[k]);
      json_puts(&d, "}");
    }
    json_puts(&d, "]}");
    if (!d.oom)
      answer_chunk(s, d.p, NULL, "");
    free(d.p);
  }
  JsonBuf u = {0};
  if (rq->usage)
    usage(&u, r, prompt_n);
  loop_json(&u, &rq->loop); /* the history's loop signals, every stream's last chunk (0.7.3) */
  answer_chunk(s, "{}", finish_of(r, m), u.p ? u.p : "");
  free(u.p);
  if (!s->gone)
    s->gone = !http_stream_event(s->fd, "[DONE]", 6);
}

void reply_whole(Answer *s, const Run *r, const Json *m, size_t prompt_n)
{
  JsonBuf b = {0};
  char t[128];
  json_puts(&b, "{\"id\": \""), json_puts(&b, s->id), json_puts(&b, "\", \"object\": \"chat.completion\"");
  snprintf(t, sizeof t, ", \"created\": %lld, \"model\": ", s->created);
  json_puts(&b, t), json_dump_string(&b, s->model, strlen(s->model));
  json_puts(&b, ", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\"");
  for (size_t k = 0; k < m->n; k++)
  {
    const Json *v = m->item[k];
    if (!strcmp(m->key[k], "role") || (v->type == JSON_ARRAY && !v->n) ||
        (!strcmp(m->key[k], "reasoning_content") && !v->len))
      continue;
    json_puts(&b, ", "), json_dump_string(&b, m->key[k], m->klen[k]), json_puts(&b, ": "), json_dump(&b, v);
  }
  json_puts(&b, "}, \"finish_reason\": "), json_dump_string(&b, finish_of(r, m), strlen(finish_of(r, m)));
  json_puts(&b, "}]");
  usage(&b, r, prompt_n);
  loop_json(&b, s->loop);
  json_puts(&b, "}");
  char head[512];
  reply_loop_head(head, sizeof head, s);
  if (b.oom)
    http_error(s->http, 500, "out of memory");
  else
    http_send_h(s->http, 200, "application/json", b.p, b.n, head);
  free(b.p);
}
