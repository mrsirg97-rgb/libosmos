#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>

#include "answer.h"

/* POST /v1/chat/completions: the request rendered (the chat encoding), tokenized, run on the engine; its tokens
   streamed as they land (answer.c) or answered whole; at DONE the completion parsed for its tool calls, then the
   finish, usage and timings (DESIGN.md, "osmos-api: the answer") */

#define WAIT_MS 500 /* how often a waiting request looks whether its client went away */

/* the client closed its end (a stream's reader gone: its run STOPped) */
static bool closed(int fd)
{
  char c;
  const ssize_t k = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  return k == 0 || (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

/* the run's tokens since `seen` as text; false when the run is done (and all of it read) */
static bool more(Api *a, Run *r, size_t *seen, Answer *s)
{
  pthread_mutex_lock(&r->mu);
  while (*seen == r->n && !r->done)
  {
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
  if (to > from && !tok)
    return s->gone = true, false; /* out of memory: given up as a client gone */
  char piece[TOKENIZER_PIECE_MAX];
  for (size_t i = 0; i < to - from; i++)
  {
    const int k = tokenizer_piece(a->tok, tok[i], true, piece);
    if (k > 0)
      json_put(&s->text, piece, (size_t)k);
  }
  free(tok);
  *seen = to;
  return !done;
}

/* the message parsed - tool calls given their ids - or, out of the format, its text as it streamed (*parsed false,
   err the parse's why) */
static Json *message(const Answer *s, const Request *rq, long long rid, bool *parsed, char err[CHAT_ERR])
{
  char jerr[JSON_ERR];
  JsonBuf b = {0};
  const ChatMode mode = rq->mode;
  if ((*parsed = s->d->parse(rq, s->text.p, s->text.n, &b, err)))
  {
    Json *m = json_parse(b.p, b.n, jerr);
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
      id->type = JSON_STRING, id->len = (size_t)snprintf(id->s, 48, "call_%lld_%zu", rid, i);
      c->item[c->n] = id, c->klen[c->n] = 2, c->n++;
    }
    return m;
  }
  free(b.p);
  /* out of the format: reasoning before </think>, the content after (the EOS dropped) */
  const char *p = s->text.p ? s->text.p : "";
  size_t n = s->text.n;
  const char *eos_text = s->d->eos, *think_end = s->d->think_end;
  const size_t eos = strlen(eos_text);
  if (n >= eos && !memcmp(p + n - eos, eos_text, eos))
    n -= eos;
  const char *end = mode == CHAT_THINKING ? memmem(p, n, think_end, strlen(think_end)) : NULL;
  const size_t rn = mode != CHAT_THINKING ? 0 : end ? (size_t)(end - p) : n;
  const size_t c0 = mode != CHAT_THINKING ? 0 : end ? rn + strlen(think_end) : n;
  json_puts(&b, "{\"role\": \"assistant\", \"content\": "), json_dump_string(&b, p + c0, n - c0);
  json_puts(&b, ", \"reasoning_content\": "), json_dump_string(&b, p, rn), json_puts(&b, ", \"tool_calls\": []}");
  Json *m = b.oom ? NULL : json_parse(b.p, b.n, jerr);
  free(b.p);
  return m;
}

/* a run that gives no answer: a streamed one's error event (no [DONE]), or a 500 */
static void fail(int fd, bool stream, const char *why)
{
  if (!stream)
  {
    http_error(fd, 500, why);
    return;
  }
  JsonBuf e = {0};
  json_puts(&e, "{\"error\": {\"message\": "), json_dump_string(&e, why, strlen(why));
  json_puts(&e, ", \"type\": \"server_error\"}}");
  if (!e.oom)
    http_stream_event(fd, e.p, e.n);
  free(e.p);
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
static void stream_end(Answer *s, const Request *rq, const Run *r, const Json *m, size_t prompt_n)
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
  answer_chunk(s, "{}", finish_of(r, m), u.p ? u.p : "");
  free(u.p);
  if (!s->gone)
    s->gone = !http_stream_event(s->fd, "[DONE]", 6);
}

static void whole(Answer *s, const Run *r, const Json *m, size_t prompt_n)
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
  json_puts(&b, "}");
  if (b.oom)
    http_error(s->fd, 500, "out of memory");
  else
    http_send(s->fd, 200, "application/json", b.p, b.n);
  free(b.p);
}

/* the prompt's ids: the rendered messages (their special tokens' text matched as tokens) */
static int32_t *prompt_ids(Api *a, const Request *rq, size_t *n, char err[API_ERR])
{
  char cerr[CHAT_ERR];
  JsonBuf text = {0};
  if (!a->d->render(rq, &text, cerr))
    return free(text.p), snprintf(err, API_ERR, "%s", cerr), NULL;
  int32_t *ids = text.oom ? NULL : malloc((text.n + 1) * sizeof *ids); /* a token is a byte or more */
  const int64_t k = ids ? tokenizer_encode(a->tok, text.p, text.n, true, ids, text.n + 1) : -1;
  free(text.p);
  if (k < 0 || (size_t)k > text.n + 1)
    return free(ids), snprintf(err, API_ERR, "out of memory"), NULL;
  *n = (size_t)k;
  return ids;
}

void api_chat(Api *a, Http *h)
{
  char err[API_ERR], jerr[JSON_ERR];
  Json *body = json_parse(h->body, h->body_n, jerr);
  if (!body || body->type != JSON_OBJECT)
  {
    snprintf(err, sizeof err, "the body is not a JSON object%s%s", body ? "" : ": ", body ? "" : jerr);
    json_free(body), http_error(h->fd, 400, err);
    return;
  }
  Request rq;
  const bool ok = request_read(body, a->d, &rq, err);
  json_free(body);
  if (!ok)
  {
    http_error(h->fd, 400, err);
    return;
  }
  size_t n = 0;
  int32_t *ids = prompt_ids(a, &rq, &n, err);
  const unsigned ctx = a->engine.max_ctx;
  if (ids && n >= ctx)
    snprintf(err, sizeof err, "the prompt is %zu tokens: the context is %u", n, ctx), free(ids), ids = NULL;
  if (ids && rq.max_tokens && n + rq.max_tokens > ctx)
    rq.max_tokens = (unsigned)(ctx - n); /* past the context: as many as it leaves (finish "length") */
  Run *r = ids ? engine_run(&a->engine, ids, n, rq.max_tokens, err) : NULL;
  free(ids);
  if (!r)
  {
    http_error(h->fd, engine_ready(&a->engine) ? 400 : 503, err);
    request_free(&rq);
    return;
  }
  Answer s = {.fd = h->fd, .stream = rq.stream, .model = a->model, .d = a->d, .created = (long long)time(NULL),
              .phase = rq.mode == CHAT_THINKING ? PHASE_REASONING : PHASE_CONTENT};
  snprintf(s.id, sizeof s.id, "chatcmpl-osmos-%lld", r->rid);
  if (rq.stream && !http_stream_open(h->fd))
    s.gone = true;
  answer_chunk(&s, "{\"role\": \"assistant\", \"content\": \"\"}", NULL, "");
  size_t seen = 0;
  while (more(a, r, &seen, &s) && !s.gone)
    answer_advance(&s);
  if (s.gone)
  {
    engine_stop(&a->engine, r);
    while (more(a, r, &seen, &s)) /* its DONE (cancel) awaited: the run is the engine's until then */
      ;
  }
  else if (!strcmp(r->finish, "error"))
    fail(h->fd, rq.stream, r->err);
  else
  {
    answer_advance(&s);
    char perr[CHAT_ERR] = "";
    bool parsed;
    Json *m = message(&s, &rq, r->rid, &parsed, perr);
    if (m && !parsed && !strcmp(r->finish, "stop") &&
        memmem(s.text.p, s.text.n, a->d->calls_open, strlen(a->d->calls_open)))
    { /* the turn ended on a calls block its parse refuses: an error, never its markup as the answer (#25) */
      snprintf(err, sizeof err, "the model's tool call did not parse (%s)", perr);
      fail(h->fd, rq.stream, err);
    }
    else if (!m)
      http_error(h->fd, 500, "out of memory");
    else if (rq.stream)
    {
      if (!(json_get(m, "tool_calls") && json_get(m, "tool_calls")->n) && s.phase == PHASE_DONE &&
          memmem(s.text.p, s.text.n, a->d->calls_open, strlen(a->d->calls_open)))
        answer_flush(&s); /* a calls block that did not parse: its text */
      else if (s.phase != PHASE_DONE)
        answer_flush(&s); /* no EOS (length): what was held */
      stream_end(&s, &rq, r, m, n);
    }
    else
      whole(&s, r, m, n);
    json_free(m);
  }
  free(s.text.p);
  request_free(&rq);
  engine_release(&a->engine, r);
}
