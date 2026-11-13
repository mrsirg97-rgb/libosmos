#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>

#include "answer.h"
#include "osmos/text/chat_qwen.h"

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
static Json *message(const Answer *s, const Request *rq, long long rid, bool *parsed, char err[CHAT_ERR])
{
  char jerr[JSON_ERR];
  JsonBuf b = {0};
  const ChatMode mode = rq->mode;
  if ((*parsed = s->d->parse(rq, s->text.p, s->shape.p, s->text.n, &b, err)))
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
  Json *m = b.oom ? NULL : json_parse(b.p, b.n, jerr);
  free(b.p);
  return m;
}

/* a run that gives no answer: a streamed one's error event (no [DONE]), or a 500. a completion its parse refused
   goes with it as its text (the EOS dropped): greedy, the same messages again are the same completion, so a client
   can only get past one by showing the model what it wrote (#34) - sent back, a client's text is defused as any */
static void fail(int fd, bool stream, const char *why, const JsonBuf *completion, const char *eos)
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
      http_error(fd, 500, why);
  }
  else if (stream)
    http_stream_event(fd, e.p, e.n);
  else
    http_send(fd, 500, "application/json", e.p, e.n);
  free(e.p);
}

/* the x-osmos-loop header line (loop.h's value, "\r\n" ended) */
static void loop_head(char *out, size_t cap, const Loop *l)
{
  char v[400];
  loop_header(v, sizeof v, l);
  snprintf(out, cap, "x-osmos-loop: %s\r\n", v);
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
  loop_json(&u, &rq->loop); /* the history's loop signals, every stream's last chunk (0.7.3) */
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
  loop_json(&b, s->loop);
  json_puts(&b, "}");
  char head[512];
  loop_head(head, sizeof head, s->loop);
  if (b.oom)
    http_error(s->fd, 500, "out of memory");
  else
    http_send_h(s->fd, 200, "application/json", b.p, b.n, head);
  free(b.p);
}

/* a deep copy of j; NULL for NULL or out of memory (*oom) */
static Json *copy(const Json *j, bool *oom)
{
  if (!j)
    return NULL;
  Json *c = calloc(1, sizeof *c);
  if (!c)
    return *oom = true, NULL;
  *c = (Json){.type = j->type, .len = j->len, .f = j->f};
  bool ok = !j->s || (c->s = malloc(j->len + 1));
  if (ok && j->s)
    memcpy(c->s, j->s, j->len + 1);
  if (ok && j->n)
    ok = (c->item = calloc(j->n, sizeof *c->item)) && (!j->key || ((c->key = calloc(j->n, sizeof *c->key)) &&
                                                                  (c->klen = malloc(j->n * sizeof *c->klen))));
  for (size_t i = 0; ok && i < j->n; c->n = ++i)
  {
    ok = (c->item[i] = copy(j->item[i], oom)) != NULL;
    if (ok && j->key)
      ok = (c->key[i] = malloc(j->klen[i] + 1)) != NULL;
    if (ok && j->key)
      memcpy(c->key[i], j->key[i], j->klen[i] + 1), c->klen[i] = j->klen[i];
  }
  if (!ok)
    return json_free(c), *oom = true, NULL;
  return c;
}

/* an array's text parts' strings as one (a template joins them: Qwen's with nothing between), defused, the bytes
   written back: a special token's text cut across two parts is text too. a part's own the string's defuse finds */
static size_t defuse_parts(Json *arr, const Tokenizer *t)
{
  JsonBuf all = {0};
  size_t parts = 0, changed = 0;
  for (size_t i = 0; i < arr->n; i++)
  {
    const Json *x = json_get(arr->item[i], "text");
    if (x && x->type == JSON_STRING)
      json_put(&all, x->s, x->len), parts++;
  }
  if (parts > 1 && !all.oom)
  {
    changed = tokenizer_defuse(t, all.p, all.n);
    size_t at = 0;
    for (size_t i = 0; i < arr->n; i++)
    {
      Json *x = (Json *)json_get(arr->item[i], "text");
      if (x && x->type == JSON_STRING)
        memcpy(x->s, all.p + at, x->len), at += x->len;
    }
  }
  free(all.p);
  return changed;
}

/* j's strings and keys defused in place (tokenizer_defuse), the bytes changed. a call's
   "arguments" as a JSON string the template reads as its object (chat_tools.c's arguments(): decoded twice at most):
   an escaped special token's text in it is that text once decoded - so its object, decoded and defused, in its place */
static size_t defuse(Json *j, const Tokenizer *t)
{
  size_t changed = j->type == JSON_STRING ? tokenizer_defuse(t, j->s, j->len) : 0;
  if (j->type == JSON_ARRAY)
    changed += defuse_parts(j, t);
  for (size_t i = 0; i < j->n; i++)
  {
    Json *v = j->item[i];
    if (j->key && !strcmp(j->key[i], "arguments") && v->type == JSON_STRING)
    {
      char jerr[JSON_ERR];
      Json *o = NULL;
      for (int d = 0; d < 2 && (o ? o : v)->type == JSON_STRING; d++)
      {
        Json *next = json_parse((o ? o : v)->s, (o ? o : v)->len, jerr);
        if (!next)
          break;
        json_free(o), o = next;
      }
      const size_t q = o && o->type == JSON_OBJECT ? defuse(o, t) : 0;
      if (q)
        json_free(v), j->item[i] = v = o, o = NULL, changed += q;
      json_free(o);
      if (q)
        continue;
    }
    changed += defuse(v, t);
    if (j->key)
      changed += tokenizer_defuse(t, j->key[i], j->klen[i]);
  }
  return changed;
}

/* the prompt's ids: the rendered messages, a special token only where the template put one. a client's string that
   holds a special token's text (a tool's result quoting a chat's markup) is that text: the request rendered again
   from its strings defused is the shape (tokenizer_encode_shaped), and the two renders must differ only in the
   bytes defused - else refused, never guessed */
static int32_t *prompt_ids(Api *a, const Request *rq, size_t *n, char err[API_ERR])
{
  char cerr[CHAT_ERR];
  JsonBuf text = {0}, shape = {0};
  if (!a->d->render(rq, &text, cerr))
    return free(text.p), snprintf(err, API_ERR, "%s", cerr), NULL;
  size_t changed = 0;
  bool oom = text.oom;
  Request sh = *rq;
  sh.messages = copy(rq->messages, &oom), sh.tools = copy(rq->tools, &oom);
  sh.branch = rq->messages; /* the template's choices the text's (a defused string reads otherwise) */
  if (sh.messages)
    changed += defuse(sh.messages, a->tok);
  if (sh.tools)
    changed += defuse(sh.tools, a->tok);
  const char *why = oom ? "out of memory" : NULL;
  if (!why && changed && !a->d->render(&sh, &shape, cerr))
    why = "the request's strings defused, its prompt refused";
  if (!why && changed && (shape.oom || shape.n != text.n))
    why = shape.oom ? "out of memory" : "the prompt's own special tokens could not be told from its strings' text";
  json_free(sh.messages), json_free(sh.tools);
  int32_t *ids = why ? NULL : malloc((text.n + 1) * sizeof *ids); /* a token is a byte or more */
  const int64_t k = !ids ? -1 : changed ? tokenizer_encode_shaped(a->tok, text.p, shape.p, text.n, ids, text.n + 1)
                                        : tokenizer_encode(a->tok, text.p, text.n, true, ids, text.n + 1);
  free(text.p), free(shape.p);
  if (k < 0 || (size_t)k > text.n + 1) /* (encode_shaped's -1: a shape not the text defused) */
    return free(ids),
           snprintf(err, API_ERR, "%s", why ? why : ids ? "the prompt's own special tokens could not be told from its "
                                                          "strings' text" : "out of memory"), NULL;
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
  const bool ok = request_read(body, a->d, a->fill_orphans, &rq, err);
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
              .phase = rq.mode == CHAT_THINKING ? PHASE_REASONING : PHASE_CONTENT, .loop = &rq.loop};
  snprintf(s.id, sizeof s.id, "chatcmpl-osmos-%lld", r->rid);
  char head[512];
  loop_head(head, sizeof head, &rq.loop);
  if (rq.stream && !http_stream_open_h(h->fd, head))
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
    fail(h->fd, rq.stream, r->err, NULL, NULL);
  else
  {
    answer_advance(&s);
    char perr[CHAT_ERR] = "";
    bool parsed;
    Json *m = message(&s, &rq, r->rid, &parsed, perr);
    if (m && !parsed && !strcmp(r->finish, "stop"))
    { /* the turn ended out of the format (a calls block, a special token's text, no </think>): an error, never its
         text as the answer (#25). cut by max_tokens ("length"), its text goes as it is */
      const bool called = memmem(s.shape.p, s.shape.n, a->d->calls_open, strlen(a->d->calls_open)) != NULL;
      snprintf(err, sizeof err, "the model's %s did not parse (%s)", called ? "tool call" : "answer", perr);
      fail(h->fd, rq.stream, err, &s.text, a->d->eos);
    }
    else if (!m)
      fail(h->fd, rq.stream, "out of memory", NULL, NULL); /* a stream's open: its error event, not a status line */
    else if (rq.stream)
    {
      if (!parsed)
        answer_flush(&s); /* cut by max_tokens: what was held, as the whole answer's content has it */
      stream_end(&s, &rq, r, m, n);
    }
    else
      whole(&s, r, m, n);
    json_free(m);
  }
  free(s.text.p), free(s.shape.p);
  request_free(&rq);
  engine_release(&a->engine, r);
}
