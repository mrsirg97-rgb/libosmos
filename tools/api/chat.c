/* POST /v1/chat/completions (api.h's api_chat): the request's messages copied, their special tokens defused, rendered
   (the chat encoding) and tokenized, its mark found, run on the engine - its answer reply.c's */
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
      char jerr[OSMOS_ERR] = "";
      Json *o = NULL;
      for (int d = 0; d < 2 && (o ? o : v)->type == JSON_STRING; d++)
      {
        Json *next = json_parse_within((o ? o : v)->s, (o ? o : v)->len, API_TREE_MAX, NULL, jerr);
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
static int32_t *prompt_ids(Api *a, const Request *rq, size_t *n, char err[OSMOS_ERR])
{
  char cerr[OSMOS_ERR] = "";
  JsonBuf text = {0}, shape = {0};
  if (!a->d->render(rq, &text, cerr))
    return free(text.p), osmos_fail(err, "%s", cerr), NULL;
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
           osmos_fail(err, "%s", why ? why : ids ? "the prompt's own special tokens could not be told from its "
                                                          "strings' text" : "out of memory"), NULL;
  *n = (size_t)k;
  return ids;
}

/* where a turn after a new user message leaves this prompt, when the template then drops its latest reasoning
   (Dialect.drops): the tokens this prompt shares with its next turn rendered (one more user message) - a point
   osmos-serve keeps (GEN mark=, 0.7.5). 0: none, or no shorter than the prompt's own point (n - 1) */
static unsigned next_mark(Api *a, const Request *rq, const int32_t *ids, size_t n)
{
  if (!a->d->drops || !a->d->drops(rq))
    return 0;
  JsonBuf b = {0};
  json_dump(&b, rq->messages);
  char jerr[OSMOS_ERR] = "", err[OSMOS_ERR] = "";
  Json *next = NULL;
  if (!b.oom && b.n > 1 && b.p[b.n - 1] == ']')
  {
    b.n--;
    json_puts(&b, ", {\"role\": \"user\", \"content\": \"next\"}]");
    next = b.oom ? NULL : json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr);
  }
  free(b.p);
  if (!next)
    return 0;
  Request nx = *rq;
  nx.messages = next, nx.branch = NULL;
  size_t m = 0;
  int32_t *nid = prompt_ids(a, &nx, &m, err);
  json_free(next);
  size_t k = 0;
  while (nid && k < m && k < n && nid[k] == ids[k])
    k++;
  free(nid);
  return k + 1 < n ? (unsigned)k : 0;
}

void api_chat(Api *a, Http *h)
{
  char err[OSMOS_ERR] = "", jerr[OSMOS_ERR] = "";
  Json *body = json_parse_within(h->body, h->body_n, API_TREE_MAX, NULL, jerr);
  if (!body || body->type != JSON_OBJECT)
  {
    osmos_fail(err, "the body is not a JSON object%s%s", body ? "" : ": ", body ? "" : jerr);
    json_free(body), http_error(h, 400, err);
    return;
  }
  Request rq;
  const bool ok = request_read(body, a->d, a->fill_orphans, a->preserve_thinking, a->temperature, a->top_p, a->top_k, &rq, err);
  json_free(body);
  if (!ok)
  {
    http_error(h, 400, err);
    return;
  }
  size_t n = 0;
  int32_t *ids = prompt_ids(a, &rq, &n, err);
  const unsigned ctx = a->engine.max_ctx;
  bool loading = false, over = false; /* the refusal's kind: a 503 to retry, OpenAI's context_length_exceeded */
  if (ids && !engine_ready(&a->engine)) /* loading: no context known yet (its 503 below, said as it is) */
    snprintf(err, sizeof err, "the engine is not ready (loading)"), free(ids), ids = NULL, loading = true;
  else if (ids && n >= ctx)
    snprintf(err, sizeof err, "the prompt is %zu tokens: the context is %u", n, ctx), free(ids), ids = NULL, over = true;
  if (ids && rq.max_tokens && n + rq.max_tokens > ctx)
    rq.max_tokens = (unsigned)(ctx - n); /* past the context: as many as it leaves (finish "length") */
  const Sampling samp = {rq.temperature, rq.top_p, rq.seed, rq.top_k};
  Run *r = ids ? engine_run(&a->engine, ids, n, rq.max_tokens, next_mark(a, &rq, ids, n), &samp, err) : NULL;
  free(ids);
  if (!r) /* 503 for loading alone (a request's own fault - its template's - a 400 then too: 0.9.2's, retried) */
  {
    if (over)
      http_error_code(h, 400, err, "messages", "context_length_exceeded");
    else
      http_error_head(h, loading ? 503 : 400, err, NULL, NULL,
                      loading ? "Retry-After: 10\r\n" : NULL); /* (a load takes minutes: a client's retries spaced) */
    request_free(&rq);
    return;
  }
  Answer s = {.fd = h->fd, .http = h, .keepalive_ms = a->keepalive_ms ? a->keepalive_ms : KEEPALIVE_MS, .stream = rq.stream, .model = a->model, .d = a->d, .created = (long long)time(NULL),
              .phase = rq.mode == CHAT_THINKING ? PHASE_REASONING : PHASE_CONTENT, .loop = &rq.loop,
              .sampled = rq.sampled, .seed = rq.seed,
              .boundary = a->d->header != NULL};
  snprintf(s.id, sizeof s.id, "chatcmpl-osmos-%s%lld", reply_boot_tag(), r->rid);
  char head[512];
  reply_loop_head(head, sizeof head, &s);
  if (rq.stream && !http_stream_open_h(h, head))
    s.gone = true;
  answer_chunk(&s, "{\"role\": \"assistant\", \"content\": \"\"}", NULL, "");
  size_t seen = 0;
  while (reply_more(a, r, &seen, &s) && !s.gone)
    answer_advance(&s);
  if (s.gone)
  {
    engine_stop(&a->engine, r);
    while (reply_more(a, r, &seen, &s)) /* its DONE (cancel) awaited: the run is the engine's until then */
      ;
  }
  else if (!strcmp(r->finish, "error"))
    reply_fail(h, rq.stream, r->err, NULL, NULL);
  else if (!strcmp(r->finish, "cancel")) /* cancelled with its client still here: the engine's QUIT (this server
                                            stopping) - a cut answer, never a "stop" (0.9.2): a stream's error event,
                                            no [DONE]; a whole answer's 503, its retry for the next server */
    reply_fail_at(h, 503, rq.stream, "the server stopped before the answer's end (osmos-api shutting down)", NULL, NULL);
  else
  {
    answer_advance(&s);
    char perr[OSMOS_ERR] = "";
    bool parsed;
    Json *m = reply_message(&s, &rq, r->rid, &parsed, perr);
    if (m && !parsed && !strcmp(r->finish, "stop"))
    { /* the turn ended out of the format (a calls block, a special token's text, no </think>): an error, never its
         text as the answer (#25). cut by max_tokens ("length"), its text goes as it is */
      const bool called = memmem(s.shape.p, s.shape.n, a->d->calls_open, strlen(a->d->calls_open)) != NULL;
      snprintf(err, sizeof err, "the model's %s did not parse (%s)", called ? "tool call" : "answer", perr);
      reply_fail(h, rq.stream, err, &s.text, a->d->eos);
    }
    else if (!m)
      reply_fail(h, rq.stream, "out of memory", NULL, NULL); /* a stream's open: its error event, not a status line */
    else if (rq.stream)
    {
      if (!parsed)
        answer_flush(&s); /* cut by max_tokens: what was held, as the whole answer's content has it */
      reply_stream_end(&s, &rq, r, m, n);
    }
    else
      reply_whole(&s, r, m, n);
    json_free(m);
  }
  free(s.text.p), free(s.shape.p);
  request_free(&rq);
  engine_release(&a->engine, r);
}
