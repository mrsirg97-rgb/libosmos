#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_args.h"
#include "osmos/text/chat_hy4.h"
#include "osmos/err.h"

/* a completion read back as the message the template renders it from (osmos/text/chat_hy4.h): the reasoning before
   the think block's end, the content before the calls, each call's arguments typed by its tool's schema. the
   template writes no whitespace between its tags: none read */

#define MAX_ARGS 64 /* a call's arguments read at most - more refused, not cut */

typedef struct Span
{
  const char *p;
  size_t n;
} Span;

static bool take(Span *s, const char *lit)
{
  const size_t n = strlen(lit);
  if (s->n < n || memcmp(s->p, lit, n))
    return false;
  s->p += n, s->n -= n;
  return true;
}

/* s's text before stop, s past it; false without one */
static bool upto(Span *s, const char *stop, Span *before)
{
  const char *k = memmem(s->p, s->n, stop, strlen(stop));
  if (!k)
    return false;
  *before = (Span){s->p, (size_t)(k - s->p)};
  const size_t past = (size_t)(k - s->p) + strlen(stop);
  s->p += past, s->n -= past;
  return true;
}

/* a piece of text holding no special token's text */
static bool clean(Span s, const char *const *refused)
{
  for (size_t i = 0; refused[i]; i++)
    if (memmem(s.p, s.n, refused[i], strlen(refused[i])))
      return false;
  return true;
}

/* one call from rest: its JSON into calls */
static bool call(Span *rest, const Json *tools, const HchatTokens *t, const char *const *refused, JsonBuf *calls,
                 ptrdiff_t off, char *err)
{
  Span fn, keys[MAX_ARGS];
  if (!take(rest, t->call))
    return osmos_fail(err, "a tool call not opened");
  const char *k = memmem(rest->p, rest->n, t->key, strlen(t->key));
  const char *c = memmem(rest->p, rest->n, t->call_end, strlen(t->call_end));
  if (!c)
    return osmos_fail(err, "a tool call not closed");
  fn = (Span){rest->p, (size_t)((k && k < c ? k : c) - rest->p)};
  rest->p += fn.n, rest->n -= fn.n;
  if (!fn.n || memchr(fn.p, '<', fn.n) || !clean(fn, refused))
    return osmos_fail(err, "a tool call's name empty, or holding '<' or a special token");
  JsonBuf args = {0};
  size_t n = 0;
  json_puts(&args, "{");
  while (take(rest, t->key))
  {
    Span key, raw;
    bool ok = upto(rest, t->key_end, &key) && take(rest, t->value) && upto(rest, t->value_end, &raw) && n < MAX_ARGS &&
              !memchr(key.p, '<', key.n) && clean(key, refused) && clean(raw, refused);
    for (size_t i = 0; ok && i < n; i++)
      ok = !(keys[i].n == key.n && !memcmp(keys[i].p, key.p, key.n));
    if (!ok)
      return free(args.p), osmos_fail(err, "a tool call's argument malformed, repeated or past the most read");
    json_puts(&args, n ? ", " : ""), json_dump_string(&args, key.p + off, key.n), json_puts(&args, ": ");
    chat_arg_value(&args, raw.p + (raw.n ? off : 0), raw.n, chat_arg_schema(tools, fn.p + off, fn.n, key.p + off, key.n));
    keys[n++] = key;
  }
  json_puts(&args, "}");
  if (!take(rest, t->call_end))
    return free(args.p), osmos_fail(err, "a tool call not closed after its arguments");
  json_puts(calls, "{\"type\": \"function\", \"function\": {\"name\": "), json_dump_string(calls, fn.p + off, fn.n);
  json_puts(calls, ", \"arguments\": "), json_dump_string(calls, args.p ? args.p : "", args.n), json_puts(calls, "}}");
  const bool oom = args.oom;
  free(args.p);
  return !oom || osmos_fail(err, "out of memory");
}

bool hchat_parse(const char *text, size_t len, bool thinking, const Json *tools, const HchatTokens *t, JsonBuf *out,
                 char err[OSMOS_ERR])
{
  return hchat_parse_shaped(text, text, len, thinking, tools, t, out, err);
}

bool hchat_parse_shaped(const char *text, const char *shape, size_t len, bool thinking, const Json *tools,
                        const HchatTokens *t, JsonBuf *out, char err[OSMOS_ERR])
{
  const ptrdiff_t off = text - shape; /* the parse reads the shape; what it gives is the text's */
  text = shape;
  if (!text || !t || !out)
    return osmos_fail(err, "no completion, tokens or output");
  const size_t en = strlen(t->end);
  if (len < en || memcmp(text + len - en, t->end, en))
    return osmos_fail(err, "no end of turn");
  const char *refused[HCHAT_REFUSED + 1];
  hchat_refused(t, refused);
  if (tools && tools->type != JSON_ARRAY)
    tools = NULL;
  Span body = {text, len - en}, reasoning = {"", 0}, content;
  if (thinking && !upto(&body, t->think_end, &reasoning))
    return osmos_fail(err, "no end of the think block in a thinking completion");
  const char *at = memmem(body.p, body.n, t->calls, strlen(t->calls));
  content = (Span){body.p, at ? (size_t)(at - body.p) : body.n};
  Span rest = {body.p + content.n, body.n - content.n};
  if (!clean(reasoning, refused) || !clean(content, refused))
    return osmos_fail(err, "a special token's text in the reasoning or the content");
  JsonBuf calls = {0};
  json_puts(&calls, "[");
  if (take(&rest, t->calls))
  {
    for (size_t k = 0; !take(&rest, t->calls_end); k++)
    {
      json_puts(&calls, k ? ", " : "");
      if (!call(&rest, tools, t, refused, &calls, off, err))
        return free(calls.p), false;
    }
    if (rest.n)
      return free(calls.p), osmos_fail(err, "text after the tool calls");
  }
  json_puts(&calls, "]");
  json_puts(out, "{\"role\": \"assistant\", \"content\": "), json_dump_string(out, content.p + (content.n ? off : 0), content.n);
  json_puts(out, ", \"reasoning_content\": "), json_dump_string(out, reasoning.p + (reasoning.n ? off : 0), reasoning.n);
  json_puts(out, ", \"tool_calls\": "), json_put(out, calls.p, calls.n), json_puts(out, "}");
  const bool oom = calls.oom || out->oom;
  free(calls.p);
  return !oom || osmos_fail(err, "out of memory");
}
