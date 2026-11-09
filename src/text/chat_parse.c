#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>

#include "chat_impl.h"

/* encoding.py's parse_message_from_completion_text and parse_tool_calls, step for step */

#define MAX_PARAMS 64 /* a call's parameters read at most - more refused, not cut (a tool schema's are a handful) */

typedef struct Span
{
  const char *p;
  size_t n;
} Span;

/* _read_until_stop: from i, the text before the earliest of the stops; *i past it; the stop's index or -1 (to the
   end) */
static int until(const char *t, size_t len, size_t *i, const char *const *stops, int n, Span *before)
{
  size_t best = len;
  int hit = -1;
  for (int k = 0; k < n; k++)
  {
    const char *f = memmem(t + *i, len - *i, stops[k], strlen(stops[k]));
    if (f && (size_t)(f - t) < best)
      best = (size_t)(f - t), hit = k;
  }
  *before = (Span){t + *i, best - *i};
  *i = hit < 0 ? len : best + strlen(stops[hit]);
  return hit;
}

static bool is(Span s, const char *lit) { return s.n == strlen(lit) && memcmp(s.p, lit, s.n) == 0; }

static bool has(Span s, const char *lit) { return memmem(s.p, s.n, lit, strlen(lit)) != NULL; }

/* a regex's `lit$` as Python matches it (no MULTILINE: `$` is the end, or before a newline that ends the text): where
   lit starts in s - at its end, or one '\n' before it - or -1 */
static long tail_at(Span s, const char *lit)
{
  const size_t l = strlen(lit);
  for (size_t nl = 0; nl <= 1; nl++)
    if (s.n >= l + nl && (!nl || s.p[s.n - 1] == '\n') && memcmp(s.p + s.n - nl - l, lit, l) == 0)
      return (long)(s.n - nl - l);
  return -1;
}

typedef struct Param
{
  Span name, value;
  bool string;
} Param;

/* ^ name="(.*?)" string="(true|false)">(.*?)<$ - the name to the first '" string="..">', the value to the '<' (tail_at) */
static bool param(Span s, Param *p)
{
  const char *open = " name=\"";
  const long end = tail_at(s, "<"); /* the value's end */
  if (s.n < strlen(open) || memcmp(s.p, open, strlen(open)) || end < 0)
    return false;
  const size_t from = strlen(open);
  for (size_t i = from; i < s.n; i++)
    for (int t = 1; t >= 0; t--)
    {
      const char *tail = t ? "\" string=\"true\">" : "\" string=\"false\">";
      const size_t tl = strlen(tail);
      if (i + tl <= (size_t)end && memcmp(s.p + i, tail, tl) == 0)
      {
        *p = (Param){{s.p + from, i - from}, {s.p + i + tl, (size_t)end - (i + tl)}, t == 1};
        return true;
      }
    }
  return false;
}

/* one DSML call into out (OpenAI's form): {"type": "function", "function": {"name", "arguments"}[, "namespace"]} */
static bool call(JsonBuf *out, Span name, const Param *ps, size_t np, char *err)
{
  char *nm = strndup(name.p, name.n);
  if (!nm)
    return chat_fail(err, "a tool call", "out of memory");
  const char *sep = strstr(nm, "::");
  JsonBuf q = {0};
  const bool ok = chat_tool_name(nm, NULL, &q, err); /* (validates: one '::' at most) */
  free(q.p);
  if (!ok)
    return free(nm), false;
  JsonBuf args = {0};
  json_puts(&args, "{");
  for (size_t i = 0; i < np; i++)
  {
    json_puts(&args, i ? ", " : "");
    json_dump_string(&args, ps[i].name.p, ps[i].name.n), json_puts(&args, ": ");
    if (ps[i].string)
      json_dump_string(&args, ps[i].value.p, ps[i].value.n);
    else
      json_put(&args, ps[i].value.p, ps[i].value.n); /* as written (decode_dsml_to_arguments) */
  }
  json_puts(&args, "}");
  json_puts(out, "{\"type\": \"function\", \"function\": {\"name\": ");
  if (sep)
    json_dump_string(out, sep + 2, strlen(sep + 2));
  else
    json_dump_string(out, nm, strlen(nm));
  json_puts(out, ", \"arguments\": "), json_dump_string(out, args.p, args.n), json_puts(out, "}");
  if (sep)
    json_puts(out, ", \"namespace\": "), json_dump_string(out, nm, (size_t)(sep - nm));
  json_puts(out, "}");
  free(args.p), free(nm);
  return true;
}

static bool calls(const char *t, size_t len, size_t *i, int *last, JsonBuf *out, char *err)
{
  const char *outer[] = {CHAT_INVOKE_OPEN, CHAT_CALLS_CLOSE}, *inner[] = {CHAT_PARAM_OPEN, CHAT_INVOKE_CLOSE};
  const char *pend[] = {"/" CHAT_DSML " parameter"};
  Span s;
  json_puts(out, "[");
  for (int k = 0; *i < len; k++)
  {
    const int stop = until(t, len, i, outer, 2, &s);
    if (!is(s, ">\n"))
      return chat_fail(err, "a tool call", "not '>\\n' before it");
    if (stop == 1)
    {
      *last = 1;
      break;
    }
    if (stop < 0)
      return chat_fail(err, "tool calls", "a missing special token");
    int at = until(t, len, i, inner, 2, &s); /* ^\s*name="(.*?)">\n$ (tail_at) */
    size_t w = 0;
    while (w < s.n && (s.p[w] == ' ' || (s.p[w] >= '\t' && s.p[w] <= '\r')))
      w++;
    const long nend = tail_at(s, "\">\n"); /* the name's end: a blank line after it the reference's too */
    if (s.n - w < 6 || memcmp(s.p + w, "name=\"", 6) || nend < (long)(w + 6))
      return chat_fail(err, "a tool call", "its name not name=\"..\"");
    const Span name = {s.p + w + 6, (size_t)nend - w - 6};
    Param ps[MAX_PARAMS];
    size_t np = 0;
    while (at == 0)
    {
      until(t, len, i, pend, 1, &s);
      if (np == MAX_PARAMS || !param(s, &ps[np]))
        return chat_fail(err, "a tool call",
                         np == MAX_PARAMS ? "more parameters than this reads" : "a malformed parameter");
      for (size_t d = 0; d < np; d++)
        if (ps[d].name.n == ps[np].name.n && memcmp(ps[d].name.p, ps[np].name.p, ps[np].name.n) == 0)
          return chat_fail(err, "a tool call", "a parameter twice");
      np++;
      at = until(t, len, i, inner, 2, &s);
      if (!is(s, ">\n"))
        return chat_fail(err, "a tool call", "not '>\\n' after a parameter");
    }
    json_puts(out, k ? ", " : "");
    if (!call(out, name, ps, np, err))
      return false;
    *last = at < 0 ? -1 : 2;
  }
  json_puts(out, "]");
  return true;
}

bool chat_parse(const char *text, size_t len, ChatMode mode, JsonBuf *out, char err[CHAT_ERR])
{
  err[0] = 0;
  const char *start = "\n\n" "<" CHAT_DSML " calls";
  const char *think_stops[] = {CHAT_THINK_END, start}, *body_stops[] = {CHAT_EOS, start}, *eos[] = {CHAT_EOS};
  Span reasoning = {"", 0}, content, rest;
  size_t i = 0;
  if (mode == CHAT_THINKING && until(text, len, &i, think_stops, 2, &reasoning) != 0)
    return chat_fail(err, "the completion", "no </think>");
  const int stop = until(text, len, &i, body_stops, 2, &content);
  JsonBuf tc = {0};
  int last = stop;
  if (stop == 1)
  {
    if (!calls(text, len, &i, &last, &tc, err))
      return free(tc.p), false;
    last = until(text, len, &i, eos, 1, &rest);
    if (rest.n)
      return free(tc.p), chat_fail(err, "the completion", "text after its tool calls");
  }
  else if (stop != 0)
    return chat_fail(err, "the completion", "no EOS");
  if (i != len || last > 0)
    return free(tc.p), chat_fail(err, "the completion", "text after its end");
  const char *special[] = {CHAT_BOS, CHAT_EOS, CHAT_THINK, CHAT_THINK_END, CHAT_DSML};
  for (int k = 0; k < 5; k++)
    if (has(content, special[k]) || has(reasoning, special[k]))
      return free(tc.p), chat_fail(err, "the completion", "a special token in its text");
  json_puts(out, "{\"role\": \"assistant\", \"content\": "), json_dump_string(out, content.p, content.n);
  json_puts(out, ", \"reasoning_content\": "), json_dump_string(out, reasoning.p, reasoning.n);
  json_puts(out, ", \"tool_calls\": ");
  if (tc.p)
    json_put(out, tc.p, tc.n);
  else
    json_puts(out, "[]");
  json_puts(out, "}");
  free(tc.p);
  return !out->oom || chat_fail(err, "the message", "out of memory");
}
