#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>

#include "chat_qwen_impl.h"

/* a completion read back as the message the template would render it from (osmos/text/chat_qwen.h): the reasoning
   before </think>, the content before the first <tool_call>, each call's parameters typed by its tool's schema */

#define MAX_PARAMS 64 /* a call's parameters read at most - more refused, not cut (a tool schema's are a handful) */

const char *const qchat_refused[] = {"<|im_start|>", QCHAT_EOS, "<|endoftext|>", NULL};

static bool starts(Span s, const char *lit) { return s.n >= strlen(lit) && !memcmp(s.p, lit, strlen(lit)); }

static Span lstrip(Span s) /* the format's own whitespace between its tags */
{
  while (s.n && strchr(" \t\r\n", *s.p))
    s.p++, s.n--;
  return s;
}

/* rest past lit (after whitespace), or false */
static bool take(Span *rest, const char *lit)
{
  Span r = lstrip(*rest);
  if (!starts(r, lit))
    return false;
  *rest = (Span){r.p + strlen(lit), r.n - strlen(lit)};
  return true;
}

/* rest's text before stop, rest past it; false without one */
static bool upto(Span *rest, const char *stop, Span *before)
{
  const char *k = memmem(rest->p, rest->n, stop, strlen(stop));
  if (!k)
    return false;
  *before = (Span){rest->p, (size_t)(k - rest->p)};
  const size_t past = (size_t)(k - rest->p) + strlen(stop);
  *rest = (Span){rest->p + past, rest->n - past};
  return true;
}

static bool name_ok(Span s) { return s.n && !memchr(s.p, '\n', s.n) && !memchr(s.p, '<', s.n); }

/* the type names the schema allows a function's parameter (its "type": one, or a list), or NULL: any */
static const Json *types_of(const Json *tools, Span fn, Span param)
{
  const Json *props = NULL;
  for (size_t i = 0; tools && i < tools->n; i++)
  {
    const Json *f = json_get(tools->item[i], "function") ? json_get(tools->item[i], "function") : tools->item[i];
    const Json *name = json_get(f, "name");
    if (name && name->type == JSON_STRING && name->len == fn.n && !memcmp(name->s, fn.p, fn.n))
      props = json_get(json_get(f, "parameters"), "properties");
  }
  for (size_t i = 0; props && props->type == JSON_OBJECT && i < props->n; i++)
    if (props->klen[i] == param.n && !memcmp(props->key[i], param.p, param.n))
      return json_get(props->item[i], "type");
  return NULL;
}

static bool allows(const Json *types, const char *kind)
{
  if (!types)
    return true;
  if (types->type == JSON_STRING)
    return !strcmp(types->s, kind);
  for (size_t i = 0; types->type == JSON_ARRAY && i < types->n; i++)
    if (types->item[i]->type == JSON_STRING && !strcmp(types->item[i]->s, kind))
      return true;
  return false;
}

static const char *kind_of(const Json *v)
{
  switch (v->type)
  {
  case JSON_TRUE:
  case JSON_FALSE:
    return "boolean";
  case JSON_NULL:
    return "null";
  case JSON_INT:
    return "integer";
  case JSON_FLOAT:
    return "number";
  case JSON_OBJECT:
    return "object";
  case JSON_ARRAY:
    return "array";
  default:
    return "string";
  }
}

/* a parameter's value into args: its JSON where that is no string and of a type allowed, else its text */
static void value(JsonBuf *args, Span raw, const Json *types)
{
  char jerr[JSON_ERR];
  Json *v = json_parse(raw.p, raw.n, jerr);
  const char *kind = v ? kind_of(v) : "string";
  const bool typed = v && strcmp(kind, "string") &&
                     (allows(types, kind) || (!strcmp(kind, "integer") && allows(types, "number")));
  if (typed)
    json_dump(args, v);
  else
    json_dump_string(args, raw.p, raw.n);
  json_free(v);
}

/* one <tool_call> block from rest: its call's JSON into calls */
static bool call(Span *rest, const Json *tools, JsonBuf *calls, char *err)
{
  Span fn, names[MAX_PARAMS];
  if (!take(rest, QCHAT_CALL_OPEN) || !take(rest, "<function=") || !upto(rest, ">", &fn) || !name_ok(fn))
    return qchat_fail(err, "a tool call's <tool_call><function=name> malformed");
  JsonBuf args = {0};
  size_t n = 0;
  json_puts(&args, "{");
  for (Span r = lstrip(*rest); starts(r, "<parameter="); r = lstrip(*rest))
  {
    Span name, raw;
    *rest = (Span){r.p + strlen("<parameter="), r.n - strlen("<parameter=")};
    bool ok = upto(rest, ">", &name) && name_ok(name) && upto(rest, "</parameter>", &raw) && n < MAX_PARAMS;
    for (size_t i = 0; ok && i < n; i++)
      ok = !(names[i].n == name.n && !memcmp(names[i].p, name.p, name.n));
    if (!ok)
      return free(args.p), qchat_fail(err, "a tool call's parameter malformed, repeated or past the most read");
    if (raw.n && raw.p[0] == '\n')
      raw.p++, raw.n--;
    if (raw.n && raw.p[raw.n - 1] == '\n')
      raw.n--;
    json_puts(&args, n ? ", " : ""), json_dump_string(&args, name.p, name.n), json_puts(&args, ": ");
    value(&args, raw, types_of(tools, fn, name));
    names[n++] = name;
  }
  json_puts(&args, "}");
  if (!take(rest, "</function>") || !take(rest, "</tool_call>"))
    return free(args.p), qchat_fail(err, "a tool call not closed by </function></tool_call>");
  json_puts(calls, "{\"type\": \"function\", \"function\": {\"name\": "), json_dump_string(calls, fn.p, fn.n);
  json_puts(calls, ", \"arguments\": "), json_dump_string(calls, args.p ? args.p : "", args.n), json_puts(calls, "}}");
  const bool oom = args.oom;
  free(args.p);
  return !oom || qchat_fail(err, "out of memory");
}

bool qchat_parse(const char *text, size_t len, ChatMode mode, const Json *tools, JsonBuf *out, char err[CHAT_ERR])
{
  const size_t en = strlen(QCHAT_EOS);
  if (!text || len < en || memcmp(text + len - en, QCHAT_EOS, en))
    return qchat_fail(err, "no end of turn (" QCHAT_EOS ")");
  Span body = {text, len - en}, reasoning = {"", 0};
  for (size_t i = 0; qchat_refused[i]; i++)
    if (memmem(body.p, body.n, qchat_refused[i], strlen(qchat_refused[i])))
      return qchat_fail(err, "a special token in the completion's text");
  if (mode == CHAT_THINKING && !upto(&body, QCHAT_THINK_END, &reasoning))
    return qchat_fail(err, "no " QCHAT_THINK_END " in a thinking completion");
  const char *at = memmem(body.p, body.n, QCHAT_CALL_OPEN, strlen(QCHAT_CALL_OPEN));
  Span content = {body.p, at ? (size_t)(at - body.p) : body.n};
  Span rest = {content.p + content.n, body.n - content.n};
  content = qchat_trim(content), reasoning = qchat_trim(reasoning);
  JsonBuf calls = {0};
  json_puts(&calls, "[");
  for (size_t k = 0; lstrip(rest).n; k++)
  {
    json_puts(&calls, k ? ", " : "");
    if (!call(&rest, tools && tools->type == JSON_ARRAY ? tools : NULL, &calls, err))
      return free(calls.p), false;
  }
  json_puts(&calls, "]");
  json_puts(out, "{\"role\": \"assistant\", \"content\": "), json_dump_string(out, content.p, content.n);
  json_puts(out, ", \"reasoning_content\": "), json_dump_string(out, reasoning.p, reasoning.n);
  json_puts(out, ", \"tool_calls\": "), json_put(out, calls.p, calls.n), json_puts(out, "}");
  const bool oom = calls.oom || out->oom;
  free(calls.p);
  return !oom || qchat_fail(err, "out of memory");
}
