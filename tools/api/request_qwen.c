#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"

/* Qwen's reader: the messages as its template takes them - `reasoning` as reasoning_content, a tool call's
   arguments given as a JSON string parsed into the object the template needs (as llama.cpp's server does) - the
   tools apart, the reasoning effort's word as asked (the template names the ones it takes), preserve_thinking */

static bool blank(const Json *s)
{
  for (size_t i = 0; i < s->len; i++)
    if (!strchr(" \t\r\n", s->s[i]))
      return false;
  return true;
}

/* a call's arguments: an object, or a JSON string of one parsed (a blank one as it is: the template takes it) */
static bool arguments(JsonBuf *b, const Json *a, char *err)
{
  if (a->type != JSON_STRING || blank(a))
    return json_dump(b, a), true;
  char jerr[JSON_ERR];
  Json *o = json_parse(a->s, a->len, jerr);
  const bool ok = o && o->type == JSON_OBJECT;
  if (ok)
    json_dump(b, o);
  json_free(o);
  return ok || request_fail(err, "tool_calls: a call's arguments are not a JSON object");
}

typedef bool (*Writer)(JsonBuf *b, const Json *v, char *err);

/* an object written back, the value of `key` by fn */
static bool object(JsonBuf *b, const Json *o, const char *key, Writer fn, char *err)
{
  json_puts(b, "{");
  for (size_t i = 0; i < o->n; i++)
  {
    json_puts(b, i ? ", " : ""), json_dump_string(b, o->key[i], o->klen[i]), json_puts(b, ": ");
    if (!strcmp(o->key[i], key) ? !fn(b, o->item[i], err) : (json_dump(b, o->item[i]), false))
      return false;
  }
  return json_puts(b, "}"), true;
}

static bool function(JsonBuf *b, const Json *f, char *err)
{
  return f->type != JSON_OBJECT ? (json_dump(b, f), true) : object(b, f, "arguments", arguments, err);
}

static bool one_call(JsonBuf *b, const Json *c, char *err)
{
  if (c->type != JSON_OBJECT)
    return json_dump(b, c), true;
  return json_get(c, "function") ? object(b, c, "function", function, err) : object(b, c, "arguments", arguments, err);
}

static bool calls(JsonBuf *b, const Json *list, char *err)
{
  if (list->type != JSON_ARRAY)
    return json_dump(b, list), true;
  json_puts(b, "[");
  for (size_t i = 0; i < list->n; i++)
    if (json_puts(b, i ? ", " : ""), !one_call(b, list->item[i], err))
      return false;
  return json_puts(b, "]"), true;
}

/* a message written back: `reasoning` as reasoning_content (unless that is given too), its calls' arguments objects */
static bool message(JsonBuf *b, const Json *m, char *err)
{
  const bool has_rc = json_get(m, "reasoning_content") != NULL;
  json_puts(b, "{");
  for (size_t i = 0, k = 0; i < m->n; i++)
  {
    const bool reasoning = !strcmp(m->key[i], "reasoning");
    if (reasoning && has_rc)
      continue;
    json_puts(b, k++ ? ", " : "");
    if (reasoning)
      json_puts(b, "\"reasoning_content\"");
    else
      json_dump_string(b, m->key[i], m->klen[i]);
    json_puts(b, ": ");
    if (!strcmp(m->key[i], "tool_calls") ? !calls(b, m->item[i], err) : (json_dump(b, m->item[i]), false))
      return false;
  }
  return json_puts(b, "}"), true;
}

bool request_qwen(const Json *body, Request *r, char err[API_ERR])
{
  const Json *msgs = json_get(body, "messages"), *eff = request_effort(body);
  const Json *pt = json_get(json_get(body, "chat_template_kwargs"), "preserve_thinking");
  if (eff && eff->type != JSON_NULL && (eff->type != JSON_STRING || eff->len >= sizeof r->level))
    return request_fail(err, "reasoning_effort: low, medium, high, xhigh or none");
  if (pt && pt->type != JSON_TRUE && pt->type != JSON_FALSE && pt->type != JSON_NULL)
    return request_fail(err, "chat_template_kwargs.preserve_thinking: true or false");
  if (eff && eff->type == JSON_STRING && r->mode == CHAT_THINKING)
    memcpy(r->level, eff->s, eff->len + 1);
  r->preserve = !pt || pt->type != JSON_FALSE;
  JsonBuf b = {0};
  json_puts(&b, "[");
  for (size_t i = 0; i < msgs->n; i++)
    if (json_puts(&b, i ? ", " : ""), !message(&b, msgs->item[i], err))
      return free(b.p), false;
  json_puts(&b, "]");
  char jerr[JSON_ERR];
  r->messages = b.oom ? NULL : json_parse(b.p, b.n, jerr);
  free(b.p);
  r->tools = request_tools(body);
  return r->messages || request_fail(err, "out of memory");
}
