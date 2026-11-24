#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "request.h"
#include "osmos/err.h"
#include "api.h"

/* the messages as Qwen's and Inkling's templates take them: `reasoning` as reasoning_content, a tool call's arguments
   given as a JSON string parsed into the object the templates need (as llama.cpp's server does; Inkling's template
   asks it: "canonicalize upstream") */

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
  char jerr[OSMOS_ERR] = "";
  Json *o = json_parse_within(a->s, a->len, API_TREE_MAX, NULL, jerr);
  const bool ok = o && o->type == JSON_OBJECT;
  if (ok)
    json_dump(b, o);
  json_free(o);
  if (!o) /* (the parse's own why: malformed, or past its memory budget) */
    return osmos_fail(err, "tool_calls: a call's arguments: %s", jerr);
  return ok || osmos_fail(err, "tool_calls: a call's arguments are not a JSON object");
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

bool request_messages(const Json *msgs, Json **out, char err[OSMOS_ERR])
{
  JsonBuf b = {0};
  json_puts(&b, "[");
  for (size_t i = 0; i < msgs->n; i++)
    if (json_puts(&b, i ? ", " : ""), !message(&b, msgs->item[i], err))
      return free(b.p), false;
  json_puts(&b, "]");
  char jerr[OSMOS_ERR] = "";
  *out = b.oom ? NULL : json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr);
  free(b.p);
  return *out || osmos_fail(err, "out of memory");
}
