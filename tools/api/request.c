#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "request.h"

/* an OpenAI chat request as DeepSeek's encoding reads it (maya's frontend.py for DeepSeek: tools onto the first
   system message, an empty one put first if none), its thinking and effort (DESIGN.md, "osmos-api: the mapping"),
   and what this server refuses by name rather than ignore - sampling (it decodes greedily), stop strings, several
   choices */

static bool fail(char *err, const char *what)
{
  snprintf(err, API_ERR, "%s", what);
  return false;
}

/* a field present and not the value greedy decoding means */
static bool sampled(const Json *v, double greedy)
{
  if (!v || v->type == JSON_NULL)
    return false;
  const double x = v->type == JSON_INT ? atof(v->s) : v->type == JSON_FLOAT ? v->f : -1;
  return x != greedy;
}

static int effort_of(const Json *v, char *err)
{
  if (v && v->type == JSON_STRING)
  {
    const char *names[] = {"minimal", "low", "medium", "high", "xhigh", "max"};
    const int budget[] = {50, 50, 75, 75, 100, 100};
    for (int i = 0; i < 6; i++)
      if (!strcmp(v->s, names[i]))
        return budget[i];
    return fail(err, "reasoning_effort: minimal, low, medium, high, xhigh, max, none or 1-100"), -1;
  }
  char cerr[CHAT_ERR];
  const int e = chat_effort(v, cerr);
  if (e < 0)
    fail(err, "reasoning_effort: an integer 1-100 or a level's name");
  return e;
}

/* a message written back with its role DeepSeek's (developer: system) and `reasoning` as reasoning_content */
static void message(JsonBuf *b, const Json *m, const Json *tools)
{
  const char *role = json_str(m, "role");
  const bool has_rc = json_get(m, "reasoning_content") != NULL;
  json_puts(b, "{\"role\": ");
  if (role && !strcmp(role, "developer"))
    role = "system";
  json_dump_string(b, role ? role : "", role ? strlen(role) : 0);
  for (size_t i = 0; i < m->n; i++)
  {
    const char *k = m->key[i];
    if (!strcmp(k, "role") || (!strcmp(k, "reasoning") && has_rc) || (tools && !strcmp(k, "tools")))
      continue;
    json_puts(b, ", ");
    if (!strcmp(k, "reasoning"))
      json_puts(b, "\"reasoning_content\"");
    else
      json_dump_string(b, k, m->klen[i]);
    json_puts(b, ": "), json_dump(b, m->item[i]);
  }
  if (tools)
    json_puts(b, ", \"tools\": "), json_dump(b, tools);
  json_puts(b, "}");
}

static bool system_role(const Json *m)
{
  const char *r = json_str(m, "role");
  return r && (!strcmp(r, "system") || !strcmp(r, "developer"));
}

bool request_read(const Json *body, Request *r, char err[API_ERR])
{
  memset(r, 0, sizeof *r);
  const Json *msgs = json_get(body, "messages"), *tools = json_get(body, "tools");
  if (!msgs || msgs->type != JSON_ARRAY || !msgs->n)
    return fail(err, "messages: a non-empty list is needed");
  if (sampled(json_get(body, "temperature"), 0) || sampled(json_get(body, "top_p"), 1))
    return fail(err, "temperature / top_p: osmos decodes greedily (temperature 0, top_p 1) - sampling is not built");
  const Json *stop = json_get(body, "stop"), *n = json_get(body, "n"), *choice = json_get(body, "tool_choice");
  if (stop && stop->type != JSON_NULL && !(stop->type == JSON_ARRAY && !stop->n))
    return fail(err, "stop: stop strings are not built (the model's own end of turn ends it)");
  if (n && n->type == JSON_INT && strcmp(n->s, "1"))
    return fail(err, "n: one choice a request");
  if (choice && !(choice->type == JSON_STRING && !strcmp(choice->s, "auto")) && choice->type != JSON_NULL)
    return fail(err, "tool_choice: auto only (a forced or a disabled call is not built)");
  /* the tools as OpenAI function tools, onto the first system message */
  JsonBuf schemas = {0}, b = {0};
  const bool has_tools = tools && tools->type == JSON_ARRAY && tools->n;
  if (has_tools)
  {
    json_puts(&schemas, "[");
    for (size_t i = 0; i < tools->n; i++)
    {
      json_puts(&schemas, i ? ", " : "");
      if (json_get(tools->item[i], "function"))
        json_dump(&schemas, tools->item[i]);
      else
        json_puts(&schemas, "{\"type\": \"function\", \"function\": "), json_dump(&schemas, tools->item[i]),
            json_puts(&schemas, "}");
    }
    json_puts(&schemas, "]");
  }
  char jerr[JSON_ERR];
  Json *tl = has_tools ? json_parse(schemas.p, schemas.n, jerr) : NULL;
  free(schemas.p);
  json_puts(&b, "[");
  if (has_tools && !system_role(msgs->item[0]))
    json_puts(&b, "{\"role\": \"system\", \"content\": \"\", \"tools\": "), json_dump(&b, tl), json_puts(&b, "}, ");
  for (size_t i = 0; i < msgs->n; i++)
  {
    if (msgs->item[i]->type != JSON_OBJECT)
      return json_free(tl), free(b.p), fail(err, "messages: each an object");
    json_puts(&b, i ? ", " : "");
    message(&b, msgs->item[i], i == 0 && has_tools && system_role(msgs->item[0]) ? tl : NULL);
  }
  json_puts(&b, "]");
  json_free(tl);
  r->messages = b.oom ? NULL : json_parse(b.p, b.n, jerr);
  free(b.p);
  if (!r->messages)
    return fail(err, "out of memory");
  /* thinking and its effort */
  const Json *ctk = json_get(body, "chat_template_kwargs"), *th = json_get(body, "thinking");
  const Json *eff = json_get(body, "reasoning_effort");
  if (!eff || eff->type == JSON_NULL)
    eff = json_get(ctk, "reasoning_effort");
  const Json *et = json_get(ctk, "enable_thinking");
  r->mode = CHAT_THINKING;
  if ((et && et->type == JSON_FALSE) || (eff && eff->type == JSON_STRING && !strcmp(eff->s, "none")) ||
      (th && json_str(th, "type") && !strcmp(json_str(th, "type"), "disabled")))
    r->mode = CHAT_PLAIN;
  r->effort = r->mode == CHAT_PLAIN ? 75 : effort_of(eff, err);
  if (r->effort < 0)
    return json_free(r->messages), r->messages = NULL, false;
  const Json *mt = json_get(body, "max_completion_tokens");
  if (!mt || mt->type == JSON_NULL)
    mt = json_get(body, "max_tokens");
  if (mt && mt->type == JSON_INT && mt->s[0] != '-' && mt->len <= 9)
    r->max_tokens = (unsigned)atol(mt->s);
  else if (mt && mt->type != JSON_NULL)
    return json_free(r->messages), r->messages = NULL, fail(err, "max_tokens: a positive integer");
  const Json *st = json_get(body, "stream"), *so = json_get(body, "stream_options");
  r->stream = st && st->type == JSON_TRUE;
  const Json *iu = json_get(so, "include_usage");
  r->usage = iu && iu->type == JSON_TRUE;
  return true;
}
