#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"

/* DeepSeek's reader (maya's frontend.py for DeepSeek): tools and the response format's schema (the encoding's
   response_format) onto the first system message, an empty one put first if none; the reasoning effort as the
   encoding's budget (DESIGN.md, "osmos-api: the mapping") */

static int effort_of(const Json *v, char *err)
{
  if (v && v->type == JSON_STRING)
  {
    const char *names[] = {"minimal", "low", "medium", "high", "xhigh", "max"};
    const int budget[] = {50, 50, 75, 75, 100, 100};
    for (int i = 0; i < 6; i++)
      if (!strcmp(v->s, names[i]))
        return budget[i];
    return request_fail(err, "reasoning_effort: minimal, low, medium, high, xhigh, max, none or 1-100"), -1;
  }
  char cerr[CHAT_ERR];
  const int e = chat_effort(v, cerr);
  if (e < 0)
    request_fail(err, "reasoning_effort: an integer 1-100 or a level's name");
  return e;
}

/* a message written back with its role DeepSeek's (developer: system) and `reasoning` as reasoning_content; tools
   and a response format's schema (JSON text, rf) given in place of its own */
static void message(JsonBuf *b, const Json *m, const Json *tools, const JsonBuf *rf)
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
    if (!strcmp(k, "role") || (!strcmp(k, "reasoning") && has_rc) || (tools && !strcmp(k, "tools")) ||
        (rf && !strcmp(k, "response_format")))
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
  if (rf)
    json_puts(b, ", \"response_format\": "), json_put(b, rf->p, rf->n);
  json_puts(b, "}");
}

static bool system_role(const Json *m)
{
  const char *r = json_str(m, "role");
  return r && (!strcmp(r, "system") || !strcmp(r, "developer"));
}

bool request_ds(const Json *body, Request *r, char err[API_ERR])
{
  const Json *msgs = json_get(body, "messages");
  JsonBuf rf = {0};
  if (!request_format(body, &rf, err))
    return free(rf.p), false;
  Json *tl = request_tools(body);
  const bool has_rf = rf.n > 0, head = tl || has_rf, first = head && system_role(msgs->item[0]);
  JsonBuf b = {0};
  char jerr[JSON_ERR];
  json_puts(&b, "[");
  if (head && !first)
  {
    json_puts(&b, "{\"role\": \"system\", \"content\": \"\"");
    if (tl)
      json_puts(&b, ", \"tools\": "), json_dump(&b, tl);
    if (has_rf)
      json_puts(&b, ", \"response_format\": "), json_put(&b, rf.p, rf.n);
    json_puts(&b, "}, ");
  }
  for (size_t i = 0; i < msgs->n; i++)
  {
    json_puts(&b, i ? ", " : "");
    message(&b, msgs->item[i], i == 0 && first ? tl : NULL, i == 0 && first && has_rf ? &rf : NULL);
  }
  json_puts(&b, "]");
  json_free(tl);
  if (rf.oom)
    b.oom = true;
  free(rf.p);
  r->messages = b.oom ? NULL : json_parse(b.p, b.n, jerr);
  free(b.p);
  if (!r->messages)
    return request_fail(err, "out of memory");
  r->effort = r->mode == CHAT_PLAIN ? 75 : effort_of(request_effort(body), err);
  return r->effort >= 0;
}
