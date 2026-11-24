#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"
#include "osmos/err.h"
#include "api.h"

/* Inkling's reader: the messages as its template takes them (request_messages: a call's arguments an object, as it
   asks; a developer message a system one, as DeepSeek's and Qwen's readers take it - the template knows no other) -
   the tools apart, the reasoning effort as asked (a name or a number: its template names and bounds them). thinking
   off (enable_thinking false, thinking disabled) is the template's "none" - effort 0, its only off */

bool request_ink2(const Json *body, Request *r, char err[OSMOS_ERR])
{
  const Json *msgs = json_get(body, "messages"), *eff = request_effort(body);
  JsonBuf rf = {0};
  const bool rf_ok = request_format(body, &rf, err), formatted = rf.n > 0;
  free(rf.p);
  if (!rf_ok)
    return false;
  if (formatted)
    return osmos_fail(err, "response_format: Inkling's template has no response format (text only)");
  static const Json off = {.type = JSON_STRING, .s = (char *)"none", .len = 4};
  if (r->mode == CHAT_PLAIN)
    eff = &off;
  if (eff && eff->type != JSON_NULL)
  {
    if (eff->type != JSON_STRING && eff->type != JSON_INT && eff->type != JSON_FLOAT)
      return osmos_fail(err, "reasoning_effort: a name (none, minimal, low, medium, high, xhigh, max) or a number");
    JsonBuf b = {0};
    char jerr[OSMOS_ERR] = "";
    json_dump(&b, eff);
    r->effort_value = b.oom ? NULL : json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr);
    free(b.p);
    if (!r->effort_value)
      return osmos_fail(err, "out of memory");
  }
  if (!request_messages(msgs, &r->messages, err))
    return false;
  for (size_t i = 0; r->messages && i < r->messages->n; i++)
  {
    Json *role = (Json *)json_get(r->messages->item[i], "role");
    char *sys = role && role->type == JSON_STRING && !strcmp(role->s, "developer") ? strdup("system") : NULL;
    if (sys)
      free(role->s), role->s = sys, role->len = 6;
  }
  r->tools = request_tools(body);
  return true;
}
