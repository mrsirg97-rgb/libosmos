#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"

/* Inkling's reader: the messages as its template takes them (request_messages: a call's arguments an object, as it
   asks) - the tools apart, the reasoning effort as asked (a name or a number: its template names and bounds them) */

bool request_ink2(const Json *body, Request *r, char err[API_ERR])
{
  const Json *msgs = json_get(body, "messages"), *eff = request_effort(body);
  JsonBuf rf = {0};
  const bool rf_ok = request_format(body, &rf, err), formatted = rf.n > 0;
  free(rf.p);
  if (!rf_ok)
    return false;
  if (formatted)
    return request_fail(err, "response_format: Inkling's template has no response format (text only)");
  if (eff && eff->type != JSON_NULL)
  {
    if (eff->type != JSON_STRING && eff->type != JSON_INT && eff->type != JSON_FLOAT)
      return request_fail(err, "reasoning_effort: a name (none, minimal, low, medium, high, xhigh, max) or a number");
    JsonBuf b = {0};
    char jerr[JSON_ERR];
    json_dump(&b, eff);
    r->effort_value = b.oom ? NULL : json_parse(b.p, b.n, jerr);
    free(b.p);
    if (!r->effort_value)
      return request_fail(err, "out of memory");
  }
  if (!request_messages(msgs, &r->messages, err))
    return false;
  r->tools = request_tools(body);
  return true;
}
