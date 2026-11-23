#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"
#include "osmos/err.h"

/* LFM3's reader: the messages as its template takes them (request_messages) - the tools apart, preserve_thinking
   (the request's, else the pack's osmos.json preserve_thinking, else off: the template's default). its template names no reasoning effort, only thinking on or off: "none"
   off, "high" on (rig's row's two: tools/install/packs.json), any other level refused, not dropped */

bool request_lfm3(const Json *body, Request *r, char err[OSMOS_ERR])
{
  const Json *msgs = json_get(body, "messages"), *eff = request_effort(body);
  const Json *pt = json_get(json_get(body, "chat_template_kwargs"), "preserve_thinking");
  JsonBuf rf = {0};
  const bool rf_ok = request_format(body, &rf, err), formatted = rf.n > 0;
  free(rf.p);
  if (!rf_ok)
    return false;
  if (formatted)
    return osmos_fail(err, "response_format: LFM3's template has no response format (text only)");
  if (eff && eff->type == JSON_STRING && strcmp(eff->s, "none") != 0 && strcmp(eff->s, "high") != 0)
    return osmos_fail(err, "reasoning_effort: LFM3's template has no effort levels (high: thinking on, none: off)");
  if (pt && pt->type != JSON_TRUE && pt->type != JSON_FALSE && pt->type != JSON_NULL)
    return osmos_fail(err, "chat_template_kwargs.preserve_thinking: true or false");
  r->preserve = pt && pt->type != JSON_NULL ? pt->type == JSON_TRUE : r->preserve_default; /* (the pack's: osmos.json) */
  if (!request_messages(msgs, &r->messages, err))
    return false;
  r->tools = request_tools(body);
  return true;
}
