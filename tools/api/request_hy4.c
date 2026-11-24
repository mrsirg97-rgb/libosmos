#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"
#include "osmos/err.h"

/* Hy4-flash's reader: the messages as its template takes them (request_messages) - the tools apart, the reasoning
   effort's word as asked (the template names the ones it takes: no_think, high, max - low too in AngelSlim's GGUF;
   "none" or thinking off its no_think), preserved_thinking */

bool request_hy4(const Json *body, Request *r, char err[OSMOS_ERR])
{
  const Json *msgs = json_get(body, "messages"), *eff = request_effort(body);
  const Json *kw = json_get(body, "chat_template_kwargs"), *pt = json_get(kw, "preserved_thinking");
  JsonBuf rf = {0};
  const bool rf_ok = request_format(body, &rf, err), formatted = rf.n > 0;
  free(rf.p);
  if (!rf_ok)
    return false;
  if (formatted)
    return osmos_fail(err, "response_format: Hy4-flash's template has no response format (text only)");
  if (eff && eff->type != JSON_NULL && (eff->type != JSON_STRING || eff->len >= sizeof r->level))
    return osmos_fail(err, "reasoning_effort: the template's names (no_think, high, max) or none");
  if (pt && pt->type != JSON_TRUE && pt->type != JSON_FALSE && pt->type != JSON_NULL)
    return osmos_fail(err, "chat_template_kwargs.preserved_thinking: true or false");
  if (eff && eff->type == JSON_STRING && !strcmp(eff->s, "no_think"))
    r->mode = CHAT_PLAIN;
  if (eff && eff->type == JSON_STRING && r->mode == CHAT_THINKING)
    memcpy(r->level, eff->s, eff->len + 1);
  r->preserve_given = pt && pt->type != JSON_NULL, r->preserve = pt && pt->type == JSON_TRUE;
  if (!request_messages(msgs, &r->messages, err))
    return false;
  r->tools = request_tools(body);
  return true;
}
