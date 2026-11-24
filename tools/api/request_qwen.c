#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"
#include "osmos/err.h"

/* Qwen's reader: the messages as its template takes them (request_messages) - the tools apart, the reasoning
   effort's word as asked (the template names the ones it takes), preserve_thinking */

bool request_qwen(const Json *body, Request *r, char err[OSMOS_ERR])
{
  const Json *msgs = json_get(body, "messages"), *eff = request_effort(body);
  const Json *pt = json_get(json_get(body, "chat_template_kwargs"), "preserve_thinking");
  JsonBuf rf = {0};
  const bool rf_ok = request_format(body, &rf, err), formatted = rf.n > 0;
  free(rf.p);
  if (!rf_ok)
    return false;
  if (formatted)
    return osmos_fail(err, "response_format: Qwen's template has no response format (text only)");
  if (eff && eff->type != JSON_NULL && (eff->type != JSON_STRING || eff->len >= sizeof r->level))
    return osmos_fail(err, "reasoning_effort: low, medium, high, xhigh or none");
  if (pt && pt->type != JSON_TRUE && pt->type != JSON_FALSE && pt->type != JSON_NULL)
    return osmos_fail(err, "chat_template_kwargs.preserve_thinking: true or false");
  if (eff && eff->type == JSON_STRING && r->mode == CHAT_THINKING)
    memcpy(r->level, eff->s, eff->len + 1);
  r->preserve = !pt || pt->type != JSON_FALSE;
  if (!request_messages(msgs, &r->messages, err))
    return false;
  r->tools = request_tools(body);
  return true;
}
