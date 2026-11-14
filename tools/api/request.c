#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dialect.h"

/* an OpenAI chat request: what every family reads alike - thinking on or off, the tokens asked, streaming - and what
   this server refuses by name rather than ignore (sampling: it decodes greedily; stop strings; several choices);
   then the family's reader (dialect.h) its messages, tools and effort (DESIGN.md, "osmos-api: the mapping") */

bool request_fail(char *err, const char *what)
{
  snprintf(err, API_ERR, "%s", what);
  return false;
}

const Json *request_effort(const Json *body)
{
  const Json *eff = json_get(body, "reasoning_effort");
  return eff && eff->type != JSON_NULL ? eff : json_get(json_get(body, "chat_template_kwargs"), "reasoning_effort");
}

Json *request_tools(const Json *body)
{
  const Json *tools = json_get(body, "tools");
  if (!tools || tools->type != JSON_ARRAY || !tools->n)
    return NULL;
  JsonBuf b = {0};
  json_puts(&b, "[");
  for (size_t i = 0; i < tools->n; i++)
  {
    json_puts(&b, i ? ", " : "");
    if (json_get(tools->item[i], "function"))
      json_dump(&b, tools->item[i]);
    else
      json_puts(&b, "{\"type\": \"function\", \"function\": "), json_dump(&b, tools->item[i]), json_puts(&b, "}");
  }
  json_puts(&b, "]");
  char jerr[JSON_ERR];
  Json *t = b.oom ? NULL : json_parse(b.p, b.n, jerr);
  free(b.p);
  return t;
}

/* the body's response_format as the schema a template is told to answer by, its JSON into out: nothing for none
   (absent, null, text), json_schema's schema, json_object's any object ({"type": "object"}); false for anything else
   (refused by name, never ignored) */
bool request_format(const Json *body, JsonBuf *out, char err[API_ERR])
{
  const Json *rf = json_get(body, "response_format");
  const char *type = json_str(rf, "type");
  if (!rf || rf->type == JSON_NULL || (type && !strcmp(type, "text")))
    return true;
  if (type && !strcmp(type, "json_object"))
    return json_puts(out, "{\"type\": \"object\"}"), true;
  const Json *schema = type && !strcmp(type, "json_schema") ? json_get(json_get(rf, "json_schema"), "schema") : NULL;
  if (!schema || schema->type != JSON_OBJECT)
    return request_fail(err, type && !strcmp(type, "json_schema") ? "response_format: json_schema needs its schema (an "
                                                                    "object)"
                                                                  : "response_format: text, json_object or json_schema");
  return json_dump(out, schema), true;
}

void request_free(Request *r)
{
  json_free(r->messages), json_free(r->tools);
  r->messages = r->tools = NULL;
}

/* a field present and not the value greedy decoding means */
static bool sampled(const Json *v, double greedy)
{
  if (!v || v->type == JSON_NULL)
    return false;
  const double x = v->type == JSON_INT ? atof(v->s) : v->type == JSON_FLOAT ? v->f : -1;
  return x != greedy;
}

bool request_read(const Json *body, const Dialect *d, bool fill_orphans, Request *r, char err[API_ERR])
{
  memset(r, 0, sizeof *r);
  r->preserve = true;
  const Json *msgs = json_get(body, "messages");
  if (!msgs || msgs->type != JSON_ARRAY || !msgs->n)
    return request_fail(err, "messages: a non-empty list is needed");
  for (size_t i = 0; i < msgs->n; i++)
    if (msgs->item[i]->type != JSON_OBJECT)
      return request_fail(err, "messages: each an object");
  if (sampled(json_get(body, "temperature"), 0) || sampled(json_get(body, "top_p"), 1))
    return request_fail(err, "temperature / top_p: osmos decodes greedily (temperature 0, top_p 1) - sampling is not built");
  const Json *stop = json_get(body, "stop"), *n = json_get(body, "n"), *choice = json_get(body, "tool_choice");
  if (stop && stop->type != JSON_NULL && !(stop->type == JSON_ARRAY && !stop->n))
    return request_fail(err, "stop: stop strings are not built (the model's own end of turn ends it)");
  if (n && n->type != JSON_NULL && sampled(n, 1)) /* (0.7.1: 2.0, a float, was taken as one) */
    return request_fail(err, "n: one choice a request");
  /* sampling's other fields: their greedy values taken (a client's defaults), any other refused by name (0.7.1:
     ignored, against this file's rule) */
  const char *const zero[] = {"presence_penalty", "frequency_penalty", "top_logprobs"};
  for (size_t k = 0; k < sizeof zero / sizeof *zero; k++)
    if (sampled(json_get(body, zero[k]), 0))
      return snprintf(err, API_ERR, "%s: osmos decodes greedily - penalties and log-probabilities are not built",
                      zero[k]), false;
  if (sampled(json_get(body, "repetition_penalty"), 1))
    return request_fail(err, "repetition_penalty: osmos decodes greedily - penalties are not built");
  const Json *bias = json_get(body, "logit_bias"), *lp = json_get(body, "logprobs");
  if (bias && bias->type != JSON_NULL && !(bias->type == JSON_OBJECT && !bias->n))
    return request_fail(err, "logit_bias: osmos decodes greedily - a bias is not built");
  if (lp && lp->type != JSON_NULL && lp->type != JSON_FALSE)
    return request_fail(err, "logprobs: log-probabilities are not built");
  if (choice && !(choice->type == JSON_STRING && !strcmp(choice->s, "auto")) && choice->type != JSON_NULL)
    return request_fail(err, "tool_choice: auto only (a forced or a disabled call is not built)");
  const Json *th = json_get(body, "thinking"), *eff = request_effort(body);
  const Json *et = json_get(json_get(body, "chat_template_kwargs"), "enable_thinking");
  r->mode = CHAT_THINKING;
  if ((et && et->type == JSON_FALSE) || (eff && eff->type == JSON_STRING && !strcmp(eff->s, "none")) ||
      (th && json_str(th, "type") && !strcmp(json_str(th, "type"), "disabled")))
    r->mode = CHAT_PLAIN;
  /* 0.7.3 (#47): the history's loop signals; its orphaned calls given a placeholder result only when asked (the
     request's osmos.fill_orphans, else the pack's) - else the prompt is the template's of what was sent */
  loop_read(msgs, &r->loop);
  const Json *os = json_get(body, "osmos"), *fo = json_get(os, "fill_orphans");
  if (os && os->type != JSON_OBJECT && os->type != JSON_NULL)
    return request_fail(err, "osmos: an object (its fill_orphans)");
  if (fo && fo->type != JSON_TRUE && fo->type != JSON_FALSE && fo->type != JSON_NULL)
    return request_fail(err, "osmos.fill_orphans: true or false");
  const bool fill = fo && fo->type != JSON_NULL ? fo->type == JSON_TRUE : fill_orphans;
  Json *filled = NULL;
  if (fill && r->loop.n_orphan_calls)
  {
    char jerr[JSON_ERR];
    JsonBuf b = {0};
    filled = loop_fill(body, &b) ? json_parse(b.p, b.n, jerr) : NULL;
    free(b.p);
    if (!filled)
      return request_fail(err, "out of memory (the orphans' results)");
    r->loop.filled = true;
  }
  const bool read = d->read(filled ? filled : body, r, err);
  json_free(filled);
  if (!read)
    return request_free(r), false;
  const Json *mt = json_get(body, "max_completion_tokens");
  if (!mt || mt->type == JSON_NULL)
    mt = json_get(body, "max_tokens");
  if (mt && mt->type == JSON_INT && mt->s[0] != '-' && mt->len <= 9 && atol(mt->s) > 0)
    r->max_tokens = (unsigned)atol(mt->s);
  else if (mt && mt->type != JSON_NULL)
    return request_free(r), request_fail(err, "max_tokens: a positive integer");
  const Json *st = json_get(body, "stream"), *so = json_get(body, "stream_options");
  r->stream = st && st->type == JSON_TRUE;
  const Json *iu = json_get(so, "include_usage");
  r->usage = iu && iu->type == JSON_TRUE;
  return true;
}
