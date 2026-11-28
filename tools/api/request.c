#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <time.h>

#include "dialect.h"
#include "osmos/err.h"
#include "api.h"

/* an OpenAI chat request: what every family reads alike - thinking on or off, the tokens asked, streaming - and what
   this server refuses by name rather than ignore (stop strings, several choices, the penalties); then the family's
   reader (dialect.h) its messages, tools and effort (DESIGN.md, "osmos-api: the mapping") */

const Json *request_effort(const Json *body)
{
  const Json *eff = json_get(body, "reasoning_effort");
  return eff && eff->type != JSON_NULL ? eff : json_get(json_get(body, "chat_template_kwargs"), "reasoning_effort");
}

bool request_tools(const Json *body, Json **out, char err[OSMOS_ERR])
{
  *out = NULL;
  const Json *tools = json_get(body, "tools");
  if (!tools || tools->type != JSON_ARRAY || !tools->n)
    return true;
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
  char jerr[OSMOS_ERR] = "";
  Json *t = b.oom ? NULL : json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr);
  free(b.p);
  if (!t)
    return osmos_fail(err, "tools: %s", jerr[0] ? jerr : "out of memory");
  *out = t;
  return true;
}

/* the body's response_format as the schema a template is told to answer by, its JSON into out: nothing for none
   (absent, null, text), json_schema's schema, json_object's any object ({"type": "object"}); false for anything else
   (refused by name, never ignored) */
bool request_format(const Json *body, JsonBuf *out, char err[OSMOS_ERR])
{
  const Json *rf = json_get(body, "response_format");
  const char *type = json_str(rf, "type");
  if (!rf || rf->type == JSON_NULL || (type && !strcmp(type, "text")))
    return true;
  if (type && !strcmp(type, "json_object"))
    return json_puts(out, "{\"type\": \"object\"}"), true;
  const Json *schema = type && !strcmp(type, "json_schema") ? json_get(json_get(rf, "json_schema"), "schema") : NULL;
  if (!schema || schema->type != JSON_OBJECT)
    return osmos_fail(err, "%s", type && !strcmp(type, "json_schema") ? "response_format: json_schema needs its schema (an "
                                                                    "object)"
                                                                  : "response_format: text, json_object or json_schema");
  return json_dump(out, schema), true;
}

void request_free(Request *r)
{
  json_free(r->messages), json_free(r->tools), json_free(r->effort_value);
  r->messages = r->tools = r->effort_value = NULL;
}

/* a field present and not the value greedy decoding means */
static bool sampled(const Json *v, double greedy)
{
  if (!v || v->type == JSON_NULL)
    return false;
  const double x = v->type == JSON_INT ? atof(v->s) : v->type == JSON_FLOAT ? v->f : -1;
  return x != greedy;
}

/* a number field: absent or null keeps *x; else a number in [lo, hi] (lo_open: above lo) */
static bool number(const Json *v, double lo, bool lo_open, double hi, double *x)
{
  if (!v || v->type == JSON_NULL)
    return true;
  if (v->type != JSON_INT && v->type != JSON_FLOAT)
    return false;
  const double y = v->type == JSON_INT ? atof(v->s) : v->f;
  if (!(lo_open ? y > lo : y >= lo) || !(y <= hi))
    return false;
  return *x = y, true;
}

/* a seed for a request that names none: 53 bits (a client's JSON number holds it exactly, to send it back) */
static unsigned long long drawn_seed(void)
{
  unsigned long long s = 0;
  if (getrandom(&s, sizeof s, 0) != (ssize_t)sizeof s)
    s = (unsigned long long)time(NULL) * 0x9E3779B97F4A7C15ull ^ (unsigned long long)clock();
  return s & ((1ull << 53) - 1);
}

/* the request's fields an allowlist (an unknown one refused: a misspelled max_token would run unbounded): what it
   reads, what it refuses past a default (n, stop, the penalties, logit_bias, logprobs), and a client's own labels it
   has nothing to do with (model, user - rig sends both - parallel_tool_calls, metadata, store when false) */
static const char *const kFields[] = {
    "messages",       "temperature",     "top_p",      "top_k",        "seed",           "stop",
    "n",              "presence_penalty", "frequency_penalty", "top_logprobs", "repetition_penalty", "logit_bias",
    "logprobs",       "tools",           "tool_choice", "thinking",    "reasoning_effort", "chat_template_kwargs",
    "osmos",          "max_tokens",      "max_completion_tokens", "stream", "stream_options", "response_format",
    "model",          "user",            "parallel_tool_calls", "metadata", "store",
};

static bool fields_known(const Json *body, char *err)
{
  for (size_t i = 0; i < body->n; i++)
  {
    bool known = false;
    for (size_t k = 0; !known && k < sizeof kFields / sizeof *kFields; k++)
      known = !strcmp(body->key[i], kFields[k]);
    if (!known)
      return osmos_fail(err, "%.64s: a field osmos does not take (refused, not ignored)", body->key[i]);
  }
  const Json *store = json_get(body, "store");
  if (store && store->type != JSON_NULL && store->type != JSON_FALSE)
    return osmos_fail(err, "store: osmos keeps no completions (false or absent)");
  return true;
}

bool request_read(const Json *body, const Dialect *d, bool fill_orphans, bool preserve_default, double temperature,
                  double top_p, unsigned top_k, Request *r, char err[OSMOS_ERR])
{
  memset(r, 0, sizeof *r);
  if (!body || body->type != JSON_OBJECT)
    return osmos_fail(err, "the body: a JSON object");
  if (!fields_known(body, err))
    return false;
  r->preserve = true, r->preserve_default = preserve_default;
  const Json *msgs = json_get(body, "messages");
  if (!msgs || msgs->type != JSON_ARRAY || !msgs->n)
    return osmos_fail(err, "messages: a non-empty list is needed");
  for (size_t i = 0; i < msgs->n; i++)
    if (msgs->item[i]->type != JSON_OBJECT)
      return osmos_fail(err, "messages: each an object");
  /* sampling - as asked, else the pack's (temperature 0 greedy); a seed as asked, else drawn */
  r->temperature = temperature, r->top_p = top_p;
  if (!number(json_get(body, "temperature"), 0, false, 100, &r->temperature))
    return osmos_fail(err, "temperature: a number, 0 (greedy) or more");
  if (!number(json_get(body, "top_p"), 0, true, 1, &r->top_p))
    return osmos_fail(err, "top_p: a number above 0, 1 at most");
  r->top_k = top_k; /* an integer; 0 or -1 (vLLM's, a client's default) every token */
  const Json *tk = json_get(body, "top_k");
  if (tk && tk->type != JSON_NULL)
  {
    if (tk->type != JSON_INT || strlen(tk->s) > 9 || atol(tk->s) < -1)
      return osmos_fail(err, "top_k: an integer, 0 or -1 (every token) or more");
    r->top_k = atol(tk->s) > 0 ? (unsigned)atol(tk->s) : 0;
  }
  const Json *seed = json_get(body, "seed");
  if (seed && seed->type != JSON_NULL && (seed->type != JSON_INT || seed->s[0] == '-' || strlen(seed->s) > 19))
    return osmos_fail(err, "seed: an integer, 0 or more");
  r->seed = seed && seed->type == JSON_INT ? strtoull(seed->s, NULL, 10) : drawn_seed();
  r->sampled = r->temperature > 0;
  const Json *stop = json_get(body, "stop"), *n = json_get(body, "n"), *choice = json_get(body, "tool_choice");
  if (stop && stop->type != JSON_NULL && !(stop->type == JSON_ARRAY && !stop->n))
    return osmos_fail(err, "stop: stop strings are not built (the model's own end of turn ends it)");
  if (n && n->type != JSON_NULL && sampled(n, 1)) /* (2.0, a float, is not one) */
    return osmos_fail(err, "n: one choice a request");
  /* sampling's other fields: their greedy values taken (a client's defaults), any other refused by name */
  const char *const zero[] = {"presence_penalty", "frequency_penalty", "top_logprobs"};
  for (size_t k = 0; k < sizeof zero / sizeof *zero; k++)
    if (sampled(json_get(body, zero[k]), 0))
      return osmos_fail(err, "%s: osmos decodes greedily - penalties and log-probabilities are not built",
                      zero[k]), false;
  if (sampled(json_get(body, "repetition_penalty"), 1))
    return osmos_fail(err, "repetition_penalty: osmos decodes greedily - penalties are not built");
  const Json *bias = json_get(body, "logit_bias"), *lp = json_get(body, "logprobs");
  if (bias && bias->type != JSON_NULL && !(bias->type == JSON_OBJECT && !bias->n))
    return osmos_fail(err, "logit_bias: osmos decodes greedily - a bias is not built");
  if (lp && lp->type != JSON_NULL && lp->type != JSON_FALSE)
    return osmos_fail(err, "logprobs: log-probabilities are not built");
  const Json *tl = json_get(body, "tools");
  const bool no_tools = !tl || tl->type == JSON_NULL || (tl->type == JSON_ARRAY && !tl->n);
  if (choice && !(choice->type == JSON_STRING && !strcmp(choice->s, "auto")) && choice->type != JSON_NULL &&
      !(no_tools && choice->type == JSON_STRING && !strcmp(choice->s, "none"))) /* (none with no tools: asks nothing) */
    return osmos_fail(err, "tool_choice: auto only (a forced or a disabled call is not built)");
  const Json *th = json_get(body, "thinking"), *eff = request_effort(body);
  const Json *et = json_get(json_get(body, "chat_template_kwargs"), "enable_thinking");
  r->mode = CHAT_THINKING;
  if ((et && et->type == JSON_FALSE) || (eff && eff->type == JSON_STRING && !strcmp(eff->s, "none")) ||
      (th && json_str(th, "type") && !strcmp(json_str(th, "type"), "disabled")))
    r->mode = CHAT_PLAIN;
  /* the history's loop signals; its orphaned calls given a placeholder result only when asked (the
     request's osmos.fill_orphans, else the pack's) - else the prompt is the template's of what was sent */
  loop_read(msgs, &r->loop);
  const Json *os = json_get(body, "osmos"), *fo = json_get(os, "fill_orphans");
  if (os && os->type != JSON_OBJECT && os->type != JSON_NULL)
    return osmos_fail(err, "osmos: an object (its fill_orphans)");
  if (fo && fo->type != JSON_TRUE && fo->type != JSON_FALSE && fo->type != JSON_NULL)
    return osmos_fail(err, "osmos.fill_orphans: true or false");
  const bool fill = fo && fo->type != JSON_NULL ? fo->type == JSON_TRUE : fill_orphans;
  Json *filled = NULL;
  if (fill && r->loop.n_orphan_calls)
  {
    char jerr[OSMOS_ERR] = "";
    JsonBuf b = {0};
    filled = loop_fill(body, &b) ? json_parse_within(b.p, b.n, API_TREE_MAX, NULL, jerr) : NULL;
    free(b.p);
    if (!filled)
      return osmos_fail(err, "out of memory (the orphans' results)");
    r->loop.filled = true;
  }
  const bool read = d->read(filled ? filled : body, r, err);
  json_free(filled);
  if (!read)
    return request_free(r), false;
  const Json *mt = json_get(body, "max_completion_tokens");
  if (!mt || mt->type == JSON_NULL)
    mt = json_get(body, "max_tokens");
  if (mt && mt->type == JSON_INT && mt->s[0] != '-' && mt->len > 9) /* past 999,999,999: as many as the context's room
                                                                       (4294967295 too) */
    r->max_tokens = 999999999u;
  else if (mt && mt->type == JSON_INT && mt->s[0] != '-' && mt->len <= 9 && atol(mt->s) > 0)
    r->max_tokens = (unsigned)atol(mt->s);
  else if (mt && mt->type != JSON_NULL)
    return request_free(r), osmos_fail(err, "max_tokens: a positive integer");
  const Json *st = json_get(body, "stream"), *so = json_get(body, "stream_options");
  r->stream = st && st->type == JSON_TRUE;
  const Json *iu = json_get(so, "include_usage");
  r->usage = iu && iu->type == JSON_TRUE;
  return true;
}
