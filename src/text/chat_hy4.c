#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/chat_hy4.h"

/* Hy4-flash's chat template rendered (osmos/text/chat_hy4.h): its macros and loop in its order */

#define MAX_CLUSTER 256 /* a run of tool messages one turn holds (more refused, not cut) */

static bool fail(char *err, const char *why)
{
  if (err)
    snprintf(err, CHAT_ERR, "%s", why);
  return false;
}

bool hchat_tokens(const char *tag, HchatTokens *t, char err[CHAT_ERR])
{
  const size_t n = tag ? strlen(tag) : 0;
  if (!n || n > HCHAT_TAG_MAX)
    return fail(err, "no special tokens' suffix, or one past HCHAT_TAG_MAX");
  for (size_t i = 0; i < n; i++)
    if (!((tag[i] >= 'a' && tag[i] <= 'z') || (tag[i] >= 'A' && tag[i] <= 'Z') || (tag[i] >= '0' && tag[i] <= '9') ||
          tag[i] == '_' || tag[i] == '-'))
      return fail(err, "a special tokens' suffix not of letters, digits, '_' and '-'");
  snprintf(t->start, sizeof t->start, "<｜hy_start:%s｜>", tag);
  snprintf(t->middle, sizeof t->middle, "<｜hy_middle:%s｜>", tag);
  snprintf(t->end, sizeof t->end, "<｜hy_end:%s｜>", tag);
  snprintf(t->mode, sizeof t->mode, "<｜reasoning_mode:%s｜>", tag);
  snprintf(t->think, sizeof t->think, "<think:%s>", tag), snprintf(t->think_end, sizeof t->think_end, "</think:%s>", tag);
  snprintf(t->calls, sizeof t->calls, "<tool_calls:%s>", tag);
  snprintf(t->calls_end, sizeof t->calls_end, "</tool_calls:%s>", tag);
  snprintf(t->call, sizeof t->call, "<tool_call:%s>", tag), snprintf(t->call_end, sizeof t->call_end, "</tool_call:%s>", tag);
  snprintf(t->key, sizeof t->key, "<arg_key:%s>", tag), snprintf(t->key_end, sizeof t->key_end, "</arg_key:%s>", tag);
  snprintf(t->value, sizeof t->value, "<arg_value:%s>", tag);
  snprintf(t->value_end, sizeof t->value_end, "</arg_value:%s>", tag);
  snprintf(t->resp, sizeof t->resp, "<tool_response:%s>", tag);
  snprintf(t->resp_end, sizeof t->resp_end, "</tool_response:%s>", tag);
  const char *release[] = {"no_think", "high", "max"};
  memset(t->efforts, 0, sizeof t->efforts);
  for (int i = 0; i < 3; i++)
    snprintf(t->efforts[i], sizeof t->efforts[i], "%s", release[i]);
  return true;
}

/* the text between `open` and the next `close` after it, or NULL */
static const char *between(const char *s, const char *open, char close, size_t *n)
{
  const char *a = s ? strstr(s, open) : NULL;
  const char *b = a ? strchr(a + strlen(open), close) : NULL;
  if (!b)
    return NULL;
  *n = (size_t)(b - a - strlen(open));
  return a + strlen(open);
}

bool hchat_from_template(const char *tmpl, HchatTokens *t, char err[CHAT_ERR])
{
  size_t n = 0, m = 0;
  const char *tag = between(tmpl, "set HYTK = ':", '\'', &n);
  const char *list = between(tmpl, "reasoning_effort not in [", ']', &m);
  char buf[HCHAT_TAG_MAX + 1];
  if (!tag || !n || n > HCHAT_TAG_MAX || !list)
    return fail(err, "not Hy4-flash's template (its HYTK suffix, or its reasoning efforts' check)");
  memcpy(buf, tag, n), buf[n] = 0;
  if (!hchat_tokens(buf, t, err))
    return false;
  memset(t->efforts, 0, sizeof t->efforts);
  int k = 0;
  for (const char *p = list; p < list + m && k < HCHAT_EFFORTS;)
  {
    const char *a = memchr(p, '\'', (size_t)(list + m - p));
    const char *b = a ? memchr(a + 1, '\'', (size_t)(list + m - a - 1)) : NULL;
    if (!b)
      break;
    if ((size_t)(b - a - 1) >= sizeof t->efforts[k])
      return fail(err, "a reasoning effort's name past the most kept");
    memcpy(t->efforts[k], a + 1, (size_t)(b - a - 1)), k++;
    p = b + 1;
  }
  return k > 0 || fail(err, "the template names no reasoning effort");
}

static bool effort_known(const HchatTokens *t, const char *e)
{
  for (int i = 0; i < HCHAT_EFFORTS && t->efforts[i][0]; i++)
    if (!strcmp(t->efforts[i], e))
      return true;
  return false;
}

void hchat_refused(const HchatTokens *t, const char *out[HCHAT_REFUSED + 1])
{
  const char *all[HCHAT_REFUSED] = {t->start, t->middle,   t->end,      t->mode,  t->think, t->think_end,
                                    t->calls, t->calls_end, t->call,    t->call_end, t->key, t->key_end,
                                    t->value, t->value_end, t->resp,    t->resp_end};
  for (int i = 0; i < HCHAT_REFUSED; i++)
    out[i] = all[i];
  out[HCHAT_REFUSED] = NULL;
}

/* Jinja's truthiness of a JSON value (absent: false) */
static bool truthy(const Json *v)
{
  if (!v)
    return false;
  switch (v->type)
  {
  case JSON_NULL:
  case JSON_FALSE:
    return false;
  case JSON_INT:
    return strcmp(v->s, "0") != 0;
  case JSON_FLOAT:
    return v->f != 0;
  case JSON_STRING:
  case JSON_ARRAY:
  case JSON_OBJECT:
    return v->type == JSON_STRING ? v->len > 0 : v->n > 0;
  default:
    return true;
  }
}

static bool equal(const Json *a, const Json *b) /* Python's == of two ids (strings or numbers, as JSON gives them) */
{
  if (!a || !b || a->type != b->type)
    return false;
  if (a->type == JSON_STRING || a->type == JSON_INT)
    return a->len == b->len && !memcmp(a->s, b->s, a->len);
  return a->type == JSON_FLOAT ? a->f == b->f : a->type == JSON_NULL || a->type == JSON_TRUE || a->type == JSON_FALSE;
}

static const char *role_of(const Json *m)
{
  const char *r = json_str(m, "role");
  return r ? r : "";
}

/* render_content: a string, text parts (or strings), null - an image or a video refused (text only) */
static bool content(JsonBuf *b, const Json *c, char *err)
{
  if (!c || c->type == JSON_NULL)
    return true;
  if (c->type == JSON_STRING)
    return json_put(b, c->s, c->len), true;
  if (c->type != JSON_ARRAY)
    return fail(err, "a message's content not a string, parts or null");
  for (size_t i = 0; i < c->n; i++)
  {
    const Json *it = c->item[i];
    if (it->type == JSON_STRING)
    {
      json_put(b, it->s, it->len);
      continue;
    }
    const char *type = json_str(it, "type");
    if (json_get(it, "image") || json_get(it, "image_url") || json_get(it, "video") || json_get(it, "video_url") ||
        (type && (!strcmp(type, "image") || !strcmp(type, "image_url") || !strcmp(type, "video") ||
                  !strcmp(type, "video_url"))))
      return fail(err, "an image or a video part (osmos reads text alone)");
    const Json *text = json_get(it, "text");
    if (!type || strcmp(type, "text") || !text || text->type != JSON_STRING)
      return fail(err, "a content part not text");
    json_put(b, text->s, text->len);
  }
  return true;
}

/* tool_to_json: the tool's keys in order but defer_loading and strict, each '"key": ' and its JSON */
static bool tool_json(JsonBuf *b, const Json *t, char *err)
{
  if (!t || t->type != JSON_OBJECT)
    return fail(err, "a tool not an object");
  json_puts(b, "{");
  bool first = true;
  for (size_t i = 0; i < t->n; i++)
  {
    if (!strcmp(t->key[i], "defer_loading") || !strcmp(t->key[i], "strict"))
      continue;
    json_puts(b, first ? "\"" : ", \""), json_put(b, t->key[i], t->klen[i]), json_puts(b, "\": ");
    json_dump(b, t->item[i]);
    first = false;
  }
  json_puts(b, "}");
  return true;
}

static const Json *fn_of(const Json *tool) /* a tool's or a call's `function`, or itself */
{
  return tool && tool->type == JSON_OBJECT && json_get(tool, "function") ? json_get(tool, "function") : tool;
}

static bool tools_prompt(JsonBuf *b, const Json *tools, const HchatTokens *t, char *err)
{
  json_puts(b, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with "
               "function signatures within <tools></tools> XML tags:\n<tools>\n");
  bool first = true;
  for (size_t i = 0; i < tools->n; i++)
  {
    const Json *f = fn_of(tools->item[i]);
    if (f && f->type == JSON_OBJECT && truthy(json_get(f, "defer_loading")))
      continue;
    json_puts(b, first ? "" : "\n");
    if (!tool_json(b, f, err))
      return false;
    first = false;
  }
  json_puts(b, "\n</tools>\n\nFor function call returns, you should first print "), json_puts(b, t->calls);
  json_puts(b, "\nFor each function call, you should return object like:\n"), json_puts(b, t->call);
  json_puts(b, "{function-name}");
  for (int k = 1; k <= 2; k++)
    json_puts(b, t->key), json_puts(b, k == 1 ? "{arg-key-1}" : "{arg-key-2}"), json_puts(b, t->key_end),
        json_puts(b, t->value), json_puts(b, k == 1 ? "{arg-value-1}" : "{arg-value-2}"), json_puts(b, t->value_end);
  json_puts(b, "..."), json_puts(b, t->call_end);
  json_puts(b, "\nAt the end of function call returns, you should print "), json_puts(b, t->calls_end);
  return true;
}

static bool head(JsonBuf *b, const HchatTokens *t, const char *role, size_t n)
{
  json_puts(b, t->start), json_put(b, role, n), json_puts(b, t->middle);
  return true;
}

/* a tool message's id: tool_call_id where present (even null), else id */
static const Json *response_id(const Json *m)
{
  const Json *id = json_get(m, "tool_call_id");
  return id ? id : json_get(m, "id");
}

static bool tool_response(JsonBuf *b, const Json *m, const HchatTokens *t, char *err)
{
  const Json *c = json_get(m, "content");
  json_puts(b, t->resp);
  if (c && c->type == JSON_ARRAY)
    for (size_t i = 0; i < c->n; i++)
    {
      const Json *it = c->item[i];
      const char *type = json_str(it, "type");
      if (type && !strcmp(type, "tool_search_output"))
        return fail(err, "a tool_search_output part (not read by this build)");
      const Json *out = it->type == JSON_OBJECT ? json_get(it, "output") : NULL;
      if (out ? !content(b, out, err) : !content(b, &(Json){.type = JSON_ARRAY, .item = (Json **)&c->item[i], .n = 1}, err))
        return false;
    }
  else if (!content(b, c, err))
    return false;
  json_puts(b, t->resp_end);
  return true;
}

/* a cluster of tool messages: in the previous turn's calls' order when each id matches exactly one, else as given */
static bool cluster(JsonBuf *b, const Json *const *msgs, size_t n, const Json *const *ids, size_t n_ids,
                    const HchatTokens *t, char *err)
{
  const Json *order[MAX_CLUSTER];
  bool matched = n_ids && n_ids == n;
  for (size_t k = 0; matched && k < n_ids; k++)
  {
    size_t hits = 0;
    for (size_t i = 0; i < n; i++)
      if (equal(response_id(msgs[i]), ids[k]))
        hits++, order[k] = msgs[i];
    matched = hits == 1;
  }
  head(b, t, "tool", 4);
  for (size_t i = 0; i < n; i++)
    if (!tool_response(b, matched ? order[i] : msgs[i], t, err))
      return false;
  json_puts(b, t->end);
  return true;
}

static bool name_ok(const Json *s) { return s && s->type == JSON_STRING && !memchr(s->s, '<', s->len); }

/* an assistant turn: its think block, content, calls; its calls' ids (all, or none: *n_ids 0) */
static bool assistant(JsonBuf *b, const Json *m, bool think_kept, const HchatTokens *t, const Json **ids,
                      size_t *n_ids, char *err)
{
  const Json *r = json_get(m, "reasoning"), *rc = json_get(m, "reasoning_content");
  const Json *text = r && r->type == JSON_STRING && r->len ? r : rc && rc->type == JSON_STRING && rc->len ? rc : NULL;
  head(b, t, "assistant", 9);
  json_puts(b, t->think);
  if (think_kept && text)
    json_put(b, text->s, text->len);
  json_puts(b, t->think_end);
  if (!content(b, json_get(m, "content"), err))
    return false;
  const Json *calls = json_get(m, "tool_calls");
  *n_ids = 0;
  if (truthy(calls))
  {
    if (calls->type != JSON_ARRAY || calls->n > MAX_CLUSTER)
      return fail(err, "an assistant's tool_calls not an array, or past the calls read");
    bool complete = true;
    for (size_t i = 0; i < calls->n; i++)
    {
      const Json *id = json_get(calls->item[i], "id");
      if (!truthy(id))
        id = json_get(calls->item[i], "tool_call_id");
      if (truthy(id))
        ids[(*n_ids)++] = id;
      else
        complete = false;
    }
    if (!complete)
      *n_ids = 0;
    json_puts(b, t->calls);
    for (size_t i = 0; i < calls->n; i++)
    {
      const Json *f = fn_of(calls->item[i]), *name = json_get(f, "name"), *args = json_get(f, "arguments");
      if (!name_ok(name))
        return fail(err, "a tool call without a name, or its name holding '<'");
      if (!args || args->type != JSON_OBJECT)
        return fail(err, "a tool call's arguments not an object");
      json_puts(b, t->call), json_put(b, name->s, name->len);
      for (size_t k = 0; k < args->n; k++)
      {
        if (memchr(args->key[k], '<', args->klen[k]))
          return fail(err, "a tool call's argument key holding '<'");
        json_puts(b, t->key), json_put(b, args->key[k], args->klen[k]), json_puts(b, t->key_end), json_puts(b, t->value);
        if (args->item[k]->type == JSON_STRING)
          json_put(b, args->item[k]->s, args->item[k]->len);
        else
          json_dump(b, args->item[k]);
        json_puts(b, t->value_end);
      }
      json_puts(b, t->call_end);
    }
    json_puts(b, t->calls_end);
  }
  json_puts(b, t->end);
  return true;
}

bool hchat_render(const Json *messages, const Json *tools, const char *effort, int preserve, const HchatTokens *t,
                  JsonBuf *out, char err[CHAT_ERR])
{
  if (!messages || messages->type != JSON_ARRAY || !t || !out)
    return fail(err, "no messages (an array), tokens or output");
  if (tools && tools->type == JSON_NULL)
    tools = NULL;
  if (tools && tools->type != JSON_ARRAY)
    return fail(err, "tools not an array");
  const char *e = effort ? effort : "high";
  if (!effort_known(t, e))
  {
    char why[CHAT_ERR];
    snprintf(why, sizeof why, "reasoning_effort error : %.64s, not one the template names", e);
    return fail(err, why);
  }
  const bool has_tools = truthy(tools), keep = preserve < 0 ? has_tools : preserve > 0;
  const bool thinks = strcmp(e, "no_think") != 0;
  /* the normed messages' count and the last user turn's place among them (a tool run one) */
  size_t last_user = (size_t)-1, normed = 0;
  for (size_t i = 0; i < messages->n; normed++)
  {
    const bool tool = !strcmp(role_of(messages->item[i]), "tool");
    if (!strcmp(role_of(messages->item[i]), "user"))
      last_user = normed;
    for (i++; tool && i < messages->n && !strcmp(role_of(messages->item[i]), "tool");)
      i++;
  }
  if (!messages->n || strcmp(role_of(messages->item[0]), "system"))
  {
    head(out, t, "system", 6);
    if (has_tools && !tools_prompt(out, tools, t, err))
      return false;
    json_puts(out, t->mode), json_puts(out, "reasoning_effort:"), json_puts(out, e), json_puts(out, t->end);
  }
  const Json *ids[MAX_CLUSTER], *run[MAX_CLUSTER];
  size_t n_ids = 0;
  bool last_assistant = false;
  for (size_t i = 0, k = 0; i < messages->n; k++)
  {
    const Json *m = messages->item[i];
    const char *role = role_of(m);
    if (!json_str(m, "role"))
      return fail(err, "a message without a role");
    if (!strcmp(role, "tool"))
    {
      size_t n = 0;
      for (; i < messages->n && !strcmp(role_of(messages->item[i]), "tool"); i++)
        if (n == MAX_CLUSTER)
          return fail(err, "a run of tool messages past the most read");
        else
          run[n++] = messages->item[i];
      if (!cluster(out, run, n, ids, n_ids, t, err))
        return false;
      n_ids = 0, last_assistant = false;
      continue;
    }
    i++;
    if (!strcmp(role, "system"))
    {
      head(out, t, "system", 6);
      if (k == 0)
      {
        JsonBuf c = {0};
        const bool ok = content(&c, json_get(m, "content"), err);
        if (ok && has_tools && !tools_prompt(out, tools, t, err))
          return free(c.p), false;
        if (ok && has_tools && c.n)
          json_puts(out, "\n\n");
        if (ok)
          json_put(out, c.p ? c.p : "", c.n);
        free(c.p);
        if (!ok)
          return false;
        json_puts(out, t->mode), json_puts(out, "reasoning_effort:"), json_puts(out, e);
      }
      else if (!content(out, json_get(m, "content"), err))
        return false;
      json_puts(out, t->end);
      n_ids = 0;
    }
    else if (!strcmp(role, "assistant"))
    {
      const bool kept = thinks && (keep || last_user == (size_t)-1 || k > last_user);
      if (!assistant(out, m, kept, t, ids, &n_ids, err))
        return false;
    }
    else
    {
      head(out, t, role, strlen(role));
      if (!content(out, json_get(m, "content"), err))
        return false;
      json_puts(out, t->end);
      n_ids = 0;
    }
    last_assistant = !strcmp(role, "assistant");
  }
  if (!last_assistant)
  {
    head(out, t, "assistant", 9), json_puts(out, t->think);
    if (!thinks)
      json_puts(out, t->think_end);
  }
  return !out->oom || fail(err, "out of memory");
}
