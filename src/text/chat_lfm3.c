#define _GNU_SOURCE /* memmem */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_qwen_impl.h"
#include "osmos/text/chat_lfm3.h"
#include "osmos/err.h"

/* LFM3's template's render (osmos/text/chat_lfm3.h), its branches in its order */

#define IM_START "<|im_start|>"

const char *const lchat_refused[] = {IM_START, LCHAT_EOS, LCHAT_BOS, NULL};

static bool is(const char *s, const char *lit) { return s && !strcmp(s, lit); }
static const char *role_of(const Json *m) { return json_str(m, "role"); }

/* parse_content: a string as is; a list's strings and text parts' text, in order (another mapping its JSON); a
   mapping a text part's text or its JSON; null nothing; anything else its JSON. an image or a video refused */
static bool media(const Json *it)
{
  const char *type = json_str(it, "type");
  return is(type, "image") || is(type, "video") || is(type, "image_url") || json_get(it, "image") ||
         json_get(it, "image_url") || json_get(it, "video");
}

static bool part(JsonBuf *b, const Json *it, char *err)
{
  if (it->type == JSON_STRING)
    return json_put(b, it->s, it->len), true;
  if (it->type == JSON_OBJECT && media(it))
    return osmos_fail(err, "an image or a video: osmos reads text only (no vision placeholder rendered)");
  if (it->type == JSON_OBJECT && is(json_str(it, "type"), "text"))
  {
    const Json *t = json_get(it, "text");
    if (t && t->type == JSON_STRING)
      json_put(b, t->s, t->len);
    else if (t && t->type != JSON_NULL)
      return osmos_fail(err, "a text part's text: a string is needed");
    return true;
  }
  return json_dump(b, it), true;
}

static bool content(JsonBuf *b, const Json *c, char *err)
{
  if (!c || c->type == JSON_NULL)
    return true;
  if (c->type != JSON_ARRAY)
    return part(b, c, err);
  for (size_t i = 0; i < c->n; i++)
    if (!part(b, c->item[i], err))
      return false;
  return true;
}

/* tool_to_json: the function's (or the tool's own) keys but defer_loading, in order, each its JSON */
static void tool_json(JsonBuf *b, const Json *fn)
{
  json_puts(b, "{");
  bool first = true;
  for (size_t i = 0; i < fn->n; i++)
  {
    if (fn->klen[i] == 13 && !memcmp(fn->key[i], "defer_loading", 13))
      continue;
    json_puts(b, first ? "" : ", "), first = false;
    json_dump_string(b, fn->key[i], fn->klen[i]), json_puts(b, ": "), json_dump(b, fn->item[i]);
  }
  json_puts(b, "}");
}

static bool truthy(const Json *v)
{
  return v && v->type != JSON_NULL && v->type != JSON_FALSE &&
         !(v->type == JSON_STRING && !v->len) && !((v->type == JSON_ARRAY || v->type == JSON_OBJECT) && !v->n);
}

static bool system_turn(JsonBuf *b, const Json *ms, const Json *tools, size_t *first, char *err)
{
  JsonBuf sys = {0};
  bool ok = true;
  *first = 0;
  if (ms->n && is(role_of(ms->item[0]), "system"))
  {
    const Json *c = json_get(ms->item[0], "content");
    if (truthy(c))
      ok = content(&sys, c, err);
    *first = 1;
  }
  size_t lines = 0;
  for (size_t i = 0; ok && tools && i < tools->n; i++)
  {
    const Json *t = tools->item[i];
    const Json *fn = json_get(t, "function") ? json_get(t, "function") : t;
    if (fn->type != JSON_OBJECT)
    {
      ok = osmos_fail(err, "tools: each a function tool (an object)");
      break;
    }
    const Json *d1 = json_get(fn, "defer_loading"), *d2 = json_get(t, "defer_loading");
    if (truthy(d1) || truthy(d2))
      continue; /* a deferred tool: the template's tool search lists it, not this turn */
    if (!lines++)
      json_puts(&sys, sys.n ? "\n\n<tools>\n" : "<tools>\n");
    else
      json_puts(&sys, "\n");
    tool_json(&sys, fn);
  }
  if (ok && lines)
    json_puts(&sys, "\n</tools>");
  if (ok && sys.n)
    json_puts(b, IM_START "system\n"), json_put(b, sys.p, sys.n), json_puts(b, LCHAT_EOS "\n");
  free(sys.p);
  return ok;
}

static bool calls(JsonBuf *b, const Json *list, char *err)
{
  for (size_t k = 0; k < list->n; k++)
  {
    const Json *tc = list->item[k];
    if (json_get(tc, "function"))
      tc = json_get(tc, "function");
    const Json *name = json_get(tc, "name");
    if (!name || name->type != JSON_STRING || !qchat_name_ok(name->s, name->len))
      return osmos_fail(err, "a tool call's function name: a string, without a newline or '<' (the format's own markup)");
    const Json *args = json_get(tc, "arguments");
    if (args && args->type == JSON_STRING)
    {
      const Span t = qchat_trim((Span){args->s, args->len});
      if (!(t.n == 0 || (t.n == 2 && !memcmp(t.p, "{}", 2)) || (t.n == 4 && !memcmp(t.p, "null", 4))))
        return osmos_fail(err, "Tool call arguments must be a mapping, got a JSON-encoded string: parse arguments with "
                         "json.loads() before applying the chat template");
    }
    json_puts(b, k ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function="), json_put(b, name->s, name->len);
    json_puts(b, ">\n");
    if (args && args->type == JSON_OBJECT)
      for (size_t i = 0; i < args->n; i++)
      {
        if (!qchat_name_ok(args->key[i], args->klen[i]))
          return osmos_fail(err, "a tool call's argument name holds a newline or '<' (the format's own markup)");
        json_puts(b, "<parameter="), json_put(b, args->key[i], args->klen[i]), json_puts(b, ">\n");
        const Json *v = args->item[i];
        if (v->type == JSON_STRING)
          json_put(b, v->s, v->len);
        else
          json_dump(b, v); /* render_value: true / false / null and a number or a container as its JSON */
        json_puts(b, "\n</parameter>\n");
      }
    json_puts(b, "</function>\n</tool_call>");
  }
  return true;
}

static const Json *thinking_of(const Json *m)
{
  const Json *t = json_get(m, "thinking");
  if (!t)
    t = json_get(m, "reasoning");
  if (!t)
    t = json_get(m, "reasoning_content");
  return t;
}

static bool assistant(JsonBuf *b, const Json *m, bool keep, char *err)
{
  JsonBuf c = {0};
  const Json *cv = json_get(m, "content");
  bool ok = !truthy(cv) || content(&c, cv, err);
  const Span text = {c.p ? c.p : "", c.n};
  const Json *th = thinking_of(m);
  if (ok && th && keep)
  {
    json_puts(b, "<think>");
    if (truthy(th))
    {
      if (th->type == JSON_STRING)
        json_put(b, th->s, th->len);
      else
        json_dump(b, th);
    }
    json_puts(b, "</think>");
  }
  else if (ok && keep && !(cv && cv->type == JSON_STRING && memmem(cv->s, cv->len, "</think>", 8)))
    json_puts(b, "<think></think>");
  if (ok && !keep)
  {
    Span rest = text;
    for (const char *p = text.p; (p = memmem(p, text.n - (size_t)(p - text.p), "</think>", 8));)
      rest = (Span){p + 8, text.n - (size_t)(p + 8 - text.p)}, p += 8;
    if (rest.p != text.p)
      rest = qchat_trim(rest);
    json_puts(b, "<think></think>"), json_put(b, rest.p, rest.n);
  }
  else if (ok)
    json_put(b, text.p, text.n);
  const Json *tc = json_get(m, "tool_calls");
  if (ok && tc && tc->type == JSON_ARRAY && tc->n)
    ok = calls(b, tc, err);
  free(c.p);
  return ok;
}

/* a tool result's items (the template's _items: a string's one text part if not empty, a list's items, else itself) */
static bool tool_items(JsonBuf *b, const Json *c, size_t *count, char *err)
{
  if (!c || c->type == JSON_NULL)
    return true;
  if (c->type == JSON_STRING)
  {
    if (c->len)
      json_puts(b, *count ? "\n" : ""), json_put(b, c->s, c->len), ++*count;
    return true;
  }
  const size_t n = c->type == JSON_ARRAY ? c->n : 1;
  for (size_t i = 0; i < n; i++)
  {
    const Json *it = c->type == JSON_ARRAY ? c->item[i] : c;
    if (it->type == JSON_OBJECT && (is(json_str(it, "type"), "tool_reference") || json_get(it, "result")))
      return osmos_fail(err, "a tool reference or a tool search's result: not served (the template's tool-search forms)");
    json_puts(b, (*count)++ ? "\n" : "");
    if (!part(b, it, err))
      return false;
  }
  return true;
}

static bool same_call(const Json *a, const Json *b)
{
  const Json *x = json_get(a, "tool_call_id"), *y = json_get(b, "tool_call_id");
  return truthy(x) && truthy(y) && x->type == JSON_STRING && y->type == JSON_STRING && x->len == y->len &&
         !memcmp(x->s, y->s, x->len);
}

static bool render(const Json *ms, const Json *tools, ChatMode mode, bool preserve, JsonBuf *b, char *err)
{
  size_t first;
  json_puts(b, LCHAT_BOS);
  if (!system_turn(b, ms, tools, &first, err))
    return false;
  long last_user = -1;
  for (size_t i = first; i < ms->n; i++)
    if (is(role_of(ms->item[i]), "user"))
      last_user = (long)(i - first);
  JsonBuf pending = {0};
  size_t pend = 0;
  bool ok = true;
  for (size_t i = first; ok && i < ms->n; i++)
  {
    const Json *m = ms->item[i];
    const char *role = role_of(m);
    if (!role)
    {
      ok = osmos_fail(err, "a message without a role");
      break;
    }
    if (json_get(m, "additional_tools"))
    {
      ok = osmos_fail(err, "additional_tools: not served (the template's tool-search forms)");
      break;
    }
    const bool tool = is(role, "tool");
    if (tool && i + 1 < ms->n && is(role_of(ms->item[i + 1]), "tool") && same_call(m, ms->item[i + 1]))
    {
      ok = tool_items(&pending, json_get(m, "content"), &pend, err);
      continue;
    }
    json_puts(b, IM_START), json_puts(b, role), json_puts(b, "\n");
    if (is(role, "assistant"))
      ok = assistant(b, m, preserve || (long)(i - first) > last_user, err);
    else if (tool)
    {
      const Json *c = json_get(m, "content");
      if (!pend && c && c->type == JSON_STRING && memmem(c->s, c->len, "<tool_response>", 15))
        json_put(b, c->s, c->len); /* a result already wrapped: as it is */
      else
      {
        JsonBuf body = {0};
        if (pend)
          json_put(&body, pending.p, pending.n);
        if (pend && c && c->type == JSON_STRING)
          ok = tool_items(&body, c, &pend, err);
        else if (pend)
          ok = tool_items(&body, c, &pend, err);
        else if (c && c->type == JSON_ARRAY)
        {
          size_t k = 0;
          ok = tool_items(&body, c, &k, err);
        }
        else
          ok = !c || c->type == JSON_NULL || content(&body, c, err);
        json_puts(b, "<tool_response>\n"), json_put(b, body.p ? body.p : "", body.n), json_puts(b, "\n</tool_response>");
        free(body.p);
        pending.n = 0, pend = 0;
      }
    }
    else
    {
      const Json *c = json_get(m, "content");
      ok = !truthy(c) || content(b, c, err);
    }
    json_puts(b, LCHAT_EOS "\n");
  }
  free(pending.p);
  if (ok)
    json_puts(b, mode == CHAT_THINKING ? IM_START "assistant\n<think>" : IM_START "assistant\n<think></think>");
  return ok;
}

bool lchat_render(const Json *messages, const Json *tools, ChatMode mode, bool preserve, JsonBuf *out,
                  char err[OSMOS_ERR])
{
  if (!messages || messages->type != JSON_ARRAY || !messages->n)
    return osmos_fail(err, "No messages provided.");
  for (size_t i = 0; i < messages->n; i++)
    if (messages->item[i]->type != JSON_OBJECT)
      return osmos_fail(err, "messages: each an object");
  if (tools && tools->type != JSON_ARRAY && tools->type != JSON_NULL)
    return osmos_fail(err, "tools: a list of function tools");
  const bool ok =
      render(messages, tools && tools->type == JSON_ARRAY && tools->n ? tools : NULL, mode, preserve, out, err);
  if (ok && out->oom)
    return osmos_fail(err, "out of memory");
  return ok;
}

bool lchat_parse_shaped(const char *text, const char *shape, size_t len, ChatMode mode, const Json *tools,
                        JsonBuf *out, char err[OSMOS_ERR])
{
  if (!text || !shape)
    return osmos_fail(err, "no completion");
  const size_t en = strlen(LCHAT_EOS), body = len >= en ? len - en : len;
  for (int i = 0; lchat_refused[i]; i++)
    if (memmem(shape, body, lchat_refused[i], strlen(lchat_refused[i])))
      return osmos_fail(err, "a special token in the completion's text");
  return qchat_parse_shaped(text, shape, len, mode, tools, out, err);
}

bool lchat_parse(const char *text, size_t len, ChatMode mode, const Json *tools, JsonBuf *out, char err[OSMOS_ERR])
{
  return lchat_parse_shaped(text, text, len, mode, tools, out, err);
}
