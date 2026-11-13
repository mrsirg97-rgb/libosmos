#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_qwen_impl.h"
#include "pretok.h"

/* the template's render (osmos/text/chat_qwen.h), its branches in its order; its refusals its own words */

#define IM_START "<|im_start|>"

static const char kXhigh[] = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                             "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
                             "and clarity in the final answer.";
static const char kLow[] = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
                           "the conclusion without unnecessary elaboration.";
static const char kToolsHead[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char kToolsTail[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- "
    "Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

bool qchat_fail(char *err, const char *why)
{
  snprintf(err, CHAT_ERR, "%s", why);
  return false;
}

static bool py_space(uint32_t c)
{
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

size_t qchat_space_lead(const char *p, size_t n)
{
  size_t at = 0, used;
  while (at < n && py_space(pretok_utf8((const unsigned char *)p + at, n - at, &used)))
    at += used;
  return at;
}

size_t qchat_space_trail(const char *p, size_t n)
{
  size_t end = n, used;
  while (end)
  {
    size_t k = end - 1;
    while (k && ((unsigned char)p[k] & 0xC0) == 0x80)
      k--;
    if (!py_space(pretok_utf8((const unsigned char *)p + k, end - k, &used)) || k + used != end)
      break;
    end = k;
  }
  return n - end;
}

Span qchat_trim(Span s)
{
  const size_t lead = qchat_space_lead(s.p, s.n);
  s.p += lead, s.n -= lead;
  s.n -= qchat_space_trail(s.p, s.n);
  return s;
}

static bool is(const char *s, const char *lit) { return s && !strcmp(s, lit); }
static const char *role_of(const Json *m) { return json_str(m, "role"); }
static bool system_role(const Json *m) { return is(role_of(m), "system") || is(role_of(m), "developer"); }

/* render_content: a message's content as text (into t, trimmed after), or the template's refusal */
static bool content(const Json *c, bool system, JsonBuf *t, char *err)
{
  if (!c || c->type == JSON_NULL)
    return true;
  if (c->type == JSON_STRING)
    return json_put(t, c->s, c->len), true;
  if (c->type != JSON_ARRAY)
    return qchat_fail(err, "Unexpected content type.");
  for (size_t i = 0; i < c->n; i++)
  {
    const Json *it = c->item[i];
    if (it->type != JSON_OBJECT)
      return qchat_fail(err, "Unexpected item type in content.");
    const char *type = json_str(it, "type");
    if (json_get(it, "image") || json_get(it, "image_url") || is(type, "image"))
      return qchat_fail(err, system ? "System message cannot contain images."
                                    : "an image: osmos reads text only (no vision placeholder rendered)");
    if (json_get(it, "video") || is(type, "video"))
      return qchat_fail(err, system ? "System message cannot contain videos."
                                    : "a video: osmos reads text only (no vision placeholder rendered)");
    const Json *text = json_get(it, "text");
    if (!text)
      return qchat_fail(err, "Unexpected item type in content.");
    if (text->type != JSON_STRING)
      return qchat_fail(err, "a text part's text: a string is needed");
    json_put(t, text->s, text->len);
  }
  return true;
}

/* a message's content trimmed: into *out (owned by *buf) */
static bool trimmed(const Json *m, bool system, JsonBuf *buf, Span *out, char *err)
{
  buf->n = 0;
  if (!content(json_get(m, "content"), system, buf, err))
    return false;
  *out = qchat_trim((Span){buf->p ? buf->p : "", buf->n});
  return true;
}

static void put(JsonBuf *b, Span s) { json_put(b, s.p, s.n); }

static bool effort_line(ChatMode mode, const char *effort, const char **line, char *err)
{
  *line = "";
  if (mode != CHAT_THINKING)
    return true;
  const char *e = effort ? effort : "xhigh";
  if (is(e, "high"))
    e = "xhigh";
  if (!is(e, "xhigh") && !is(e, "medium") && !is(e, "low"))
  {
    snprintf(err, CHAT_ERR, "Unexpected reasoning effort %s. Supported types are xhigh (default), medium, and low.", e);
    return false;
  }
  *line = is(e, "xhigh") ? kXhigh : is(e, "low") ? kLow : "";
  return true;
}

static void header(JsonBuf *b, const Json *tools, Span sys, const char *line)
{
  const bool has_line = line[0] != 0;
  if (tools)
  {
    json_puts(b, IM_START "system\n");
    if (has_line)
      json_puts(b, line), json_puts(b, "\n\n");
    json_puts(b, kToolsHead);
    for (size_t i = 0; i < tools->n; i++)
      json_puts(b, "\n"), json_dump(b, tools->item[i]);
    json_puts(b, kToolsTail);
    if (sys.n)
      json_puts(b, "\n\n"), put(b, sys);
    json_puts(b, QCHAT_EOS "\n");
  }
  else if (sys.n || has_line)
  {
    json_puts(b, IM_START "system\n");
    if (has_line)
      json_puts(b, line), json_puts(b, sys.n ? "\n\n" : "");
    put(b, sys), json_puts(b, QCHAT_EOS "\n");
  }
}

/* the last user message that is not tool responses alone (ns.last_query_index) */
static bool last_query(const Json *ms, size_t *last, JsonBuf *buf, char *err)
{
  *last = ms->n - 1;
  for (size_t k = ms->n; k-- > 0;)
  {
    if (!is(role_of(ms->item[k]), "user"))
      continue;
    Span c;
    if (!trimmed(ms->item[k], false, buf, &c, err))
      return false;
    const char *open = "<tool_response>", *close = "</tool_response>";
    const size_t on = strlen(open), cn = strlen(close);
    if (!(c.n >= on && !memcmp(c.p, open, on) && c.n >= cn && !memcmp(c.p + c.n - cn, close, cn)))
    {
      *last = k;
      break;
    }
  }
  return true;
}

static bool calls(JsonBuf *b, const Json *list, Span text, char *err)
{
  for (size_t k = 0; k < list->n; k++)
  {
    const Json *tc = list->item[k];
    if (json_get(tc, "function"))
      tc = json_get(tc, "function");
    const Json *name = json_get(tc, "name");
    if (!name || name->type != JSON_STRING)
      return qchat_fail(err, "Tool call is missing a function name.");
    if (!qchat_name_ok(name->s, name->len))
      return qchat_fail(err, "a tool call's function name holds a newline or '<' (the format's own markup)");
    json_puts(b, k ? "\n<tool_call>\n<function=" : text.n ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
    json_put(b, name->s, name->len), json_puts(b, ">\n");
    const Json *args = json_get(tc, "arguments");
    if (args && args->type == JSON_OBJECT)
      for (size_t i = 0; i < args->n; i++)
      {
        if (!qchat_name_ok(args->key[i], args->klen[i]))
          return qchat_fail(err, "a tool call's argument name holds a newline or '<' (the format's own markup)");
        json_puts(b, "<parameter="), json_put(b, args->key[i], args->klen[i]), json_puts(b, ">\n");
        const Json *v = args->item[i];
        if (v->type == JSON_STRING)
          json_put(b, v->s, v->len);
        else
          json_dump(b, v);
        json_puts(b, "\n</parameter>\n");
      }
    else if (args && args->type == JSON_STRING && qchat_trim((Span){args->s, args->len}).n)
    {
      snprintf(err, CHAT_ERR, "Tool call arguments for function \"%s\" were passed as a JSON string. Parse them "
                              "into an object before calling apply_chat_template.", name->s);
      return false;
    }
    else if (args && args->type != JSON_STRING && args->type != JSON_NULL)
    {
      snprintf(err, CHAT_ERR, "Tool call arguments for function \"%s\" must be an object/mapping or a JSON string.",
               name->s);
      return false;
    }
    json_puts(b, "</function>\n</tool_call>");
  }
  return true;
}

static bool message(JsonBuf *b, const Json *ms, size_t i, size_t last, bool preserve, JsonBuf *buf, char *err)
{
  const Json *m = ms->item[i];
  const char *role = role_of(m);
  Span c;
  if (!trimmed(m, false, buf, &c, err))
    return false;
  if (system_role(m))
    return qchat_fail(err, "System message must be at the beginning.");
  if (is(role, "user"))
    return json_puts(b, IM_START "user\n"), put(b, c), json_puts(b, QCHAT_EOS "\n"), true;
  if (is(role, "assistant"))
  {
    const Json *rc = json_get(m, "reasoning_content");
    const Span r = qchat_trim(rc && rc->type == JSON_STRING ? (Span){rc->s, rc->len} : (Span){"", 0});
    json_puts(b, IM_START "assistant\n");
    if (preserve || i > last)
      json_puts(b, "<think>\n"), put(b, r), json_puts(b, "\n</think>\n\n");
    put(b, c);
    const Json *tc = json_get(m, "tool_calls");
    if (tc && tc->type == JSON_ARRAY && !calls(b, tc, c, err))
      return false;
    return json_puts(b, QCHAT_EOS "\n"), true;
  }
  if (is(role, "tool"))
  {
    if (i > 0 && !is(role_of(ms->item[i - 1]), "tool"))
      json_puts(b, IM_START "user");
    json_puts(b, "\n<tool_response>\n"), put(b, c), json_puts(b, "\n</tool_response>");
    if (i + 1 == ms->n || !is(role_of(ms->item[i + 1]), "tool"))
      json_puts(b, QCHAT_EOS "\n");
    return true;
  }
  return qchat_fail(err, "Unexpected message role.");
}

static bool render(const Json *ms, const Json *branch, const Json *tools, ChatMode mode, const char *effort,
                   bool preserve, JsonBuf *b, JsonBuf *buf, char *err)
{
  JsonBuf sys = {0};
  size_t n_sys = 0;
  for (; n_sys < ms->n && system_role(ms->item[n_sys]); n_sys++)
  {
    Span c;
    if (!trimmed(ms->item[n_sys], true, buf, &c, err))
      return free(sys.p), false;
    if (c.n)
      json_puts(&sys, sys.n ? "\n" : ""), put(&sys, c);
  }
  const char *line;
  size_t last;
  bool ok = effort_line(mode, effort, &line, err);
  if (ok)
    header(b, tools, (Span){sys.p ? sys.p : "", sys.n}, line);
  free(sys.p);
  ok = ok && last_query(branch, &last, buf, err); /* on the branch's text: the template's choice the text's */
  for (size_t i = n_sys; ok && i < ms->n; i++)
    ok = message(b, ms, i, last, preserve, buf, err);
  if (ok)
    json_puts(b, mode == CHAT_THINKING ? IM_START "assistant\n<think>\n" : IM_START "assistant\n<think>\n\n</think>\n\n");
  return ok;
}

bool qchat_render(const Json *messages, const Json *tools, ChatMode mode, const char *effort, bool preserve,
                  JsonBuf *out, char err[CHAT_ERR])
{
  return qchat_render_with(messages, messages, tools, mode, effort, preserve, out, err);
}

bool qchat_render_with(const Json *messages, const Json *branch, const Json *tools, ChatMode mode, const char *effort,
                       bool preserve, JsonBuf *out, char err[CHAT_ERR])
{
  if (!branch || branch->type != JSON_ARRAY || !messages || messages->type != JSON_ARRAY || branch->n != messages->n)
    return qchat_fail(err, "a branch's messages not the messages' count");
  if (!messages || messages->type != JSON_ARRAY || !messages->n)
    return qchat_fail(err, "No messages provided.");
  for (size_t i = 0; i < messages->n; i++)
    if (messages->item[i]->type != JSON_OBJECT)
      return qchat_fail(err, "messages: each an object");
  if (tools && tools->type != JSON_ARRAY && tools->type != JSON_NULL)
    return qchat_fail(err, "tools: a list of function tools");
  JsonBuf buf = {0};
  const bool ok = render(messages, branch, tools && tools->type == JSON_ARRAY && tools->n ? tools : NULL, mode, effort,
                         preserve, out, &buf, err);
  free(buf.p);
  if (ok && (out->oom || buf.oom))
    return qchat_fail(err, "out of memory");
  return ok;
}
