#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/chat_ink2.h"
#include "osmos/text/chat_qwen.h"
#include "osmos/err.h"

/* the template's render (osmos/text/chat_ink2.h), its branches in its order; its refusals its own words */

/* Python's truth of a value: null, false, 0, "", [] and {} false (NULL: an absent key's Undefined, false too) */
static bool truthy(const Json *j)
{
  if (!j)
    return false;
  switch (j->type)
  {
  case JSON_NULL:
  case JSON_FALSE:
    return false;
  case JSON_TRUE:
    return true;
  case JSON_INT:
    return strcmp(j->s, "0") != 0;
  case JSON_FLOAT:
    return j->f != 0;
  case JSON_STRING:
    return j->len > 0;
  case JSON_ARRAY:
  case JSON_OBJECT:
    return j->n > 0;
  }
  return false;
}

static bool is_str(const Json *j) { return j && j->type == JSON_STRING; }
static bool is_obj(const Json *j) { return j && j->type == JSON_OBJECT; }
static bool str_is(const Json *j, const char *s) { return is_str(j) && !strcmp(j->s, s); }
static void put_str(JsonBuf *b, const Json *s) { json_put(b, s->s, s->len); }

static const JsonStyle kCompact = {true, false}, kSorted = {true, true};

/* "Thinking effort level: " and the effort: a name (trimmed) the map's, else the value as Python's float() */
static bool effort_line(const Json *e, JsonBuf *out, char *err)
{
  static const struct
  {
    const char *name;
    double v;
  } map[] = {{"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7}, {"high", 0.9}, {"xhigh", 0.99},
             {"max", 0.99}};
  double v = 0.9;
  if (e && e->type != JSON_NULL)
  {
    if (e->type == JSON_STRING)
    {
      const char *p = e->s + qchat_space_lead(e->s, e->len), *q = e->s + e->len; /* jinja's trim: str.strip */
      q -= q > p ? qchat_space_trail(p, (size_t)(q - p)) : 0;
      size_t i = 0;
      for (; i < sizeof map / sizeof *map; i++)
        if ((size_t)(q - p) == strlen(map[i].name) && !memcmp(p, map[i].name, (size_t)(q - p)))
          break;
      if (i == sizeof map / sizeof *map)
      {
        osmos_fail(err, "Unknown reasoning_effort: %.*s", (int)e->len, e->s);
        return false;
      }
      v = map[i].v;
    }
    else if (e->type == JSON_INT)
      v = strtod(e->s, NULL);
    else if (e->type == JSON_FLOAT)
      v = e->f;
    else if (e->type == JSON_TRUE || e->type == JSON_FALSE)
      v = e->type == JSON_TRUE;
    else
      return osmos_fail(err, "reasoning_effort: a name or a number");
  }
  if (v < 0.0 || v > 0.99)
    return osmos_fail(err, "reasoning_effort must be in [0.0, 0.99]");
  json_puts(out, "<|message_system|><|content_text|>Thinking effort level: ");
  if (v == 0.0)
    json_puts(out, "0");
  else
  {
    const Json f = {.type = JSON_FLOAT, .f = v};
    json_dump(out, &f);
  }
  json_puts(out, ICHAT_END);
  return true;
}

static bool tools_block(const Json *tools, JsonBuf *out, char *err)
{
  if (tools->type != JSON_ARRAY)
    return osmos_fail(err, "tools: an array");
  /* each spec built as the template's dict, then the list dumped sorted and compact */
  Json *specs = calloc(1, sizeof *specs);
  bool ok = specs && (specs->item = calloc(tools->n, sizeof *specs->item));
  if (ok)
    specs->type = JSON_ARRAY;
  static const char *const keys[4] = {"description", "name", "parameters", "type"};
  static size_t klen[4] = {11, 4, 10, 4};
  static const Json empty_str = {.type = JSON_STRING, .s = "", .len = 0};
  static const Json empty_obj = {.type = JSON_OBJECT};
  static const Json function = {.type = JSON_STRING, .s = "function", .len = 8};
  for (size_t i = 0; ok && i < tools->n; i++)
  {
    const Json *tool = tools->item[i];
    if (!is_obj(tool))
      ok = osmos_fail(err, "a tool: an object");
    const Json *fn = ok && json_get(tool, "function") ? json_get(tool, "function") : tool;
    if (ok && !is_obj(fn))
      ok = osmos_fail(err, "a tool's function: an object");
    const Json *name = ok ? json_get(fn, "name") : NULL;
    if (ok && !name)
      ok = osmos_fail(err, "a tool's function without a name");
    Json *spec = ok ? calloc(1, sizeof *spec) : NULL;
    if (ok && !(spec && (spec->item = calloc(4, sizeof *spec->item))))
      ok = osmos_fail(err, "out of memory"), free(spec), spec = NULL;
    if (!ok)
      break;
    const Json *d = json_get(fn, "description"), *p = json_get(fn, "parameters"), *t = json_get(tool, "type");
    spec->type = JSON_OBJECT, spec->n = 4, spec->key = (char **)keys, spec->klen = klen;
    spec->item[0] = (Json *)(truthy(d) ? d : &empty_str), spec->item[1] = (Json *)name;
    spec->item[2] = (Json *)(truthy(p) ? p : &empty_obj), spec->item[3] = (Json *)(truthy(t) ? t : &function);
    specs->item[specs->n++] = spec;
  }
  if (ok)
  {
    json_puts(out, "<|message_system|>tool_declare<|content_xml|>");
    json_dump_with(out, specs, kSorted);
    json_puts(out, ICHAT_END);
  }
  for (size_t i = 0; specs && specs->item && i < specs->n; i++) /* the specs' values are the request's: theirs */
    free(specs->item[i]->item), free(specs->item[i]);
  if (specs)
    free(specs->item);
  free(specs);
  return ok || (err[0] ? false : osmos_fail(err, "out of memory"));
}

/* a tool result's header name: the name of each call (of every assistant's) whose id is its tool_call_id */
static bool result_names(const Json *messages, const Json *id, JsonBuf *head, char *err)
{
  if (!is_str(id))
    return osmos_fail(err, "a tool message's tool_call_id: a string");
  for (size_t i = 0; i < messages->n; i++)
  {
    const Json *prev = messages->item[i], *calls = json_get(prev, "tool_calls");
    if (!str_is(json_get(prev, "role"), "assistant") || !truthy(calls))
      continue;
    if (calls->type != JSON_ARRAY)
      return osmos_fail(err, "tool_calls: an array");
    for (size_t k = 0; k < calls->n; k++)
    {
      const Json *tc = calls->item[k], *tid = is_obj(tc) ? json_get(tc, "id") : NULL;
      if (!is_str(tid) || tid->len != id->len || memcmp(tid->s, id->s, id->len))
        continue;
      const Json *fn = json_get(tc, "function");
      if (!fn)
        return osmos_fail(err, "a tool call without its function");
      const Json *name = is_obj(fn) ? json_get(fn, "name") : NULL;
      if (name && !is_str(name))
        return osmos_fail(err, "a tool call's name: a string");
      if (name)
        put_str(head, name);
    }
  }
  return true;
}

static void block(JsonBuf *out, const JsonBuf *head, const char *kind, const char *body, size_t n)
{
  json_put(out, head->p, head->n), json_puts(out, kind), json_put(out, body, n), json_puts(out, ICHAT_END);
}

static bool parts(const Json *content, const char *role, const JsonBuf *head, JsonBuf *out, char *err)
{
  if (content->type != JSON_ARRAY)
    return osmos_fail(err, "%s", content->type == JSON_OBJECT ? "content an object: a string or a list of parts"
                                                  : "content: a string or a list of parts");
  for (size_t i = 0; i < content->n; i++)
  {
    const Json *part = content->item[i];
    if (is_str(part))
    {
      block(out, head, ICHAT_TEXT, part->s, part->len);
      continue;
    }
    if (!is_obj(part))
      return osmos_fail(err, "a content part: a string or an object");
    const Json *type = json_get(part, "type");
    if (!type || str_is(type, "text") || str_is(type, "input_text"))
    {
      const Json *text = json_get(part, "text");
      if (is_str(text))
        block(out, head, ICHAT_TEXT, text->s, text->len);
      else
        block(out, head, ICHAT_TEXT, "", 0);
    }
    else if (str_is(type, "image") || str_is(type, "input_image") || str_is(type, "image_url"))
      return osmos_fail(err, "%s", strcmp(role, "user") && strcmp(role, "tool")
                           ? "Image content is only supported for user and tool messages"
                           : "an image part: this build reads text alone");
    else if (str_is(type, "audio") || str_is(type, "input_audio") || str_is(type, "audio_url"))
      return osmos_fail(err, "Inkling 2 does not accept audio input");
    else if (is_str(type))
    {
      osmos_fail(err, "Unsupported content part type: %.*s", (int)type->len, type->s);
      return false;
    }
    else
      return osmos_fail(err, "a content part's type: a string");
  }
  return true;
}

static bool calls_blocks(const Json *msg, const Json *calls, JsonBuf *out, char *err)
{
  if (calls->type != JSON_ARRAY)
    return osmos_fail(err, "tool_calls: an array");
  const Json *mname = json_get(msg, "name");
  for (size_t i = 0; i < calls->n; i++)
  {
    const Json *tc = calls->item[i], *fn = is_obj(tc) ? json_get(tc, "function") : NULL;
    const Json *name = is_obj(fn) ? json_get(fn, "name") : NULL;
    if (!is_str(name))
      return osmos_fail(err, "tool call function name must be a string");
    if (truthy(mname) && (!is_str(mname) || mname->len != name->len || memcmp(mname->s, name->s, name->len)))
    {
      osmos_fail(err, "assistant name %.*s does not match its tool call %.*s",
               is_str(mname) ? (int)mname->len : 0, is_str(mname) ? mname->s : "", (int)name->len, name->s);
      return false;
    }
    const Json *args = json_get(fn, "arguments");
    static const Json none = {.type = JSON_OBJECT};
    if (!truthy(args))
      args = &none;
    if (is_str(args))
      return osmos_fail(err, "tool call arguments must be a parsed object, not a JSON string; canonicalize upstream");
    if (!is_obj(args))
      return osmos_fail(err, "tool call arguments must be an object");
    json_puts(out, ICHAT_MODEL), put_str(out, name), json_puts(out, ICHAT_CALL "{\"name\":");
    json_dump_with(out, name, kSorted);
    json_puts(out, ",\"args\":{");
    /* the keys sorted (dictsort, case-sensitive: code point order), each value compact and unsorted */
    size_t stack[64], *order = args->n <= 64 ? stack : malloc(args->n * sizeof *order);
    if (!order)
      return osmos_fail(err, "out of memory");
    for (size_t a = 0; a < args->n; a++)
    {
      size_t k = a;
      for (; k > 0; k--)
      {
        const size_t o = order[k - 1], m = args->klen[o] < args->klen[a] ? args->klen[o] : args->klen[a];
        const int c = memcmp(args->key[o], args->key[a], m);
        if (c < 0 || (c == 0 && args->klen[o] <= args->klen[a]))
          break;
        order[k] = order[k - 1];
      }
      order[k] = a;
    }
    for (size_t k = 0; k < args->n; k++)
    {
      if (k)
        json_puts(out, ",");
      json_dump_string(out, args->key[order[k]], args->klen[order[k]]);
      json_puts(out, ":");
      json_dump_with(out, args->item[order[k]], kCompact);
    }
    if (order != stack)
      free(order);
    json_puts(out, "}}" ICHAT_END);
  }
  return true;
}

static bool message(const Json *messages, const Json *msg, const char *role, JsonBuf *out, char *err)
{
  static const char *const tokens[][2] = {{"user", "<|message_user|>"},
                                          {"assistant", ICHAT_MODEL},
                                          {"system", "<|message_system|>"},
                                          {"tool", "<|message_tool|>"}};
  JsonBuf head = {0};
  for (size_t i = 0; i < 4; i++)
    if (!strcmp(role, tokens[i][0]))
      json_puts(&head, tokens[i][1]);
  const bool assistant = !strcmp(role, "assistant");
  const Json *name = json_get(msg, "name"), *id = json_get(msg, "tool_call_id");
  bool ok = true;
  if (!strcmp(role, "tool") && truthy(name))
    ok = is_str(name) ? (put_str(&head, name), true) : osmos_fail(err, "a tool message's name: a string");
  else if (!strcmp(role, "tool") && truthy(id))
    ok = result_names(messages, id, &head, err);
  else if (assistant && truthy(name))
    ok = is_str(name) ? (put_str(&head, name), true) : osmos_fail(err, "an assistant's name: a string");
  const Json *reasoning = json_get(msg, "reasoning_content"), *content = json_get(msg, "content");
  const Json *calls = json_get(msg, "tool_calls");
  if (ok && assistant && truthy(reasoning))
  {
    if (is_str(reasoning))
      block(out, &head, ICHAT_THINKING, reasoning->s, reasoning->len);
    else
      ok = osmos_fail(err, "reasoning_content: a string");
  }
  if (ok && is_str(content) && (truthy(content) || !(assistant && (truthy(calls) || truthy(reasoning)))))
    block(out, &head, ICHAT_TEXT, content->s, content->len);
  else if (ok && !is_str(content) && truthy(content))
    ok = parts(content, role, &head, out, err);
  if (ok && assistant && truthy(calls))
    ok = calls_blocks(msg, calls, out, err);
  if (ok && assistant)
    json_puts(out, ICHAT_EOS);
  free(head.p);
  return ok && (!head.oom || osmos_fail(err, "out of memory"));
}

bool ichat_render(const Json *messages, const Json *tools, const Json *effort, JsonBuf *out, char err[OSMOS_ERR])
{
  if (!messages || messages->type != JSON_ARRAY || !messages->n)
    return osmos_fail(err, "Cannot apply chat template to an empty conversation. Provide at least one message.");
  if (tools && truthy(tools) && !tools_block(tools, out, err))
    return false;
  bool effort_done = false;
  for (size_t i = 0; i < messages->n; i++)
  {
    const Json *msg = messages->item[i], *role = is_obj(msg) ? json_get(msg, "role") : NULL;
    if (!is_obj(msg))
      return osmos_fail(err, "a message: an object");
    if (!str_is(role, "user") && !str_is(role, "assistant") && !str_is(role, "system") && !str_is(role, "tool"))
    {
      if (is_str(role))
        osmos_fail(err, "Unknown message role: %.*s", (int)role->len, role->s);
      else
        osmos_fail(err, "Unknown message role");
      return false;
    }
    if (!effort_done && !str_is(role, "system"))
    {
      if (!effort_line(effort, out, err))
        return false;
      effort_done = true;
    }
    if (!message(messages, msg, role->s, out, err))
      return false;
  }
  if (!effort_done && !effort_line(effort, out, err))
    return false;
  return !out->oom || osmos_fail(err, "out of memory");
}
