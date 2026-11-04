#include <stdlib.h>
#include <string.h>

#include "chat_impl.h"

/* encoding.py's TOOLS_TEMPLATE, its tokens and tag names in place */
static const char kToolsHead[] =
    "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing "
    "a \"" CHAT_CALLS_OPEN "\" block like the following:\n\n" CHAT_CALLS_OPEN "\n" CHAT_INVOKE_OPEN
    " name=\"$TOOL_NAME\">\n" CHAT_PARAM_OPEN " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE"
    CHAT_PARAM_CLOSE "\n...\n" CHAT_INVOKE_CLOSE ">\n" CHAT_INVOKE_OPEN " name=\"$TOOL_NAME2\">\n...\n"
    CHAT_INVOKE_CLOSE ">\n" CHAT_CALLS_CLOSE "\n\nString parameters should be specified as is and set "
    "`string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and "
    "set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by " CHAT_THINK "), you MUST output your "
    "complete reasoning inside " CHAT_THINK "..." CHAT_THINK_END " BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after " CHAT_THINK_END " with tool calls or final response.\n\n### Available Tool "
    "Schemas\n\n";
static const char kToolsTail[] =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

bool chat_tool_name(const char *name, const char *ns, JsonBuf *out, char *err)
{
  if (!name)
    return chat_fail(err, "a tool", "no name");
  const char *sep = strstr(name, "::");
  size_t plen = 0;
  if (sep)
  {
    plen = (size_t)(sep - name);
    if (ns && (strlen(ns) != plen || strncmp(ns, name, plen)))
      return chat_fail(err, name, "conflicting tool namespaces");
    ns = NULL; /* the prefix is the namespace */
    if (strstr(sep + 2, "::"))
      return chat_fail(err, name, "a tool name with '::' in it");
    json_put(out, name, plen), json_puts(out, "::"), json_puts(out, sep + 2);
    return true;
  }
  if (ns && strstr(ns, "::"))
    return chat_fail(err, ns, "a tool namespace with '::' in it");
  if (ns)
    json_puts(out, ns), json_puts(out, "::");
  json_puts(out, name);
  return true;
}

/* one function as tools_from_openai_format leaves it, to_json: its keys in order, "name" qualified, "namespace"
   gone, a namespace object's description before its own (appended last if it had none) */
static bool function(JsonBuf *b, const Json *tool, char *err)
{
  const Json *fn = json_get(tool, "function");
  if (!fn || fn->type != JSON_OBJECT)
    return chat_fail(err, "a tool", "no function object");
  const Json *ns = json_get(tool, "namespace");
  if (!ns || ns->type == JSON_NULL)
    ns = json_get(fn, "namespace");
  const char *ns_name = !ns                       ? NULL
                        : ns->type == JSON_STRING ? ns->s
                        : ns->type == JSON_OBJECT ? json_str(ns, "name")
                                                  : NULL;
  const char *ns_desc = ns && ns->type == JSON_OBJECT ? json_str(ns, "description") : NULL;
  if (ns_desc && !ns_desc[0])
    ns_desc = NULL;
  JsonBuf q = {0};
  if (!chat_tool_name(json_str(fn, "name"), ns && ns->type != JSON_NULL ? ns_name : NULL, &q, err))
    return free(q.p), false;
  json_puts(b, "{");
  bool first = true, had_desc = false;
  for (size_t i = 0; i < fn->n; i++)
  {
    const char *k = fn->key[i];
    if (strcmp(k, "namespace") == 0)
      continue;
    json_puts(b, first ? "" : ", "), first = false;
    json_dump_string(b, k, fn->klen[i]), json_puts(b, ": ");
    if (strcmp(k, "name") == 0)
      json_dump_string(b, q.p, q.n);
    else if (strcmp(k, "description") == 0 && ns_desc)
    {
      JsonBuf d = {0};
      json_puts(&d, ns_desc), json_puts(&d, "\n");
      if (fn->item[i]->type == JSON_STRING)
        json_put(&d, fn->item[i]->s, fn->item[i]->len);
      json_dump_string(b, d.p, d.n), free(d.p), had_desc = true;
    }
    else
      json_dump(b, fn->item[i]);
  }
  if (ns_desc && !had_desc)
  {
    JsonBuf d = {0};
    json_puts(&d, ns_desc), json_puts(&d, "\n");
    json_puts(b, first ? "" : ", "), json_puts(b, "\"description\": "), json_dump_string(b, d.p, d.n), free(d.p);
  }
  json_puts(b, "}");
  free(q.p);
  return true;
}

bool chat_tools(JsonBuf *b, const Json *tools, char *err)
{
  if (tools->type != JSON_ARRAY)
    return chat_fail(err, "tools", "not a list");
  json_puts(b, kToolsHead);
  for (size_t i = 0; i < tools->n; i++)
  {
    if (i)
      json_puts(b, "\n");
    if (!function(b, tools->item[i], err))
      return false;
  }
  json_puts(b, kToolsTail);
  return true;
}

/* a call's arguments as encode_arguments_to_dsml takes them: an object, or a JSON string of one (read twice, for a
   double-encoded one), else {"arguments": as given} */
static void arguments(JsonBuf *b, const Json *args)
{
  Json *own = NULL;
  const Json *a = args;
  char err[JSON_ERR];
  for (int i = 0; i < 2 && a && a->type == JSON_STRING; i++)
  {
    Json *next = json_parse(a->s, a->len, err);
    if (!next)
      break;
    json_free(own);
    a = own = next;
  }
  const bool obj = a && a->type == JSON_OBJECT;
  const size_t n = obj ? a->n : 1;
  for (size_t i = 0; i < n; i++)
  {
    const Json *v = obj ? a->item[i] : args;
    if (i)
      json_puts(b, "\n");
    json_puts(b, CHAT_PARAM_OPEN " name=\"");
    if (obj)
      json_put(b, a->key[i], a->klen[i]);
    else
      json_puts(b, "arguments");
    json_puts(b, v->type == JSON_STRING ? "\" string=\"true\">" : "\" string=\"false\">");
    if (v->type == JSON_STRING)
      json_put(b, v->s, v->len);
    else
      json_dump(b, v);
    json_puts(b, CHAT_PARAM_CLOSE);
  }
  json_free(own);
}

bool chat_calls(JsonBuf *b, const Json *calls, char *err)
{
  if (calls->type != JSON_ARRAY)
    return chat_fail(err, "tool_calls", "not a list");
  json_puts(b, "\n\n" CHAT_CALLS_OPEN "\n");
  for (size_t i = 0; i < calls->n; i++)
  {
    const Json *c = calls->item[i], *fn = json_get(c, "function"), *args = json_get(fn, "arguments");
    const char *ns = json_str(c, "namespace");
    if (!ns || !ns[0])
      ns = json_str(fn, "namespace");
    if (!fn || !args)
      return chat_fail(err, "a tool call", "no function or no arguments");
    if (i)
      json_puts(b, "\n");
    json_puts(b, CHAT_INVOKE_OPEN " name=\"");
    if (!chat_tool_name(json_str(fn, "name"), ns && ns[0] ? ns : NULL, b, err))
      return false;
    json_puts(b, "\">\n");
    arguments(b, args);
    json_puts(b, "\n" CHAT_INVOKE_CLOSE ">");
  }
  json_puts(b, "\n" CHAT_CALLS_CLOSE);
  return true;
}
