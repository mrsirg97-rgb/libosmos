#include <string.h>

#include "chat_args.h"

/* a function's parameter's schema, or NULL (no such tool or parameter) */
const Json *chat_arg_schema(const Json *tools, const char *fn, size_t fn_n, const char *param, size_t param_n)
{
  const Json *props = NULL;
  for (size_t i = 0; tools && i < tools->n; i++)
  {
    const Json *f = json_get(tools->item[i], "function") ? json_get(tools->item[i], "function") : tools->item[i];
    const Json *name = json_get(f, "name");
    if (name && name->type == JSON_STRING && name->len == fn_n && !memcmp(name->s, fn, fn_n))
      props = json_get(json_get(f, "parameters"), "properties");
  }
  for (size_t i = 0; props && props->type == JSON_OBJECT && i < props->n; i++)
    if (props->klen[i] == param_n && !memcmp(props->key[i], param, param_n))
      return props->item[i];
  return NULL;
}

/* kind among a "type" (one, or a list); *named whether it names any */
static bool listed(const Json *types, const char *kind, bool *named)
{
  if (types && types->type == JSON_STRING)
    return *named = true, !strcmp(types->s, kind);
  for (size_t i = 0; types && types->type == JSON_ARRAY && i < types->n; i++)
    if (types->item[i]->type == JSON_STRING && (*named = true) && !strcmp(types->item[i]->s, kind))
      return true;
  return false;
}

/* the schema allows kind: its "type", or an anyOf / oneOf's (pydantic's Optional[str]: anyOf string, null). a
   schema that names no type - or none - a string's alone, as vLLM's qwen3coder parser takes it */
static bool allows(const Json *schema, const char *kind)
{
  bool named = false;
  if (listed(json_get(schema, "type"), kind, &named))
    return true;
  for (int a = 0; a < 2; a++)
  {
    const Json *alts = json_get(schema, a ? "oneOf" : "anyOf");
    for (size_t i = 0; alts && alts->type == JSON_ARRAY && i < alts->n; i++)
      if (listed(json_get(alts->item[i], "type"), kind, &named))
        return true;
  }
  return !named && !strcmp(kind, "string");
}

static const char *kind_of(const Json *v)
{
  switch (v->type)
  {
  case JSON_TRUE:
  case JSON_FALSE:
    return "boolean";
  case JSON_NULL:
    return "null";
  case JSON_INT:
    return "integer";
  case JSON_FLOAT:
    return "number";
  case JSON_OBJECT:
    return "object";
  case JSON_ARRAY:
    return "array";
  default:
    return "string";
  }
}

/* a parameter's value into args: its JSON where that is no string and of a type allowed, else its text */
void chat_arg_value(JsonBuf *args, const char *raw, size_t n, const Json *schema)
{
  char jerr[OSMOS_ERR] = "";
  Json *v = json_parse(raw, n, jerr);
  const char *kind = v ? kind_of(v) : "string";
  const bool typed = v && strcmp(kind, "string") &&
                     (allows(schema, kind) || (!strcmp(kind, "integer") && allows(schema, "number")));
  if (typed)
    json_dump(args, v);
  else
    json_dump_string(args, raw, n);
  json_free(v);
}
