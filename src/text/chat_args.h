/* a tool call's arguments read back from a template's text (chat_qwen_parse.c's, chat_hy4_parse.c's): a parameter's
   schema in the tools given, and its value - JSON where that is no string and of a type the schema allows, else the
   text itself (as a template writes a string raw) */
#pragma once

#include <stddef.h>

#include "osmos/text/json.h"

/* a function's parameter's schema in tools (OpenAI's, or bare {name, parameters}), or NULL: no such tool or parameter */
const Json *chat_arg_schema(const Json *tools, const char *fn, size_t fn_n, const char *param, size_t param_n);
/* the value's JSON into args where it parses to no string and the schema allows its type (its "type", or an anyOf /
   oneOf's; an integer where a number is), else the text as a JSON string. a schema naming no type: a string's */
void chat_arg_value(JsonBuf *args, const char *raw, size_t n, const Json *schema);
