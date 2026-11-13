#ifndef OSMOS_TEXT_JSON_H
#define OSMOS_TEXT_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define JSON_ERR 128
#define JSON_MAX_DEPTH 256 /* nesting refused past this (Python's own json gives up near its recursion limit, 1000) */

typedef enum JsonType
{
  JSON_NULL,
  JSON_FALSE,
  JSON_TRUE,
  JSON_INT,
  JSON_FLOAT,
  JSON_STRING,
  JSON_ARRAY,
  JSON_OBJECT,
} JsonType;

typedef struct Json
{
  JsonType type;
  char *s;            /* a string's UTF-8 (NUL-terminated, len its bytes); an int's digits as Python writes them */
  size_t len;
  double f;           /* a float's value */
  struct Json **item; /* an array's values, an object's (in the order their keys first came) */
  char **key;         /* an object's keys (UTF-8, NUL-terminated) */
  size_t *klen;
  size_t n;
} Json;

typedef struct JsonBuf /* text written: grown as needed; oom set (and nothing more written) when it could not grow */
{
  char *p;
  size_t n, cap;
  bool oom;
} JsonBuf;

Json *json_parse(const char *text, size_t len, char err[JSON_ERR]);
void json_free(Json *j);
const Json *json_get(const Json *obj, const char *key);
const char *json_str(const Json *obj, const char *key);

void json_dump(JsonBuf *b, const Json *j);
void json_dump_string(JsonBuf *b, const char *s, size_t len);
size_t json_utf8_len(const unsigned char *p, size_t left);
void json_put(JsonBuf *b, const char *s, size_t len);
void json_puts(JsonBuf *b, const char *s);

#endif

/*
json_parse
  RFC 8259 JSON (UTF-8 in, every escape, \u pairs joined), read as Python's json.loads reads it: an int its digits
  (any length), a number with a fraction or an exponent a double, an object's later duplicate key's value in the
  first's place. NULL with why for anything else - NaN and Infinity too (Python reads them; JSON has none), a number past a double's range (Python's inf: its
  dump Infinity), a lone
  surrogate, invalid UTF-8, nesting past JSON_MAX_DEPTH.

json_free
  j and everything under it. NULL is nothing to do.

json_get / json_str
  an object's value of that key, or NULL (not an object, no such key); json_str the value's text if a string.

json_dump
  j written as Python's json.dumps(value, ensure_ascii=False) writes it - ", " between items, ": " after keys, a
  float's shortest round-trip digits (repr: fixed from 1e-4 below 1e16, else d.dde+XX; ".0" on a whole one), an int
  its digits, a string's ", \, control characters escaped (\n \r \t \b \f, else \u00xx) and the rest raw. (DeepSeek's
  chat encoding writes tool schemas and arguments so: the prompt's bytes depend on it.)

json_dump_string
  len bytes of UTF-8 as a JSON string, json_dump's escapes; each invalid sequence's maximal subpart (Unicode's, as
  Python's errors="replace" takes it) U+FFFD - a model's bytes (a character cut by max_tokens) valid JSON still.

json_utf8_len
  a valid UTF-8 sequence's length at p (shortest form, no surrogate, at most U+10FFFF; left > 0), or 0.

json_put / json_puts
  bytes appended as they are.
*/
