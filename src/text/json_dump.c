#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/json.h"

/* the writer: Python's json.dumps(value, ensure_ascii=False), byte for byte (json.h) */

void json_put(JsonBuf *b, const char *s, size_t len)
{
  if (b->oom)
    return;
  if (b->n + len + 1 > b->cap)
  {
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + len + 1)
      cap *= 2;
    char *p = realloc(b->p, cap);
    if (!p)
    {
      b->oom = true;
      return;
    }
    b->p = p, b->cap = cap;
  }
  memcpy(b->p + b->n, s, len);
  b->n += len;
  b->p[b->n] = 0;
}

void json_puts(JsonBuf *b, const char *s) { json_put(b, s, strlen(s)); }

/* the bytes at p of an invalid sequence's maximal subpart: its lead and the continuations a valid sequence could
   have gone on with (E0's second byte A0..BF, ED's 80..9F, F0's 90..BF, F4's 80..8F), one at least */
static size_t subpart(const unsigned char *p, size_t left)
{
  const unsigned char b = p[0];
  const size_t n = b >= 0xC2 && b <= 0xDF ? 2 : b >= 0xE0 && b <= 0xEF ? 3 : b >= 0xF0 && b <= 0xF4 ? 4 : 1;
  const unsigned char lo = b == 0xE0 ? 0xA0 : b == 0xF0 ? 0x90 : 0x80, hi = b == 0xED ? 0x9F : b == 0xF4 ? 0x8F : 0xBF;
  size_t k = 1;
  while (k < n && k < left && p[k] >= (k == 1 ? lo : 0x80) && p[k] <= (k == 1 ? hi : 0xBF))
    k++;
  return k;
}

void json_dump_string(JsonBuf *b, const char *s, size_t len)
{
  json_put(b, "\"", 1);
  size_t run = 0; /* the bytes since the last escape, written at once */
  for (size_t i = 0; i < len; i++)
  {
    const unsigned char c = (unsigned char)s[i];
    if (c >= 0x80)
    {
      const size_t k = json_utf8_len((const unsigned char *)s + i, len - i);
      if (k)
      {
        run += k, i += k - 1;
        continue;
      }
      json_put(b, s + i - run, run), run = 0;
      json_puts(b, "\xEF\xBF\xBD"); /* U+FFFD */
      i += subpart((const unsigned char *)s + i, len - i) - 1;
      continue;
    }
    const char *e = c == '"' ? "\\\"" : c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\r' ? "\\r" : c == '\t' ? "\\t"
                  : c == '\b' ? "\\b" : c == '\f' ? "\\f" : NULL;
    if (!e && c >= 0x20)
    {
      run++;
      continue;
    }
    json_put(b, s + i - run, run), run = 0;
    if (e)
      json_puts(b, e);
    else
    {
      char u[8];
      snprintf(u, sizeof u, "\\u%04x", c);
      json_puts(b, u);
    }
  }
  json_put(b, s + len - run, run);
  json_put(b, "\"", 1);
}

/* Python's repr of a double: its shortest round-trip digits; fixed between 1e-4 and 1e16, else d.dde+XX; a whole
   fixed one ".0" */
static void dump_float(JsonBuf *b, double f)
{
  if (isnan(f) || isinf(f))
    return json_puts(b, isnan(f) ? "NaN" : f < 0 ? "-Infinity" : "Infinity");
  if (f == 0)
    return json_puts(b, signbit(f) ? "-0.0" : "0.0");
  char e[40], digits[24];
  for (int prec = 1; prec <= 17; prec++)
  {
    snprintf(e, sizeof e, "%.*e", prec - 1, f);
    if (strtod(e, NULL) == f)
      break;
  }
  const char *p = e + (e[0] == '-');
  size_t nd = 0;
  for (; *p && *p != 'e'; p++)
    if (*p != '.')
      digits[nd++] = *p;
  while (nd > 1 && digits[nd - 1] == '0')
    nd--;
  digits[nd] = 0;
  const int exp = atoi(p + 1);
  char out[400];
  size_t o = 0;
  if (f < 0)
    out[o++] = '-';
  if (exp >= -4 && exp < 16)
  {
    if (exp < 0)
    {
      out[o++] = '0', out[o++] = '.';
      for (int z = 0; z < -exp - 1; z++)
        out[o++] = '0';
      memcpy(out + o, digits, nd), o += nd;
    }
    else
    {
      for (int i = 0; i <= exp; i++)
        out[o++] = (size_t)i < nd ? digits[i] : '0';
      out[o++] = '.';
      if ((size_t)exp + 1 < nd)
        memcpy(out + o, digits + exp + 1, nd - (size_t)exp - 1), o += nd - (size_t)exp - 1;
      else
        out[o++] = '0';
    }
  }
  else
  {
    out[o++] = digits[0];
    if (nd > 1)
      out[o++] = '.', memcpy(out + o, digits + 1, nd - 1), o += nd - 1;
    o += (size_t)snprintf(out + o, sizeof out - o, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
  }
  json_put(b, out, o);
}

void json_dump(JsonBuf *b, const Json *j)
{
  switch (j->type)
  {
  case JSON_NULL:
    return json_puts(b, "null");
  case JSON_FALSE:
    return json_puts(b, "false");
  case JSON_TRUE:
    return json_puts(b, "true");
  case JSON_INT:
    return json_put(b, j->s, j->len);
  case JSON_FLOAT:
    return dump_float(b, j->f);
  case JSON_STRING:
    return json_dump_string(b, j->s, j->len);
  case JSON_ARRAY:
  case JSON_OBJECT:
    json_puts(b, j->type == JSON_ARRAY ? "[" : "{");
    for (size_t i = 0; i < j->n; i++)
    {
      if (i)
        json_puts(b, ", ");
      if (j->type == JSON_OBJECT)
        json_dump_string(b, j->key[i], j->klen[i]), json_puts(b, ": ");
      json_dump(b, j->item[i]);
    }
    json_puts(b, j->type == JSON_ARRAY ? "]" : "}");
  }
}
