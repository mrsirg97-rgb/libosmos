#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/json.h"

/* the reader: RFC 8259 by recursive descent, Python's json.loads' meaning for what it reads (json.h) */

typedef struct Cur
{
  const char *start, *p, *end;
  char *err;
  int depth;
} Cur;

static Json *fail(Cur *c, const char *why)
{
  if (!c->err[0])
    snprintf(c->err, JSON_ERR, "json: %s at byte %td", why, c->p - c->start);
  return NULL;
}

static void ws(Cur *c)
{
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r'))
    c->p++;
}

static bool take(Cur *c, const char *lit)
{
  const size_t n = strlen(lit);
  if ((size_t)(c->end - c->p) < n || memcmp(c->p, lit, n))
    return false;
  c->p += n;
  return true;
}

static Json *node(JsonType t)
{
  Json *j = calloc(1, sizeof *j);
  if (j)
    j->type = t;
  return j;
}

static int hex4(const char *p)
{
  int v = 0;
  for (int i = 0; i < 4; i++)
  {
    const char h = p[i];
    const int d = h >= '0' && h <= '9'   ? h - '0'
                  : h >= 'a' && h <= 'f' ? h - 'a' + 10
                  : h >= 'A' && h <= 'F' ? h - 'A' + 10
                                         : -1;
    if (d < 0)
      return -1;
    v = v << 4 | d;
  }
  return v;
}

static size_t put_utf8(char *o, uint32_t c)
{
  if (c < 0x80)
    return o[0] = (char)c, 1;
  if (c < 0x800)
    return o[0] = (char)(0xC0 | c >> 6), o[1] = (char)(0x80 | (c & 0x3F)), 2;
  if (c < 0x10000)
    return o[0] = (char)(0xE0 | c >> 12), o[1] = (char)(0x80 | (c >> 6 & 0x3F)), o[2] = (char)(0x80 | (c & 0x3F)), 3;
  o[0] = (char)(0xF0 | c >> 18), o[1] = (char)(0x80 | (c >> 12 & 0x3F)), o[2] = (char)(0x80 | (c >> 6 & 0x3F));
  o[3] = (char)(0x80 | (c & 0x3F));
  return 4;
}

size_t json_utf8_len(const unsigned char *p, size_t left)
{
  const unsigned char b = p[0];
  const size_t n = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
  if (!n || n > left)
    return 0;
  uint32_t c = n == 1 ? b : b & (0x7F >> n);
  for (size_t i = 1; i < n; i++)
  {
    if ((p[i] & 0xC0) != 0x80)
      return 0;
    c = c << 6 | (p[i] & 0x3F);
  }
  const uint32_t least[5] = {0, 0, 0x80, 0x800, 0x10000};
  return c < least[n] || (c >= 0xD800 && c <= 0xDFFF) || c > 0x10FFFF ? 0 : n;
}

/* a string's text (after its opening quote) into a new buffer: *len its bytes */
static char *string(Cur *c, size_t *len)
{
  char *o = malloc((size_t)(c->end - c->p) + 1);
  size_t n = 0;
  if (!o)
    return fail(c, "out of memory"), NULL;
  while (c->p < c->end && *c->p != '"')
  {
    const unsigned char b = (unsigned char)*c->p;
    if (b < 0x20)
      return free(o), fail(c, "a control character in a string"), NULL;
    if (b != '\\')
    {
      const size_t k = json_utf8_len((const unsigned char *)c->p, (size_t)(c->end - c->p));
      if (!k)
        return free(o), fail(c, "invalid UTF-8"), NULL;
      memcpy(o + n, c->p, k), n += k, c->p += k;
      continue;
    }
    if (++c->p >= c->end)
      break;
    const char e = *c->p++;
    const char *simple = strchr("\"\\/bfnrt", e), *to = "\"\\/\b\f\n\r\t";
    if (simple && e)
    {
      o[n++] = to[simple - "\"\\/bfnrt"];
      continue;
    }
    int u = e == 'u' && c->end - c->p >= 4 ? hex4(c->p) : -1;
    if (u < 0)
      return free(o), fail(c, "an invalid escape"), NULL;
    c->p += 4;
    uint32_t cp = (uint32_t)u;
    if (cp >= 0xD800 && cp <= 0xDBFF) /* a high surrogate: its low one next, or a lone surrogate */
    {
      const int lo = c->end - c->p >= 6 && c->p[0] == '\\' && c->p[1] == 'u' ? hex4(c->p + 2) : -1;
      if (lo < 0xDC00 || lo > 0xDFFF)
        return free(o), fail(c, "a lone surrogate"), NULL;
      cp = 0x10000 + ((cp - 0xD800) << 10) + (uint32_t)(lo - 0xDC00);
      c->p += 6;
    }
    else if (cp >= 0xDC00 && cp <= 0xDFFF)
      return free(o), fail(c, "a lone surrogate"), NULL;
    n += put_utf8(o + n, cp);
  }
  if (c->p >= c->end)
    return free(o), fail(c, "an unterminated string"), NULL;
  c->p++;
  o[n] = 0;
  *len = n;
  return o;
}

static Json *number(Cur *c)
{
  const char *s = c->p;
  bool flt = false;
  if (c->p < c->end && *c->p == '-')
    c->p++;
  if (c->p < c->end && *c->p == '0')
    c->p++;
  else if (c->p < c->end && *c->p >= '1' && *c->p <= '9')
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
      c->p++;
  else
    return fail(c, "a malformed number");
  if (c->p < c->end && *c->p == '.')
  {
    const char *d = ++c->p;
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
      c->p++;
    if (c->p == d)
      return fail(c, "a malformed number");
    flt = true;
  }
  if (c->p < c->end && (*c->p == 'e' || *c->p == 'E'))
  {
    c->p++;
    if (c->p < c->end && (*c->p == '+' || *c->p == '-'))
      c->p++;
    const char *d = c->p;
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
      c->p++;
    if (c->p == d)
      return fail(c, "a malformed number");
    flt = true;
  }
  const size_t n = (size_t)(c->p - s);
  Json *j = node(flt ? JSON_FLOAT : JSON_INT);
  char *text = j ? malloc(n + 1) : NULL;
  if (!text)
    return free(j), fail(c, "out of memory");
  memcpy(text, s, n), text[n] = 0;
  if (flt)
  {
    j->f = strtod(text, NULL), free(text);
    if (isinf(j->f)) /* Python's inf, written back as Infinity: JSON no reader takes (json.h) */
      return free(j), fail(c, "a number past a double's range");
  }
  else if (strcmp(text, "-0") == 0) /* Python's int: no negative zero */
    j->s = (free(text), strdup("0")), j->len = 1;
  else
    j->s = text, j->len = n;
  if (!flt && !j->s)
    return free(j), fail(c, "out of memory");
  return j;
}

static Json *value(Cur *c);

static bool grow(Json *j, size_t cap)
{
  Json **it = realloc(j->item, cap * sizeof *it);
  if (it)
    j->item = it;
  if (j->type == JSON_OBJECT && it)
  {
    char **k = realloc(j->key, cap * sizeof *k);
    size_t *kl = k ? realloc(j->klen, cap * sizeof *kl) : NULL;
    if (k)
      j->key = k;
    if (kl)
      j->klen = kl;
    return k && kl;
  }
  return it != NULL;
}

static Json *container(Cur *c, JsonType t)
{
  const char close = t == JSON_ARRAY ? ']' : '}';
  Json *j = node(t);
  size_t cap = 0;
  if (!j)
    return fail(c, "out of memory");
  ws(c);
  if (c->p < c->end && *c->p == close)
    return c->p++, j;
  for (;;)
  {
    char *key = NULL;
    size_t klen = 0;
    if (t == JSON_OBJECT)
    {
      ws(c);
      if (c->p >= c->end || *c->p != '"' || (c->p++, !(key = string(c, &klen))))
        return json_free(j), fail(c, "an object key not a string");
      ws(c);
      if (!take(c, ":"))
        return free(key), json_free(j), fail(c, "no ':' after a key");
    }
    Json *v = value(c);
    if (!v)
      return free(key), json_free(j), NULL;
    size_t at = j->n; /* a duplicate key: its value in the first's place (Python's dict) */
    for (size_t i = 0; key && i < j->n; i++)
      if (j->klen[i] == klen && memcmp(j->key[i], key, klen) == 0)
        at = i;
    if (at < j->n)
      json_free(j->item[at]), j->item[at] = v, free(key);
    else
    {
      if (j->n == cap && !grow(j, cap = cap ? 2 * cap : 4))
        return free(key), json_free(v), json_free(j), fail(c, "out of memory");
      j->item[j->n] = v;
      if (key)
        j->key[j->n] = key, j->klen[j->n] = klen;
      j->n++;
    }
    ws(c);
    if (take(c, ","))
      continue;
    if (c->p < c->end && *c->p == close)
      return c->p++, j;
    return json_free(j), fail(c, t == JSON_ARRAY ? "no ',' or ']'" : "no ',' or '}'");
  }
}

static Json *value(Cur *c)
{
  ws(c);
  if (c->p >= c->end)
    return fail(c, "no value");
  if (++c->depth > JSON_MAX_DEPTH)
    return fail(c, "nested too deep");
  Json *j;
  const char b = *c->p;
  if (b == '{' || b == '[')
    c->p++, j = container(c, b == '{' ? JSON_OBJECT : JSON_ARRAY);
  else if (b == '"')
  {
    c->p++;
    size_t n;
    char *s = string(c, &n);
    j = s ? node(JSON_STRING) : NULL;
    if (j)
      j->s = s, j->len = n;
    else if (s)
      free(s), fail(c, "out of memory");
  }
  else if (take(c, "true"))
    j = node(JSON_TRUE);
  else if (take(c, "false"))
    j = node(JSON_FALSE);
  else if (take(c, "null"))
    j = node(JSON_NULL);
  else
    j = number(c);
  c->depth--;
  return j;
}

Json *json_parse(const char *text, size_t len, char err[JSON_ERR])
{
  err[0] = 0;
  Cur c = {text, text, text + len, err, 0};
  Json *j = text ? value(&c) : NULL;
  if (!j)
    return fail(&c, "no value");
  ws(&c);
  if (c.p != c.end)
    return json_free(j), fail(&c, "data after the value");
  return j;
}

void json_free(Json *j)
{
  if (!j)
    return;
  for (size_t i = 0; i < j->n; i++)
  {
    json_free(j->item[i]);
    if (j->key)
      free(j->key[i]);
  }
  free(j->item), free(j->key), free(j->klen), free(j->s);
  free(j);
}

const Json *json_get(const Json *obj, const char *key)
{
  if (!obj || obj->type != JSON_OBJECT)
    return NULL;
  const size_t n = strlen(key);
  for (size_t i = 0; i < obj->n; i++)
    if (obj->klen[i] == n && memcmp(obj->key[i], key, n) == 0)
      return obj->item[i];
  return NULL;
}

const char *json_str(const Json *obj, const char *key)
{
  const Json *v = json_get(obj, key);
  return v && v->type == JSON_STRING ? v->s : NULL;
}
