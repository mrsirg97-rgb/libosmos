#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/json.h"
#include "osmos/err.h"

/* the reader: RFC 8259 by recursive descent, Python's json.loads' meaning for what it reads (json.h) */

typedef struct Cur
{
  const char *start, *p, *end;
  char *err;
  int depth;
  size_t budget, used; /* the tree's bytes at most (json_parse_within), and taken: every allocation counted */
} Cur;

/* the bytes an allocation takes counted against the budget: false past it */
static bool spend(Cur *c, size_t n)
{
  if (n > c->budget - c->used)
    return false;
  c->used += n;
  return true;
}

static Json *fail(Cur *c, const char *why) { return osmos_fail(c->err, "json: %s at byte %td", why, c->p - c->start), NULL; }

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

static Json *node(Cur *c, JsonType t)
{
  if (!spend(c, sizeof(Json)))
    return fail(c, "past its memory budget");
  Json *j = calloc(1, sizeof *j);
  if (!j)
    return fail(c, "out of memory");
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

/* a string's text (after its opening quote) into a new buffer of its own bytes: *len them. its span to the closing
   quote first (an escape only shrinks it) - 0.16.0: the input's whole rest each, a million short strings reserving
   terabytes of address space */
static char *string(Cur *c, size_t *len)
{
  const char *q = c->p;
  while (q < c->end && *q != '"')
    q += *q == '\\' && q + 1 < c->end ? 2 : 1;
  const size_t span = (size_t)(q - c->p);
  if (!spend(c, span + 1))
    return fail(c, "past its memory budget"), NULL;
  char *o = malloc(span + 1);
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
    if (cp >= 0xD800 && cp <= 0xDBFF) /* a high surrogate: its low one next - or a lone one, U+FFFD */
    {
      const int lo = c->end - c->p >= 6 && c->p[0] == '\\' && c->p[1] == 'u' ? hex4(c->p + 2) : -1;
      if (lo >= 0xDC00 && lo <= 0xDFFF)
        cp = 0x10000 + ((cp - 0xD800) << 10) + (uint32_t)(lo - 0xDC00), c->p += 6;
      else
        cp = 0xFFFD;
    }
    else if (cp >= 0xDC00 && cp <= 0xDFFF) /* a lone surrogate (a JS string cut between a pair's halves, as
                                              JSON.stringify writes it): U+FFFD, as lenient decoders take it - 0.11.2,
                                              refused: the conversation's every later request refused with it */
      cp = 0xFFFD;
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
  Json *j = node(c, flt ? JSON_FLOAT : JSON_INT);
  if (!j)
    return NULL; /* (its cause said) */
  if (!spend(c, n + 1))
    return free(j), fail(c, "past its memory budget");
  char *text = malloc(n + 1);
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

/* a container's slots to cap: false, its cause said (the budget, or out of memory) */
static bool grow(Cur *c, Json *j, size_t cap)
{
  const size_t was = j->n, each = sizeof *j->item + (j->type == JSON_OBJECT ? sizeof *j->key + sizeof *j->klen : 0);
  if (!spend(c, (cap - was) * each)) /* (the new slots: the old ones counted when they came) */
    return fail(c, "past its memory budget"), false;
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
    return (k && kl) || (fail(c, "out of memory"), false);
  }
  return it || (fail(c, "out of memory"), false);
}

/* an object's keys in order: bytes, then length, then where they came - each key's first place at its run's head */
static bool key_before(const Json *j, size_t a, size_t b)
{
  const size_t n = j->klen[a] < j->klen[b] ? j->klen[a] : j->klen[b];
  const int c = memcmp(j->key[a], j->key[b], n);
  return c ? c < 0 : j->klen[a] != j->klen[b] ? j->klen[a] < j->klen[b] : a < b;
}

static void merge_sort(const Json *j, size_t *x, size_t *tmp, size_t n)
{
  if (n < 2)
    return;
  const size_t h = n / 2;
  merge_sort(j, x, tmp, h), merge_sort(j, x + h, tmp, n - h);
  size_t a = 0, b = h, o = 0;
  while (a < h && b < n)
    tmp[o++] = key_before(j, x[b], x[a]) ? x[b++] : x[a++];
  while (a < h)
    tmp[o++] = x[a++];
  while (b < n)
    tmp[o++] = x[b++];
  memcpy(x, tmp, n * sizeof *x);
}

/* an object's duplicate keys made one (Python's dict: the first's place, the last's value) once it closes - by a
   merge sort of its keys, n log n whatever the keys (a key compared with every earlier one was n^2: one 1 MB body
   held a core 67 s) */
static bool dedupe(Json *j)
{
  if (j->n < 2)
    return true;
  size_t *x = malloc(2 * j->n * sizeof *x);
  if (!x)
    return false;
  for (size_t i = 0; i < j->n; i++)
    x[i] = i;
  merge_sort(j, x, x + j->n, j->n);
  bool dup = false;
  for (size_t r = 0; r < j->n;)
  {
    size_t e = r + 1;
    while (e < j->n && j->klen[x[e]] == j->klen[x[r]] && !memcmp(j->key[x[e]], j->key[x[r]], j->klen[x[r]]))
      e++;
    if (e - r > 1) /* x[r] the first place, x[e - 1] the last value */
    {
      dup = true;
      json_free(j->item[x[r]]), j->item[x[r]] = j->item[x[e - 1]];
      for (size_t k = r + 1; k < e; k++)
      {
        if (k < e - 1)
          json_free(j->item[x[k]]);
        free(j->key[x[k]]), j->item[x[k]] = NULL, j->key[x[k]] = NULL;
      }
    }
    r = e;
  }
  free(x);
  if (!dup)
    return true;
  size_t o = 0;
  for (size_t i = 0; i < j->n; i++)
    if (j->key[i])
      j->item[o] = j->item[i], j->key[o] = j->key[i], j->klen[o] = j->klen[i], o++;
  j->n = o;
  return true;
}

static Json *container(Cur *c, JsonType t)
{
  const char close = t == JSON_ARRAY ? ']' : '}';
  Json *j = node(c, t);
  size_t cap = 0;
  if (!j)
    return NULL; /* (its cause said) */
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
    if (j->n == cap && !grow(c, j, cap = cap ? 2 * cap : 4)) /* (its cause said) */
      return free(key), json_free(v), json_free(j), NULL;
    j->item[j->n] = v;
    if (key)
      j->key[j->n] = key, j->klen[j->n] = klen;
    j->n++;
    ws(c);
    if (take(c, ","))
      continue;
    if (c->p < c->end && *c->p == close)
      return c->p++, t == JSON_OBJECT && !dedupe(j) ? json_free(j), fail(c, "out of memory") : j;
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
    j = s ? node(c, JSON_STRING) : NULL;
    if (j)
      j->s = s, j->len = n;
    else if (s)
      free(s), fail(c, "out of memory");
  }
  else if (take(c, "true"))
    j = node(c, JSON_TRUE);
  else if (take(c, "false"))
    j = node(c, JSON_FALSE);
  else if (take(c, "null"))
    j = node(c, JSON_NULL);
  else
    j = number(c);
  c->depth--;
  return j;
}

Json *json_parse_within(const char *text, size_t len, size_t budget, size_t *used, char err[OSMOS_ERR])
{
  err[0] = 0;
  Cur c = {text, text, text + len, err, 0, budget, 0};
  Json *j = text ? value(&c) : NULL;
  if (used)
    *used = c.used;
  if (!j)
    return fail(&c, "no value");
  ws(&c);
  if (c.p != c.end)
    return json_free(j), fail(&c, "data after the value");
  return j;
}

Json *json_parse(const char *text, size_t len, char err[OSMOS_ERR])
{
  return json_parse_within(text, len, SIZE_MAX, NULL, err);
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
