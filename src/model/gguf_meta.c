#include <stdio.h>
#include <string.h>

#include "gguf_meta.h"

bool meta_fail(GgufMeta *m, const char *key, const char *why)
{
  snprintf(m->err, GGUF_META_ERR, "%s: %s", key, why);
  return false;
}

/* the key's other spelling (the family's aliases), or NULL */
static const char *alias(const GgufMeta *m, const char *key)
{
  for (size_t i = 0; i < m->n_aliases; i++)
    if (strcmp(key, m->aliases[i][0]) == 0 || strcmp(key, m->aliases[i][1]) == 0)
      return m->aliases[i][strcmp(key, m->aliases[i][0]) == 0];
  return NULL;
}

static bool is_int(GgufValueType t)
{
  return t == GGUF_UINT8 || t == GGUF_INT8 || t == GGUF_UINT16 || t == GGUF_INT16 || t == GGUF_UINT32 ||
         t == GGUF_INT32 || t == GGUF_UINT64 || t == GGUF_INT64;
}

/* the same value in both spellings: integers (or lists of them) of any width equal as numbers; anything else the
   same type and bytes */
static bool same_value(const GgufKv *a, const GgufKv *b)
{
  const bool arrays = a->type == GGUF_ARRAY && b->type == GGUF_ARRAY;
  if ((is_int(a->type) && is_int(b->type)) || (arrays && is_int(a->item) && is_int(b->item)))
  {
    const size_t n = arrays ? a->n : 1;
    if (arrays && a->n != b->n)
      return false;
    for (size_t i = 0; i < n; i++)
    {
      int64_t x, y;
      if (!gguf_int(a, i, &x) || !gguf_int(b, i, &y) || x != y)
        return false;
    }
    return true;
  }
  if (a->type != b->type)
    return false;
  switch (a->type)
  {
  case GGUF_STRING: return strcmp(a->v.s, b->v.s) == 0;
  case GGUF_FLOAT32:
  case GGUF_FLOAT64: return a->v.f == b->v.f;
  case GGUF_BOOL: return a->v.u == b->v.u;
  default: return false; /* no other pair is spelled two ways */
  }
}

/* the key, or its other spelling; both: they must agree */
static const GgufKv *lookup(GgufMeta *m, const char *key)
{
  const char *other = alias(m, key);
  const GgufKv *a = gguf_kv(m->g, key), *b = other ? gguf_kv(m->g, other) : NULL;
  char both[2 * 96];
  snprintf(both, sizeof both, other ? "%s (or %s)" : "%s", key, other ? other : "");
  if (a && b && !same_value(a, b))
    return meta_fail(m, both, "in both spellings, unequal"), NULL;
  if (!a && !b)
    return meta_fail(m, both, "missing"), NULL;
  return a ? a : b;
}

/* a key in either spelling (an optional one's: absent is not a failure) */
bool meta_present(GgufMeta *m, const char *key)
{
  const char *other = alias(m, key);
  return gguf_kv(m->g, key) || (other && gguf_kv(m->g, other));
}

/* GGUF_UINT32 asks for an integer scalar of any width; another type, that type */
const GgufKv *meta_find(GgufMeta *m, const char *key, GgufValueType type)
{
  const GgufKv *kv = lookup(m, key);
  if (!kv)
    return NULL;
  if (type == GGUF_UINT32 ? !is_int(kv->type) : kv->type != type)
    return meta_fail(m, kv->key, "of another value type"), NULL;
  return kv;
}

bool meta_uint_in(GgufMeta *m, const char *key, int lo, int hi, int *out)
{
  const GgufKv *kv = meta_find(m, key, GGUF_UINT32);
  int64_t v;
  if (!kv)
    return false;
  if (!gguf_int(kv, 0, &v) || v < lo || v > hi)
    return meta_fail(m, kv->key, "out of range");
  *out = (int)v;
  return true;
}

bool meta_uint_is(GgufMeta *m, const char *key, uint64_t want)
{
  const GgufKv *kv = meta_find(m, key, GGUF_UINT32);
  int64_t v;
  return kv && ((gguf_int(kv, 0, &v) && v >= 0 && (uint64_t)v == want) ||
                meta_fail(m, kv->key, "a value this build does not run"));
}

bool meta_flag_set(GgufMeta *m, const char *key)
{
  const GgufKv *kv = meta_find(m, key, GGUF_BOOL);
  return kv && (kv->v.u || meta_fail(m, key, "false; this build runs it true"));
}

bool meta_real(GgufMeta *m, const char *key, float *out)
{
  const GgufKv *kv = meta_find(m, key, GGUF_FLOAT32);
  if (!kv)
    return false;
  if (!(kv->v.f > 0))
    return meta_fail(m, key, "not positive");
  *out = (float)kv->v.f;
  return true;
}

bool meta_text_is(GgufMeta *m, const char *key, const char *want)
{
  const GgufKv *kv = meta_find(m, key, GGUF_STRING);
  return kv && (strcmp(kv->v.s, want) == 0 || meta_fail(m, key, "a value this build does not run"));
}

/* an array of integers: n (at least min, at most cap) into out, each read at its own width */
bool meta_ints(GgufMeta *m, const char *key, size_t min, size_t cap, int64_t *out, size_t *n)
{
  const GgufKv *kv = meta_find(m, key, GGUF_ARRAY);
  if (!kv)
    return false;
  if (kv->n < min || kv->n > cap)
    return meta_fail(m, kv->key, "of a length this model does not have");
  if (!is_int(kv->item))
    return meta_fail(m, kv->key, "not of integers");
  for (size_t i = 0; i < kv->n; i++)
    if (!gguf_int(kv, i, &out[i]))
      return meta_fail(m, kv->key, "out of range");
  *n = kv->n;
  return true;
}

/* a list of layers, each within them and once, as flags */
bool meta_layers(GgufMeta *m, const char *key, int n_layers, bool *set, int *count)
{
  int64_t v[GGUF_META_LIST_MAX];
  size_t n = 0;
  if (!meta_ints(m, key, 1, GGUF_META_LIST_MAX, v, &n))
    return false;
  for (size_t i = 0; i < n; i++)
  {
    if (v[i] < 0 || v[i] >= n_layers)
      return meta_fail(m, key, "a layer past the model's");
    if (set[v[i]])
      return meta_fail(m, key, "a layer twice");
    set[v[i]] = true;
  }
  if (count)
    *count = (int)n;
  return true;
}

