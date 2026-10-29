#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "gguf_read.h"

bool gguf_fail(Reader *r, const char *what)
{
  if (r->err && !r->err[0])
    snprintf(r->err, GGUF_ERR, "%s: offset %" PRIu64 ": %s", r->path, r->pos, what);
  return false;
}

bool gguf_take(Reader *r, void *dst, uint64_t n)
{
  if (n > r->size - r->pos)
    return gguf_fail(r, "past the end of the file");
  if (n && fread(dst, 1, (size_t)n, r->f) != n)
    return gguf_fail(r, "read failed");
  r->pos += n;
  return true;
}

bool gguf_u32(Reader *r, uint32_t *v) { return gguf_take(r, v, 4); }
bool gguf_u64(Reader *r, uint64_t *v) { return gguf_take(r, v, 8); }

char *gguf_string(Reader *r)
{
  uint64_t n = 0;
  if (!gguf_u64(r, &n))
    return NULL;
  if (n > r->size - r->pos)
  {
    gguf_fail(r, "a string past the end of the file");
    return NULL;
  }
  char *s = malloc((size_t)n + 1);
  if (!s)
  {
    gguf_fail(r, "out of memory");
    return NULL;
  }
  if (!gguf_take(r, s, n))
  {
    free(s);
    return NULL;
  }
  s[n] = 0;
  return s;
}

static uint64_t scalar_width(GgufValueType t)
{
  switch (t)
  {
  case GGUF_UINT8:
  case GGUF_INT8:
  case GGUF_BOOL:
    return 1;
  case GGUF_UINT16:
  case GGUF_INT16:
    return 2;
  case GGUF_UINT32:
  case GGUF_INT32:
  case GGUF_FLOAT32:
    return 4;
  case GGUF_UINT64:
  case GGUF_INT64:
  case GGUF_FLOAT64:
    return 8;
  default:
    return 0;
  }
}

static bool scalar(Reader *r, GgufValueType t, GgufKv *kv)
{
  uint8_t raw[8] = {0};
  const uint64_t w = scalar_width(t);
  if (!w)
    return gguf_fail(r, "an unknown value type");
  if (!gguf_take(r, raw, w))
    return false;
  int64_t sv = 0;
  switch (t)
  {
  case GGUF_INT8: sv = (int8_t)raw[0]; kv->v.i = sv; break;
  case GGUF_INT16: { int16_t x; memcpy(&x, raw, 2); kv->v.i = x; break; }
  case GGUF_INT32: { int32_t x; memcpy(&x, raw, 4); kv->v.i = x; break; }
  case GGUF_INT64: memcpy(&kv->v.i, raw, 8); break;
  case GGUF_FLOAT32: { float x; memcpy(&x, raw, 4); kv->v.f = x; break; }
  case GGUF_FLOAT64: memcpy(&kv->v.f, raw, 8); break;
  default: { uint64_t x = 0; memcpy(&x, raw, w); kv->v.u = x; break; }
  }
  return true;
}

bool gguf_value(Reader *r, GgufKv *kv)
{
  uint32_t t = 0;
  if (!gguf_u32(r, &t))
    return false;
  kv->type = (GgufValueType)t;
  if (t == GGUF_STRING)
    return (kv->v.s = gguf_string(r)) != NULL;
  if (t != GGUF_ARRAY)
    return scalar(r, kv->type, kv);
  uint32_t item = 0;
  uint64_t n = 0;
  if (!gguf_u32(r, &item) || !gguf_u64(r, &n))
    return false;
  kv->item = (GgufValueType)item;
  if (item == GGUF_STRING)
  {
    if (n > (r->size - r->pos) / 8)
      return gguf_fail(r, "an array longer than the file");
    char **s = calloc(n ? (size_t)n : 1, sizeof *s);
    if (!s)
      return gguf_fail(r, "out of memory");
    kv->v.items = s;
    for (uint64_t i = 0; i < n; i++, kv->n = i)
      if (!(s[i] = gguf_string(r)))
        return false;
    kv->n = n;
    return true;
  }
  const uint64_t w = scalar_width(kv->item);
  if (!w)
    return gguf_fail(r, "an array of an unknown value type");
  if (n > (r->size - r->pos) / w)
    return gguf_fail(r, "an array longer than the file");
  if (!(kv->v.items = malloc(n ? (size_t)(n * w) : 1)))
    return gguf_fail(r, "out of memory");
  kv->n = n;
  return gguf_take(r, kv->v.items, n * w);
}

void gguf_kv_free(GgufKv *kv)
{
  free(kv->key);
  if (kv->type == GGUF_STRING)
    free(kv->v.s);
  else if (kv->type == GGUF_ARRAY)
  {
    if (kv->item == GGUF_STRING && kv->v.items)
      for (uint64_t i = 0; i < kv->n; i++)
        free(((char **)kv->v.items)[i]);
    free(kv->v.items);
  }
}

