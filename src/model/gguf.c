#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"
#include "gguf_read.h"

#define GGUF_MAGIC 0x46554747u
#define GGUF_VERSION 3
#define GGUF_ALIGN 32

static bool gguf_signed(GgufValueType t)
{
  return t == GGUF_INT8 || t == GGUF_INT16 || t == GGUF_INT32 || t == GGUF_INT64;
}

bool gguf_int(const GgufKv *kv, size_t i, int64_t *out)
{
  if (kv->type != GGUF_ARRAY)
  {
    if (!gguf_signed(kv->type) && kv->v.u > INT64_MAX)
      return false;
    *out = gguf_signed(kv->type) ? kv->v.i : (int64_t)kv->v.u;
    return true;
  }
  const void *p = kv->v.items;
  switch (kv->item)
  {
  case GGUF_UINT8: *out = ((const uint8_t *)p)[i]; return true;
  case GGUF_INT8: *out = ((const int8_t *)p)[i]; return true;
  case GGUF_UINT16: *out = ((const uint16_t *)p)[i]; return true;
  case GGUF_INT16: *out = ((const int16_t *)p)[i]; return true;
  case GGUF_UINT32: *out = ((const uint32_t *)p)[i]; return true;
  case GGUF_INT32: *out = ((const int32_t *)p)[i]; return true;
  case GGUF_INT64: *out = ((const int64_t *)p)[i]; return true;
  case GGUF_UINT64:
    if (((const uint64_t *)p)[i] > INT64_MAX)
      return false;
    *out = (int64_t)((const uint64_t *)p)[i];
    return true;
  default: return false;
  }
}

uint64_t gguf_bytes(uint32_t type, uint64_t values)
{
  const QTypeInfo *t = qtype_info(type); /* the types a file may hold (qtype.h); anything else is refused */
  if (!t || !(t->paths & QTYPE_FILE) || values % t->values)
    return 0;
  const uint64_t blocks = values / t->values;
  if (blocks > UINT64_MAX / t->bytes)
    return 0;
  return blocks * t->bytes;
}

/* one file's header into g: its kv (kept for the first file only) and its tensors (appended) */
static bool read_file(Gguf *g, int file, Reader *r, uint64_t *split_count, uint64_t *split_no)
{
  uint32_t magic = 0, version = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  if (!gguf_u32(r, &magic) || !gguf_u32(r, &version) || !gguf_u64(r, &n_tensors) || !gguf_u64(r, &n_kv))
    return false;
  if (magic != GGUF_MAGIC)
    return gguf_fail(r, "not a GGUF file");
  if (version != GGUF_VERSION)
    return gguf_fail(r, "not GGUF version 3");
  if (n_kv > (r->size - r->pos) / 12 || n_tensors > (r->size - r->pos) / 24)
    return gguf_fail(r, "more entries than the file holds");
  uint64_t align = GGUF_ALIGN;
  GgufKv *kv = calloc(n_kv ? (size_t)n_kv : 1, sizeof *kv);
  if (!kv)
    return gguf_fail(r, "out of memory");
  size_t got = 0;
  bool ok = true;
  while (got < n_kv)
  {
    GgufKv *k = &kv[got++];
    if (!(k->key = gguf_string(r)) || !gguf_value(r, k))
    {
      ok = false;
      break;
    }
    if (strcmp(k->key, "general.alignment") == 0)
      align = k->type == GGUF_UINT32 ? k->v.u : 0;
    else if (strcmp(k->key, "split.count") == 0 || strcmp(k->key, "split.no") == 0)
    { /* an unsigned integer (gguf-split's u16): another type refused by name (0.7.1: a string's pointer bits read) */
      if (k->type != GGUF_UINT16 && k->type != GGUF_UINT32 && k->type != GGUF_UINT64)
      {
        ok = gguf_fail(r, "split.count or split.no not an unsigned integer");
        break;
      }
      *(strcmp(k->key, "split.count") == 0 ? split_count : split_no) = k->v.u;
    }
  }
  if (ok && (align == 0 || (align & (align - 1))))
    ok = gguf_fail(r, "an alignment that is not a power of two");
  if (ok && file == 0)
  {
    g->kv = kv;
    g->n_kv = got;
    kv = NULL;
  }
  for (size_t i = 0; kv && i < got; i++)
    gguf_kv_free(&kv[i]);
  free(kv);
  if (!ok)
    return false;
  GgufTensor *t = realloc(g->tensors, (g->n_tensors + (size_t)n_tensors + 1) * sizeof *t);
  if (!t)
    return gguf_fail(r, "out of memory");
  g->tensors = t;
  const size_t first = g->n_tensors;
  for (uint64_t i = 0; i < n_tensors; i++)
  {
    GgufTensor *x = &g->tensors[g->n_tensors];
    memset(x, 0, sizeof *x);
    if (!(x->name = gguf_string(r)))
      return false;
    g->n_tensors++;
    if (!gguf_u32(r, &x->n_dims))
      return false;
    if (x->n_dims < 1 || x->n_dims > GGUF_MAX_DIMS)
      return gguf_fail(r, "a tensor of more than 4 dims or none");
    uint64_t values = 1;
    for (uint32_t d = 0; d < x->n_dims; d++)
    {
      if (!gguf_u64(r, &x->dims[d]))
        return false;
      if (x->dims[d] == 0 || values > UINT64_MAX / x->dims[d])
        return gguf_fail(r, "a tensor's dims are 0 or overflow");
      values *= x->dims[d];
    }
    if (!gguf_u32(r, &x->type) || !gguf_u64(r, &x->offset))
      return false;
    const QTypeInfo *ts = qtype_info(x->type);
    if (!ts || !(ts->paths & QTYPE_FILE))
      return gguf_fail(r, "a tensor type this build does not know");
    if (x->dims[0] % ts->values || !(x->bytes = gguf_bytes(x->type, values)))
      return gguf_fail(r, "a tensor's rows are not whole blocks");
    if (x->offset % align)
      return gguf_fail(r, "a tensor's offset is not aligned");
    x->file = file;
  }
  const uint64_t data = (r->pos + align - 1) / align * align;
  for (size_t i = first; i < g->n_tensors; i++)
  {
    GgufTensor *x = &g->tensors[i];
    if (x->offset > r->size || data > r->size - x->offset || x->bytes > r->size - data - x->offset)
    {
      r->pos = data;
      return gguf_fail(r, "a tensor's bytes leave the file");
    }
    x->offset += data;
  }
  /* no two tensors' bytes overlap (a type read as more bytes than the file gives it runs into the next: upstream's
     43, F8_E4M3, taken for unsloth's MXFP8) */
  for (size_t i = first; i < g->n_tensors; i++)
    for (size_t j = i + 1; j < g->n_tensors; j++)
    {
      const GgufTensor *a = &g->tensors[i], *b = &g->tensors[j];
      if (a->offset < b->offset + b->bytes && b->offset < a->offset + a->bytes)
        return gguf_fail(r, "two tensors' bytes overlap");
    }
  return true;
}

/* a split model's shard `no` (1-based) of `count`: the first shard's path with its -00001-of-NNNNN replaced (NNNNN the
   header's count, as llama.cpp's gguf-split writes it); NULL for a first shard named otherwise (0.7.1: its name's tail
   was written past when shorter, and a count spelt otherwise opened siblings of another name) */
static char *shard_path(const char *first, int no, int count)
{
  char want[32], num[32];
  snprintf(want, sizeof want, "-00001-of-%05d", count);
  snprintf(num, sizeof num, "-%05d-of-%05d", no, count);
  const char *at = strstr(first, want);
  if (!at || strlen(num) != strlen(want))
    return NULL;
  char *p = strdup(first);
  if (p)
    memcpy(p + (at - first), num, strlen(num));
  return p;
}

static bool open_file(Gguf *g, int file, const char *path, char *err, uint64_t *split_count, uint64_t *split_no)
{
  Reader r = {0};
  r.path = path;
  r.err = err;
  struct stat st;
  if (stat(path, &st) != 0 || !(r.f = fopen(path, "rb")))
    return gguf_fail(&r, "cannot open");
  r.size = (uint64_t)st.st_size;
  g->sizes[file] = r.size;
  const bool ok = read_file(g, file, &r, split_count, split_no);
  fclose(r.f);
  return ok;
}

static Gguf *create(const char *path, bool siblings, char err[GGUF_ERR])
{
  if (err)
    err[0] = 0;
  Gguf *g = calloc(1, sizeof *g);
  if (!g)
    return NULL;
  uint64_t count = 0, no = 0;
  g->paths = calloc(1, sizeof *g->paths);
  g->sizes = calloc(1, sizeof *g->sizes);
  if (!g->paths || !g->sizes || !(g->paths[0] = strdup(path)))
  {
    gguf_destroy(g);
    return NULL;
  }
  g->n_files = 1;
  if (!open_file(g, 0, path, err, &count, &no))
  {
    gguf_destroy(g);
    return NULL;
  }
  if (count > 1 && siblings)
  {
    Reader r = {NULL, path, 0, 0, err};
    if (no != 0 || count > 999 || !strstr(path, "-00001-of-"))
    {
      gguf_fail(&r, "a split model is opened by its first shard (-00001-of-)");
      gguf_destroy(g);
      return NULL;
    }
    char **paths = realloc(g->paths, (size_t)count * sizeof *paths);
    uint64_t *sizes = paths ? realloc(g->sizes, (size_t)count * sizeof *sizes) : NULL;
    if (paths)
      g->paths = paths;
    if (sizes)
      g->sizes = sizes;
    if (!paths || !sizes)
    {
      gguf_destroy(g);
      return NULL;
    }
    for (int i = 1; i < (int)count; i++)
    {
      uint64_t c = 0, n = 0;
      g->paths[i] = shard_path(path, i + 1, (int)count);
      if (!g->paths[i])
      {
        if (err)
          snprintf(err, GGUF_ERR, "%s: a split model's first shard is named -00001-of-%05d", path, (int)count);
        gguf_destroy(g);
        return NULL;
      }
      g->n_files = i + 1;
      bool ok = open_file(g, i, g->paths[i], err, &c, &n);
      if (ok && (c != count || n != (uint64_t)i))
      {
        Reader rr = {NULL, g->paths[i], 0, 0, err};
        ok = gguf_fail(&rr, "a shard of another split");
      }
      if (!ok)
      {
        gguf_destroy(g);
        return NULL;
      }
    }
  }
  for (size_t i = 0; i < g->n_tensors; i++)
    for (size_t j = i + 1; j < g->n_tensors; j++)
      if (strcmp(g->tensors[i].name, g->tensors[j].name) == 0)
      {
        Reader r = {NULL, g->paths[g->tensors[j].file], 0, 0, err};
        gguf_fail(&r, "a tensor name twice");
        gguf_destroy(g);
        return NULL;
      }
  return g;
}

Gguf *gguf_create(const char *path, char err[GGUF_ERR])
{
  return create(path, true, err);
}

Gguf *gguf_header(const char *path, char err[GGUF_ERR])
{
  return create(path, false, err);
}

void gguf_destroy(Gguf *g)
{
  if (!g)
    return;
  for (size_t i = 0; i < g->n_kv; i++)
    gguf_kv_free(&g->kv[i]);
  free(g->kv);
  for (size_t i = 0; i < g->n_tensors; i++)
    free(g->tensors[i].name);
  free(g->tensors);
  for (int i = 0; g->paths && i < g->n_files; i++)
    free(g->paths[i]);
  free(g->paths);
  free(g->sizes);
  free(g);
}

const GgufTensor *gguf_tensor(const Gguf *g, const char *name)
{
  for (size_t i = 0; g && name && i < g->n_tensors; i++)
    if (strcmp(g->tensors[i].name, name) == 0)
      return &g->tensors[i];
  return NULL;
}

const GgufKv *gguf_kv(const Gguf *g, const char *key)
{
  for (size_t i = 0; g && key && i < g->n_kv; i++)
    if (strcmp(g->kv[i].key, key) == 0)
      return &g->kv[i];
  return NULL;
}
