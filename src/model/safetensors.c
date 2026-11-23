/* a safetensors file as a Gguf (osmos/model/safetensors.h): its JSON header's tensors, their bytes where the header
   puts them - read the way a GGUF's are (gguf_bytes, gpu_read_tensor), no metadata */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "osmos/model/qtype.h"
#include "osmos/model/safetensors.h"
#include "osmos/text/json.h"
#include "osmos/err.h"

#define ST_MAX_HEADER (100u << 20) /* the header's bytes at most (the format's own limit: 100 MB) */

static Gguf *fail(Gguf *g, char *err, const char *path, const char *why)
{
  osmos_fail(err, "%s: %s", path, why);
  gguf_destroy(g);
  return NULL;
}

/* a dtype's qtype id: the ones osmos reads (BF16, F16, F32), the rest refused */
static int dtype_of(const char *s)
{
  if (!s)
    return -1;
  if (!strcmp(s, "BF16"))
    return QTYPE_BF16;
  if (!strcmp(s, "F32"))
    return QTYPE_F32;
  return -1;
}

/* a JSON integer's value: digits alone, under 2^63 */
static bool uint_of(const Json *j, uint64_t *v)
{
  if (!j || j->type != JSON_INT || !j->s || j->s[0] == '-' || j->len == 0 || j->len > 18)
    return false;
  char *end = NULL;
  errno = 0;
  *v = strtoull(j->s, &end, 10);
  return errno == 0 && end == j->s + j->len;
}

Gguf *st_create(const char *path, char err[OSMOS_ERR])
{
  err[0] = 0;
  Gguf *g = calloc(1, sizeof *g);
  if (!g || !(g->paths = calloc(1, sizeof *g->paths)) || !(g->sizes = calloc(1, sizeof *g->sizes)) ||
      !(g->paths[0] = strdup(path)))
    return fail(g, err, path, "out of memory");
  g->n_files = 1;
  struct stat st;
  FILE *f = stat(path, &st) == 0 ? fopen(path, "rb") : NULL;
  if (!f)
    return fail(g, err, path, "cannot open");
  g->sizes[0] = (uint64_t)st.st_size;
  uint8_t len8[8];
  uint64_t hn = 0;
  if (fread(len8, 1, 8, f) != 8)
  {
    fclose(f);
    return fail(g, err, path, "shorter than a safetensors header");
  }
  for (int i = 7; i >= 0; i--)
    hn = hn << 8 | len8[i];
  if (hn == 0 || hn > ST_MAX_HEADER || hn > g->sizes[0] - 8)
  {
    fclose(f);
    return fail(g, err, path, "a header length out of range");
  }
  char *head = malloc(hn);
  const bool got = head && fread(head, 1, hn, f) == hn;
  fclose(f);
  char jerr[JSON_ERR] = "";
  Json *j = got ? json_parse(head, hn, jerr) : NULL;
  free(head);
  if (!j || j->type != JSON_OBJECT)
  {
    json_free(j);
    return fail(g, err, path, "its header not a JSON object");
  }
  const uint64_t data = 8 + hn, room = g->sizes[0] - data;
  if (!(g->tensors = calloc(j->n ? j->n : 1, sizeof *g->tensors)))
  {
    json_free(j);
    return fail(g, err, path, "out of memory");
  }
  const char *why = NULL;
  for (size_t i = 0; !why && i < j->n; i++)
  {
    if (!strcmp(j->key[i], "__metadata__"))
      continue;
    const Json *t = j->item[i], *shape = json_get(t, "shape"), *offs = json_get(t, "data_offsets");
    const int type = dtype_of(json_str(t, "dtype"));
    GgufTensor *x = &g->tensors[g->n_tensors];
    uint64_t b0 = 0, b1 = 0, values = 1;
    if (t->type != JSON_OBJECT || !shape || shape->type != JSON_ARRAY || !offs || offs->type != JSON_ARRAY ||
        offs->n != 2 || !uint_of(offs->item[0], &b0) || !uint_of(offs->item[1], &b1))
    {
      why = "a tensor without its dtype, shape or data_offsets";
      break;
    }
    if (type < 0)
    {
      why = "a dtype this build does not read (BF16, F32)";
      break;
    }
    if (shape->n < 1 || shape->n > GGUF_MAX_DIMS)
    {
      why = "a tensor of more than 4 dims or none";
      break;
    }
    x->n_dims = (uint32_t)shape->n;
    for (size_t d = 0; d < shape->n; d++) /* ggml's order: the row's values first */
    {
      uint64_t v = 0;
      if (!uint_of(shape->item[d], &v) || v == 0 || values > UINT64_MAX / v)
      {
        why = "a tensor's dims are 0 or overflow";
        break;
      }
      x->dims[shape->n - 1 - d] = v, values *= v;
    }
    if (why)
      break;
    x->type = (uint32_t)type, x->bytes = gguf_bytes(x->type, values);
    if (!x->bytes || b1 < b0 || b1 - b0 != x->bytes || b1 > room)
    {
      why = "a tensor's bytes not its dtype's for its shape, or past the file";
      break;
    }
    if (!(x->name = strdup(j->key[i])))
    {
      why = "out of memory";
      break;
    }
    x->offset = data + b0, x->file = 0;
    g->n_tensors++;
  }
  json_free(j);
  for (size_t a = 0; !why && a < g->n_tensors; a++)
    for (size_t b = a + 1; !why && b < g->n_tensors; b++)
    {
      const GgufTensor *p = &g->tensors[a], *q = &g->tensors[b];
      if (p->offset < q->offset + q->bytes && q->offset < p->offset + p->bytes)
        why = "two tensors' bytes overlap";
    }
  return why ? fail(g, err, path, why) : g;
}
