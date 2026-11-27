/* a tensor read from its file (osmos/model/tensor.h) */
#include "osmos/model/tensor.h"

#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/model/qtype.h"
#include "osmos/cpu/par.h"

#define TENSOR_F16 1u /* ggml's F16: a file's small vectors (qtype.h has none: no kernel takes it) */

bool tensor_read_fd(int fd, const GgufTensor *t, void *out)
{
  for (uint64_t at = 0; at < t->bytes;)
  {
    const ssize_t n = pread(fd, (uint8_t *)out + at, t->bytes - at, (off_t)(t->offset + at));
    if (n <= 0)
      return false;
    at += (uint64_t)n;
  }
  return true;
}

bool tensor_read(const Gguf *g, const GgufTensor *t, void *out)
{
  const int fd = open(g->paths[t->file], O_RDONLY);
  if (fd < 0)
    return false;
  const bool ok = tensor_read_fd(fd, t, out);
  close(fd);
  return ok;
}

/* an IEEE half's value, exactly (a subnormal's, an infinity's, a NaN's) */
static float half(uint16_t h)
{
  const uint32_t s = (uint32_t)(h >> 15) << 31, e = h >> 10 & 31, m = h & 1023;
  if (!e)
  {
    const float v = ldexpf((float)m, -24);
    return s ? -v : v;
  }
  const uint32_t u = e == 31 ? s | 0x7f800000u | m << 13 : s | (e + 112) << 23 | m << 13;
  float v;
  memcpy(&v, &u, 4);
  return v;
}

typedef struct ToF32
{
  const uint8_t *b;
  float *v;
  uint32_t es, type;
} ToF32;

static void to_f32(void *ctx, uint64_t i0, uint64_t i1)
{
  const ToF32 *c = ctx;
  for (uint64_t i = i0; i < i1; i++)
    if (c->es == 4)
      memcpy(&c->v[i], c->b + 4 * i, 4);
    else
    {
      const uint16_t h = (uint16_t)(c->b[2 * i] | c->b[2 * i + 1] << 8);
      const uint32_t u = (uint32_t)h << 16;
      if (c->type == QTYPE_BF16)
        memcpy(&c->v[i], &u, 4);
      else
        c->v[i] = half(h);
    }
}

float *tensor_f32(const Gguf *g, const GgufTensor *t, uint64_t *n)
{
  const uint32_t es = t->type == QTYPE_F32 ? 4 : t->type == TENSOR_F16 || t->type == QTYPE_BF16 ? 2 : 0;
  if (!es)
    return NULL;
  *n = t->bytes / es;
  uint8_t *b = malloc(t->bytes ? t->bytes : 1);
  float *v = malloc(*n * 4 + 4);
  if (!b || !v || !tensor_read(g, t, b))
    return free(b), free(v), NULL;
  cpu_par_rows(*n, 1u << 20, to_f32, &(ToF32){b, v, es, t->type}); /* (each value alone: the serial loop's) */
  free(b);
  return v;
}
