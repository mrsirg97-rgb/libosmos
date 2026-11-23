/* osmos/gpu/upload.h: a tensor read into GPU memory through a pinned piece */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include "osmos/err.h"
#include "osmos/gpu/upload.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/quant.h"

const char *gpu_read_tensor(OsmosLedger *l, const Gguf *g, const GgufTensor *t, void *pin, uint64_t piece, void **dev)
{
  const int fd = open(g->paths[t->file], O_RDONLY);
  if (fd < 0)
    return "its file not opened";
  const hipError_t r = ledger_malloc(l, dev, t->bytes ? t->bytes : 1);
  const char *why = r == hipSuccess ? NULL : r == hipErrorOutOfMemory && ledger_why(l)[0] ? ledger_why(l) : "GPU memory refused";
  for (uint64_t at = 0; !why && at < t->bytes; at += piece)
  {
    const size_t n = t->bytes - at < piece ? (size_t)(t->bytes - at) : (size_t)piece;
    if (pread(fd, pin, n, (off_t)(t->offset + at)) != (ssize_t)n)
      why = "its file's read failed";
    else if (hipMemcpy((uint8_t *)*dev + at, pin, n, hipMemcpyHostToDevice) != hipSuccess)
      why = "the copy to GPU memory failed";
  }
  close(fd);
  return why;
}

bool gpu_upload_table(OsmosLedger *l, const Gguf *g, uint64_t piece, gpu_upload_skip skip, void *arg, void ***dev,
                      char *err)
{
  void *pin = NULL;
  if (!(*dev = calloc(g->n_tensors ? g->n_tensors : 1, sizeof **dev)) || ledger_host_malloc(l, &pin, piece, 0) != hipSuccess)
  {
    if (err)
      osmos_fail(err, "out of memory (the weights' table, or a pinned piece)");
    return false;
  }
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    if (skip && skip(t, arg))
      continue;
    const char *why = gpu_read_tensor(l, g, t, pin, piece, &(*dev)[i]);
    if ((ok = !why) == false && err)
      osmos_fail(err, "%s: not read into GPU memory (%s)", t->name, why);
  }
  ledger_host_free(l, pin);
  return ok;
}

void gpu_free_table(OsmosLedger *l, const Gguf *g, void **dev, const bool *shared)
{
  for (size_t i = 0; dev && i < g->n_tensors; i++)
    if ((!shared || !shared[i]) && dev[i])
      ledger_free(l, dev[i]);
  free(dev);
}

bool gpu_q8_rows(OsmosLedger *l, uint32_t type, const void *src, uint64_t row_bytes, uint32_t V, uint32_t D, void **out)
{
  const uint32_t chunk = 4096;
  const uint64_t rb = q4x_kv_row_bytes(D);
  void *q = NULL, *spare = NULL;
  float *tmp = NULL;
  bool ok = ledger_malloc(l, &q, (uint64_t)V * rb) == hipSuccess &&
            ledger_malloc(l, &spare, (uint64_t)chunk * rb) == hipSuccess &&
            ledger_malloc(l, (void **)&tmp, (uint64_t)chunk * D * sizeof(float)) == hipSuccess;
  for (uint32_t r = 0; ok && r < V; r += chunk)
  {
    const uint32_t m = V - r < chunk ? V - r : chunk;
    ok = quant_dequant(type, (const uint8_t *)src + (uint64_t)r * row_bytes, (uint64_t)m * D, tmp, NULL) &&
         q4x_kv_put(tmp, tmp, m, D, (uint8_t *)q + (uint64_t)r * rb, spare, NULL);
  }
  ok = ok && hipDeviceSynchronize() == hipSuccess;
  if (spare)
    ledger_free(l, spare);
  if (tmp)
    ledger_free(l, tmp);
  if (!ok && q)
    ledger_free(l, q), q = NULL;
  *out = q;
  return ok;
}
