/* osmos/gpu/upload.h: a tensor read into GPU memory through a pinned piece */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include "osmos/err.h"
#include "osmos/gpu/upload.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/quant.h"

static const char *read_tensor_fd(OsmosLedger *l, int fd, const GgufTensor *t, void *pin, uint64_t piece, void **dev)
{
  const hipError_t r = ledger_malloc(l, dev, t->bytes ? t->bytes : 1);
  const char *why = r == hipSuccess ? NULL : r == hipErrorOutOfMemory && ledger_why(l)[0] ? ledger_why(l) : "GPU memory refused";
  for (uint64_t at = 0; !why && at < t->bytes; at += piece)
  {
    const size_t n = t->bytes - at < piece ? (size_t)(t->bytes - at) : (size_t)piece;
    size_t got = 0;
    while (got < n)
    {
      const ssize_t k = pread(fd, (uint8_t *)pin + got, n - got, (off_t)(t->offset + at + got));
      if (k < 0 && errno == EINTR)
        continue;
      if (k <= 0)
      {
        why = "its file's read failed";
        break;
      }
      got += (size_t)k;
    }
    if (!why && hipMemcpy((uint8_t *)*dev + at, pin, n, hipMemcpyHostToDevice) != hipSuccess)
      why = "the copy to GPU memory failed";
  }
  return why;
}

const char *gpu_read_tensor(OsmosLedger *l, const Gguf *g, const GgufTensor *t, void *pin, uint64_t piece, void **dev)
{
  const int fd = open(g->paths[t->file], O_RDONLY);
  if (fd < 0)
    return "its file not opened";
  const char *why = read_tensor_fd(l, fd, t, pin, piece, dev);
  close(fd);
  return why;
}

bool gpu_upload_table(OsmosLedger *l, const Gguf *g, uint64_t piece, gpu_upload_skip skip, void *arg, void ***dev,
                      char *err)
{
  void *pin = NULL;
  int *fds = g->n_files > 0 ? malloc((size_t)g->n_files * sizeof *fds) : NULL;
  if (!(*dev = calloc(g->n_tensors ? g->n_tensors : 1, sizeof **dev)) || (g->n_files > 0 && !fds) ||
      ledger_host_malloc(l, &pin, piece, 0) != hipSuccess)
  {
    free(fds);
    if (err)
      osmos_fail(err, "out of memory (the weights' table, or a pinned piece)");
    return false;
  }
  for (int f = 0; f < g->n_files; f++)
    fds[f] = -1;
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    if (skip && skip(t, arg))
      continue;
    if (fds[t->file] < 0)
      fds[t->file] = open(g->paths[t->file], O_RDONLY);
    const char *why = fds[t->file] < 0 ? "its file not opened" : read_tensor_fd(l, fds[t->file], t, pin, piece, &(*dev)[i]);
    if ((ok = !why) == false && err)
      osmos_fail(err, "%s: not read into GPU memory (%s)", t->name, why);
  }
  for (int f = 0; f < g->n_files; f++)
    if (fds[f] >= 0)
    {
      (void)posix_fadvise(fds[f], 0, 0, POSIX_FADV_DONTNEED);
      close(fds[f]);
    }
  free(fds);
  ledger_host_free(l, pin);
  return ok;
}

uint64_t gpu_table_bytes(const Gguf *g, void *const *dev, const bool *shared)
{
  uint64_t n = 0;
  for (size_t i = 0; g && dev && i < g->n_tensors; i++)
    if (dev[i] && (!shared || !shared[i]))
      n += g->tensors[i].bytes ? g->tensors[i].bytes : 1; /* (gpu_read_tensor's: a 0-byte tensor one byte) */
  return n;
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
  void *q = NULL;
  float *tmp = NULL;
  bool ok = ledger_malloc(l, &q, (uint64_t)V * rb) == hipSuccess &&
            ledger_malloc(l, (void **)&tmp, (uint64_t)chunk * D * sizeof(float)) == hipSuccess;
  for (uint32_t r = 0; ok && r < V; r += chunk)
  {
    const uint32_t m = V - r < chunk ? V - r : chunk;
    ok = quant_dequant(type, (const uint8_t *)src + (uint64_t)r * row_bytes, (uint64_t)m * D, tmp, NULL) &&
         q4x_kv_put(tmp, NULL, m, D, (uint8_t *)q + (uint64_t)r * rb, NULL, NULL);
  }
  ok = ok && hipDeviceSynchronize() == hipSuccess;
  if (tmp)
    ledger_free(l, tmp);
  if (!ok && q)
    ledger_free(l, q), q = NULL;
  *out = q;
  return ok;
}
