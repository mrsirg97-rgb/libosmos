/* osmos/gpu/upload.h: a tensor read into GPU memory through a pinned piece */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include "osmos/gpu/upload.h"

const char *gpu_read_tensor(const Gguf *g, const GgufTensor *t, void *pin, uint64_t piece, void **dev)
{
  const int fd = open(g->paths[t->file], O_RDONLY);
  if (fd < 0)
    return "its file not opened";
  const char *why = hipMalloc(dev, t->bytes ? t->bytes : 1) == hipSuccess ? NULL : "GPU memory refused";
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

bool gpu_upload_table(const Gguf *g, uint64_t piece, gpu_upload_skip skip, void *arg, void ***dev, char *err,
                      size_t err_n)
{
  void *pin = NULL;
  if (!(*dev = calloc(g->n_tensors ? g->n_tensors : 1, sizeof **dev)) || hipHostMalloc(&pin, piece, 0) != hipSuccess)
  {
    if (err)
      snprintf(err, err_n, "%s", "out of memory (the weights' table, or a pinned piece)");
    return false;
  }
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    if (skip && skip(t, arg))
      continue;
    const char *why = gpu_read_tensor(g, t, pin, piece, &(*dev)[i]);
    if ((ok = !why) == false && err)
      snprintf(err, err_n, "%s: not read into GPU memory (%s)", t->name, why);
  }
  hipHostFree(pin);
  return ok;
}

void gpu_free_table(const Gguf *g, void **dev, const bool *shared)
{
  for (size_t i = 0; dev && i < g->n_tensors; i++)
    if (!shared || !shared[i])
      hipFree(dev[i]);
  free(dev);
}
