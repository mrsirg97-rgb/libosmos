/* osmos/gpu/upload.h: a tensor read into GPU memory through a pinned piece */
#include <fcntl.h>
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
