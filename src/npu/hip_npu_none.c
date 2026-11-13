/* the NPU's calls without it (a build without NPU=1): each fails, npu_built false */
#include <stdio.h>

#include "osmos/npu/npu.h"

bool npu_built(void) { return false; }

Npu *npu_open(const char *dir, uint32_t D, uint32_t FF, uint32_t max_parts, char err[NPU_ERR])
{
  (void)dir, (void)D, (void)FF, (void)max_parts;
  snprintf(err, NPU_ERR, "npu: osmos was built without it (make NPU=1)");
  return NULL;
}

void npu_close(Npu *n) { (void)n; }

bool npu_region(Npu *n, void *base, uint64_t bytes, char err[NPU_ERR])
{
  (void)n, (void)base, (void)bytes;
  snprintf(err, NPU_ERR, "npu: not built");
  return false;
}

void *npu_a(const Npu *n) { return (void)n, NULL; }
bool npu_output(Npu *n, void *dev, uint64_t bytes, char err[NPU_ERR])
{
  (void)n, (void)dev, (void)bytes;
  snprintf(err, NPU_ERR, "npu: not built");
  return false;
}

bool npu_has(const Npu *n, NpuKind kind, uint32_t qtype) { return (void)n, (void)kind, (void)qtype, false; }

bool npu_run(Npu *n, NpuKind kind, uint32_t qtype, uint32_t parts, const void *const *w0, const void *const *w1,
             uint64_t w_bytes, void *const *c0, void *const *c1, char err[NPU_ERR])
{
  (void)n, (void)kind, (void)qtype, (void)parts, (void)w0, (void)w1, (void)w_bytes, (void)c0, (void)c1;
  snprintf(err, NPU_ERR, "npu: not built");
  return false;
}

NpuStats npu_stats(const Npu *n) { return (void)n, (NpuStats){0}; }
