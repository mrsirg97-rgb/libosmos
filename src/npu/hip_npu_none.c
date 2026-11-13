/* the NPU's calls without it (a build without NPU=1): each fails, npu_built false */
#include <stdio.h>

#include "osmos/npu/npu.h"

bool npu_built(void) { return false; }

Npu *npu_open(const char *dir, uint32_t K, uint32_t N, uint32_t max_parts, char err[NPU_ERR])
{
  (void)dir, (void)K, (void)N, (void)max_parts;
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
void *npu_c(const Npu *n) { return (void)n, NULL; }

bool npu_gate_up(Npu *n, uint32_t parts, const void *const *wg, const void *const *wu, uint64_t w_bytes,
                 char err[NPU_ERR])
{
  (void)n, (void)parts, (void)wg, (void)wu, (void)w_bytes;
  snprintf(err, NPU_ERR, "npu: not built");
  return false;
}

NpuStats npu_stats(const Npu *n) { return (void)n, (NpuStats){0}; }
