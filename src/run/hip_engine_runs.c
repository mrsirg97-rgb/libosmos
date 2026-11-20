/* the families' shared engine glue (engine_impl.h): a batch's runs, the GPU's room - each family's step its own */
#include <stdio.h>

#include <hip/hip_runtime_api.h>

#include "engine_impl.h"

bool osmos_runs_consecutive(OsmosSeq *const *seqs, uint32_t n, char err[OSMOS_ERR])
{
  for (uint32_t i = 1; i < n; i++)
    if (seqs[i - 1] != seqs[i])
      for (uint32_t j = 0; j + 1 < i; j++)
        if (seqs[j] == seqs[i])
          return snprintf(err, OSMOS_ERR, "%s", "a sequence's tokens not consecutive in the batch"), false;
  return true;
}

uint32_t osmos_runs(OsmosSeq *const *seqs, uint32_t n, uint32_t at, uint32_t first[OSMOS_FAMILY_RUNS],
                    uint32_t len[OSMOS_FAMILY_RUNS], uint32_t *runs)
{
  uint32_t r = 0, k = 0;
  for (; at + k < n && r < OSMOS_FAMILY_RUNS; r++)
  {
    uint32_t m = 1;
    while (at + k + m < n && seqs[at + k + m] == seqs[at + k])
      m++;
    first[r] = at + k, len[r] = m, k += m;
  }
  *runs = r;
  return k;
}

bool osmos_gpu_room(uint64_t *bytes)
{
  size_t free_b = 0, total = 0;
  if (hipMemGetInfo(&free_b, &total) != hipSuccess)
    return false;
  *bytes = free_b;
  return true;
}
