/* a step's kept rows' logits (osmos/gpu/kept.h) */
#include <hip/hip_runtime_api.h>

#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"

const float *kept_row(const int32_t *slot, uint32_t rows, const float *logits, uint32_t V, uint32_t row)
{
  return row < rows && slot[row] >= 0 ? logits + (uint64_t)slot[row] * V : NULL;
}

const char *kept_greedy(const int32_t *slot, uint32_t rows, const float *logits, uint32_t V, uint32_t row0, uint32_t n,
                        int32_t *dev_ids, void *argmax, void *stream, int32_t *ids)
{
  if (!ids || !n || row0 + n > rows)
    return "no forward or ids, or rows past the last step's";
  for (uint32_t r = 0; r < n; r++) /* kept rows' slots in their order: rows row0 .. one run of slots */
    if (slot[row0 + r] < 0 || slot[row0 + r] != slot[row0] + (int32_t)r)
      return "a row the last step kept no logits of";
  if (!ops_argmax(logits + (uint64_t)slot[row0] * V, V, n, V, dev_ids, argmax, stream) ||
      hipMemcpyAsync(ids, dev_ids, n * 4, hipMemcpyDeviceToHost, (hipStream_t)stream) != hipSuccess ||
      hipStreamSynchronize((hipStream_t)stream) != hipSuccess)
    return "a GPU call failed (the greedy pick)";
  for (uint32_t r = 0; r < n; r++)
    if (ids[r] < 0)
      return "a NaN in a row's logits";
  return NULL;
}
