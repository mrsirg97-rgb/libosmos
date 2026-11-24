/* a step's kept rows (osmos/gpu/kept.h) */
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"

const float *kept_row(const KeptView *k, uint32_t row)
{
  return k && row < k->rows && k->slot[row] >= 0 ? k->logits + (uint64_t)k->slot[row] * k->V : NULL;
}

const char *kept_greedy(const KeptView *k, uint32_t row0, uint32_t n, int32_t *ids)
{
  if (!k || !ids || !n || row0 + n < row0 || row0 + n > k->rows)
    return "no forward or ids, or rows past the last step's";
  for (uint32_t r = 0; r < n; r++) /* kept rows' slots in their order: rows row0 .. one run of slots */
    if (k->slot[row0 + r] < 0 || k->slot[row0 + r] != k->slot[row0] + (int32_t)r)
      return "a row the last step kept no logits of";
  const hipStream_t st = (hipStream_t)k->stream;
  int32_t *out = k->host_ids ? k->host_ids : ids;
  if (!ops_argmax(k->logits + (uint64_t)k->slot[row0] * k->V, k->V, n, k->V, k->dev_ids, k->argmax, k->stream) ||
      hipMemcpyAsync(out, k->dev_ids, n * 4, hipMemcpyDeviceToHost, st) != hipSuccess ||
      !(k->settle ? k->settle(k->settle_ctx) : hipStreamSynchronize(st) == hipSuccess))
    return "a GPU call failed (the greedy pick)";
  if (out != ids)
    memcpy(ids, out, n * 4);
  for (uint32_t r = 0; r < n; r++)
    if (ids[r] < 0)
      return "a NaN in a row's logits";
  return NULL;
}
