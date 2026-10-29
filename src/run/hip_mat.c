#include "forward_impl.h"

bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys)
{
  if (mm_known(t->type))
    return mm_x16(x, xs, n, cols, f->b.x16, f->b.overflow, NULL) &&
           mm_run(t->type, w, rows, cols, f->b.x16, n, y, ys, NULL);
  for (uint32_t t0 = 0; t0 < n; t0 += MV_MAX_TOKENS)
  {
    const uint32_t k = n - t0 < MV_MAX_TOKENS ? n - t0 : MV_MAX_TOKENS;
    if (!mv_strided(t->type, w, rows, cols, x + (uint64_t)t0 * xs, xs, k, y + (uint64_t)t0 * ys, ys, NULL))
      return false;
  }
  return true;
}
