#include "forward_impl.h"

bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys)
{
  if (mm_known(t->type))
    return mm_x16(x, xs, n, cols, f->b.x16, f->b.overflow, NULL) &&
           mm_run(t->type, w, rows, cols, f->b.x16, n, y, ys, NULL);
  return mv_strided(t->type, w, rows, cols, x, xs, n, y, ys, NULL); /* past MV_MAX_TOKENS wide tiles: n-invariant */
}
