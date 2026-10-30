#include "forward_impl.h"

/* the quantized types a chunk of MM_MAX_TOKENS at a time through the one x16 (each token's sums its own: the bits
   of one call); past MV_MAX_TOKENS the matvec's wide tiles: n-invariant */
bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys)
{
  if (!mm_known(t->type))
    return mv_strided(t->type, w, rows, cols, x, xs, n, y, ys, NULL);
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS)
  {
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    ok = mm_x16(x + (uint64_t)k0 * xs, xs, m, cols, f->b.x16, f->b.overflow, NULL) &&
         mm_run(t->type, w, rows, cols, f->b.x16, m, y + (uint64_t)k0 * ys, ys, NULL);
  }
  return ok;
}
