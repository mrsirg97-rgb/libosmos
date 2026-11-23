#include "forward_impl.h"

/* a weight's product by its type alone: mm for the quantized types (mm_tokens: MM_MAX_TOKENS a launch through the one
   x16, each token's sums its own), the matvec for the rest (past MV_MAX_TOKENS its wide tiles: n-invariant) */
bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys)
{
  return fwd_mat_ready(f, t, w, rows, cols, x, xs, n, y, ys, false);
}

/* two matrices of one type on the same x (the shared expert's gate and up; q_a and kv): the x16 once and both in one
   launch (mm_run2: each the bits of fwd_mat alone); a type without mm, or two types, as fwd_mat each */
bool fwd_mat2(Forward *f, const GgufTensor *t0, const void *w0, uint32_t rows0, float *y0, uint32_t ys0,
              const GgufTensor *t1, const void *w1, uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols,
              const float *x, uint32_t xs, uint32_t n, bool ready)
{
  const uint32_t type = fwd_type(f, t0);
  if (!mm_known(type) || type != fwd_type(f, t1))
    return fwd_mat(f, t0, w0, rows0, cols, x, xs, n, y0, ys0) && fwd_mat(f, t1, w1, rows1, cols, x, xs, n, y1, ys1);
  return mm_tokens(type, w0, rows0, y0, ys0, w1, rows1, y1, ys1, cols, x, xs, n, f->b.x16, f->b.overflow, ready, NULL);
}

/* fwd_mat on an x whose x16 may be in f->b.x16 already (`ready`: an op wrote it beside x - ops_*_x16): the conversion
   skipped within one x16's tokens */
bool fwd_mat_ready(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x,
                   uint32_t xs, uint32_t n, float *y, uint32_t ys, bool ready)
{
  const uint32_t type = fwd_type(f, t);
  if (!mm_known(type))
    return mv_strided(type, w, rows, cols, x, xs, n, y, ys, NULL);
  return mm_tokens(type, w, rows, y, ys, NULL, 0, NULL, 0, cols, x, xs, n, f->b.x16, f->b.overflow, ready, NULL);
}
