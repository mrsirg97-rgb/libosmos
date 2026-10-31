#include "forward_impl.h"

/* the quantized types a chunk of MM_MAX_TOKENS at a time through the one x16 (each token's sums its own: the bits
   of one call); past MV_MAX_TOKENS the matvec's wide tiles: n-invariant */
bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys)
{
  const uint32_t type = fwd_type(f, t);
  if (!mm_known(type))
    return mv_strided(type, w, rows, cols, x, xs, n, y, ys, NULL);
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS)
  {
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    ok = mm_x16(x + (uint64_t)k0 * xs, xs, m, cols, f->b.x16, f->b.overflow, NULL) &&
         mm_run(type, w, rows, cols, f->b.x16, m, y + (uint64_t)k0 * ys, ys, NULL);
  }
  return ok;
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
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS)
  {
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    ok = ((ready && n <= MM_MAX_TOKENS) || mm_x16(x + (uint64_t)k0 * xs, xs, m, cols, f->b.x16, f->b.overflow, NULL)) &&
         mm_run2(type, w0, rows0, y0 + (uint64_t)k0 * ys0, ys0, w1, rows1, y1 + (uint64_t)k0 * ys1, ys1, cols,
                 f->b.x16, m, NULL);
  }
  return ok;
}

/* fwd_mat on an x whose x16 is in f->b.x16 already (`ready`: an op wrote it beside x - ops_*_x16): the conversion
   skipped; else, or past one x16's tokens, fwd_mat */
bool fwd_mat_ready(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x,
                   uint32_t xs, uint32_t n, float *y, uint32_t ys, bool ready)
{
  const uint32_t type = fwd_type(f, t);
  if (!ready || !mm_known(type) || n > MM_MAX_TOKENS)
    return fwd_mat(f, t, w, rows, cols, x, xs, n, y, ys);
  return mm_run(type, w, rows, cols, f->b.x16, n, y, ys, NULL);
}
