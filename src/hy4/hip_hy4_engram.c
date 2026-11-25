/* a hy4 step's Engram rows: each token's n-gram (it and the two before it, the pad before its sequence's start)
   hashed (engram_rows), its 64 rows read from the table's file (rowread: threads of pread into pinned memory), then to
   the GPU and decoded - [Engram layer][token][column][dim] */
#include <hip/hip_runtime_api.h>

#include "osmos/gpu/quant.h"
#include "hy4_impl.h"

/* the step's rows' file offsets, every Engram layer's (no read yet: hy4_engram_layer reads each as its layer is queued) */
bool hy4_engram_offsets(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n)
{
  const Hy4Geometry *g = &f->w->geo;
  const int L = g->n_engram, cols = engram_cols(f->eg), ng = engram_ngram(f->eg);
  const GgufTensor *t = f->w->engram_table;
  int64_t rows[HY4_MAX_ENGRAM * 64];
  int32_t last[ENGRAM_MAX_NGRAM];
  if (L * cols > (int)(sizeof rows / sizeof *rows))
    return false;
  for (uint32_t r = 0; r < runs; r++)
    for (uint32_t k = 0; k < run[r].len; k++)
    {
      const Hy4Seq *s = run[r].s;
      const int32_t p = s->pos + (int32_t)k;
      const int have = p + 1 < ng ? p + 1 : ng;
      for (int j = 0; j < have; j++)
        last[j] = s->tokens[p - j];
      if (!engram_rows(f->eg, last, have, rows))
        return false;
      const uint64_t tok = run[r].off + k;
      for (int e = 0; e < L; e++)
        for (int c = 0; c < cols; c++)
          f->eg_off[((uint64_t)e * n + tok) * cols + c] =
              t->offset + (f->eg_first[e] + (uint64_t)rows[e * cols + c]) * f->eg_row_bytes;
    }
  return true;
}

/* Engram layer e's rows of the step (n tokens): read (rowread: threads of pread into its part of the pinned stage),
   then their copy to the GPU and their decode queued on the step's stream - called as the layer is queued, so the
   reads of a later Engram layer run while the GPU works the layers before it */
bool hy4_engram_layer(Hy4Forward *f, int e, uint32_t n)
{
  const Hy4Geometry *g = &f->w->geo;
  const GgufTensor *t = f->w->engram_table;
  const uint64_t cols = (uint64_t)engram_cols(f->eg), count = n * cols, at = (uint64_t)e * count;
  const double t0 = f->pf.on ? hy4_now() : 0;
  bool ok = rowread(f->eg_read, f->eg_fd, f->eg_off + at, (uint32_t)count, (uint32_t)f->eg_row_bytes,
                    f->eg_stage + at * f->eg_row_bytes);
  const double t1 = f->pf.on ? hy4_now() : 0;
  ok = ok && hipMemcpyAsync((uint8_t *)f->eg_raw + at * f->eg_row_bytes, f->eg_stage + at * f->eg_row_bytes,
                            count * f->eg_row_bytes, hipMemcpyHostToDevice, (hipStream_t)f->st) == hipSuccess &&
       quant_dequant(t->type, (uint8_t *)f->eg_raw + at * f->eg_row_bytes, count * (uint64_t)g->engram_dim,
                     f->eg_rows + at * (uint64_t)g->engram_dim, f->st);
  if (f->pf.on)
    f->pf.rowread_s += t1 - t0, f->pf.copy_s += hy4_now() - t1, f->pf.rows += count, f->pf.bytes += count * f->eg_row_bytes;
  return ok;
}
