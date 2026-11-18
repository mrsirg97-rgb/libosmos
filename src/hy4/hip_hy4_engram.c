/* a hy4 step's Engram rows: each token's n-gram (it and the two before it, the pad before its sequence's start)
   hashed (engram_rows), its 64 rows read from the table's file (rowread: threads of pread into pinned memory), then to
   the GPU and decoded - [Engram layer][token][column][dim] */
#include <hip/hip_runtime_api.h>

#include "osmos/gpu/quant.h"
#include "hy4_impl.h"

bool hy4_engram_read(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n)
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
  const uint64_t count = (uint64_t)L * n * cols;
  return rowread(f->eg_read, f->eg_fd, f->eg_off, (uint32_t)count, (uint32_t)f->eg_row_bytes, f->eg_stage) &&
         hipMemcpy(f->eg_raw, f->eg_stage, count * f->eg_row_bytes, hipMemcpyHostToDevice) == hipSuccess &&
         quant_dequant(t->type, f->eg_raw, count * (uint64_t)g->engram_dim, f->eg_rows, f->st);
}
