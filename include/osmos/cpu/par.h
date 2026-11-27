#ifndef CPU_PAR_H
#define CPU_PAR_H

#include <stdbool.h>
#include <stdint.h>

#define CPU_PAR_THREADS 16 /* a load's conversions' threads: the box's cores (32 threads, 16 cores) */

typedef void (*CpuParFn)(void *ctx, uint64_t r0, uint64_t r1);
void cpu_par_rows(uint64_t rows, uint64_t least, CpuParFn fn, void *ctx);

#endif

/*
cpu_par_rows
  fn over rows [0, rows) in up to CPU_PAR_THREADS contiguous runs at once, each at least `least` rows (fewer runs past
  that: a small matrix on the calling thread alone); returns when every run has. each row is computed as it would be
  alone, so the bytes are the serial loop's. a thread not started runs its rows on the calling thread.
*/
