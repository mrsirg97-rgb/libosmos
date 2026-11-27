/* rows over threads (osmos/cpu/par.h) */
#include "osmos/cpu/par.h"

#include <pthread.h>

typedef struct Run
{
  CpuParFn fn;
  void *ctx;
  uint64_t r0, r1;
} Run;

static void *run(void *a)
{
  Run *r = a;
  r->fn(r->ctx, r->r0, r->r1);
  return NULL;
}

void cpu_par_rows(uint64_t rows, uint64_t least, CpuParFn fn, void *ctx)
{
  uint64_t k = least ? rows / least : rows;
  k = k < 1 ? 1 : k > CPU_PAR_THREADS ? CPU_PAR_THREADS : k;
  Run r[CPU_PAR_THREADS];
  pthread_t th[CPU_PAR_THREADS];
  bool started[CPU_PAR_THREADS] = {false};
  for (uint64_t i = 0; i < k; i++)
  {
    r[i] = (Run){fn, ctx, rows * i / k, rows * (i + 1) / k};
    if (i)
      started[i] = pthread_create(&th[i], NULL, run, &r[i]) == 0;
  }
  run(&r[0]);
  for (uint64_t i = 1; i < k; i++)
    if (started[i])
      pthread_join(th[i], NULL);
    else
      run(&r[i]);
}
