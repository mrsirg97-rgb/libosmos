#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "osmos/cpu/par.h"

/* cpu_par_rows: every row once, whatever the rows and the least a run takes (one run past the rows: the caller's) */

static void mark(void *ctx, uint64_t r0, uint64_t r1)
{
  unsigned char *seen = ctx;
  for (uint64_t r = r0; r < r1; r++)
    __atomic_fetch_add(&seen[r], 1, __ATOMIC_RELAXED);
}

int main(void)
{
  printf("[test_par] start\n");
  const uint64_t rows[] = {0, 1, 7, 255, 256, 4097, 100003};
  const uint64_t least[] = {0, 1, 256, 1u << 20};
  for (size_t i = 0; i < sizeof rows / sizeof *rows; i++)
    for (size_t j = 0; j < sizeof least / sizeof *least; j++)
    {
      unsigned char *seen = calloc(rows[i] + 1, 1);
      assert(seen);
      cpu_par_rows(rows[i], least[j], mark, seen);
      for (uint64_t r = 0; r < rows[i]; r++)
        assert(seen[r] == 1);
      free(seen);
    }
  printf("[test_par] ok\n");
  return 0;
}
