#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/mem/arena.h"

#define G ARENA_GRAIN

static uint8_t *mem(uint64_t bytes)
{
  uint8_t *m = aligned_alloc(G, bytes);
  assert(m);
  return m;
}

static void test_refusals(void)
{
  uint8_t *m = mem(8 * G);
  assert(!arena_create(NULL, 8 * G) && !arena_create(m + 1, 8 * G) && !arena_create(m, 8 * G + 1));
  assert(!arena_create(m, 0));
  Arena *a = arena_create(m, 8 * G);
  assert(a && !arena_alloc(a, 0) && !arena_alloc(a, 9 * G) && !arena_free(a, m) && !arena_free(a, NULL));
  arena_destroy(a);
  arena_destroy(NULL);
  free(m);
  printf("  refusals: an unaligned or empty range, 0 bytes, more than it holds, a block never given\n");
}

/* first fit, rounded to grains; freed blocks join their neighbours; a double free refused */
static void test_fit_and_join(void)
{
  uint8_t *m = mem(16 * G);
  Arena *a = arena_create(m, 16 * G);
  uint8_t *x = arena_alloc(a, 1), *y = arena_alloc(a, 3 * G), *z = arena_alloc(a, G + 1);
  assert(x == m && y == m + G && z == m + 4 * G && arena_used(a) == 6 * G && arena_largest(a) == 10 * G);
  assert(arena_free(a, y) && !arena_free(a, y) && arena_used(a) == 3 * G);
  uint8_t *w = arena_alloc(a, 2 * G); /* first fit: y's hole */
  assert(w == m + G);
  assert(arena_free(a, x) && arena_free(a, w) && arena_free(a, z)); /* all joined: one extent again */
  assert(arena_used(a) == 0 && arena_largest(a) == 16 * G && arena_alloc(a, 16 * G) == m);
  arena_destroy(a);
  free(m);
  printf("  fit and join: first fit in grains, a freed block joined with its neighbours, a double free refused\n");
}

/* random allocs and frees against a model: no two blocks overlap, each inside the range, the counts exact, and once
   all are freed the whole range is one extent again */
static void test_random(void)
{
  enum { GRAINS = 4096, N = 300, STEPS = 50000 };
  uint8_t *m = mem((uint64_t)GRAINS * G);
  Arena *a = arena_create(m, (uint64_t)GRAINS * G);
  uint8_t *blk[N] = {0};
  uint64_t len[N] = {0}, used = 0;
  srand(11);
  int ok = 0, refused = 0;
  for (int s = 0; s < STEPS; s++)
  {
    const int i = rand() % N;
    if (blk[i])
    {
      assert(arena_free(a, blk[i]));
      used -= len[i], blk[i] = NULL;
    }
    else
    {
      const uint64_t want = 1 + (uint64_t)(rand() % (40 * G));
      uint8_t *p = arena_alloc(a, want);
      if (!p)
      {
        assert(arena_largest(a) < (want + G - 1) / G * G), refused++;
        continue;
      }
      len[i] = (want + G - 1) / G * G, blk[i] = p, used += len[i], ok++;
      assert(p >= m && p + len[i] <= m + (uint64_t)GRAINS * G && (uintptr_t)p % G == 0);
      memset(p, i, len[i]); /* a block's bytes its own: an overlap would show below */
    }
    assert(arena_used(a) == used);
  }
  for (int i = 0; i < N; i++)
    if (blk[i])
    {
      for (uint64_t b = 0; b < len[i]; b += G)
        assert(blk[i][b] == (uint8_t)i);
      assert(arena_free(a, blk[i]));
    }
  assert(arena_used(a) == 0 && arena_largest(a) == (uint64_t)GRAINS * G);
  arena_destroy(a);
  free(m);
  printf("  random: %d steps, %d allocs (%d refused, none while a large enough extent was free), no overlap, all joined back\n",
         STEPS, ok, refused);
}

int main(void)
{
  printf("[test_arena] start\n");
  test_refusals();
  test_fit_and_join();
  test_random();
  printf("[test_arena] ok\n");
  return 0;
}
