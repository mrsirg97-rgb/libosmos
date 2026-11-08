#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "osmos/run/ahead.h"

/* ahead_union: each token's first k guesses, their union; ids out of range skipped; the count */

int main(void)
{
  printf("[test_ahead] start\n");
  enum { E = 8, T = 3, K = 3 };
  const int32_t g[T * K] = {1, 4, 7,   /* token 0 */
                            4, -1, 2,  /* token 1: an id out of range (below) */
                            9, 1, 0};  /* token 2: one past E */
  bool named[E];
  assert(ahead_union(g, T, K, K, E, named) == 5); /* 1 4 7 2 0 */
  const bool all[E] = {true, true, true, false, true, false, false, true};
  for (int e = 0; e < E; e++)
    assert(named[e] == all[e]);
  assert(ahead_union(g, T, K, 1, E, named) == 2); /* the first guess each: 1, 4, (9 skipped) */
  const bool first[E] = {false, true, false, false, true, false, false, false};
  for (int e = 0; e < E; e++)
    assert(named[e] == first[e]);
  assert(ahead_union(g, 0, K, K, E, named) == 0); /* no tokens: none */
  for (int e = 0; e < E; e++)
    assert(!named[e]);
  printf("  union: each token's first k, out-of-range ids skipped, counted\n");
  printf("[test_ahead] ok\n");
  return 0;
}
