/* the experts as units (expert_units) against the slicing it replaced (0.12.3's weights_experts / q4x_weights_experts
   / ink2_weights_experts: each the same, Inkling's with its fused gate-up halves): every unit's three parts, a fused
   layer's, and the refusals - a tensor missing, slices not equal, a fused slice not halves - nothing written */
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "osmos/model/expert_units.h"

#define L 3
#define E 5

int main(void)
{
  printf("[test_expert_units] start\n");
  GgufTensor t[L][3];
  ExpertTensors lt[L];
  const int fds[2] = {7, 9};
  for (int l = 0; l < L; l++)
  {
    for (int k = 0; k < 3; k++)
      t[l][k] = (GgufTensor){.file = (l + k) % 2, .offset = 1000u * (l * 3 + k + 1), .bytes = (uint64_t)E * (64 + 16 * k)};
    lt[l] = (ExpertTensors){&t[l][0], &t[l][1], &t[l][2], false};
  }
  PoolUnit u[L * E];
  assert(expert_units(lt, L, E, NULL, NULL, 0) == L * E);
  assert(expert_units(lt, L, E, fds, u, L * E) == L * E);
  for (int i = 0; i < L * E; i++)
    for (int k = 0; k < 3; k++) /* 0.12.3's: offset + (i % E) slice, slice */
    {
      const GgufTensor *x = &t[i / E][k];
      const uint64_t slice = x->bytes / E;
      assert(u[i].n_parts == 3 && u[i].part[k].fd == fds[x->file] && u[i].part[k].offset == x->offset + (i % E) * slice &&
             u[i].part[k].bytes == slice);
    }
  lt[1].fused = true, lt[1].up = NULL; /* Inkling's: gate's slice its gate, then its up */
  assert(expert_units(lt, L, E, fds, u, L * E) == L * E);
  for (int e = 0; e < E; e++)
  {
    const GgufTensor *g = &t[1][0];
    const uint64_t slice = g->bytes / E, half = slice / 2;
    const PoolUnit *x = &u[E + e];
    assert(x->part[0].offset == g->offset + e * slice && x->part[0].bytes == half);
    assert(x->part[1].offset == g->offset + e * slice + half && x->part[1].bytes == half);
    assert(x->part[2].bytes == t[1][2].bytes / E);
  }
  memset(u, 0xAB, sizeof u);
  PoolUnit before[L * E];
  memcpy(before, u, sizeof u);
  t[1][0].bytes = (uint64_t)E * 63 + 1; /* not equal slices */
  assert(!expert_units(lt, L, E, fds, u, L * E));
  t[1][0].bytes = (uint64_t)E * 63; /* equal slices, a fused one odd: not halves */
  assert(!expert_units(lt, L, E, fds, u, L * E));
  lt[1].fused = false, lt[1].up = NULL; /* a tensor missing */
  assert(!expert_units(lt, L, E, fds, u, L * E));
  assert(!memcmp(before, u, sizeof u)); /* nothing written */
  printf("  %d layers x %d experts: every unit's parts the old slicing's; a fused layer's halves; refusals write nothing\n", L, E);
  printf("[test_expert_units] ok\n");
  return 0;
}
