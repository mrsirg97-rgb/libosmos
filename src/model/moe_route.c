/* a step's picks grouped by expert (osmos/model/moe_route.h): one counting sort, then the runs, slots and entries */
#include <stdlib.h>
#include <string.h>

#include "osmos/model/moe_route.h"

uint64_t moe_route_slots(uint64_t picks, uint64_t experts, uint32_t tile) { return picks + experts * (tile - 1); }

bool moe_route(const int32_t *idx, uint32_t picks, uint32_t k, uint32_t experts, uint32_t tile, MoeRoute *r)
{
  uint32_t *at = r->estart; /* expert e's rows from at[e]: its picks' count, their running sum, then its end */
  memset(at, 0, ((size_t)experts + 1) * sizeof *at);
  for (uint32_t m = 0; m < picks; m++)
  {
    if (idx[m] < 0 || (uint32_t)idx[m] >= experts)
      return false;
    at[idx[m] + 1]++;
  }
  for (uint32_t e = 0; e < experts; e++)
    at[e + 1] += at[e];
  for (uint32_t m = 0; m < picks; m++) /* pick order: an expert's rows its picks in order */
  {
    const uint32_t q = at[idx[m]]++;
    r->by[q] = m, r->row[m] = q;
  }
  uint32_t first = 0, runs = 0, slots = 0;
  for (uint32_t e = 0; e < experts; e++) /* (at[e] now e's end) */
  {
    const uint32_t end = at[e];
    if (end == first)
      continue;
    r->expert[runs] = e, r->start[runs] = first, r->count[runs] = end - first, r->slot0[runs] = slots;
    for (uint32_t q = first; q < end; q++)
      r->gidx[slots + q - first] = (int32_t)(r->by[q] / k), r->hidx[slots + q - first] = (int32_t)q;
    const uint32_t padded = (end - first + tile - 1) / tile * tile;
    for (uint32_t s = slots + end - first; s < slots + padded; s++)
      r->gidx[s] = r->hidx[s] = -1;
    slots += padded, runs++, first = end;
  }
  r->runs = runs, r->slots = slots;
  return true;
}

static int cmp_u64(const void *a, const void *b)
{
  const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

bool moe_entries(const MoeRoute *r, uint32_t chunk, MoeEntry *out, uint64_t *key, uint32_t *n)
{
  /* a key a part: its count, run and first row (MOE_CHUNK_BITS, MOE_RUN_BITS, 32) - sorted, the count's order then
     the build order (run, then part) - then each decoded into out */
  if (!chunk || chunk >= (1u << MOE_CHUNK_BITS) || r->runs >= (1u << MOE_RUN_BITS))
    return false;
  uint32_t ne = 0;
  for (uint32_t j = 0; j < r->runs; j++)
    for (uint32_t off = 0; off < r->count[j]; off += chunk)
    {
      const uint64_t cnt = r->count[j] - off < chunk ? r->count[j] - off : chunk;
      key[ne++] = cnt << (64 - MOE_CHUNK_BITS) | (uint64_t)j << 32 | off;
    }
  qsort(key, ne, sizeof *key, cmp_u64);
  for (uint32_t i = 0; i < ne; i++)
    out[i] = (MoeEntry){(uint32_t)(key[i] >> 32) & ((1u << MOE_RUN_BITS) - 1), (uint32_t)key[i],
                        (uint32_t)(key[i] >> (64 - MOE_CHUNK_BITS))};
  *n = ne;
  return true;
}
