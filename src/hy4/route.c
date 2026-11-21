/* a Hy4 read's routing tables on the host (osmos/hy4/route.h): one counting sort of the picks by expert */
#include <string.h>

#include "osmos/hy4/route.h"

bool hy4_read_route(const int32_t *idx, const float *w, uint32_t n, uint32_t k, uint32_t experts, uint32_t chunk,
                    uint32_t tile, uint32_t lat, uint32_t ffe, Hy4ReadRoute *r)
{
  const uint32_t picks = n * k;
  uint32_t *start = r->estart; /* expert e's rows from start[e]: its picks' count, then their running sum */
  memset(start, 0, ((size_t)experts + 1) * sizeof *start);
  for (uint32_t i = 0; i < picks; i++)
  {
    if (idx[i] < 0 || (uint32_t)idx[i] >= experts)
      return false;
    start[idx[i] + 1]++;
  }
  for (uint32_t e = 0; e < experts; e++)
    start[e + 1] += start[e];
  for (uint32_t i = 0; i < picks; i++) /* (token, slot) order: an expert's rows its tokens ascending */
  {
    const uint32_t q = start[idx[i]]++;
    r->gidx[q] = (int32_t)(i / k), r->sidx[i] = (int32_t)q, r->wsel[q] = w[i];
  }
  uint32_t q = 0, e_n = 0, s_n = 0; /* (start[e] now e's end) */
  for (uint32_t e = 0; e < experts; e++)
  {
    const uint32_t first = q;
    q = start[e];
    if (q == first)
      continue;
    for (uint32_t c = first; c < q; c += chunk, e_n++)
    {
      const uint32_t cnt = q - c < chunk ? q - c : chunk, slot = s_n + (c - first);
      r->cnt[e_n] = (int32_t)cnt, r->expert[e_n] = e;
      r->x[e_n] = slot * lat, r->y[e_n] = c * ffe, r->xd[e_n] = slot * ffe, r->yd[e_n] = c * lat;
    }
    for (uint32_t j = first; j < q; j++)
      r->rgidx[s_n + (j - first)] = r->gidx[j], r->rhidx[s_n + (j - first)] = (int32_t)j;
    const uint32_t padded = (q - first + tile - 1) / tile * tile;
    for (uint32_t s = s_n + (q - first); s < s_n + padded; s++)
      r->rgidx[s] = r->rhidx[s] = -1;
    s_n += padded;
  }
  r->entries = e_n, r->slots = s_n;
  return true;
}
