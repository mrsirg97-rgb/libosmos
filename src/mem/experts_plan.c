/* the experts' plan (osmos/mem/experts.h): room, stream and warm set for a budget, every family's */
#include <stdlib.h>
#include <string.h>

#include "osmos/mem/experts.h"

uint64_t experts_stream_room(const PoolUnit *units, uint32_t layers, uint32_t experts, const bool *warm,
                             uint64_t grain)
{
  uint64_t most = 0, whole = 0;
  for (uint32_t l = 0; l < layers; l++)
  {
    uint64_t layer = 0, cold = 0;
    for (uint32_t e = 0; e < experts; e++)
    {
      const uint64_t b = pool_unit_bytes(&units[(size_t)l * experts + e], grain);
      layer += b, cold += warm[(size_t)l * experts + e] ? 0 : b;
    }
    most = cold > most ? cold : most, whole = layer > whole ? layer : whole;
  }
  return 2 * most > whole ? 2 * most : whole;
}

bool experts_plan(const PoolUnit *units, uint32_t layers, uint32_t experts, const uint32_t *order, uint64_t room,
                  uint64_t budget, uint64_t fixed, uint64_t grain, ExpertsPlan *p)
{
  const size_t n = (size_t)layers * experts;
  if (!units || !order || !p || !n)
    return false;
  bool *warm = calloc(n, 1);
  if (!warm)
    return false;
  uint64_t all = 0, largest = 0;
  bool every = true;
  for (size_t i = 0; i < n; i++)
  {
    const uint64_t b = pool_unit_bytes(&units[i], grain);
    all += b, largest = b > largest ? b : largest;
    every = every && order[i] < n && !warm[order[i]], warm[order[i] < n ? order[i] : 0] = true;
  }
  if (!every)
    return free(warm), false;
  const uint64_t given = room, floor = (uint64_t)experts * largest;
  uint64_t stream_room = 2 * floor;
  size_t n_warm = 0;
  for (int pass = 0; pass < EXPERTS_PASSES; pass++)
  {
    room = given;
    if (budget) /* the build's whole budget: the experts' pool what the rest leaves (fixed, the stream, the slack) - the
                   budget a ceiling: the two slots' round-up to a whole grain each counted too */
    {
      const uint64_t rest = fixed + stream_room + 4 * largest + 2 * grain;
      room = budget > rest ? budget - rest : 0;
    }
    room = room > all ? all : room < floor ? floor : room;
    n_warm = 0;
    for (uint64_t took = 0; n_warm < n; n_warm++)
    {
      const uint64_t b = pool_unit_bytes(&units[order[n_warm]], grain);
      if (took + b > room)
        break;
      took += b;
    }
    memset(warm, 0, n);
    for (size_t i = 0; i < n_warm; i++)
      warm[order[i]] = true;
    const uint64_t want = n_warm == n ? 3 * largest : experts_stream_room(units, layers, experts, warm, grain);
    const bool settled = want >= stream_room;
    stream_room = want < stream_room ? want : stream_room;
    if (settled || !budget)
      break;
  }
  free(warm);
  *p = (ExpertsPlan){room, stream_room, largest, all, n_warm};
  return true;
}
