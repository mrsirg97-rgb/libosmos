/* osmos/model/moe_route.h - a step's router picks grouped by expert: every MoE family's host tables (DeepSeek's,
   Inkling's, Hy4-flash's), one counting sort */
#pragma once
#include <stdbool.h>
#include <stdint.h>

/* the tables (the caller's memory: picks for by / row, experts for the runs and estart + 1, slots_max for gidx and
   hidx - moe_route_slots) */
typedef struct MoeRoute
{
  uint32_t *by;        /* [picks] sorted row r's pick: experts ascending, an expert's picks in pick order */
  uint32_t *row;       /* [picks] pick m's sorted row */
  uint32_t *expert;    /* [runs] a run's expert (each distinct pick one run) */
  uint32_t *start;     /* [runs] its first sorted row */
  uint32_t *count;     /* [runs] its rows */
  uint32_t *slot0;     /* [runs] its first slot (an expert's slots its rows padded to whole tiles) */
  int32_t *gidx;       /* [slots] a slot's token (its pick / k), -1 a tile's pad */
  int32_t *hidx;       /* [slots] a slot's sorted row, -1 a pad */
  uint32_t *estart;    /* [experts + 1] the sort's own */
  uint32_t runs, slots;
} MoeRoute;

/* an entry: a run's rows off .. off + cnt - 1 (a part of at most a chunk) */
typedef struct MoeEntry
{
  uint32_t run, off, cnt;
} MoeEntry;

/*
moe_route
  the picks idx [picks] (k a token: pick m token m / k) grouped by expert into r, its slots tiles of `tile`. false for
  a pick outside 0 .. experts - 1 (nothing else trusted).
moe_route_slots
  the most slots a step's picks take: picks + experts (tile - 1).
moe_entries
  r's runs in parts of at most chunk rows, out in ascending count, a count's in build order (run, then part): a
  launch a run of one column-tile count. *n its entries (at most picks / chunk + runs: out's and key's room). false
  for a chunk 0 or past MOE_CHUNK_BITS, or runs past MOE_RUN_BITS.
*/
#define MOE_CHUNK_BITS 12 /* moe_entries' most chunk: under 4,096 rows (MM_MAX_TOKENS' 2,048) */
#define MOE_RUN_BITS 20   /* its most runs (experts a step) */
bool moe_route(const int32_t *idx, uint32_t picks, uint32_t k, uint32_t experts, uint32_t tile, MoeRoute *r);
uint64_t moe_route_slots(uint64_t picks, uint64_t experts, uint32_t tile);
bool moe_entries(const MoeRoute *r, uint32_t chunk, MoeEntry *out, uint64_t *key, uint32_t *n);
