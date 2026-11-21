/* osmos/hy4/route.h - a read's routing (the host's, between the router and the experts' matrix cores): each pick's
   row in its expert's run, the runs cut into mm_group's entries */
#pragma once
#include <stdbool.h>
#include <stdint.h>

/* the tables (the caller's: hy4_read_entries / hy4_read_slots of the step's room; estart n_experts + 1) */
typedef struct Hy4ReadRoute
{
  int32_t *gidx;      /* [picks] a sorted row's token */
  int32_t *sidx;      /* [picks] a (token, slot)'s sorted row */
  float *wsel;        /* [picks] a sorted row's weight */
  int32_t *cnt;       /* [entries] an entry's rows */
  uint32_t *expert;   /* [entries] its expert */
  uint32_t *x, *y;    /* [entries] gate / up: its x16 row (x lat values a slot) and output row (ffe a row) */
  uint32_t *xd, *yd;  /* [entries] down: its x16 row (ffe a slot) and output row (lat a row) */
  int32_t *rgidx;     /* [slots] a slot's token (-1: a tile's pad) */
  int32_t *rhidx;     /* [slots] a slot's sorted row (-1: a pad) */
  uint32_t *estart;   /* [n_experts + 1] the sort's scratch */
  uint32_t entries, slots;
} Hy4ReadRoute;

/*
hy4_read_route
  the picks idx [n][k] (weights w) grouped by expert, ascending, each expert's rows its tokens ascending (a token's
  slots in order): gidx / sidx / wsel; each expert's rows cut into entries of at most chunk (in build order) with
  their x16 slots - an expert's slots a whole number of MM_TILE (tile; the pad -1) - and output rows. one counting
  sort: O(picks + experts). false for a pick outside 0 .. experts - 1.
*/
bool hy4_read_route(const int32_t *idx, const float *w, uint32_t n, uint32_t k, uint32_t experts, uint32_t chunk,
                    uint32_t tile, uint32_t lat, uint32_t ffe, Hy4ReadRoute *r);
