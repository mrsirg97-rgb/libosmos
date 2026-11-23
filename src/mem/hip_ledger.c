/* the ledger (osmos/mem/ledger.h) */
#include "osmos/mem/ledger_hip.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "osmos/err_hip.h"
#include "osmos/run/fit.h"

/* the room's reading (DESIGN.md, "the build ledger"): memory taken is counted at once - HIP's,
   pinned, mapped slots alike - so a reading never overstates the room; memory freed from mapped slots is counted back
   over ~1.25 ms a GiB (32 GiB: 40 ms, in steps of 1.9-6.7 GiB a 10 ms), so a reading just after a free is short. a
   refusal is final only after the room is read again once it stands still */
#define LEDGER_NAP_MS 10           /* a reading's interval: the count-back's steps seen at it */
#define LEDGER_STILL (64ull << 20) /* between the idle reading's steps (7 MiB at most) and the count-back's (1.9 GiB) */
#define LEDGER_WAIT_NAPS 45        /* 3x the longest count-back: the GPU's whole 115 GiB at 1.25 ms a GiB */

typedef struct
{
  void *p;
  uint64_t bytes;
} Entry;

struct OsmosLedger
{
  uint64_t room, taken, kept; /* the room past what it holds (read, plus taken); what it holds; the RAM kept */
  bool read;                  /* its room this machine's (false: a test's, never read) */
  uint32_t n, cap;            /* the table's entries and slots (a power of 2, grown at half full) */
  Entry *e;
  char why[192];
};

static uint64_t room_now(uint64_t kept)
{
  size_t f = 0, t = 0;
  if (hipMemGetInfo(&f, &t) != hipSuccess)
    return 0;
  return fit_room((uint64_t)f, fit_ram_available(), kept);
}

static void nap(long ms)
{
  struct timespec ts = {0, ms * 1000000L};
  nanosleep(&ts, NULL);
}

static uint64_t room_settled(uint64_t kept)
{
  uint64_t a = room_now(kept);
  for (int i = 0; i < LEDGER_WAIT_NAPS; i++)
  {
    nap(LEDGER_NAP_MS);
    const uint64_t b = room_now(kept);
    if ((b > a ? b - a : a - b) < LEDGER_STILL)
      return b < a ? b : a;
    a = b;
  }
  return a;
}

OsmosLedger *ledger_create(uint64_t ram_kept, char err[OSMOS_ERR])
{
  OsmosLedger *l = calloc(1, sizeof *l);
  if (!l)
    return osmos_fail(err, "out of memory (the ledger)"), NULL;
  l->kept = ram_kept, l->read = true;
  l->room = room_now(ram_kept);
  return l;
}

OsmosLedger *ledger_create_room(uint64_t room, uint64_t ram_kept)
{
  OsmosLedger *l = calloc(1, sizeof *l);
  if (l)
    l->room = room, l->kept = ram_kept;
  return l;
}

void ledger_destroy(OsmosLedger *l)
{
  if (!l)
    return;
  free(l->e);
  free(l);
}

void ledger_read(OsmosLedger *l)
{
  if (l && l->read)
    l->room = room_now(l->kept) + l->taken;
}

static void settle(OsmosLedger *l)
{
  if (l->read)
    l->room = room_settled(l->kept) + l->taken;
}

uint64_t ledger_left(const OsmosLedger *l)
{
  if (!l)
    return UINT64_MAX;
  const uint64_t used = l->taken + FIT_HEADROOM;
  return l->room > used ? l->room - used : 0;
}

uint64_t ledger_left_settled(OsmosLedger *l)
{
  if (l)
    settle(l);
  return ledger_left(l);
}

uint64_t ledger_taken(const OsmosLedger *l) { return l ? l->taken : 0; }

const char *ledger_why(const OsmosLedger *l) { return l ? l->why : ""; }

bool ledger_check(const OsmosLedger *l, hipError_t e, char err[OSMOS_ERR], const char *what)
{
  if (e == hipErrorOutOfMemory && l && l->why[0])
    return osmos_fail(err, "%s: refused by the ledger: %s", what, l->why);
  return osmos_hip(e, err, what);
}

static uint32_t slot(const void *p, uint32_t cap)
{
  const uint64_t h = (uint64_t)(uintptr_t)p * 0x9e3779b97f4a7c15ull; /* (Fibonacci hashing: 2^64 / the golden ratio) */
  return (uint32_t)(h >> 32) & (cap - 1);
}

static void place(Entry *e, uint32_t cap, Entry x)
{
  uint32_t i = slot(x.p, cap);
  while (e[i].p)
    i = (i + 1) & (cap - 1);
  e[i] = x;
}

/* the table twice its slots (open addressing stays short under half full) - false: no memory */
static bool grow(OsmosLedger *l)
{
  const uint32_t cap = l->cap ? 2 * l->cap : 64;
  Entry *e = calloc(cap, sizeof *e);
  if (!e)
    return false;
  for (uint32_t i = 0; i < l->cap; i++)
    if (l->e[i].p)
      place(e, cap, l->e[i]);
  free(l->e), l->e = e, l->cap = cap;
  return true;
}

/* bytes as a reader reads them: GiB, or MiB under one */
static const char *size_of(char out[24], uint64_t b)
{
  if (b >= (1ull << 30))
    snprintf(out, 24, "%.2f GiB", (double)b / (1ull << 30));
  else
    snprintf(out, 24, "%.1f MiB", (double)b / (1ull << 20));
  return out;
}

/* the allocation's bytes past the room: refused (why kept), else counted - a refusal only on a settled reading */
static bool take(OsmosLedger *l, size_t bytes)
{
  if (!l)
    return true;
  if ((uint64_t)bytes > ledger_left(l))
    settle(l);
  if ((uint64_t)bytes > ledger_left(l))
  {
    char asked[24], left[24];
    snprintf(l->why, sizeof l->why, "%s asked, %s left (of the GPU's memory or the RAM past %.1f GiB kept)",
             size_of(asked, bytes), size_of(left, ledger_left(l)), (double)l->kept / (1ull << 30));
    return false;
  }
  l->taken += bytes;
  l->why[0] = 0; /* (a refusal said once: never another failure's) */
  return true;
}

/* the pointer kept with its bytes (no memory for the table: counted, never given back - the ledger stricter, never
   looser) */
static void note(OsmosLedger *l, void *p, size_t bytes)
{
  if (!l || !p || (2 * (l->n + 1) > l->cap && !grow(l)))
    return;
  place(l->e, l->cap, (Entry){p, bytes});
  l->n++;
}

static void give_back(OsmosLedger *l, void *p)
{
  if (!l || !p || !l->n)
    return;
  const uint32_t m = l->cap - 1;
  for (uint32_t i = slot(p, l->cap); l->e[i].p; i = (i + 1) & m)
    if (l->e[i].p == p)
    {
      l->taken -= l->e[i].bytes;
      l->n--;
      uint32_t j = i; /* the run after it re-placed (deletion from open addressing) */
      for (;;)
      {
        l->e[j] = (Entry){NULL, 0};
        uint32_t k = j;
        for (;;)
        {
          k = (k + 1) & m;
          if (!l->e[k].p)
            return;
          const uint32_t h = slot(l->e[k].p, l->cap);
          if (j <= k ? (h <= j || h > k) : (h <= j && h > k))
            break;
        }
        l->e[j] = l->e[k];
        j = k;
      }
    }
}

bool ledger_reserve(OsmosLedger *l, size_t bytes) { return take(l, bytes); }
void ledger_unreserve(OsmosLedger *l, size_t bytes)
{
  if (l)
    l->taken -= bytes <= l->taken ? bytes : l->taken;
}
void ledger_keep(OsmosLedger *l, void *p, size_t bytes) { note(l, p, bytes); }
void ledger_release(OsmosLedger *l, void *p) { give_back(l, p); }

hipError_t ledger_malloc(OsmosLedger *l, void **p, size_t bytes)
{
  if (!take(l, bytes))
    return hipErrorOutOfMemory;
  const hipError_t r = hipMalloc(p, bytes);
  if (r != hipSuccess)
    ledger_unreserve(l, bytes);
  else
    note(l, *p, bytes);
  return r;
}

hipError_t ledger_host_malloc(OsmosLedger *l, void **p, size_t bytes, unsigned int flags)
{
  if (!take(l, bytes))
    return hipErrorOutOfMemory;
  const hipError_t r = hipHostMalloc(p, bytes, flags);
  if (r != hipSuccess)
    ledger_unreserve(l, bytes);
  else
    note(l, *p, bytes);
  return r;
}

hipError_t ledger_free(OsmosLedger *l, void *p)
{
  give_back(l, p);
  return hipFree(p);
}

hipError_t ledger_host_free(OsmosLedger *l, void *p)
{
  give_back(l, p);
  return hipHostFree(p);
}
