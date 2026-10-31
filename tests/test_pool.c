#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/mem/pool.h"

#define KB 1024u
#define MAX_ALLOCS 4096
#define MAX_READS 4096
#define MAX_EVENTS 200000

/* the memory the binary injects is HSA's; this one is malloc's, every allocation tracked: a free names a live one */
typedef struct
{
  uint8_t *at[MAX_ALLOCS];
  uint64_t bytes[MAX_ALLOCS];
  int n, allocs, frees;
  uint64_t total;
  int refuse; /* the next allocs fail, as a driver could */
} Mem;

static void *fake_alloc(uint64_t bytes, void *ctx)
{
  Mem *m = ctx;
  if (m->refuse)
    return NULL;
  assert(m->n < MAX_ALLOCS && bytes > 0 && bytes % POOL_ALIGN == 0);
  uint8_t *p = aligned_alloc(POOL_ALIGN, bytes);
  assert(p);
  memset(p, 0xEE, bytes); /* not the file's: a unit pinned before its read would show */
  m->at[m->n] = p, m->bytes[m->n] = bytes, m->n++, m->allocs++, m->total += bytes;
  return p;
}

static void fake_free(void *dev, void *ctx)
{
  Mem *m = ctx;
  for (int i = 0; i < m->n; i++)
    if (m->at[i] == dev)
    {
      m->total -= m->bytes[i];
      free(dev);
      m->n--, m->frees++;
      m->at[i] = m->at[m->n], m->bytes[i] = m->bytes[m->n];
      return;
    }
  assert(!"a free of memory never allocated");
}

/* reads are the test's: recorded with where they go, finished by hand (the file's bytes copied in, then
   pool_ready) */
typedef struct
{
  uint32_t unit;
  PoolRange part[POOL_PARTS];
  void *dst[POOL_PARTS];
  int n;
  bool urgent;
} Read;

typedef struct
{
  Read r[MAX_READS];
  int n;
  uint32_t hurried[MAX_READS]; /* the units hurry was told of, in order */
  int n_hurried;
} Reads;

static void fake_fetch(uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], bool urgent,
                       void *ctx)
{
  Reads *rs = ctx;
  assert(rs->n < MAX_READS && n >= 1 && n <= POOL_PARTS);
  Read *r = &rs->r[rs->n++];
  r->unit = unit, r->n = n, r->urgent = urgent;
  for (int k = 0; k < n; k++)
  {
    assert(parts[k].fd >= 0 && parts[k].bytes > 0 && dst[k] && (uintptr_t)dst[k] % POOL_ALIGN == 0);
    r->part[k] = parts[k], r->dst[k] = dst[k];
  }
}

static void fake_hurry(uint32_t unit, void *ctx)
{
  Reads *rs = ctx;
  assert(rs->n_hurried < MAX_READS);
  rs->hurried[rs->n_hurried++] = unit;
}

typedef struct
{
  PoolEvent ev[MAX_EVENTS];
  int n;
} Log;

static void on_event(const PoolEvent *ev, void *arg)
{
  Log *l = arg;
  assert(l->n < MAX_EVENTS);
  if (l->n)
    assert(ev->at_ns >= l->ev[l->n - 1].at_ns);
  l->ev[l->n++] = *ev;
}

static int count(const Log *l, PoolEventKind k, uint32_t unit)
{
  int c = 0;
  for (int i = 0; i < l->n; i++)
    c += l->ev[i].kind == k && l->ev[i].unit == unit;
  return c;
}

static char dir[] = "/tmp/test_pool_XXXXXX";

static uint8_t byte_at(uint64_t i) { return (uint8_t)((i * 13 + 5) & 255); }

/* a file of `bytes`, byte i = byte_at(i) */
static int file(const char *name, size_t bytes)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (size_t i = 0; i < bytes; i++)
    fputc(byte_at(i), f);
  fclose(f);
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0);
  return fd;
}

static Mem mem;
static Reads reads;
static Log events;

static Pool *make(const PoolUnit *u, size_t n, uint64_t budget, char *err)
{
  for (int i = 0; i < mem.n; i++)
    free(mem.at[i]);
  memset(&mem, 0, sizeof mem);
  memset(&reads, 0, sizeof reads);
  memset(&events, 0, sizeof events);
  return pool_create(u, n, budget, (Slots){fake_alloc, fake_free, &mem}, (Fetcher){fake_fetch, &reads, fake_hurry},
                     on_event, &events, err);
}

/* the k-th read begun ends: the unit must still be reading (its memory never taken from under its read); `ok` copies
   the file's bytes in first */
static uint32_t finish(Pool *p, int k, bool ok)
{
  const Read r = reads.r[k];
  reads.r[k] = reads.r[--reads.n];
  assert(pool_state(p, r.unit) == POOL_READING);
  for (int i = 0; ok && i < r.n; i++)
    assert(pread(r.part[i].fd, r.dst[i], r.part[i].bytes, (off_t)r.part[i].offset) == (ssize_t)r.part[i].bytes);
  pool_ready(p, r.unit, ok);
  return r.unit;
}

static int find_read(uint32_t unit)
{
  for (int k = 0; k < reads.n; k++)
    if (reads.r[k].unit == unit)
      return k;
  return -1;
}

static uint64_t unit_bytes(const PoolUnit *u)
{
  uint64_t b = 0;
  for (int k = 0; k < u->n_parts; k++)
    b += (u->part[k].bytes + POOL_ALIGN - 1) / POOL_ALIGN * POOL_ALIGN;
  return b;
}

/* what must hold after every call: the pool's bytes the allocator's live ones, within the budget; every pinned unit's
   parts its file's bytes; no two units in memory (pinned or reading) overlap */
static void check(const Pool *p, const PoolUnit *u, size_t n, uint64_t budget)
{
  assert(pool_bytes(p) == mem.total && mem.total <= budget);
  uintptr_t lo[MAX_ALLOCS], hi[MAX_ALLOCS];
  int live = 0;
  for (size_t i = 0; i < n; i++)
  {
    const void *dev[POOL_PARTS] = {0};
    const PoolState st = pool_state(p, (uint32_t)i);
    if (st == POOL_READING)
    {
      const int k = find_read((uint32_t)i);
      assert(k >= 0 && !pool_dev(p, (uint32_t)i, dev));
      lo[live] = (uintptr_t)reads.r[k].dst[0], hi[live++] = (uintptr_t)reads.r[k].dst[0] + unit_bytes(&u[i]);
      continue;
    }
    if (st != POOL_PINNED)
    {
      assert(!pool_dev(p, (uint32_t)i, dev) && find_read((uint32_t)i) < 0);
      continue;
    }
    assert(pool_dev(p, (uint32_t)i, dev));
    lo[live] = (uintptr_t)dev[0], hi[live++] = (uintptr_t)dev[0] + unit_bytes(&u[i]);
    for (int k = 0; k < u[i].n_parts; k++)
    {
      const uint8_t *b = dev[k];
      const uint64_t off = u[i].part[k].offset, bytes = u[i].part[k].bytes;
      assert((uintptr_t)b % POOL_ALIGN == 0);
      for (uint64_t j = 0; j < bytes; j += 1 + bytes / 7)
        assert(b[j] == byte_at(off + j));
      assert(b[bytes - 1] == byte_at(off + bytes - 1));
    }
  }
  for (int a = 0; a < live; a++)
    for (int b = a + 1; b < live; b++)
      assert(hi[a] <= lo[b] || hi[b] <= lo[a]);
}

static void test_refusals(int fd)
{
  char err[POOL_ERR];
  const PoolUnit none = {.n_parts = 0};
  const PoolUnit four = {.n_parts = 4};
  const PoolUnit zero = {.part = {{fd, 0, 0}}, .n_parts = 1};
  const PoolUnit past = {.part = {{fd, 1 << 20, 1}}, .n_parts = 1};
  const PoolUnit closed = {.part = {{9999, 0, 1}}, .n_parts = 1};
  const PoolUnit big = {.part = {{fd, 0, 64 * KB}}, .n_parts = 1};
  const PoolUnit *bad[] = {&none, &four, &zero, &past, &closed, &big};
  for (int i = 0; i < 6; i++)
  {
    err[0] = 0;
    assert(make(bad[i], 1, 16 * KB, err) == NULL && err[0]);
  }
  pool_destroy(NULL);
  printf("  refusals: no parts, four parts, 0 bytes, past the file, a closed file, larger than the budget\n");
}

static void test_cold_to_pinned(int fd)
{
  char err[POOL_ERR] = "";
  const PoolUnit u[] = {{.part = {{fd, 100, 3 * KB}, {fd, 9 * KB + 7, 50}}, .n_parts = 2},
                        {.part = {{fd, 20 * KB, KB}}, .n_parts = 1}};
  Pool *p = make(u, 2, 64 * KB, err);
  assert(p && pool_state(p, 0) == POOL_COLD && pool_bytes(p) == 0);
  const void *dev[POOL_PARTS] = {0};
  assert(pool_pin(p, 0, dev) == POOL_WAIT && dev[0] == NULL);
  assert(pool_state(p, 0) == POOL_READING && reads.n == 1 && reads.r[0].unit == 0 && reads.r[0].urgent);
  assert(reads.n_hurried == 0);
  assert(pool_pin(p, 0, dev) == POOL_WAIT && reads.n == 1); /* a second hold, the same read */
  assert(pool_bytes(p) == 3 * KB + POOL_ALIGN); /* 3 KiB, then 50 B from the next boundary */
  check(p, u, 2, 64 * KB);
  finish(p, 0, true);
  assert(pool_state(p, 0) == POOL_PINNED && count(&events, POOL_EV_PIN, 0) == 1);
  assert(pool_dev(p, 0, dev) && (const uint8_t *)dev[1] == (const uint8_t *)dev[0] + 3 * KB);
  check(p, u, 2, 64 * KB);
  assert(pool_pin(p, 0, dev) == POOL_OK); /* pinned: at once */
  for (int i = 0; i < 4; i++)
    pool_release(p, 0, 0); /* three holds, then none left: nothing */
  assert(pool_state(p, 0) == POOL_PINNED && mem.allocs == 1);
  pool_ready(p, 0, true); /* not reading: nothing */
  pool_ready(p, 1, true);
  assert(pool_state(p, 1) == POOL_COLD);
  pool_destroy(p);
  assert(mem.n == 0 && mem.total == 0 && count(&events, POOL_EV_UNPIN, 0) == 1);
  printf("  cold to pinned: WAIT, one read for two holds, PIN, the file's bytes at aligned parts; destroy frees all\n");
}

static void test_room(int fd)
{
  char err[POOL_ERR] = "";
  PoolUnit u[4];
  for (int i = 0; i < 4; i++)
    u[i] = (PoolUnit){.part = {{fd, (uint64_t)(10 * i) * KB, 4 * KB}}, .n_parts = 1};
  const uint64_t budget = 8 * KB; /* two units */
  Pool *p = make(u, 4, budget, err);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < 2; i++)
    assert(pool_pin(p, i, dev) == POOL_WAIT), finish(p, find_read(i), true);
  pool_release(p, 1, 0); /* 1 released first, then 0: 1 is the least recently released */
  pool_release(p, 0, 0);
  assert(pool_pin(p, 2, dev) == POOL_WAIT && pool_state(p, 1) == POOL_COLD && pool_state(p, 0) == POOL_PINNED);
  assert(count(&events, POOL_EV_UNPIN, 1) == 1 && mem.allocs == 2); /* 1's memory is 2's: no allocation */
  check(p, u, 4, budget);
  /* 2 reading, 0 not held: 3 takes 0's room */
  assert(pool_pin(p, 3, dev) == POOL_WAIT && pool_state(p, 0) == POOL_COLD && mem.allocs == 2);
  check(p, u, 4, budget);
  /* both reading: 0 does not fit - no hold, nothing changed */
  assert(pool_pin(p, 0, dev) == POOL_FULL && count(&events, POOL_EV_REFUSED, 0) == 1);
  assert(pool_state(p, 2) == POOL_READING && pool_state(p, 3) == POOL_READING && reads.n == 2);
  pool_release(p, 0, 0); /* the refused pin took no hold: nothing */
  finish(p, find_read(2), true), finish(p, find_read(3), true);
  check(p, u, 4, budget);
  /* pinned and held: still no room */
  assert(pool_pin(p, 0, dev) == POOL_FULL && pool_state(p, 2) == POOL_PINNED && pool_state(p, 3) == POOL_PINNED);
  pool_destroy(p);
  assert(mem.n == 0);
  printf("  room: the least recently released goes, its memory reused; a held or reading one never; FULL changes "
         "nothing\n");
}

/* units of two sizes: memory of the wrong size is kept spare, freed when room is wanted, reused when its size is */
static void test_sizes(int fd)
{
  char err[POOL_ERR] = "";
  const PoolUnit u[] = {{.part = {{fd, 0, 4 * KB}}, .n_parts = 1},
                        {.part = {{fd, 8 * KB, 4 * KB}}, .n_parts = 1},
                        {.part = {{fd, 16 * KB, 2 * KB}}, .n_parts = 1},
                        {.part = {{fd, 32 * KB, 6 * KB}}, .n_parts = 1}};
  const uint64_t budget = 10 * KB;
  Pool *p = make(u, 4, budget, err);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < 3; i++) /* 4 + 4 + 2: the budget's all */
    assert(pool_pin(p, i, dev) == POOL_WAIT), finish(p, find_read(i), true), pool_release(p, i, i);
  assert(mem.allocs == 3 && pool_bytes(p) == budget);
  /* 3 (6 KiB): furthest ahead first - 2 (2 KiB), then 1 (4 KiB), each freed (not its size): room for a new 6 */
  assert(pool_pin(p, 3, dev) == POOL_WAIT);
  assert(pool_state(p, 2) == POOL_COLD && pool_state(p, 1) == POOL_COLD && pool_state(p, 0) == POOL_PINNED);
  assert(mem.allocs == 4 && mem.frees == 2);
  finish(p, find_read(3), true);
  check(p, u, 4, budget);
  pool_release(p, 3, 9);
  /* 2 (2 KiB) back: room from 3 (furthest), freed - its size is not 2's */
  assert(pool_pin(p, 2, dev) == POOL_WAIT && pool_state(p, 3) == POOL_COLD && mem.frees == 3);
  finish(p, find_read(2), true);
  check(p, u, 4, budget);
  pool_destroy(p);
  assert(mem.n == 0);
  printf("  sizes: room from the furthest first whatever its size, memory of another size freed, of its size reused\n");
}

/* a failed read: cold again, its memory spare, its holds kept; a pin reads it again, into that memory */
static void test_failed(int fd)
{
  char err[POOL_ERR] = "";
  const PoolUnit u[] = {{.part = {{fd, 0, 4 * KB}}, .n_parts = 1}, {.part = {{fd, 4 * KB, 4 * KB}}, .n_parts = 1}};
  Pool *p = make(u, 2, 8 * KB, err);
  assert(p);
  const void *dev[POOL_PARTS];
  assert(pool_pin(p, 0, dev) == POOL_WAIT);
  finish(p, 0, false);
  assert(pool_state(p, 0) == POOL_COLD && count(&events, POOL_EV_FAILED, 0) == 1 && !pool_dev(p, 0, dev));
  assert(pool_bytes(p) == 4 * KB && mem.n == 1); /* spare */
  assert(pool_pin(p, 0, dev) == POOL_WAIT && mem.allocs == 1); /* read again, into the spare */
  finish(p, 0, true);
  check(p, u, 2, 8 * KB);
  pool_release(p, 0, 0), pool_release(p, 0, 0), pool_release(p, 0, 0); /* two holds, one of them the failed read's */
  /* the allocator refuses: FULL, REFUSED */
  mem.refuse = 1;
  assert(pool_pin(p, 1, dev) == POOL_FULL && count(&events, POOL_EV_REFUSED, 1) == 1 && pool_state(p, 1) == 0);
  mem.refuse = 0;
  assert(pool_pin(p, 1, dev) == POOL_WAIT);
  finish(p, 0, true);
  check(p, u, 2, 8 * KB);
  pool_destroy(p);
  assert(mem.n == 0);
  printf("  failed: a failed read cold again, its memory spare, read again into it; the allocator's refusal FULL\n");
}

/* the prefetch: a want reads a cold unit without a hold, taking room only from units wanted later than it */
static void test_want(int fd)
{
  char err[POOL_ERR] = "";
  PoolUnit u[4];
  for (int i = 0; i < 4; i++)
    u[i] = (PoolUnit){.part = {{fd, (uint64_t)(10 * i) * KB, 4 * KB}}, .n_parts = 1};
  const uint64_t budget = 8 * KB; /* two units */
  Pool *p = make(u, 4, budget, err);
  assert(p);
  const void *dev[POOL_PARTS];
  pool_want(p, 0, 5); /* room free: read, no hold - not urgent */
  assert(pool_state(p, 0) == POOL_READING && reads.n == 1 && !reads.r[0].urgent);
  pool_want(p, 0, 5); /* reading: no second read */
  assert(reads.n == 1 && reads.n_hurried == 0);
  assert(pool_pin(p, 0, dev) == POOL_WAIT); /* waited on while read ahead: hurried */
  assert(reads.n == 1 && reads.n_hurried == 1 && reads.hurried[0] == 0);
  finish(p, 0, true);
  pool_release(p, 0, 5);
  assert(pool_pin(p, 0, dev) == POOL_OK && reads.n_hurried == 1); /* the prefetched unit a hit: nothing hurried */
  pool_release(p, 0, 5);
  assert(pool_pin(p, 1, dev) == POOL_WAIT);
  finish(p, 0, true);
  pool_release(p, 1, 10);
  check(p, u, 4, budget);
  /* full: 0 wanted at 5, 1 at 10. a want at 12 takes no room (both wanted sooner); at 7 it takes 1's */
  pool_want(p, 2, 12);
  assert(pool_state(p, 2) == POOL_COLD && reads.n == 0 && pool_state(p, 1) == POOL_PINNED);
  pool_want(p, 2, 7);
  assert(pool_state(p, 2) == POOL_READING && pool_state(p, 1) == POOL_COLD && pool_state(p, 0) == POOL_PINNED);
  finish(p, 0, true);
  check(p, u, 4, budget);
  /* a want on a pinned unit corrects its time: 0 now wanted at 20, so a want at 9 takes 0's room, not 2's (at 7) */
  pool_want(p, 0, 20);
  pool_want(p, 3, 9);
  assert(pool_state(p, 3) == POOL_READING && pool_state(p, 0) == POOL_COLD && pool_state(p, 2) == POOL_PINNED);
  finish(p, 0, true);
  /* a held unit: nothing - its release says when */
  assert(pool_pin(p, 2, dev) == POOL_OK);
  pool_want(p, 2, 99);
  pool_want(p, 1, 10); /* no room: 2 held, 3 wanted at 9, sooner than 10 */
  assert(pool_state(p, 1) == POOL_COLD);
  check(p, u, 4, budget);
  pool_destroy(p);
  assert(mem.n == 0);
  printf("  want: a prefetch reads without a hold, takes room only from units wanted later, corrects a pinned one's "
         "time\n");
}

/* units held in turn, a round after round (the forward's layers), each released as the next is held, wanted again a
   round on: the one released longest ago is the one wanted soonest */
static int cycle(int fd, bool ahead)
{
  enum { N = 6, ROUNDS = 4 };
  char err[POOL_ERR] = "";
  PoolUnit u[N];
  for (int i = 0; i < N; i++)
    u[i] = (PoolUnit){.part = {{fd, (uint64_t)(10 * i) * KB, 4 * KB}}, .n_parts = 1};
  const uint64_t budget = 4 * 4 * KB; /* four of the six */
  Pool *p = make(u, N, budget, err);
  assert(p);
  const void *dev[POOL_PARTS];
  int hits = 0;
  for (uint64_t visit = 0; visit < (uint64_t)N * ROUNDS; visit++)
  {
    const uint32_t i = (uint32_t)(visit % N);
    hits += visit >= N && pool_state(p, i) == POOL_PINNED;
    if (pool_pin(p, i, dev) == POOL_WAIT)
      finish(p, find_read(i), true);
    check(p, u, N, budget);
    if (visit) /* the one before: wanted again a round after its visit (or, not ahead, all alike: the oldest goes) */
      pool_release(p, (i + N - 1) % N, ahead ? visit - 1 + N : 0);
  }
  assert(mem.allocs == 4); /* every unit the same size: room is always memory reused */
  pool_destroy(p);
  assert(mem.n == 0);
  return hits;
}

static void test_cycle(int fd)
{
  const int lru = cycle(fd, false), ahead = cycle(fd, true);
  /* least recently released: each unit is gone by its turn, every round. furthest ahead: a round keeps all but the
     two the budget cannot - the held one's room and the one taken for it */
  assert(lru == 0);
  assert(ahead >= 3 * (4 - 2));
  printf("  a cycle of 6 units, room for 4, 3 rounds after the first: %d hits least recently released, %d furthest "
         "ahead\n", lru, ahead);
}

/* property: random calls on units of random unaligned ranges in two files, reads ending in any order, some failing,
   the allocator refusing now and then; the invariants after every one */
static void test_random(int fd_a, int fd_b)
{
  enum { N = 50, STEPS = 20000 };
  PoolUnit u[N];
  uint64_t s = 0x2545f4914f6cdd1dull;
#define RND() (s ^= s << 13, s ^= s >> 7, s ^= s << 17, s)
  for (int i = 0; i < N; i++)
  {
    u[i].n_parts = 1 + (int)(RND() % 3);
    for (int k = 0; k < u[i].n_parts; k++)
    {
      const int fd = RND() % 2 ? fd_a : fd_b;
      const uint64_t bytes = 1 + RND() % (6 * KB);
      u[i].part[k] = (PoolRange){fd, RND() % ((1 << 20) - bytes), bytes};
    }
  }
  char err[POOL_ERR] = "";
  const uint64_t budget = 120 * KB;
  Pool *p = make(u, N, budget, err);
  if (!p)
    fprintf(stderr, "  pool_create: %s\n", err);
  assert(p);
  int holds[N] = {0};
  int ok = 0, wait = 0, full = 0, failed = 0;
  for (int step = 0; step < STEPS; step++)
  {
    const uint32_t i = (uint32_t)(RND() % N);
    const void *dev[POOL_PARTS];
    mem.refuse = RND() % 50 == 0;
    switch (RND() % 7) /* pins, releases and reads' ends alike, and a want now and then */
    {
    case 6:
      pool_want(p, i, RND() % 4);
      break;
    case 0:
    case 1:
    {
      const PoolPin r = pool_pin(p, i, dev);
      ok += r == POOL_OK, wait += r == POOL_WAIT, full += r == POOL_FULL;
      if (r != POOL_FULL)
        holds[i]++;
      break;
    }
    case 2:
    case 3:
      if (holds[i])
      {
        pool_release(p, i, RND() % 4); /* wanted again at any of a few times: ties and not */
        holds[i]--;
      }
      break;
    default:
      if (reads.n) /* a read ends, any of those begun; one in ten fails */
      {
        const bool good = RND() % 10 != 0;
        failed += !good;
        finish(p, (int)(RND() % (uint64_t)reads.n), good);
      }
    }
    mem.refuse = 0;
    check(p, u, N, budget);
  }
  pool_destroy(p);
  assert(mem.n == 0 && mem.total == 0);
  printf("  random: %d calls (%d OK, %d WAIT, %d FULL, %d reads failed), every invariant after each; all freed\n", STEPS,
         ok, wait, full, failed);
}

int main(void)
{
  printf("[test_pool] start\n");
  assert(mkdtemp(dir));
  const int a = file("a.bin", 1 << 20), b = file("b.bin", 1 << 20);
  test_refusals(a);
  test_cold_to_pinned(a);
  test_room(a);
  test_sizes(a);
  test_failed(a);
  test_want(a);
  test_cycle(a);
  test_random(a, b);
  close(a);
  close(b);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_pool] ok\n");
  return 0;
}
