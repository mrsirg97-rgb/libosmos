#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/mem/pool.h"

#define PAGE 4096u
#define MAX_REGS 4096
#define MAX_EVENTS 200000

/* the pinner the binary injects is HIP's; this one behaves as its rules do on this APU: page-aligned ranges only, a
   range may not overlap one registered already, an unpin names a registration's start */
typedef struct
{
  uintptr_t start[MAX_REGS];
  size_t bytes[MAX_REGS];
  int n;
  uint64_t total;
  int refuse; /* the next pins fail, as a driver could */
} Regs;

static bool fake_pin(void *host, size_t bytes, void *ctx)
{
  Regs *r = ctx;
  const uintptr_t h = (uintptr_t)host;
  assert(h % PAGE == 0 && bytes % PAGE == 0 && bytes > 0);
  if (r->refuse)
    return false;
  for (int i = 0; i < r->n; i++)
    assert(h + bytes <= r->start[i] || h >= r->start[i] + r->bytes[i]); /* HIP refuses overlap: never asked */
  assert(r->n < MAX_REGS);
  r->start[r->n] = h;
  r->bytes[r->n] = bytes;
  r->n++;
  r->total += bytes;
  return true;
}

static void fake_unpin(void *host, void *ctx)
{
  Regs *r = ctx;
  for (int i = 0; i < r->n; i++)
    if (r->start[i] == (uintptr_t)host)
    {
      r->total -= r->bytes[i];
      r->start[i] = r->start[r->n - 1];
      r->bytes[i] = r->bytes[r->n - 1];
      r->n--;
      return;
    }
  assert(!"unpin of a range never pinned");
}

static bool covered(const Regs *r, uintptr_t a, uintptr_t b)
{
  for (uintptr_t p = a / PAGE * PAGE; p < b; p += PAGE)
  {
    bool in = false;
    for (int i = 0; i < r->n && !in; i++)
      in = p >= r->start[i] && p < r->start[i] + r->bytes[i];
    if (!in)
      return false;
  }
  return true;
}

/* reads are the test's: recorded, finished by hand with pool_ready */
typedef struct
{
  uint32_t unit[MAX_REGS];
  int n;
} Reads;

static void fake_fetch(uint32_t unit, const PoolRange *parts, int n, void *ctx)
{
  Reads *r = ctx;
  for (int i = 0; i < n; i++)
    assert(parts[i].fd >= 0 && parts[i].bytes > 0);
  assert(r->n < MAX_REGS);
  r->unit[r->n++] = unit;
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

/* a file of `bytes`, byte i = (i * 13 + 5) & 255 */
static int file(const char *name, size_t bytes)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (size_t i = 0; i < bytes; i++)
    fputc((int)((i * 13 + 5) & 255), f);
  fclose(f);
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0);
  return fd;
}

static Regs regs;
static Reads reads;
static Log events;

static Pool *make(const PoolUnit *u, size_t n, uint64_t budget, char *err)
{
  memset(&regs, 0, sizeof regs);
  memset(&reads, 0, sizeof reads);
  memset(&events, 0, sizeof events);
  return pool_create(u, n, budget, (Pinner){fake_pin, fake_unpin, &regs}, (Fetcher){fake_fetch, &reads}, on_event,
                     &events, err);
}

/* what must hold after every call: registrations disjoint (the fake asserts it), every pinned unit's pages
   registered, the pinned bytes the registrations' and within the budget */
static void check(const Pool *p, const PoolUnit *u, size_t n, uint64_t budget)
{
  assert(pool_pinned(p) == regs.total && regs.total <= budget);
  for (size_t i = 0; i < n; i++)
  {
    const void *dev[POOL_PARTS] = {0};
    if (pool_state(p, (uint32_t)i) != POOL_PINNED)
    {
      assert(!pool_dev(p, (uint32_t)i, dev));
      continue;
    }
    assert(pool_dev(p, (uint32_t)i, dev));
    for (int k = 0; k < u[i].n_parts; k++)
    {
      const uintptr_t a = (uintptr_t)dev[k];
      assert(covered(&regs, a, a + u[i].part[k].bytes));
      const uint8_t *b = dev[k];
      const uint64_t off = u[i].part[k].offset;
      assert(b[0] == (uint8_t)((off * 13 + 5) & 255)); /* the pointer is the file's own bytes */
      assert(b[u[i].part[k].bytes - 1] == (uint8_t)(((off + u[i].part[k].bytes - 1) * 13 + 5) & 255));
    }
  }
}

static void test_refusals(int fd)
{
  char err[POOL_ERR];
  const PoolUnit none = {.n_parts = 0};
  const PoolUnit four = {.n_parts = 4};
  const PoolUnit zero = {.part = {{fd, 0, 0}}, .n_parts = 1};
  const PoolUnit past = {.part = {{fd, 1 << 20, 1}}, .n_parts = 1};
  const PoolUnit big = {.part = {{fd, 0, 64 * PAGE}}, .n_parts = 1};
  const PoolUnit *bad[] = {&none, &four, &zero, &past, &big};
  for (int i = 0; i < 5; i++)
  {
    err[0] = 0;
    assert(make(bad[i], 1, 16 * PAGE, err) == NULL && err[0]);
  }
  pool_destroy(NULL);
  printf("  refusals: no parts, four parts, 0 bytes, past the file, larger than the budget\n");
}

static void test_cold_to_pinned(int fd)
{
  char err[POOL_ERR] = "";
  const PoolUnit u[] = {{.part = {{fd, 100, 3 * PAGE}, {fd, 9 * PAGE + 7, 50}}, .n_parts = 2},
                        {.part = {{fd, 20 * PAGE, PAGE}}, .n_parts = 1}};
  Pool *p = make(u, 2, 64 * PAGE, err);
  assert(p);
  assert(pool_state(p, 0) == POOL_COLD);
  const void *dev[POOL_PARTS] = {0};
  assert(pool_pin(p, 0, dev) == POOL_WAIT && dev[0] == NULL);
  assert(pool_state(p, 0) == POOL_READING && reads.n == 1 && reads.unit[0] == 0);
  assert(pool_pin(p, 0, dev) == POOL_WAIT && reads.n == 1); /* a second hold, the same read */
  check(p, u, 2, 64 * PAGE);
  pool_ready(p, 0);
  assert(pool_state(p, 0) == POOL_PINNED && count(&events, POOL_EV_PIN, 0) == 1);
  assert(pool_dev(p, 0, dev) && dev[0] && dev[1]);
  check(p, u, 2, 64 * PAGE);
  assert(pool_pinned(p) == 5 * PAGE); /* [0, 4) pages and page 9, whole pages */
  pool_release(p, 0);
  pool_release(p, 0);
  pool_release(p, 0); /* no hold left: nothing */
  assert(pool_state(p, 0) == POOL_PINNED);
  /* want: a cold unit's read once; read, it is pinned at once without a read */
  const uint32_t both[] = {0, 1};
  pool_want(p, both, 2);
  pool_want(p, both, 2);
  assert(reads.n == 2 && reads.unit[1] == 1 && count(&events, POOL_EV_READ, 1) == 1);
  pool_ready(p, 1);
  assert(pool_state(p, 1) == POOL_READ && count(&events, POOL_EV_READY, 1) == 1);
  assert(pool_pin(p, 1, dev) == POOL_OK && dev[0] && reads.n == 2);
  check(p, u, 2, 64 * PAGE);
  pool_destroy(p);
  assert(regs.n == 0 && regs.total == 0);
  assert(count(&events, POOL_EV_UNPIN, 0) == 1 && count(&events, POOL_EV_UNPIN, 1) == 1);
  printf("  cold to pinned: WAIT, one read for two holds, PIN, the file's own bytes; READ then OK; destroy unpins\n");
}

static void test_room(int fd)
{
  char err[POOL_ERR] = "";
  PoolUnit u[4];
  for (int i = 0; i < 4; i++)
    u[i] = (PoolUnit){.part = {{fd, (uint64_t)(10 * i) * PAGE, 4 * PAGE}}, .n_parts = 1};
  const uint64_t budget = 8 * PAGE; /* two units */
  Pool *p = make(u, 4, budget, err);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < 4; i++)
    pool_ready(p, i), pool_want(p, &i, 1), pool_ready(p, i);
  assert(pool_pin(p, 0, dev) == POOL_OK && pool_pin(p, 1, dev) == POOL_OK);
  pool_release(p, 1); /* 1 released first, then 0: 1 is the least recently released */
  pool_release(p, 0);
  assert(pool_pin(p, 2, dev) == POOL_OK);
  assert(pool_state(p, 1) == POOL_READ && pool_state(p, 0) == POOL_PINNED && count(&events, POOL_EV_UNPIN, 1) == 1);
  check(p, u, 4, budget);
  /* 2 held, 0 not: 3 takes 0's room */
  assert(pool_pin(p, 3, dev) == POOL_OK && pool_state(p, 0) == POOL_READ);
  check(p, u, 4, budget);
  /* both held: 0 does not fit - no hold, nothing changed */
  const int regs_before = regs.n;
  assert(pool_pin(p, 0, dev) == POOL_FULL && count(&events, POOL_EV_REFUSED, 0) == 1);
  assert(regs.n == regs_before && pool_state(p, 2) == POOL_PINNED && pool_state(p, 3) == POOL_PINNED);
  pool_release(p, 0); /* the refused pin took no hold: nothing */
  check(p, u, 4, budget);
  pool_destroy(p);
  assert(regs.n == 0);
  printf("  room: the least recently released goes first; a held unit never; FULL changes nothing\n");
}

static void test_shared_pages(int fd)
{
  char err[POOL_ERR] = "";
  /* three experts back to back, none page-aligned: each pair shares a page; a fourth elsewhere, 2 pages */
  const uint64_t s = 3 * PAGE + 1000;
  const PoolUnit u[] = {{.part = {{fd, 500, s}}, .n_parts = 1},
                        {.part = {{fd, 500 + s, s}}, .n_parts = 1},
                        {.part = {{fd, 500 + 2 * s, s}}, .n_parts = 1},
                        {.part = {{fd, 100 * PAGE, 2 * PAGE}}, .n_parts = 1}};
  const uint64_t span = (500 + 3 * s + PAGE - 1) / PAGE * PAGE; /* the three's pages, each once */
  const uint64_t budget = span + PAGE;
  Pool *p = make(u, 4, budget, err);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < 4; i++)
    pool_want(p, &i, 1), pool_ready(p, i);
  for (uint32_t i = 0; i < 3; i++)
  {
    assert(pool_pin(p, i, dev) == POOL_OK);
    check(p, u, 4, budget);
  }
  assert(pool_pinned(p) == span);
  /* the middle one released: its registration also holds a page its held neighbour reads - no room from it */
  pool_release(p, 1);
  const int regs_before = regs.n;
  assert(pool_pin(p, 3, dev) == POOL_FULL && regs.n == regs_before && pool_state(p, 1) == POOL_PINNED);
  check(p, u, 4, budget);
  /* the first released too: the middle goes first (frees nothing its neighbour reads), then the first - room */
  pool_release(p, 0);
  assert(pool_pin(p, 3, dev) == POOL_OK);
  assert(pool_state(p, 0) == POOL_READ && pool_state(p, 1) == POOL_READ && pool_state(p, 2) == POOL_PINNED);
  check(p, u, 4, budget);
  pool_destroy(p);
  assert(regs.n == 0);
  printf("  shared pages: a page registered once, never unregistered under a unit that reads it; FULL then room\n");
}

/* property: random calls on units of random unaligned ranges in two files; the invariants after every one */
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
      const uint64_t bytes = 1 + RND() % (6 * PAGE);
      u[i].part[k] = (PoolRange){fd, RND() % ((1 << 20) - bytes), bytes};
    }
  }
  char err[POOL_ERR] = "";
  const uint64_t budget = 120 * PAGE;
  Pool *p = make(u, N, budget, err);
  if (!p)
    fprintf(stderr, "  pool_create: %s\n", err);
  assert(p);
  int holds[N] = {0};
  int ok = 0, wait = 0, full = 0;
  for (int step = 0; step < STEPS; step++)
  {
    const uint32_t i = (uint32_t)(RND() % N);
    const void *dev[POOL_PARTS];
    switch (RND() % 5)
    {
    case 0:
      pool_want(p, &i, 1);
      break;
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
        pool_release(p, i);
        holds[i]--;
      }
      break;
    default:
      if (reads.n) /* a read ends, any of those begun */
      {
        const int k = (int)(RND() % (uint64_t)reads.n);
        const uint32_t done = reads.unit[k];
        reads.unit[k] = reads.unit[--reads.n];
        pool_ready(p, done);
      }
    }
    check(p, u, N, budget);
  }
  pool_destroy(p);
  assert(regs.n == 0 && regs.total == 0);
  printf("  random: %d calls (%d OK, %d WAIT, %d FULL), every invariant after each; nothing left registered\n", STEPS,
         ok, wait, full);
}

int main(void)
{
  printf("[test_pool] start\n");
  assert(mkdtemp(dir));
  const int a = file("a.bin", 1 << 20), b = file("b.bin", 1 << 20);
  test_refusals(a);
  test_cold_to_pinned(a);
  test_room(a);
  test_shared_pages(a);
  test_random(a, b);
  close(a);
  close(b);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_pool] ok\n");
  return 0;
}
