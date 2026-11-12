#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/mem/pool.h"
#include "osmos/mem/reads.h"

#define UNIT 4096
#define MAX_EVENTS 8192

static char dir[] = "/tmp/test_reads_XXXXXX";

typedef struct
{
  PoolEvent ev[MAX_EVENTS];
  int n;
} Log;

static void on_event(const PoolEvent *ev, void *arg)
{
  Log *log = arg;
  assert(log->n < MAX_EVENTS);
  log->ev[log->n++] = *ev;
}

static void *fake_alloc(uint64_t bytes, void *ctx) { return (void)ctx, aligned_alloc(POOL_ALIGN, bytes); }
static void fake_free(void *dev, void *ctx) { (void)ctx, free(dev); }

static int file(const char *name, uint32_t units)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (size_t i = 0; i < (size_t)units * UNIT; i++)
    fputc((int)(i & 255), f);
  fclose(f);
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0);
  return fd;
}

static PoolUnit *units_of(int fd, uint32_t n)
{
  PoolUnit *u = calloc(n, sizeof *u);
  assert(u);
  for (uint32_t i = 0; i < n; i++)
    u[i] = (PoolUnit){.part = {{fd, (uint64_t)i * UNIT, UNIT}}, .n_parts = 1};
  return u;
}

/* the consumer's wait for a unit: pumped until it is no longer reading */
static void wait_for(Reads *r, Pool *p, uint32_t unit)
{
  char err[READS_ERR];
  while (pool_state(p, unit) == POOL_READING)
    assert(reads_pump(r, (Pool *[READS_POOLS]){p}, true, err));
}

static void check_bytes(Pool *p, uint32_t unit)
{
  const void *dev[POOL_PARTS];
  assert(pool_dev(p, unit, dev));
  const uint8_t *b = dev[0];
  for (uint32_t j = 0; j < UNIT; j++)
    assert(b[j] == (uint8_t)(unit * UNIT + j)); /* the file's bytes, read straight into the unit's memory */
}

static void test_refusals(int fd)
{
  char err[READS_ERR];
  assert(!reads_create(0, 4, err) && err[0]);
  assert(!reads_create(8, 0, err) && !reads_create(8, 65, err));
  reads_destroy(NULL);
  assert(!reads_pump(NULL, NULL, false, err));
  assert(!reads_fetcher(NULL, -1).fetch && !reads_fetcher(NULL, READS_POOLS).fetch);
  (void)fd;
  printf("  refusals: no units, 0 or 65 readers\n");
}

/* many reads on many readers, pins among the reads ahead: every one pinned once with its bytes */
static void test_many(int fd, uint32_t n, int threads)
{
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, n);
  Reads *r = reads_create(n, threads, err);
  assert(r);
  Log log = {0};
  Pool *p = pool_create(u, n, (uint64_t)n * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, on_event,
                        &log, perr);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < n; i++)
  {
    if (i % 3)
      pool_want(p, i, n - i); /* needs out of the asked order */
    else
      assert(pool_pin(p, i, dev) == POOL_WAIT);
    assert(reads_out(r) <= (int)i + 1);
  }
  int pinned = 0;
  while (pinned < (int)n)
  {
    assert(reads_pump(r, (Pool *[READS_POOLS]){p}, true, err));
    pinned = 0;
    for (int i = 0; i < log.n; i++)
      pinned += log.ev[i].kind == POOL_EV_PIN;
  }
  for (uint32_t i = 0; i < n; i++)
    assert(pool_state(p, i) == POOL_PINNED), check_bytes(p, i);
  assert(pinned == (int)n && reads_out(r) == 0);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  many: %u reads on %d readers - each pinned once, its bytes\n", n, threads);
}

/* a lone read (a decode's miss: none other out or waiting) split across the readers; reads asked together (a prompt's
   next layer) whole, one reader each - the file's readahead kept */
static void test_split(int fd)
{
  enum { U = 16 };
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, U);
  Reads *r = reads_create(U, 4, err);
  assert(r);
  Pool *p = pool_create(u, U, (uint64_t)U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, NULL, NULL,
                        perr);
  assert(p);
  const void *dev[POOL_PARTS];
  assert(pool_pin(p, 0, dev) == POOL_WAIT);
  wait_for(r, p, 0);
  assert(reads_split(r) == 1); /* alone: split */
  for (uint32_t i = 1; i < U; i++)
    pool_want(p, i, i);
  for (uint32_t i = 1; i < U; i++)
    wait_for(r, p, i), check_bytes(p, i);
  assert(reads_split(r) == 2); /* of the 15 asked together only the first, given out before the rest were asked */
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  split: a lone read across the readers, reads asked together whole\n");
}

/* a pin (a miss the forward waits on: need 0) split across the readers unless another pin is out, among reads ahead
   read whole; a read ahead then pinned (its need moved to 0) completes as the rest - each pinned once with its bytes
   (the order the readers take them: test_fetchq) */
static void test_pins(int fd)
{
  enum { U = 12 };
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, U);
  Reads *r = reads_create(U, 1, err);
  assert(r);
  Pool *p = pool_create(u, U, (uint64_t)U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, NULL, NULL,
                        perr);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < 6; i++)
    pool_want(p, i, 10 + i);
  const uint64_t split0 = reads_split(r);
  assert(pool_pin(p, 8, dev) == POOL_WAIT);
  assert(reads_split(r) == split0 + 1); /* the first pin out: split */
  assert(pool_pin(p, 9, dev) == POOL_WAIT);
  assert(reads_split(r) == split0 + 1); /* another pin out: whole */
  (void)pool_pin(p, 5, dev);            /* a read ahead, now waited on */
  const uint32_t asked[] = {0, 1, 2, 3, 4, 5, 8, 9};
  for (size_t i = 0; i < sizeof asked / sizeof *asked; i++)
    wait_for(r, p, asked[i]), check_bytes(p, asked[i]);
  assert(reads_out(r) == 0);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  pins: split unless another pin is out; a read ahead pinned completes with the rest\n");
}

/* a read that fails (its file gone from under it) ends FAILED, cold again - not lost, nothing waits forever */
static void test_failure(void)
{
  char err[READS_ERR], perr[POOL_ERR];
  const int fd = file("gone.bin", 2);
  PoolUnit *u = units_of(fd, 2);
  Reads *r = reads_create(2, 2, err);
  assert(r);
  Log log = {0};
  Pool *p = pool_create(u, 2, 2 * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, on_event, &log, perr);
  assert(p);
  close(fd);
  const void *dev[POOL_PARTS];
  assert(pool_pin(p, 1, dev) == POOL_WAIT);
  wait_for(r, p, 1);
  assert(pool_state(p, 1) == POOL_COLD && log.ev[log.n - 1].kind == POOL_EV_FAILED);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  failure: a read whose file is gone ends FAILED, the unit cold again\n");
}

/* destroyed with reads out and waiting: the readers stopped, the waiting dropped - no end runs after (asan: none
   leaks) */
static void test_destroy_busy(int fd)
{
  enum { U = 400 };
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, U);
  Reads *r = reads_create(U, 2, err);
  assert(r);
  Pool *p = pool_create(u, U, (uint64_t)U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, NULL, NULL,
                        perr);
  assert(p);
  for (uint32_t i = 0; i < U; i++)
    pool_want(p, i, i);
  assert(reads_out(r) > 0);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  destroy busy: %d reads asked, the readers stopped and the waiting dropped\n", U);
}

/* two pools on one Reads, as the forward's membrane and stream: each read's end its own pool's, the bytes its file's;
   a unit out for one, fetched for the other, stops the reads (the forward never reads one unit into both); an end
   for a pool the pump is not given fails it closed */
static void test_two_pools(int fd)
{
  enum { U = 8 };
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, U);
  const void *dev[POOL_PARTS];
  Reads *r = reads_create(U, 2, err);
  assert(r);
  Pool *a = pool_create(u, U, U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, NULL, NULL, perr);
  Pool *b = pool_create(u, U, U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 1), 0, NULL, NULL, perr);
  assert(a && b);
  Pool *both[READS_POOLS] = {a, b};
  assert(pool_pin(a, 1, dev) == POOL_WAIT && pool_pin(b, 2, dev) == POOL_WAIT);
  for (int i = 0; i < 2000 && (pool_state(a, 1) == POOL_READING || pool_state(b, 2) == POOL_READING); i++)
    assert(reads_pump(r, both, false, err)), usleep(1000); /* bounded: an end gone to the wrong pool never comes */
  assert(pool_state(a, 1) == POOL_PINNED && pool_state(b, 2) == POOL_PINNED);
  assert(pool_state(a, 2) == POOL_COLD && pool_state(b, 1) == POOL_COLD);
  check_bytes(a, 1), check_bytes(b, 2);
  pool_release(a, 1, 0), pool_release(b, 2, 0);
  /* an end for b while the pump is given a alone: closed */
  assert(pool_pin(b, 3, dev) == POOL_WAIT);
  assert(!reads_pump(r, (Pool *[READS_POOLS]){a}, true, err) && err[0]);
  reads_destroy(r);
  pool_destroy(a), pool_destroy(b);
  /* one unit out for both: closed */
  r = reads_create(U, 2, err);
  a = pool_create(u, U, U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 0), 0, NULL, NULL, perr);
  b = pool_create(u, U, U * UNIT, (Slots){fake_alloc, fake_free, NULL, NULL, NULL}, reads_fetcher(r, 1), 0, NULL, NULL, perr);
  assert(r && a && b);
  Pool *again[READS_POOLS] = {a, b};
  assert(pool_pin(a, 4, dev) == POOL_WAIT && pool_pin(b, 4, dev) == POOL_WAIT);
  assert(!reads_pump(r, again, false, err));
  reads_destroy(r);
  pool_destroy(a), pool_destroy(b);
  free(u);
  printf("  two pools: each read's end its own pool's; one unit out for both, or an end for a pool not given, fails\n");
}

int main(void)
{
  printf("[test_reads] start\n");
  assert(mkdtemp(dir));
  enum { N = 2000 };
  const int fd = file("f.bin", N);
  test_refusals(fd);
  test_many(fd, 200, 1);
  test_many(fd, N, 8);
  test_split(fd);
  test_pins(fd);
  test_failure();
  test_destroy_busy(fd);
  test_two_pools(fd);
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_reads] ok\n");
  return 0;
}
