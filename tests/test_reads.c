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
    assert(reads_pump(r, p, true, err));
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
  assert(!reads_create(0, 4, 4, err) && err[0]);
  assert(!reads_create(8, 0, 4, err) && !reads_create(8, 65, 4, err));
  assert(!reads_create(8, 4, 0, err));
  reads_destroy(NULL);
  assert(!reads_pump(NULL, NULL, false, err));
  (void)fd;
  printf("  refusals: no units, 0 or 65 readers, nothing in flight\n");
}

/* one read out at a time: the next given out is the one needed soonest, ties in the order asked, a need corrected
   while waiting (a want's next, a pin's now) the need then - whatever the readers' timing */
static void test_order(int fd)
{
  enum { U = 8 };
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, U);
  Reads *r = reads_create(U, 1, 1, err);
  assert(r);
  Log log = {0};
  Pool *p = pool_create(u, U, (uint64_t)U * UNIT, (Slots){fake_alloc, fake_free, NULL}, reads_fetcher(r), on_event,
                        &log, perr);
  assert(p);
  const void *dev[POOL_PARTS];
  pool_want(p, 0, 10); /* the slot free: given out at once */
  assert(reads_inflight(r) == 1 && reads_waiting(r) == 0);
  pool_want(p, 1, 6);
  pool_want(p, 2, 1);
  pool_want(p, 3, 4);
  pool_want(p, 4, 1);              /* the same need as 2, asked after it */
  pool_want(p, 6, 8);
  pool_want(p, 6, 2);              /* corrected: needed sooner */
  pool_want(p, 7, 3);
  pool_want(p, 7, 9);              /* corrected: needed later */
  assert(pool_pin(p, 5, dev) == POOL_WAIT); /* needed now */
  assert(reads_inflight(r) == 1 && reads_waiting(r) == U - 1);
  for (uint32_t i = 0; i < U; i++)
    wait_for(r, p, i);
  const uint32_t want[U] = {0, 5, 2, 4, 6, 3, 1, 7};
  int k = 0;
  for (int i = 0; i < log.n; i++)
    if (log.ev[i].kind == POOL_EV_PIN)
      assert(log.ev[i].unit == want[k++]);
  assert(k == U && reads_inflight(r) == 0 && reads_waiting(r) == 0);
  assert(reads_split(r) == 2); /* the first (given out alone) and the last (none left waiting); the rest went whole */
  for (uint32_t i = 0; i < U; i++)
    check_bytes(p, i);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  order: one out at a time - by need, ties as asked, a need corrected while waiting the need then\n");
}

/* many reads on many readers: never more than `inflight` out, every one pinned once with its bytes */
static void test_many(int fd, uint32_t n, int threads, int inflight)
{
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, n);
  Reads *r = reads_create(n, threads, inflight, err);
  assert(r);
  Log log = {0};
  Pool *p = pool_create(u, n, (uint64_t)n * UNIT, (Slots){fake_alloc, fake_free, NULL}, reads_fetcher(r), on_event,
                        &log, perr);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < n; i++)
  {
    if (i % 3)
      pool_want(p, i, n - i); /* needs out of the asked order */
    else
      assert(pool_pin(p, i, dev) == POOL_WAIT);
    assert(reads_inflight(r) <= inflight);
  }
  int pinned = 0;
  while (pinned < (int)n)
  {
    assert(reads_pump(r, p, true, err));
    assert(reads_inflight(r) <= inflight);
    pinned = 0;
    for (int i = 0; i < log.n; i++)
      pinned += log.ev[i].kind == POOL_EV_PIN;
  }
  for (uint32_t i = 0; i < n; i++)
    assert(pool_state(p, i) == POOL_PINNED), check_bytes(p, i);
  assert(pinned == (int)n && reads_inflight(r) == 0 && reads_waiting(r) == 0);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  many: %u reads on %d readers, at most %d out - each pinned once, its bytes\n", n, threads, inflight);
}

/* a lone read (a decode's miss: none other out or waiting) split across the readers; reads asked together (a prompt's
   next layer) whole, one reader each - the file's readahead kept */
static void test_split(int fd)
{
  enum { U = 16 };
  char err[READS_ERR], perr[POOL_ERR];
  PoolUnit *u = units_of(fd, U);
  Reads *r = reads_create(U, 4, 4, err);
  assert(r);
  Pool *p = pool_create(u, U, (uint64_t)U * UNIT, (Slots){fake_alloc, fake_free, NULL}, reads_fetcher(r), NULL, NULL,
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

/* a read that fails (its file gone from under it) ends FAILED, cold again - not lost, nothing waits forever */
static void test_failure(void)
{
  char err[READS_ERR], perr[POOL_ERR];
  const int fd = file("gone.bin", 2);
  PoolUnit *u = units_of(fd, 2);
  Reads *r = reads_create(2, 2, 2, err);
  assert(r);
  Log log = {0};
  Pool *p = pool_create(u, 2, 2 * UNIT, (Slots){fake_alloc, fake_free, NULL}, reads_fetcher(r), on_event, &log, perr);
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
  Reads *r = reads_create(U, 2, 4, err);
  assert(r);
  Pool *p = pool_create(u, U, (uint64_t)U * UNIT, (Slots){fake_alloc, fake_free, NULL}, reads_fetcher(r), NULL, NULL,
                        perr);
  assert(p);
  for (uint32_t i = 0; i < U; i++)
    pool_want(p, i, i);
  assert(reads_waiting(r) > 0);
  reads_destroy(r);
  pool_destroy(p);
  free(u);
  printf("  destroy busy: %d reads asked, the readers stopped and the waiting dropped\n", U);
}

int main(void)
{
  printf("[test_reads] start\n");
  assert(mkdtemp(dir));
  enum { N = 2000 };
  const int fd = file("f.bin", N);
  test_refusals(fd);
  test_order(fd);
  test_many(fd, 200, 4, 3);
  test_many(fd, N, 8, 8);
  test_many(fd, N, 8, 64);
  test_split(fd);
  test_failure();
  test_destroy_busy(fd);
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_reads] ok\n");
  return 0;
}
