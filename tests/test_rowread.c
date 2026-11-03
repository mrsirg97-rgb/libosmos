#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/mem/rowread.h"

#define FILE_BYTES (4u << 20)

static char dir[] = "/tmp/test_rowread_XXXXXX";

static uint8_t byte_at(uint64_t i) { return (uint8_t)((i * 31 + 7) & 255); }

static int file(const char *name)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (uint64_t i = 0; i < FILE_BYTES; i++)
    fputc(byte_at(i), f);
  fclose(f);
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0);
  return fd;
}

static void test_refusals(int fd)
{
  assert(!rowread_create(0) && !rowread_create(ROWREAD_MAX_THREADS + 1));
  rowread_destroy(NULL);
  assert(!rowread(NULL, fd, NULL, 0, 144, NULL));
  Rowread *r = rowread_create(4);
  assert(r && rowread(r, fd, NULL, 0, 144, NULL)); /* no rows: nothing to do */
  rowread_destroy(r);
  printf("  refusals: 0 or too many workers, no reader; no rows nothing to do\n");
}

/* random rows, every one's bytes in its place, for 1 worker and many, call after call (the workers reused) */
static void test_rows(int fd)
{
  enum { N = 20000, BYTES = 144, CALLS = 50 };
  static uint64_t off[N];
  static uint8_t dst[N * BYTES];
  uint64_t s = 0x9e3779b97f4a7c15ull;
  for (int threads = 1; threads <= 64; threads *= 8)
  {
    Rowread *r = rowread_create(threads);
    assert(r);
    for (int call = 0; call < CALLS; call++)
    {
      const uint32_t n = 1 + (uint32_t)((s ^= s << 13, s ^= s >> 7, s ^= s << 17) % N);
      for (uint32_t i = 0; i < n; i++)
        off[i] = (s ^= s << 13, s ^= s >> 7, s ^= s << 17) % (FILE_BYTES - BYTES);
      memset(dst, 0, (size_t)n * BYTES);
      assert(rowread(r, fd, off, n, BYTES, dst));
      for (uint32_t i = 0; i < n; i++)
        for (uint32_t b = 0; b < BYTES; b += 13)
          assert(dst[(uint64_t)i * BYTES + b] == byte_at(off[i] + b));
    }
    rowread_destroy(r);
  }
  printf("  rows: %d calls of up to %d random rows, 1, 8 and 64 workers, every row's bytes in its place\n", 3 * CALLS,
         N);
}

/* many calls of a row or two, 64 workers: the caller reads each itself before most workers wake, so a worker often
   wakes for a batch already done - and must be out of it before the next call's fields are written (a race
   ThreadSanitizer sees, `make tsan`, if they are not) */
static void test_tiny(int fd)
{
  enum { CALLS = 20000, BYTES = 144 };
  Rowread *r = rowread_create(ROWREAD_MAX_THREADS);
  assert(r);
  uint64_t off[2];
  uint8_t dst[2][2 * BYTES];
  for (uint32_t call = 0; call < CALLS; call++)
  {
    const uint32_t n = 1 + call % 2;
    uint8_t *d = dst[call % 2];
    for (uint32_t i = 0; i < n; i++)
      off[i] = ((uint64_t)call * 4099 + i * 977) % (FILE_BYTES - BYTES);
    assert(rowread(r, fd, off, n, BYTES, d));
    for (uint32_t i = 0; i < n; i++)
      assert(d[i * BYTES] == byte_at(off[i]) && d[i * BYTES + BYTES - 1] == byte_at(off[i] + BYTES - 1));
  }
  rowread_destroy(r);
  printf("  tiny: %d calls of a row or two, %d workers, each row in its place\n", CALLS, ROWREAD_MAX_THREADS);
}

/* a row past the file's end comes short: false, the others still read */
static void test_short(int fd)
{
  Rowread *r = rowread_create(8);
  assert(r);
  uint64_t off[64];
  uint8_t dst[64 * 100];
  for (int i = 0; i < 64; i++)
    off[i] = (uint64_t)i * 1000;
  off[17] = FILE_BYTES - 10; /* 10 bytes of 100 */
  assert(!rowread(r, fd, off, 64, 100, dst));
  assert(dst[63 * 100] == byte_at(63000) && dst[0] == byte_at(0));
  off[17] = 17000;
  assert(rowread(r, fd, off, 64, 100, dst)); /* the next call is clean */
  const int dup_fd = dup(fd);
  close(dup_fd);
  assert(!rowread(r, dup_fd, off, 64, 100, dst)); /* a descriptor that is no more */
  rowread_destroy(r);
  printf("  short: a row past the file's end, or a closed file, fails the call; the next is clean\n");
}

/* rowread_start, then the caller's own work, then rowread_wait: every row in its place; a second start while one is
   pending refused, a wait with none pending true; a failed row through the wait; destroyed with a batch pending:
   the batch finished first (its rows written while dst lives) */
static void test_async(int fd)
{
  enum { N = 5000, BYTES = 144 };
  static uint64_t off[N];
  static uint8_t dst[N * BYTES];
  Rowread *r = rowread_create(16);
  assert(r && rowread_wait(r)); /* none pending */
  for (int round = 0; round < 20; round++)
  {
    for (uint32_t i = 0; i < N; i++)
      off[i] = ((uint64_t)(round + 1) * 7919 * i + 13 * round) % (FILE_BYTES - BYTES);
    memset(dst, 0, sizeof dst);
    assert(rowread_start(r, fd, off, N, BYTES, dst));
    assert(!rowread_start(r, fd, off, N, BYTES, dst)); /* one at a time */
    volatile uint64_t spin = 0; /* the caller's own work meanwhile */
    for (int i = 0; i < 100000; i++)
      spin += (uint64_t)i;
    assert(rowread_wait(r));
    for (uint32_t i = 0; i < N; i++)
      assert(dst[(uint64_t)i * BYTES] == byte_at(off[i]) && dst[(uint64_t)i * BYTES + BYTES - 1] ==
                                                                   byte_at(off[i] + BYTES - 1));
  }
  off[3] = FILE_BYTES - 10;
  assert(rowread_start(r, fd, off, N, BYTES, dst) && !rowread_wait(r)); /* a short row: the wait says so */
  assert(rowread_wait(r));                                              /* and then none pending */
  rowread_destroy(r);
  /* destroyed at once after a start, before its workers may have woken: every row still read */
  enum { QUICK = 2000, QN = 64 };
  for (int k = 0; k < QUICK; k++)
  {
    Rowread *q = rowread_create(8);
    assert(q);
    for (uint32_t i = 0; i < QN; i++)
      off[i] = ((uint64_t)k * 131 + i * 4093) % (FILE_BYTES - BYTES);
    memset(dst, 0, (size_t)QN * BYTES);
    assert(rowread_start(q, fd, off, QN, BYTES, dst));
    rowread_destroy(q);
    for (uint32_t i = 0; i < QN; i++)
      assert(dst[(uint64_t)i * BYTES + BYTES - 1] == byte_at(off[i] + BYTES - 1));
  }
  printf("  async: start, work, wait - every row in place; one batch at a time; a short row through the wait; "
         "destroyed at once after a start (2,000 times), every row read\n");
}

/* a worker's share: a small batch spread over every reader (decode's 48 rows among 65: one each, not 3 workers' 16),
   a long one at most ROWREAD_TAKE, never 0 */
static void test_take(void)
{
  assert(rowread_take(48, 65) == 1 && rowread_take(1, 65) == 1 && rowread_take(0, 65) == 1);
  assert(rowread_take(130, 65) == 2 && rowread_take(131, 65) == 3);
  assert(rowread_take(98304, 65) == ROWREAD_TAKE && rowread_take(65 * ROWREAD_TAKE, 65) == ROWREAD_TAKE);
  assert(rowread_take(48, 1) == ROWREAD_TAKE && rowread_take(5, 1) == 5 && rowread_take(48, 0) == 1);
  printf("  take: 48 rows among 65 readers one each; 130 two; a long batch %d; one reader all, up to %d\n",
         ROWREAD_TAKE, ROWREAD_TAKE);
}

int main(void)
{
  printf("[test_rowread] start\n");
  assert(mkdtemp(dir));
  const int fd = file("f.bin");
  test_take();
  test_refusals(fd);
  test_rows(fd);
  test_tiny(fd);
  test_short(fd);
  test_async(fd);
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_rowread] ok\n");
  return 0;
}
