#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/mem/fetchq.h"
#include "osmos/mem/pool.h"

#define N 2000
#define CHUNKED_FILE (8u << 20) /* test_chunks' file: parts of several FETCHQ_CHUNK */

static char dir[] = "/tmp/test_fetchq_XXXXXX";
static atomic_int seen[N];
static atomic_int n_ok, n_bad, after_destroy;
static atomic_bool destroyed;

static void done(uint32_t unit, bool ok, void *arg)
{
  (void)arg;
  if (atomic_load(&destroyed))
    atomic_fetch_add(&after_destroy, 1);
  if (unit < N)
    atomic_fetch_add(&seen[unit], 1);
  atomic_fetch_add(ok ? &n_ok : &n_bad, 1);
}

static int file(const char *name, size_t bytes)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (size_t i = 0; i < bytes; i++)
    fputc((int)(i & 255), f);
  fclose(f);
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0);
  return fd;
}

static void reset(void)
{
  for (int i = 0; i < N; i++)
    atomic_store(&seen[i], 0);
  atomic_store(&n_ok, 0);
  atomic_store(&n_bad, 0);
  atomic_store(&after_destroy, 0);
  atomic_store(&destroyed, false);
}

static void test_refusals(void)
{
  assert(fetchq_create(0, N, done, NULL) == NULL);
  assert(fetchq_create(65, N, done, NULL) == NULL);
  assert(fetchq_create(4, N, NULL, NULL) == NULL);
  assert(fetchq_create(4, 0, done, NULL) == NULL);
  fetchq_destroy(NULL);
  printf("  refusals: 0 or 65 workers, no done, no units\n");
}

/* every unit read once, by 4 workers at once, each part's bytes where it was told */
static void test_once(int fd)
{
  reset();
  Fetchq *q = fetchq_create(4, N, done, NULL);
  assert(q);
  static uint8_t buf[N][2][3000];
  for (uint32_t u = 0; u < N; u++)
  {
    const PoolRange parts[2] = {{fd, (u * 4096u) % ((1 << 20) - 3000), 3000},
                                {fd, (u * 7919u) % ((1 << 20) - 100), 100}};
    void *const dst[POOL_PARTS] = {buf[u][0], buf[u][1], NULL};
    assert(fetchq_read(q, u, parts, 1 + (int)(u % 2), dst, false, 1));
  }
  while (atomic_load(&n_ok) + atomic_load(&n_bad) < N)
    usleep(1000);
  fetchq_destroy(q);
  for (uint32_t u = 0; u < N; u++)
  {
    assert(atomic_load(&seen[u]) == 1);
    const uint64_t at = (u * 4096u) % ((1 << 20) - 3000);
    assert(buf[u][0][0] == (uint8_t)at && buf[u][0][2999] == (uint8_t)(at + 2999));
    const uint64_t at1 = (u * 7919u) % ((1 << 20) - 100);
    assert(u % 2 == 0 || (buf[u][1][0] == (uint8_t)at1 && buf[u][1][99] == (uint8_t)(at1 + 99)));
  }
  assert(atomic_load(&n_ok) == N && atomic_load(&n_bad) == 0);
  printf("  once: %d units, 4 workers, each done exactly once, every part's bytes in its own place\n", N);
}

/* parts of many chunks (FETCHQ_CHUNK), and a chunk and a half, and the exact one, and a few bytes: every byte in its
   place and each unit done once - its chunks read by many workers at once, the last one's worker calling done */
static void test_chunks(int fd)
{
  reset();
  Fetchq *q = fetchq_create(4, N, done, NULL);
  assert(q);
  enum { UNITS = 24 };
  const uint64_t sizes[3] = {FETCHQ_CHUNK * 5 / 2, FETCHQ_CHUNK, 7};
  uint8_t *buf[UNITS][3];
  for (uint32_t u = 0; u < UNITS; u++)
  {
    PoolRange parts[3];
    void *dst[POOL_PARTS] = {NULL, NULL, NULL};
    for (int k = 0; k < 3; k++)
    {
      const uint64_t at = (u * 7919u + (uint64_t)k * 104729u) % (CHUNKED_FILE - sizes[k]);
      parts[k] = (PoolRange){fd, at, sizes[k]};
      dst[k] = buf[u][k] = malloc(sizes[k]);
      assert(buf[u][k]);
    }
    assert(fetchq_read(q, u, parts, 3, dst, u % 2 == 0, 1)); /* split and whole: the same bytes */
  }
  while (atomic_load(&n_ok) + atomic_load(&n_bad) < UNITS)
    usleep(1000);
  fetchq_destroy(q);
  for (uint32_t u = 0; u < UNITS; u++)
  {
    assert(atomic_load(&seen[u]) == 1);
    for (int k = 0; k < 3; k++)
    {
      const uint64_t at = (u * 7919u + (uint64_t)k * 104729u) % (CHUNKED_FILE - sizes[k]);
      for (uint64_t i = 0; i < sizes[k]; i++)
        assert(buf[u][k][i] == (uint8_t)(at + i));
      free(buf[u][k]);
    }
  }
  assert(atomic_load(&n_ok) == UNITS && atomic_load(&n_bad) == 0);
  printf("  chunks: %d units of parts %.1f, 1 chunk and 7 bytes, half split, half whole - every byte in its place, each done once\n", UNITS,
         (double)sizes[0] / FETCHQ_CHUNK);
}

static void test_failure(int fd)
{
  reset();
  const int dup_fd = dup(fd);
  close(dup_fd); /* a descriptor that is no more */
  Fetchq *q = fetchq_create(2, N, done, NULL);
  assert(q);
  const PoolRange bad = {dup_fd, 0, 4096};
  static uint8_t buf[4096];
  void *const dst[POOL_PARTS] = {buf, NULL, NULL};
  assert(fetchq_read(q, 7, &bad, 1, dst, true, 0));
  while (atomic_load(&n_ok) + atomic_load(&n_bad) < 1)
    usleep(1000);
  fetchq_destroy(q);
  assert(atomic_load(&n_bad) == 1 && atomic_load(&seen[7]) == 1);
  printf("  failure: a read on a closed descriptor reports ok = false\n");
}

/* destroyed with thousands queued: it stops, and nothing is called once it returned */
static void test_destroy_busy(int fd)
{
  reset();
  enum { JOBS = 20000, BYTES = 4096 };
  Fetchq *q = fetchq_create(4, JOBS, done, NULL);
  assert(q);
  uint8_t *buf = malloc((size_t)JOBS * BYTES); /* each read its own place */
  assert(buf);
  for (uint32_t u = 0; u < JOBS; u++)
  {
    const PoolRange r = {fd, (u * 4096u) % ((1 << 20) - BYTES), BYTES};
    void *const dst[POOL_PARTS] = {buf + (size_t)u * BYTES, NULL, NULL};
    assert(fetchq_read(q, u, &r, 1, dst, false, 1));
  }
  fetchq_destroy(q);
  free(buf);
  atomic_store(&destroyed, true);
  usleep(20000);
  assert(atomic_load(&after_destroy) == 0);
  printf("  destroy busy: stopped with %d of %d read; none after it returned\n", atomic_load(&n_ok), JOBS);
}

/* the order a worker takes: one worker held inside unit 0's done (so the queue is what it is when the test lets it
   go), the rest queued behind it at their needs - taken by need (0, a miss waited on, first), ties in the order given;
   a need moved while queued (fetchq_need) the need then. a second read of a queued unit refused */
static pthread_mutex_t gate_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cv = PTHREAD_COND_INITIALIZER;
static bool gate_open;
static uint32_t order[N];
static atomic_int n_order;

static void done_gated(uint32_t unit, bool ok, void *arg)
{
  (void)arg, (void)ok;
  order[atomic_fetch_add(&n_order, 1)] = unit;
  pthread_mutex_lock(&gate_mu);
  while (unit == 0 && !gate_open)
    pthread_cond_wait(&gate_cv, &gate_mu);
  pthread_mutex_unlock(&gate_mu);
}

static void test_order(int fd)
{
  enum { READS = 10, BYTES = 4096 };
  atomic_store(&n_order, 0);
  gate_open = false;
  Fetchq *q = fetchq_create(1, N, done_gated, NULL);
  assert(q);
  static uint8_t buf[READS][BYTES];
  /* unit: its need */
  static const uint64_t need[READS] = {5, 7, 3, 9, 3, 0, 8, 6, 0, 4};
  for (uint32_t u = 0; u < READS; u++)
  {
    const PoolRange r = {fd, (uint64_t)u * BYTES, BYTES};
    void *const dst[POOL_PARTS] = {buf[u], NULL, NULL};
    assert(fetchq_read(q, u, &r, 1, dst, u == 6, need[u])); /* 6: split, its chunks side by side */
    while (u == 0 && atomic_load(&n_order) == 0) /* the worker in unit 0's done before the rest are queued */
      usleep(1000);
    if (u == 1)
      assert(!fetchq_read(q, 1, &r, 1, dst, false, 1)); /* queued already: refused */
  }
  fetchq_need(q, 1, 0);  /* a read ahead now waited on: among the pins, by the order it was given (before 5, 8) */
  fetchq_need(q, 3, 2);  /* sooner: before 2 and 4 (3) */
  fetchq_need(q, 9, 20); /* later: last */
  fetchq_need(q, 0, 0);  /* begun (held): nothing to do */
  pthread_mutex_lock(&gate_mu);
  gate_open = true;
  pthread_cond_broadcast(&gate_cv);
  pthread_mutex_unlock(&gate_mu);
  while (atomic_load(&n_order) < READS)
    usleep(1000);
  fetchq_destroy(q);
  const uint32_t want[READS] = {0, 1, 5, 8, 3, 2, 4, 7, 6, 9};
  for (int i = 0; i < READS; i++)
    assert(order[i] == want[i]);
  for (uint32_t u = 0; u < READS; u++)
    assert(buf[u][0] == (uint8_t)(u * BYTES) && buf[u][BYTES - 1] == (uint8_t)(u * BYTES + BYTES - 1));
  printf("  order: queued behind a held read, taken by need (0 first), ties as given, a moved need the need then\n");
}

int main(void)
{
  printf("[test_fetchq] start\n");
  assert(mkdtemp(dir));
  const int fd = file("f.bin", 1 << 20);
  test_refusals();
  test_once(fd);
  const int big = file("big.bin", CHUNKED_FILE);
  test_chunks(big);
  close(big);
  test_failure(fd);
  test_destroy_busy(fd);
  test_order(fd);
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_fetchq] ok\n");
  return 0;
}
