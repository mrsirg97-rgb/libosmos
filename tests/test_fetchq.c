#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/mem/fetchq.h"
#include "substrate/mem/pool.h"

#define N 2000

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
  assert(fetchq_create(0, done, NULL) == NULL);
  assert(fetchq_create(65, done, NULL) == NULL);
  assert(fetchq_create(4, NULL, NULL) == NULL);
  fetchq_destroy(NULL);
  printf("  refusals: 0 or 65 workers, no done\n");
}

/* every unit read once, by 4 workers at once */
static void test_once(int fd)
{
  reset();
  Fetchq *q = fetchq_create(4, done, NULL);
  assert(q);
  const Fetcher f = fetchq_fetcher(q);
  for (uint32_t u = 0; u < N; u++)
  {
    const PoolRange parts[2] = {{fd, (u * 4096u) % (1 << 20), 3000}, {fd, (u * 7919u) % (1 << 20), 100}};
    f.fetch(u, parts, 1 + (int)(u % 2), f.ctx);
  }
  while (atomic_load(&n_ok) + atomic_load(&n_bad) < N)
    usleep(1000);
  fetchq_destroy(q);
  for (int i = 0; i < N; i++)
    assert(atomic_load(&seen[i]) == 1);
  assert(atomic_load(&n_ok) == N && atomic_load(&n_bad) == 0);
  printf("  once: %d units, 4 workers, each done exactly once, every read whole\n", N);
}

static void test_failure(int fd)
{
  reset();
  const int dup_fd = dup(fd);
  close(dup_fd); /* a descriptor that is no more */
  Fetchq *q = fetchq_create(2, done, NULL);
  assert(q);
  const Fetcher f = fetchq_fetcher(q);
  const PoolRange bad = {dup_fd, 0, 4096};
  f.fetch(7, &bad, 1, f.ctx);
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
  Fetchq *q = fetchq_create(4, done, NULL);
  assert(q);
  const Fetcher f = fetchq_fetcher(q);
  for (uint32_t u = 0; u < 20000; u++)
  {
    const PoolRange r = {fd, (u * 4096u) % (1 << 20), 65536};
    f.fetch(u % N, &r, 1, f.ctx);
  }
  fetchq_destroy(q);
  atomic_store(&destroyed, true);
  usleep(20000);
  assert(atomic_load(&after_destroy) == 0);
  printf("  destroy busy: stopped with %d of 20000 read; none after it returned\n", atomic_load(&n_ok));
}

/* the pool fed by the fetcher, its ends on the consumer: a worker's done queues the unit, the consumer drains them
   into pool_ready - the binary's engine event in small */
typedef struct
{
  pthread_mutex_t mu;
  uint32_t unit[N];
  int n;
} Inbox;

static Inbox inbox = {PTHREAD_MUTEX_INITIALIZER, {0}, 0};

static void post(uint32_t unit, bool ok, void *arg)
{
  (void)ok, (void)arg;
  pthread_mutex_lock(&inbox.mu);
  inbox.unit[inbox.n++] = unit;
  pthread_mutex_unlock(&inbox.mu);
}

static bool fake_pin(void *host, size_t bytes, void *ctx) { return (void)host, (void)bytes, (void)ctx, true; }
static void fake_unpin(void *host, void *ctx) { (void)host, (void)ctx; }

static void test_with_pool(int fd)
{
  enum { U = 200 };
  PoolUnit units[U];
  for (int i = 0; i < U; i++)
    units[i] = (PoolUnit){.part = {{fd, (uint64_t)i * 4096, 4096}}, .n_parts = 1};
  Fetchq *q = fetchq_create(4, post, NULL);
  assert(q);
  char err[POOL_ERR];
  Pool *p = pool_create(units, U, 1 << 20, (Pinner){fake_pin, fake_unpin, NULL}, fetchq_fetcher(q), NULL, NULL, err);
  assert(p);
  const void *dev[POOL_PARTS];
  for (uint32_t i = 0; i < U; i++)
    assert(pool_pin(p, i, dev) == POOL_WAIT);
  int pinned = 0;
  while (pinned < U)
  {
    pthread_mutex_lock(&inbox.mu);
    const int n = inbox.n;
    uint32_t got[N];
    memcpy(got, inbox.unit, (size_t)n * sizeof *got);
    inbox.n = 0;
    pthread_mutex_unlock(&inbox.mu);
    for (int i = 0; i < n; i++)
    {
      pool_ready(p, got[i]);
      assert(pool_state(p, got[i]) == POOL_PINNED && pool_dev(p, got[i], dev));
      pinned++;
    }
    if (!n)
      usleep(1000);
  }
  fetchq_destroy(q);
  pool_destroy(p);
  printf("  with the pool: %d units WAIT, read by the workers, pinned on the consumer\n", U);
}

int main(void)
{
  printf("[test_fetchq] start\n");
  assert(mkdtemp(dir));
  const int fd = file("f.bin", 1 << 20);
  test_refusals();
  test_once(fd);
  test_failure(fd);
  test_destroy_busy(fd);
  test_with_pool(fd);
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_fetchq] ok\n");
  return 0;
}
