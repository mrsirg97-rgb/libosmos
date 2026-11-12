#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/mem/resident.h"

#define FILE_BYTES (8u << 20)
#define MAX_ALLOCS 64

static char dir[] = "/tmp/test_resident_XXXXXX";

static uint8_t byte_at(uint64_t i) { return (uint8_t)((i * 13 + 5) & 255); }

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

/* the memory: malloc's, every allocation tracked; `refuse_at` the allocation that fails (0: none) */
typedef struct
{
  void *at[MAX_ALLOCS];
  uint64_t bytes[MAX_ALLOCS];
  int n, made, refuse_at;
} Mem;

static void *fake_alloc(uint64_t bytes, void *ctx)
{
  Mem *m = ctx;
  if (++m->made == m->refuse_at)
    return NULL;
  assert(m->n < MAX_ALLOCS);
  m->bytes[m->n] = bytes;
  return m->at[m->n++] = aligned_alloc(POOL_ALIGN, bytes);
}

static void fake_free(void *dev, void *ctx)
{
  Mem *m = ctx;
  for (int i = 0; i < m->n; i++)
    if (m->at[i] == dev)
    {
      free(dev);
      m->at[i] = m->at[--m->n], m->bytes[i] = m->bytes[m->n];
      return;
    }
  assert(!"a free of memory never allocated");
}

static void test_refusals(int fd)
{
  Mem m = {0};
  char err[POOL_ERR];
  const Slots s = {fake_alloc, fake_free, &m, NULL, NULL};
  const PoolUnit bad[] = {{.n_parts = 0}, {.n_parts = 4}, {.part = {{fd, 0, 0}}, .n_parts = 1},
                          {.part = {{fd, FILE_BYTES - 1, 2}}, .n_parts = 1}, {.part = {{9999, 0, 1}}, .n_parts = 1}};
  for (int i = 0; i < 5; i++)
  {
    err[0] = 0;
    assert(!resident_create(&bad[i], 1, s, 2, err) && err[0] && m.n == 0);
  }
  /* the allocator refuses the second arena (units of the whole file, past an arena's worth): nothing kept */
  enum { BIG = RESIDENT_ARENA / FILE_BYTES + 2 };
  static PoolUnit big[BIG];
  for (int i = 0; i < BIG; i++)
    big[i] = (PoolUnit){.part = {{fd, 0, FILE_BYTES}}, .n_parts = 1};
  m.refuse_at = 2;
  assert(!resident_create(big, BIG, s, 2, err) && strstr(err, "refused") && m.n == 0);
  resident_destroy(NULL);
  printf("  refusals: no parts, four parts, 0 bytes, past the file, a closed file, the allocator's no\n");
}

/* every part its file's bytes, aligned, within one allocation; the bytes held the allocations' */
static void verify(const Resident *r, const PoolUnit *u, int n, const Mem *mp)
{
  const Mem m = *mp;
  uint64_t held = 0;
  for (int i = 0; i < m.n; i++)
    held += m.bytes[i];
  assert(resident_bytes(r) == held);
  for (int i = 0; i < n; i++)
  {
    const void *dev[POOL_PARTS];
    assert(resident_dev(r, (size_t)i, dev));
    for (int k = 0; k < u[i].n_parts; k++)
    {
      const uint8_t *b = dev[k];
      assert((uintptr_t)b % POOL_ALIGN == 0);
      bool inside = false; /* within one allocation, whole */
      for (int a = 0; a < m.n && !inside; a++)
        inside = b >= (uint8_t *)m.at[a] && b + u[i].part[k].bytes <= (uint8_t *)m.at[a] + m.bytes[a];
      assert(inside);
      for (uint64_t j = 0; j < u[i].part[k].bytes; j += 1 + u[i].part[k].bytes / 9)
        assert(b[j] == byte_at(u[i].part[k].offset + j));
      assert(b[u[i].part[k].bytes - 1] == byte_at(u[i].part[k].offset + u[i].part[k].bytes - 1));
    }
  }
}

/* units of 1-3 unaligned parts: each part's file bytes at an aligned place, units in arenas, none over an arena's end
   (one larger than an arena its own); every allocation freed by destroy */
static void test_layout(int fd)
{
  enum { N = 300 };
  static PoolUnit u[N];
  uint64_t s = 0x2545f4914f6cdd1dull;
#define RND() (s ^= s << 13, s ^= s >> 7, s ^= s << 17, s)
  for (int i = 0; i < N; i++)
  {
    u[i].n_parts = 1 + (int)(RND() % 3);
    for (int k = 0; k < u[i].n_parts; k++)
    {
      const uint64_t bytes = 1 + RND() % (64 * 1024);
      u[i].part[k] = (PoolRange){fd, RND() % (FILE_BYTES - bytes), bytes};
    }
  }
  u[7] = (PoolUnit){.part = {{fd, 3, RESIDENT_ARENA / 512}}, .n_parts = 1}; /* small enough for the file */
  Mem m = {0};
  char err[POOL_ERR] = "";
  Resident *r = resident_create(u, N, (Slots){fake_alloc, fake_free, &m, NULL, NULL}, 4, err);
  if (!r)
    fprintf(stderr, "  %s\n", err);
  assert(r);
  verify(r, u, N, &m);
  const void *dev[POOL_PARTS];
  assert(!resident_dev(r, N, dev));
  const int arenas = m.n;
  resident_destroy(r);
  assert(m.n == 0);
  /* none: an empty set */
  r = resident_create(u, 0, (Slots){fake_alloc, fake_free, &m, NULL, NULL}, 4, err);
  assert(r && resident_bytes(r) == 0 && !resident_dev(r, 0, dev));
  resident_destroy(r);
  /* past an arena: units of the whole file, the second arena begun where the first cannot hold the next */
  enum { BIG = RESIDENT_ARENA / FILE_BYTES + 2 };
  static PoolUnit big[BIG];
  for (int i = 0; i < BIG; i++)
    big[i] = (PoolUnit){.part = {{fd, 0, FILE_BYTES}}, .n_parts = 1};
  r = resident_create(big, BIG, (Slots){fake_alloc, fake_free, &m, NULL, NULL}, 4, err);
  assert(r && m.n == 2 && resident_bytes(r) == 2 * RESIDENT_ARENA);
  verify(r, big, BIG, &m);
  resident_destroy(r);
  assert(m.n == 0);
  printf("  layout: %d units of 1-3 unaligned parts in %d arena(s), %d of a whole file in 2: each part its file's "
         "bytes, aligned, within one allocation; destroy frees all\n", N, arenas, BIG);
}

/* room: a part of fd -1 laid out after the file's part, not read - its memory as the allocator gave it (0xAB here),
   the file's part its bytes; a room part with an offset refused */
static void *alloc_ab(uint64_t bytes, void *ctx)
{
  void *p = fake_alloc(bytes, ctx);
  if (p)
    memset(p, 0xAB, bytes);
  return p;
}

static void test_room(int fd)
{
  const uint64_t file_b = 1000, room_b = 5000;
  const PoolUnit u[2] = {{.part = {{fd, 17, file_b}, {-1, 0, room_b}}, .n_parts = 2},
                         {.part = {{-1, 0, 64}}, .n_parts = 1}};
  Mem m = {0};
  char err[POOL_ERR] = "";
  Resident *r = resident_create(u, 2, (Slots){alloc_ab, fake_free, &m, NULL, NULL}, 4, err);
  assert(r);
  const void *dev[POOL_PARTS];
  assert(resident_dev(r, 0, dev));
  const uint8_t *a = dev[0], *b = dev[1];
  assert(b == a + (file_b + POOL_ALIGN - 1) / POOL_ALIGN * POOL_ALIGN);
  for (uint64_t i = 0; i < file_b; i++)
    assert(a[i] == byte_at(17 + i));
  for (uint64_t i = 0; i < room_b; i++)
    assert(b[i] == 0xAB);
  assert(resident_dev(r, 1, dev) && ((const uint8_t *)dev[0])[63] == 0xAB); /* a unit all room: nothing read */
  resident_destroy(r);
  assert(m.n == 0);
  const PoolUnit bad = {.part = {{-1, 8, 64}}, .n_parts = 1};
  assert(!resident_create(&bad, 1, (Slots){alloc_ab, fake_free, &m, NULL, NULL}, 4, err) && m.n == 0);
  printf("  room: a part of no file laid out after the file's, not read; with an offset refused\n");
}

int main(void)
{
  printf("[test_resident] start\n");
  assert(mkdtemp(dir));
  const int fd = file("f.bin");
  test_refusals(fd);
  test_layout(fd);
  test_room(fd);
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_resident] ok\n");
  return 0;
}
