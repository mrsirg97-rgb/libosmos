#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/mem/membrane.h"

/* the membrane on real files and real readers (membrane.h; DESIGN.md, "0.6.2: the membrane a module of its own"): its
   memory malloc's, a promotion's copy memcpy - what the forward gives it from the GPU */

enum { L = 4, E = 8, PARTS = 3, UNIT = 4096 };
static const uint64_t part_bytes[PARTS] = {1024, 2048, 1024}; /* a gate, an up, a down: UNIT together */

static char dir[] = "/tmp/test_membrane_XXXXXX";
static int copies;

static uint8_t byte_at(uint64_t off) { return (uint8_t)((off * 2654435761u) >> 13); }

static void *fake_alloc(uint64_t bytes, void *ctx) { return (void)ctx, aligned_alloc(POOL_ALIGN, bytes); }
static void fake_free(void *dev, void *ctx) { (void)ctx, free(dev); }

static bool copy_fails;

static bool fake_copy(void *dst, const void *src, uint64_t bytes, void *ctx)
{
  if (copy_fails)
    return false;
  memcpy(dst, src, bytes);
  ++*(int *)ctx;
  return true;
}

static int file(void)
{
  char path[512];
  snprintf(path, sizeof path, "%s/experts.bin", dir);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (uint64_t i = 0; i < (uint64_t)L * E * UNIT; i++)
    fputc(byte_at(i), f);
  fclose(f);
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0);
  return fd;
}

static uint64_t part_offset(uint32_t unit, int k)
{
  uint64_t at = (uint64_t)unit * UNIT;
  for (int j = 0; j < k; j++)
    at += part_bytes[j];
  return at;
}

static PoolUnit units[L * E];

static void make_units(int fd)
{
  for (uint32_t u = 0; u < L * E; u++)
  {
    units[u].n_parts = PARTS;
    for (int k = 0; k < PARTS; k++)
      units[u].part[k] = (PoolRange){fd, part_offset(u, k), part_bytes[k]};
  }
}

static uint64_t ahead_limit; /* the readers' bound on reads ahead (main runs every test without one, then under one) */

static MembraneOptions options(uint32_t room_units, uint32_t stream_units, const uint32_t *warm, size_t n_warm)
{
  const uint64_t ub = pool_unit_bytes(&units[0], 0);
  return (MembraneOptions){units, L, E, room_units * ub, stream_units * ub, {fake_alloc, fake_free, NULL},
                           {fake_alloc, fake_free, NULL}, 0, 4, warm, n_warm, fake_copy, &copies, ahead_limit};
}

static Membrane *make(uint32_t room_units, uint32_t stream_units, const uint32_t *warm, size_t n_warm)
{
  char err[MEMBRANE_ERR] = "";
  const MembraneOptions o = options(room_units, stream_units, warm, n_warm);
  Membrane *m = membrane_create(&o, err);
  if (!m)
    printf("  membrane_create: %s\n", err);
  assert(m);
  return m;
}

static void check_dev(uint32_t unit, const void *dev[POOL_PARTS])
{
  for (int k = 0; k < PARTS; k++)
  {
    const uint8_t *b = dev[k];
    for (uint64_t j = 0; j < part_bytes[k]; j++)
      assert(b[j] == byte_at(part_offset(unit, k) + j)); /* the file's bytes, each part its own */
  }
}

/* a unit held and waited for: its bytes the file's */
static void take(Membrane *m, uint32_t unit)
{
  char err[MEMBRANE_ERR] = "";
  const void *dev[POOL_PARTS];
  assert(membrane_hold(m, unit, err));
  assert(membrane_wait(m, unit, dev, err));
  check_dev(unit, dev);
}

static void visit(Membrane *m, uint32_t layer, bool passing)
{
  char err[MEMBRANE_ERR] = "";
  assert(membrane_visit(m, layer, passing, err));
}

static bool ready(const Membrane *m, uint32_t unit)
{
  const void *dev[POOL_PARTS];
  if (!membrane_ready(m, unit, dev))
    return false;
  check_dev(unit, dev);
  return true;
}

static uint64_t waits(const Membrane *m)
{
  MembraneStats s;
  membrane_stats(m, &s);
  return s.waits[0] + s.waits[1];
}

static void test_refusals(void)
{
  char err[MEMBRANE_ERR] = "";
  MembraneOptions o = options(8, 8, NULL, 0);
  assert(!membrane_create(NULL, err) && err[0]);
  o.units = NULL;
  assert(!membrane_create(&o, err));
  o = options(8, 8, NULL, 0), o.n_layers = 0;
  assert(!membrane_create(&o, err));
  o = options(8, 8, NULL, 0), o.per_layer = 0;
  assert(!membrane_create(&o, err));
  o = options(8, 8, NULL, 0), o.copy = NULL;
  assert(!membrane_create(&o, err));
  o = options(8, 8, NULL, 0), o.room = 1; /* under a unit: the pool's refusal */
  assert(!membrane_create(&o, err));
  const uint32_t past[] = {L * E};
  o = options(8, 8, past, 1);
  assert(!membrane_create(&o, err));
  membrane_destroy(NULL);
  Membrane *m = make(8, 8, NULL, 0);
  assert(!membrane_visit(m, L, false, err) && err[0]); /* a layer past the model's */
  membrane_destroy(m);
  printf("  refusals: no options, units, layers or copy; a room under a unit; a warm unit or a layer out of range\n");
}

/* a decode's experts: read once, in memory a round on with no wait; the same layer visited again keeps its holds */
static void test_decode_keeps(void)
{
  Membrane *m = make(8, 8, NULL, 0);
  for (uint32_t l = 0; l < 2; l++)
  {
    visit(m, l, false);
    for (uint32_t e = 0; e < 4; e++)
      take(m, l * E + e);
    visit(m, l, false); /* a sub-batch: the holds stay */
    for (uint32_t e = 0; e < 4; e++)
      assert(ready(m, l * E + e));
  }
  visit(m, 2, false), visit(m, 3, false);
  const uint64_t before = waits(m);
  for (uint32_t l = 0; l < 2; l++)
  {
    visit(m, l, false);
    for (uint32_t e = 0; e < 4; e++)
      assert(ready(m, l * E + e)), take(m, l * E + e);
  }
  assert(waits(m) == before);
  membrane_end(m);
  membrane_destroy(m);
  printf("  a decode's units: their bytes the file's, in memory a round on, no read waited for\n");
}

/* a prompt's step passes through: its misses in the stream, the membrane's units all still there */
static void test_prompt_passes(void)
{
  Membrane *m = make(8, 8, NULL, 0);
  visit(m, 0, false);
  for (uint32_t e = 0; e < E; e++)
    take(m, e); /* the membrane's room, full */
  for (uint32_t l = 1; l < L; l++) /* a prompt's steps: most of every other layer, and layer 0's own again */
  {
    visit(m, l, true);
    for (uint32_t e = 0; e < 6; e++)
      take(m, l * E + e);
  }
  visit(m, 0, true);
  for (uint32_t e = 0; e < E; e++)
    take(m, e);
  membrane_end(m);
  for (uint32_t e = 0; e < E; e++)
    assert(ready(m, e));
  membrane_destroy(m);
  printf("  a prompt's steps: read into the stream, the membrane's units untouched\n");
}

/* a decode's pick the stream holds is copied into the membrane: later prompts flush the stream, not it */
static void test_promotion(void)
{
  copies = 0;
  Membrane *m = make(8, 4, NULL, 0);
  visit(m, 0, true);
  take(m, 1);
  visit(m, 1, false);
  assert(copies == 0);
  visit(m, 0, false);
  take(m, 1);
  assert(copies == PARTS); /* its parts, once */
  for (uint32_t l = 2; l < L; l++) /* two prompt steps, the stream's room each */
  {
    visit(m, l, true);
    for (uint32_t e = 0; e < 4; e++)
      take(m, l * E + e);
  }
  membrane_end(m);
  assert(ready(m, 1) && copies == PARTS);
  visit(m, 0, false);
  take(m, 1); /* in the membrane now: no second copy */
  assert(copies == PARTS);
  membrane_end(m);
  membrane_destroy(m);
  printf("  promotion: a decode's unit copied out of the stream once, kept when prompts flush it\n");
}

/* a promotion whose copy fails leaves a unit pinned with bytes never written: every call refused until the step's
   end, which drops it - the next pick promotes it again from the stream's copy, or reads it from its file */
static void test_copy_fails(void)
{
  char err[MEMBRANE_ERR] = "";
  const void *dev[POOL_PARTS];
  Membrane *m = make(8, 4, NULL, 0);
  visit(m, 0, true);
  take(m, 1);
  visit(m, 1, false);
  visit(m, 0, false);
  copy_fails = true;
  assert(!membrane_hold(m, 1, err) && err[0]);
  err[0] = '\0';
  assert(!membrane_hold(m, 1, err) && err[0]); /* not "held already": its bytes are not there */
  assert(!membrane_ready(m, 1, dev) && !membrane_wait(m, 1, dev, err) && !membrane_in_memory(m, 1));
  assert(!membrane_visit(m, 1, false, err) && !membrane_begin(m, 2) && !membrane_warm(m, err));
  membrane_end(m);
  visit(m, 0, false);
  assert(!membrane_hold(m, 1, err)); /* the copy still failing: refused again, dropped again */
  membrane_end(m);
  copy_fails = false;
  copies = 0;
  visit(m, 0, false);
  take(m, 1); /* the stream's copy promoted: the file's bytes */
  assert(copies == PARTS);
  for (uint32_t l = 2; l < L; l++) /* the stream flushed: the membrane's copy the one read */
  {
    visit(m, l, true);
    for (uint32_t e = 0; e < 4; e++)
      take(m, l * E + e);
  }
  membrane_end(m);
  assert(ready(m, 1));
  membrane_destroy(m);
  printf("  a promotion's copy failed: refused until the step's end, the unit dropped, sound when next picked\n");
}

/* the guess: a unit's read begun a visit ahead and scored a hit if still reading; an unguessed pick a miss */
static void test_guess(void)
{
  Membrane *m = make(8, 8, NULL, 0);
  visit(m, 0, false);
  membrane_guess(m, E + 2);
  visit(m, 1, false);
  membrane_score(m, E + 2); /* guessed: in memory by now (not streamed), or still reading (streamed, a hit) */
  membrane_score(m, E + 3); /* not guessed: streamed, no hit */
  MembraneStats s;
  membrane_stats(m, &s);
  assert(s.guess_read == 1 && s.streamed - s.guessed == 1 && s.guessed <= 1);
  take(m, E + 2), take(m, E + 3);
  membrane_guess(m, 2 * E + 2); /* a guess of one already wanted: no second read */
  membrane_guess(m, 2 * E + 2);
  membrane_stats(m, &s);
  assert(s.guess_read == 2);
  membrane_end(m);
  membrane_destroy(m);
  printf("  the guess: its read begun a visit ahead, once; scored against the picks\n");
}

/* the warm start: in memory with no hold, the first named the last to go */
static void test_warm(void)
{
  const uint32_t warm[] = {E + 5, 2 * E + 1, 3 * E + 2};
  char err[MEMBRANE_ERR] = "";
  Membrane *m = make(3, 4, warm, 3);
  assert(membrane_warm(m, err));
  for (int i = 0; i < 3; i++)
    assert(ready(m, warm[i]));
  visit(m, 0, false);
  take(m, 0);
  assert(!ready(m, warm[2]) && ready(m, warm[1]) && ready(m, warm[0]));
  take(m, 1);
  assert(!ready(m, warm[1]) && ready(m, warm[0]));
  membrane_end(m);
  membrane_destroy(m);
  printf("  the warm start: read in before the first visit, the first named the last to go\n");
}

/* a warm start whose read fails (its file cut short after create): refused, and the units it did read held no longer
   - the room whole for the next request's visits (0.7.1: the pins it took stayed, every later step refused) */
static void test_warm_failed(void)
{
  char path[512], err[MEMBRANE_ERR] = "";
  snprintf(path, sizeof path, "%s/short.bin", dir);
  FILE *f = fopen(path, "wb");
  assert(f);
  for (uint64_t i = 0; i < 2 * UNIT; i++)
    fputc(byte_at(i), f);
  fclose(f);
  const int fd = open(path, O_RDWR);
  assert(fd >= 0);
  const uint32_t warm[] = {E + 5, 2 * E + 1, 3 * E + 2};
  const PoolUnit kept = units[warm[2]];
  for (int k = 0; k < PARTS; k++)
    units[warm[2]].part[k] = (PoolRange){fd, part_offset(0, k), part_bytes[k]};
  Membrane *m = make(3, 4, warm, 3);
  assert(ftruncate(fd, 1) == 0); /* the read of warm[2] fails */
  assert(!membrane_warm(m, err) && strstr(err, "read"));
  units[warm[2]] = kept;
  visit(m, 0, false); /* the whole room held by one visit: none of it still the warm start's */
  take(m, 0), take(m, 1), take(m, 2);
  membrane_end(m);
  membrane_destroy(m);
  close(fd);
  printf("  a warm start's failed read: refused, the room whole again for the next visit\n");
}

/* a membrane with room for every unit, all of them warm: whole - every unit's pointers there, the same ones after
   visits that hold them; a room one unit short, or the warm start not yet read: not whole */
static void test_whole(void)
{
  enum { N = L * E };
  uint32_t all[N];
  for (uint32_t u = 0; u < N; u++)
    all[u] = u;
  char err[MEMBRANE_ERR] = "";
  Membrane *m = make(N, 2, all, N);
  assert(!membrane_whole(m)); /* nothing read yet */
  assert(membrane_warm(m, err) && membrane_whole(m));
  const void *before[N][POOL_PARTS], *after[POOL_PARTS];
  for (uint32_t u = 0; u < N; u++)
    assert(membrane_ready(m, u, before[u]));
  for (uint32_t l = 0; l < L; l++) /* a round of visits holding every unit: nothing read, nothing moved */
  {
    visit(m, l, l & 1);
    for (uint32_t e = 0; e < E; e++)
      take(m, l * E + e);
  }
  membrane_end(m);
  MembraneStats st;
  membrane_stats(m, &st);
  assert(membrane_whole(m) && st.streamed == 0 && st.waits[0] == 0);
  for (uint32_t u = 0; u < N; u++)
    assert(membrane_ready(m, u, after) && !memcmp(after, before[u], sizeof after));
  membrane_destroy(m);
  m = make(N - 1, 2, all, N - 1); /* a unit short: its room not every unit's */
  assert(membrane_warm(m, err) && !membrane_whole(m));
  membrane_destroy(m);
  assert(!membrane_whole(NULL));
  printf("  whole: room for every unit and all warm - their pointers the same after a round of visits; a unit short, "
         "or not yet read: not whole\n");
}

/* more held in a visit than the room: refused with why, and the next visit sound */
static void test_full(void)
{
  char err[MEMBRANE_ERR] = "";
  Membrane *m = make(2, 2, NULL, 0);
  visit(m, 0, false);
  take(m, 0), take(m, 1);
  assert(!membrane_hold(m, 2, err) && err[0]);
  const void *dev[POOL_PARTS];
  assert(!membrane_ready(m, 2, dev));
  assert(ready(m, 0) && ready(m, 1)); /* the held ones stand */
  visit(m, 1, false);
  take(m, E), take(m, E + 1);
  membrane_end(m);
  membrane_destroy(m);
  printf("  a visit past the room: the hold refused with why, nothing else moved\n");
}

/* a walk of visits - decodes and prompts, guesses and reads ahead - every held unit's bytes the file's */
static void test_walk(void)
{
  Membrane *m = make(6, 8, NULL, 0);
  uint32_t seed = 12345;
#define RND(n) ((seed = seed * 1664525u + 1013904223u) >> 8) % (n)
  for (int v = 0; v < 400; v++)
  {
    const uint32_t l = (uint32_t)v % L, next = (l + 1) % L;
    const bool passing = RND(4) == 0;
    visit(m, l, passing);
    const uint32_t n = passing ? 6 : 3, first = RND(E);
    for (uint32_t i = 0; i < n; i++)
      membrane_score(m, l * E + (first + i) % E);
    if (!passing && RND(2))
      membrane_guess(m, next * E + RND(E));
    for (uint32_t i = 0; i < n; i++)
      take(m, l * E + (first + i) % E);
    if (passing)
      for (uint32_t i = 0; i < 2; i++)
        membrane_ahead(m, next * E + RND(E));
  }
#undef RND
  membrane_end(m);
  MembraneStats s;
  membrane_stats(m, &s);
  assert(s.guessed <= s.streamed);
  membrane_destroy(m);
  printf("  a walk of 400 visits: every held unit's bytes the file's\n");
}

int main(void)
{
  printf("[test_membrane] start\n");
  assert(mkdtemp(dir));
  const int fd = file();
  make_units(fd);
  assert(pool_unit_bytes(&units[0], 0) == UNIT);
  for (int bound = 0; bound < 2; bound++) /* reads ahead a job at a time: which units are where, and when, the same */
  {
    ahead_limit = bound ? 1 : 0;
    printf(bound ? " under a bound on reads ahead (a job at a time):\n" : " no bound on reads ahead:\n");
    test_refusals();
    test_decode_keeps();
    test_prompt_passes();
    test_promotion();
    test_copy_fails();
    test_guess();
    test_warm();
    test_warm_failed();
    test_whole();
    test_full();
    test_walk();
  }
  close(fd);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_membrane] ok\n");
  return 0;
}
