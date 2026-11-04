/* the kept conversations' store (substrate/serve/slots.h): states as opaque bytes, tokens as the prefixes to match.
   RAM alone (the minimum, the longest prefix, `beyond`, take, the cap, the same tokens replaced); the folder past RAM
   (a conversation written there and taken back, its files gone); flush and a new store over the folder (a restart);
   a store of another identity, a damaged .meta, a .bin of the wrong size and a .bin alone - each removed at start;
   the folder's cap (its least recently used removed, never the one just written); a conversation older than `days` */
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "substrate/serve/slots.h"

enum { STATE = 4096, N = 2000 };

static char dir[] = "/tmp/test_slots_XXXXXX";
static char err[SLOTS_ERR];

/* conversation k's tokens: its own from token 2 on (any two diverge there), n of them */
static int32_t *conv(int k, uint32_t n)
{
  int32_t *t = malloc(n * sizeof *t);
  assert(t);
  for (uint32_t i = 0; i < n; i++)
    t[i] = i < 2 ? (int32_t)i : (int32_t)(k * 100003 + i * 7) % 129000;
  return t;
}

static uint8_t *state(int k, uint64_t bytes)
{
  uint8_t *b = malloc(bytes);
  assert(b);
  for (uint64_t i = 0; i < bytes; i++)
    b[i] = (uint8_t)(k * 31 + i * 13);
  return b;
}

static SlotsOptions opts(const char *d, uint32_t cap, double gb, double days, uint64_t id)
{
  return (SlotsOptions){d, cap, gb, 1024, days, id};
}

static void put(Slots *s, int k, uint32_t n, uint64_t bytes, bool want_kept)
{
  int32_t *t = conv(k, n);
  uint8_t *b = state(k, bytes);
  bool kept = !want_kept;
  assert(slots_put(s, t, n, b, bytes, &kept, err) && kept == want_kept);
  free(t), free(b);
}

/* conversation k found by a prompt that continues it (its tokens and 50 more), taken, its bytes its own */
static void take_back(Slots *s, int k, uint32_t n, uint64_t bytes)
{
  int32_t *p = conv(k, n + 50);
  uint32_t len = 0;
  const int64_t id = slots_find(s, p, n + 50, 0, &len);
  assert(id >= 0 && len == n);
  assert(slots_find(s, p, n + 50, n, &len) < 0); /* nothing longer than what the caller holds */
  void *st = NULL;
  uint64_t got = 0;
  assert(slots_take(s, id, &st, &got, err) && got == bytes);
  uint8_t *b = state(k, bytes);
  assert(memcmp(st, b, bytes) == 0);
  free(st), free(b);
  assert(slots_find(s, p, n + 50, 0, &len) < 0); /* taken: gone */
  free(p);
}

static int files(const char *suffix)
{
  DIR *d = opendir(dir);
  assert(d);
  int n = 0;
  for (struct dirent *e; (e = readdir(d));)
    n += strlen(e->d_name) > strlen(suffix) && !strcmp(e->d_name + strlen(e->d_name) - strlen(suffix), suffix);
  closedir(d);
  return n;
}

/* the folder's first file ending in suffix (its path into out) */
static void first(const char *suffix, char *out, size_t cap)
{
  DIR *d = opendir(dir);
  assert(d);
  for (struct dirent *e; (e = readdir(d));)
    if (strlen(e->d_name) > strlen(suffix) && !strcmp(e->d_name + strlen(e->d_name) - strlen(suffix), suffix))
    {
      snprintf(out, cap, "%s/%s", dir, e->d_name);
      closedir(d);
      return;
    }
  assert(0);
}

static void clear(void)
{
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -f %s/*", dir);
  assert(system(cmd) == 0);
}

int main(void)
{
  printf("[test_slots] start\n");
  assert(mkdtemp(dir));
  SlotsOptions o = opts(NULL, 0, 0, 0, 7);
  assert(!slots_create(NULL, err) && !slots_create(&o, err));
  printf("  refusals: no options, a cap of 0\n");

  /* RAM alone */
  o = opts(NULL, 2, 0, 0, 7);
  Slots *s = slots_create(&o, err);
  assert(s);
  put(s, 1, 10, STATE, false); /* under the minimum: not kept */
  put(s, 1, N, STATE, true);
  put(s, 1, N, STATE, true); /* the same tokens: replaced, not a second */
  assert(slots_in_ram(s) == 1);
  put(s, 2, N + 300, STATE * 2, true);
  take_back(s, 1, N, STATE);
  take_back(s, 2, N + 300, STATE * 2);
  put(s, 3, N, STATE, true), put(s, 4, N, STATE, true), put(s, 5, N, STATE, true); /* past the cap: 3 dropped */
  assert(slots_in_ram(s) == 2);
  uint32_t len;
  int32_t *c3 = conv(3, N);
  assert(slots_find(s, c3, N, 0, &len) < 0);
  free(c3);
  assert(!slots_flush(s, err)); /* no folder */
  slots_destroy(s);
  printf("  RAM: the minimum, the same tokens replaced, the longest prefix past `beyond` taken with its bytes, the "
         "cap's least recently used dropped\n");

  /* the folder past RAM */
  o = opts(dir, 1, 0, 0, 7);
  assert((s = slots_create(&o, err)));
  put(s, 1, N, STATE, true), put(s, 2, N, STATE, true); /* 1 to the folder */
  assert(slots_in_ram(s) == 1 && slots_in_dir(s) == 1 && files(".bin") == 1 && files(".meta") == 1);
  assert(slots_dir_bytes(s) == STATE);
  take_back(s, 1, N, STATE); /* from the folder: its files gone */
  assert(slots_in_dir(s) == 0 && files(".bin") == 0 && files(".meta") == 0 && files(".tmp") == 0);
  /* a flush, and a new store over the folder: a restart */
  put(s, 3, N + 10, STATE, true);
  assert(slots_flush(s, err) && slots_in_dir(s) == 2 && slots_in_ram(s) == 0 && files(".bin") == 2);
  slots_destroy(s);
  assert((s = slots_create(&o, err)) && slots_in_dir(s) == 2);
  take_back(s, 3, N + 10, STATE);
  take_back(s, 2, N, STATE);
  slots_destroy(s);
  printf("  the folder: RAM's least recently used written there and taken back (its files removed); a flush, a new "
         "store over it, every conversation found\n");

  /* removed at start: another identity's; a damaged .meta; a .bin not its size; a .bin alone */
  assert((s = slots_create(&o, err)));
  put(s, 1, N, STATE, true), put(s, 2, N, STATE, true), slots_flush(s, err);
  slots_destroy(s);
  SlotsOptions other = opts(dir, 1, 0, 0, 8);
  assert((s = slots_create(&other, err)) && slots_in_dir(s) == 0 && files(".bin") == 0 && files(".meta") == 0);
  slots_destroy(s);
  char path[700];
  for (int damage = 0; damage < 3; damage++)
  {
    clear();
    assert((s = slots_create(&o, err)));
    put(s, 1, N, STATE, true), put(s, 2, N, STATE, true), slots_flush(s, err);
    slots_destroy(s);
    if (damage == 0) /* a .meta's byte */
    {
      first(".meta", path, sizeof path);
      FILE *f = fopen(path, "r+b");
      assert(f && fseek(f, 40, SEEK_SET) == 0);
      fputc(0x5A, f);
      fclose(f);
    }
    else if (damage == 1) /* a .bin cut short */
    {
      first(".bin", path, sizeof path);
      assert(truncate(path, STATE - 1) == 0);
    }
    else /* a .meta removed: its .bin alone */
    {
      first(".meta", path, sizeof path);
      assert(unlink(path) == 0);
    }
    assert((s = slots_create(&o, err)) && slots_in_dir(s) == 1 && files(".bin") == 1 && files(".meta") == 1);
    slots_destroy(s);
  }
  printf("  at start, removed: another identity's conversations, a damaged .meta, a .bin cut short, a .bin alone\n");

  /* the folder's cap: two states' bytes; a third written removes the least recently used */
  clear();
  SlotsOptions capped = opts(dir, 1, 2.0 * STATE / (1u << 30), 0, 7);
  assert((s = slots_create(&capped, err)));
  put(s, 1, N, STATE, true), put(s, 2, N, STATE, true), put(s, 3, N, STATE, true), put(s, 4, N, STATE, true);
  assert(slots_in_dir(s) == 2 && slots_dir_bytes(s) <= 2 * STATE); /* 1 removed; 2, 3 in the folder, 4 in RAM */
  int32_t *c1 = conv(1, N), *c2 = conv(2, N);
  assert(slots_find(s, c1, N, 0, &len) < 0 && slots_find(s, c2, N, 0, &len) >= 0);
  free(c1), free(c2);
  put(s, 5, N, STATE * 3, true); /* 4 to the folder; then 5, larger than the cap alone, written last at a flush */
  int32_t *c5 = conv(5, N);
  assert(slots_flush(s, err) && slots_find(s, c5, N, 0, &len) >= 0 && slots_in_dir(s) == 1);
  free(c5);
  slots_destroy(s);
  printf("  the folder's cap: its least recently used removed, the one just written kept (even past the cap)\n");

  /* older than `days`: removed at start */
  clear();
  assert((s = slots_create(&o, err)));
  put(s, 1, N, STATE, true), slots_flush(s, err);
  slots_destroy(s);
  sleep(2);
  SlotsOptions young = opts(dir, 1, 0, 1.0 / 86400.0, 7); /* a second */
  assert((s = slots_create(&young, err)) && slots_in_dir(s) == 0 && files(".bin") == 0);
  slots_destroy(s);
  printf("  a conversation unused past `days`: removed at start\n");

  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_slots] ok\n");
  return 0;
}
