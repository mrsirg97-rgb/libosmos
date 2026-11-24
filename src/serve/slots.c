/* the kept conversations' store (osmos/serve/slots.h, slots_impl.h): each in RAM or the folder, its files
   written under a temporary name and renamed, the least recently used trimmed, the folder indexed at start */
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "osmos/serve/slots.h"
#include "osmos/err.h"
#include "slots_impl.h"

uint64_t slots_fnv(const void *p, uint64_t n, uint64_t h)
{
  const uint8_t *b = p;
  for (uint64_t i = 0; i < n; i++)
    h = (h ^ b[i]) * FNV_PRIME;
  return h;
}

bool slots_fail(char *err, const char *why) { return osmos_fail(err, "slots: %s", why); }

static uint64_t now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* a strictly later stamp than any before it in this run (two puts within a clock tick still ordered) */
uint64_t slots_stamp(void)
{
  static uint64_t last;
  const uint64_t t = now_ns();
  last = t > last ? t : last + 1;
  return last;
}

void slots_path(const Slots *s, const char *name, const char *ext, char *out, size_t cap)
{
  snprintf(out, cap, "%s/%s%s", s->dir, name, ext);
}

void slots_unlink_entry(const Slots *s, const char *name)
{
  char p[4096];
  slots_path(s, name, ".bin", p, sizeof p), unlink(p);
  slots_path(s, name, ".meta", p, sizeof p), unlink(p);
}

void slots_drop(Slots *s, uint32_t i)
{
  free(s->e[i].tokens), free(s->e[i].state);
  s->e[i] = s->e[--s->n];
}

bool slots_grow(Slots *s)
{
  if (s->n < s->cap)
    return true;
  const uint32_t cap = s->cap ? 2 * s->cap : 16;
  Entry *e = realloc(s->e, cap * sizeof *e);
  if (!e)
    return false;
  s->e = e, s->cap = cap;
  return true;
}

/* bytes to a file under a temporary name, then renamed: the name holds the whole file or none */
static bool write_file(const char *p, const void *a, uint64_t na, const void *b, uint64_t nb)
{
  char tmp[4200];
  snprintf(tmp, sizeof tmp, "%s.tmp", p);
  FILE *f = fopen(tmp, "wb");
  if (!f)
    return false;
  const bool ok = fwrite(a, 1, na, f) == na && (!nb || fwrite(b, 1, nb, f) == nb) && fflush(f) == 0 &&
                  fsync(fileno(f)) == 0;
  if (fclose(f) != 0 || !ok || rename(tmp, p) != 0)
  {
    unlink(tmp);
    return false;
  }
  return true;
}

bool slots_write_meta(const Slots *s, const Entry *e)
{
  const Meta m = {SLOTS_META_MAGIC, s->o.identity, e->bytes, e->used, e->n, 0};
  const uint64_t tb = (uint64_t)e->n * sizeof *e->tokens;
  uint8_t *buf = malloc(sizeof m + tb + sizeof(uint64_t));
  if (!buf)
    return false;
  memcpy(buf, &m, sizeof m), memcpy(buf + sizeof m, e->tokens, tb);
  const uint64_t sum = slots_fnv(buf, sizeof m + tb, FNV_SEED);
  memcpy(buf + sizeof m + tb, &sum, sizeof sum);
  char p[4096];
  slots_path(s, e->name, ".meta", p, sizeof p);
  const bool ok = write_file(p, buf, sizeof m + tb + sizeof sum, NULL, 0);
  free(buf);
  return ok;
}

/* entry i from RAM to the folder (its .bin, then its .meta: a .meta names a whole .bin) */
bool slots_to_dir(Slots *s, uint32_t i)
{
  Entry *e = &s->e[i];
  char p[4096];
  slots_path(s, e->name, ".bin", p, sizeof p);
  if (!write_file(p, e->state, e->bytes, NULL, 0) || !slots_write_meta(s, e))
  {
    slots_unlink_entry(s, e->name);
    return false;
  }
  free(e->state);
  e->state = NULL;
  return true;
}

uint32_t slots_in_ram(const Slots *s)
{
  uint32_t k = 0;
  for (uint32_t i = 0; s && i < s->n; i++)
    k += s->e[i].state != NULL;
  return k;
}

uint32_t slots_in_dir(const Slots *s) { return s ? s->n - slots_in_ram(s) : 0; }

uint64_t slots_ram_bytes(const Slots *s)
{
  uint64_t b = 0;
  for (uint32_t i = 0; s && i < s->n; i++)
    b += s->e[i].state ? s->e[i].bytes : 0;
  return b;
}

uint64_t slots_dir_bytes(const Slots *s)
{
  uint64_t b = 0;
  for (uint32_t i = 0; s && i < s->n; i++)
    b += s->e[i].state ? 0 : s->e[i].bytes;
  return b;
}

/* the least recently used entry in RAM (dir false) or in the folder (true), other than `keep`; -1 for none */
int64_t slots_lru(const Slots *s, bool dir, int64_t keep)
{
  int64_t best = -1;
  for (uint32_t i = 0; i < s->n; i++)
    if ((s->e[i].state == NULL) == dir && (int64_t)i != keep && (best < 0 || s->e[i].used < s->e[best].used))
      best = i;
  return best;
}

/* the folder under gb: its least recently used removed, never `keep` */
void slots_trim_dir(Slots *s, int64_t keep)
{
  if (!s->dir || s->o.gb <= 0)
    return;
  const uint64_t cap = (uint64_t)(s->o.gb * (double)(1ull << 30));
  while (slots_dir_bytes(s) > cap)
  {
    const int64_t i = slots_lru(s, true, keep);
    if (i < 0)
      return;
    slots_unlink_entry(s, s->e[i].name);
    slots_drop(s, (uint32_t)i);
    if (keep == (int64_t)s->n) /* (drop moved the last entry into i) */
      keep = i;
  }
}

/* RAM past its caps (conversations, bytes): the least recently used to the folder (or dropped without one); false
   for a write that failed */
bool slots_trim_ram(Slots *s)
{
  while (slots_in_ram(s) > s->o.cap || slots_ram_bytes(s) > s->o.ram_bytes)
  {
    const int64_t i = slots_lru(s, false, -1);
    if (!s->dir)
    {
      slots_drop(s, (uint32_t)i);
      continue;
    }
    if (!slots_to_dir(s, (uint32_t)i))
    {
      slots_drop(s, (uint32_t)i);
      return false;
    }
    slots_trim_dir(s, i);
  }
  return true;
}

/* a .meta read and checked: its tokens (malloc'd) and fixed part; false for anything not this store's or damaged -
   *foreign for a whole one of another identity (another model's, or this one's with other options: another server's,
   left alone) */
static bool read_meta(const Slots *s, const char *p, Meta *m, int32_t **tokens, bool *foreign)
{
  *foreign = false;
  FILE *f = fopen(p, "rb");
  if (!f)
    return false;
  bool ok = fread(m, sizeof *m, 1, f) == 1 && m->magic == SLOTS_META_MAGIC;
  if (ok && m->identity != s->o.identity)
    return fclose(f), *foreign = true, false;
  ok = ok && m->n > 0 && m->n <= (1u << 26);
  *tokens = ok ? malloc((uint64_t)m->n * sizeof **tokens) : NULL;
  uint64_t sum = 0;
  ok = ok && *tokens && fread(*tokens, sizeof **tokens, m->n, f) == m->n && fread(&sum, sizeof sum, 1, f) == 1 &&
       fgetc(f) == EOF;
  fclose(f);
  if (ok)
  {
    uint64_t h = slots_fnv(m, sizeof *m, FNV_SEED);
    h = slots_fnv(*tokens, (uint64_t)m->n * sizeof **tokens, h);
    ok = h == sum;
  }
  if (!ok)
    free(*tokens), *tokens = NULL;
  return ok;
}

static bool has_ext(const char *name, const char *ext)
{
  const size_t a = strlen(name), b = strlen(ext);
  return a > b && !strcmp(name + a - b, ext);
}

/* a file older than an hour: another server's write would be newer (a write's .tmp, a .bin before its .meta) */
#define STALE_NS (3600ull * 1000000000ull)
static bool stale(const char *p, uint64_t now)
{
  struct stat st;
  if (stat(p, &st) != 0)
    return false;
  const uint64_t m = (uint64_t)st.st_mtim.tv_sec * 1000000000ull + (uint64_t)st.st_mtim.tv_nsec;
  return now > m && now - m > STALE_NS;
}

/* the folder indexed: each .meta checked against its .bin and the clock; this store's broken or old slots removed -
   another identity's left, and a write's files left until stale (two servers may share one folder) */
static bool index_dir(Slots *s, char *err)
{
  DIR *d = opendir(s->dir);
  if (!d)
    return slots_fail(err, "the folder cannot be read");
  const uint64_t now = now_ns();
  for (struct dirent *de; (de = readdir(d));)
  {
    const char *nm = de->d_name;
    if (strncmp(nm, "slot-", 5) != 0)
      continue;
    char p[4096];
    snprintf(p, sizeof p, "%s/%s", s->dir, nm);
    if (has_ext(nm, ".tmp")) /* a write that never ended - or another server's, still going */
    {
      if (stale(p, now))
        unlink(p);
      continue;
    }
    if (!has_ext(nm, ".meta"))
      continue;
    char stem[NAME];
    const size_t ls = strlen(nm) - strlen(".meta");
    if (ls >= sizeof stem)
      continue;
    memcpy(stem, nm, ls), stem[ls] = 0;
    Meta m;
    int32_t *t = NULL;
    char bp[4096];
    slots_path(s, stem, ".bin", bp, sizeof bp);
    struct stat st;
    const bool young = s->o.days <= 0;
    bool foreign = false;
    bool ok = read_meta(s, p, &m, &t, &foreign) && stat(bp, &st) == 0 && (uint64_t)st.st_size == m.bytes;
    if (foreign)
      continue; /* another identity's slot: its server's to keep or drop */
    ok = ok && (young || (double)(now > m.used ? now - m.used : 0) <= s->o.days * 86400e9);
    if (!ok || !slots_grow(s))
    {
      free(t);
      slots_unlink_entry(s, stem);
      continue;
    }
    Entry *e = &s->e[s->n++];
    *e = (Entry){t, m.n, NULL, m.bytes, m.used, {0}};
    memcpy(e->name, stem, ls + 1);
  }
  closedir(d);
  /* a .bin with no entry (its .meta gone or removed above) */
  if (!(d = opendir(s->dir)))
    return slots_fail(err, "the folder cannot be read");
  for (struct dirent *de; (de = readdir(d));)
  {
    const char *nm = de->d_name;
    if (strncmp(nm, "slot-", 5) != 0 || !has_ext(nm, ".bin"))
      continue;
    bool known = false;
    for (uint32_t i = 0; i < s->n && !known; i++)
      known = strlen(s->e[i].name) == strlen(nm) - 4 && !strncmp(s->e[i].name, nm, strlen(nm) - 4);
    char p[4096], mp[4096];
    snprintf(p, sizeof p, "%s/%s", s->dir, nm);
    snprintf(mp, sizeof mp, "%.*s.meta", (int)(strlen(p) - 4), p);
    if (!known && access(mp, F_OK) != 0 && stale(p, now)) /* no .meta, and not a write still going */
      unlink(p);
  }
  closedir(d);
  slots_trim_dir(s, -1);
  return true;
}

Slots *slots_create(const SlotsOptions *o, char err[OSMOS_ERR])
{
  if (!o || !o->cap)
    return slots_fail(err, "no options, or a cap of 0"), NULL;
  Slots *s = calloc(1, sizeof *s);
  if (!s)
    return slots_fail(err, "out of memory"), NULL;
  s->o = *o;
  if (o->dir)
  {
    if (!(s->dir = strdup(o->dir)) || (mkdir(s->dir, 0755) != 0 && errno != EEXIST) || !index_dir(s, err))
    {
      if (s->dir && err && !err[0])
        slots_fail(err, "the folder cannot be made");
      slots_destroy(s);
      return NULL;
    }
  }
  return s;
}

void slots_destroy(Slots *s)
{
  if (!s)
    return;
  for (uint32_t i = 0; i < s->n; i++)
    free(s->e[i].tokens), free(s->e[i].state);
  free(s->e), free(s->dir), free(s);
}
