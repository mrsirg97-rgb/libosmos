#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "osmos/serve/slots.h"

/* the kept conversations (slots.h): an array of entries, each in RAM (its state) or in the folder (its files), every
   one's tokens in memory for the prefix match. a few dozen at most: linear scans */

#define FNV_SEED 1469598103934665603ull
#define FNV_PRIME 1099511628211ull
#define NAME 32 /* "slot-" + 16 hex digits + an extension */

typedef struct Entry
{
  int32_t *tokens;
  uint32_t n;
  uint8_t *state; /* RAM's (NULL: in the folder) */
  uint64_t bytes;
  uint64_t used;  /* when last used (ns, the wall clock: kept in the .meta across runs) */
  char name[NAME]; /* the folder's files' stem (slot-<hash>), its tokens' hash */
} Entry;

struct Slots
{
  SlotsOptions o;
  char *dir;
  Entry *e;
  uint32_t n, cap;
};

typedef struct Meta /* a .meta's fixed part; then the tokens (int32 x n), then the FNV-1a of everything before it */
{
  uint64_t magic, identity, bytes, used;
  uint32_t n, pad;
} Meta;

static uint64_t fnv(const void *p, uint64_t n, uint64_t h)
{
  const uint8_t *b = p;
  for (uint64_t i = 0; i < n; i++)
    h = (h ^ b[i]) * FNV_PRIME;
  return h;
}

static bool fail(char *err, const char *why)
{
  if (err)
    snprintf(err, SLOTS_ERR, "slots: %s", why);
  return false;
}

static uint64_t now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* a strictly later stamp than any before it in this run (two puts within a clock tick still ordered) */
static uint64_t stamp(void)
{
  static uint64_t last;
  const uint64_t t = now_ns();
  last = t > last ? t : last + 1;
  return last;
}

static void path(const Slots *s, const char *name, const char *ext, char *out, size_t cap)
{
  snprintf(out, cap, "%s/%s%s", s->dir, name, ext);
}

static void unlink_entry(const Slots *s, const char *name)
{
  char p[4096];
  path(s, name, ".bin", p, sizeof p), unlink(p);
  path(s, name, ".meta", p, sizeof p), unlink(p);
}

static void drop(Slots *s, uint32_t i)
{
  free(s->e[i].tokens), free(s->e[i].state);
  s->e[i] = s->e[--s->n];
}

static bool grow(Slots *s)
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

static bool write_meta(const Slots *s, const Entry *e)
{
  const Meta m = {SLOTS_META_MAGIC, s->o.identity, e->bytes, e->used, e->n, 0};
  const uint64_t tb = (uint64_t)e->n * sizeof *e->tokens;
  uint8_t *buf = malloc(sizeof m + tb + sizeof(uint64_t));
  if (!buf)
    return false;
  memcpy(buf, &m, sizeof m), memcpy(buf + sizeof m, e->tokens, tb);
  const uint64_t sum = fnv(buf, sizeof m + tb, FNV_SEED);
  memcpy(buf + sizeof m + tb, &sum, sizeof sum);
  char p[4096];
  path(s, e->name, ".meta", p, sizeof p);
  const bool ok = write_file(p, buf, sizeof m + tb + sizeof sum, NULL, 0);
  free(buf);
  return ok;
}

/* entry i from RAM to the folder (its .bin, then its .meta: a .meta names a whole .bin) */
static bool to_dir(Slots *s, uint32_t i)
{
  Entry *e = &s->e[i];
  char p[4096];
  path(s, e->name, ".bin", p, sizeof p);
  if (!write_file(p, e->state, e->bytes, NULL, 0) || !write_meta(s, e))
  {
    unlink_entry(s, e->name);
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
static int64_t lru(const Slots *s, bool dir, int64_t keep)
{
  int64_t best = -1;
  for (uint32_t i = 0; i < s->n; i++)
    if ((s->e[i].state == NULL) == dir && (int64_t)i != keep && (best < 0 || s->e[i].used < s->e[best].used))
      best = i;
  return best;
}

/* the folder under gb: its least recently used removed, never `keep` */
static void trim_dir(Slots *s, int64_t keep)
{
  if (!s->dir || s->o.gb <= 0)
    return;
  const uint64_t cap = (uint64_t)(s->o.gb * (double)(1ull << 30));
  while (slots_dir_bytes(s) > cap)
  {
    const int64_t i = lru(s, true, keep);
    if (i < 0)
      return;
    unlink_entry(s, s->e[i].name);
    drop(s, (uint32_t)i);
    if (keep == (int64_t)s->n) /* (drop moved the last entry into i) */
      keep = i;
  }
}

/* RAM past its caps (conversations, bytes): the least recently used to the folder (or dropped without one); false
   for a write that failed */
static bool trim_ram(Slots *s)
{
  while (slots_in_ram(s) > s->o.cap || slots_ram_bytes(s) > s->o.ram_bytes)
  {
    const int64_t i = lru(s, false, -1);
    if (!s->dir)
    {
      drop(s, (uint32_t)i);
      continue;
    }
    if (!to_dir(s, (uint32_t)i))
    {
      drop(s, (uint32_t)i);
      return false;
    }
    trim_dir(s, i);
  }
  return true;
}

/* a .meta read and checked: its tokens (malloc'd) and fixed part; false for anything not this store's or damaged */
static bool read_meta(const Slots *s, const char *p, Meta *m, int32_t **tokens)
{
  FILE *f = fopen(p, "rb");
  if (!f)
    return false;
  bool ok = fread(m, sizeof *m, 1, f) == 1 && m->magic == SLOTS_META_MAGIC && m->identity == s->o.identity &&
            m->n > 0 && m->n <= (1u << 26);
  *tokens = ok ? malloc((uint64_t)m->n * sizeof **tokens) : NULL;
  uint64_t sum = 0;
  ok = ok && *tokens && fread(*tokens, sizeof **tokens, m->n, f) == m->n && fread(&sum, sizeof sum, 1, f) == 1 &&
       fgetc(f) == EOF;
  fclose(f);
  if (ok)
  {
    uint64_t h = fnv(m, sizeof *m, FNV_SEED);
    h = fnv(*tokens, (uint64_t)m->n * sizeof **tokens, h);
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

/* the folder indexed: each .meta checked against its .bin and the clock; whatever is not a whole slot removed */
static bool index_dir(Slots *s, char *err)
{
  DIR *d = opendir(s->dir);
  if (!d)
    return fail(err, "the folder cannot be read");
  const uint64_t now = now_ns();
  for (struct dirent *de; (de = readdir(d));)
  {
    const char *nm = de->d_name;
    if (strncmp(nm, "slot-", 5) != 0)
      continue;
    char p[4096];
    snprintf(p, sizeof p, "%s/%s", s->dir, nm);
    if (has_ext(nm, ".tmp")) /* a write that never ended */
    {
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
    path(s, stem, ".bin", bp, sizeof bp);
    struct stat st;
    const bool young = s->o.days <= 0;
    bool ok = read_meta(s, p, &m, &t) && stat(bp, &st) == 0 && (uint64_t)st.st_size == m.bytes;
    ok = ok && (young || (double)(now > m.used ? now - m.used : 0) <= s->o.days * 86400e9);
    if (!ok || !grow(s))
    {
      free(t);
      unlink_entry(s, stem);
      continue;
    }
    Entry *e = &s->e[s->n++];
    *e = (Entry){t, m.n, NULL, m.bytes, m.used, {0}};
    memcpy(e->name, stem, ls + 1);
  }
  closedir(d);
  /* a .bin with no entry (its .meta gone or removed above) */
  if (!(d = opendir(s->dir)))
    return fail(err, "the folder cannot be read");
  for (struct dirent *de; (de = readdir(d));)
  {
    const char *nm = de->d_name;
    if (strncmp(nm, "slot-", 5) != 0 || !has_ext(nm, ".bin"))
      continue;
    bool known = false;
    for (uint32_t i = 0; i < s->n && !known; i++)
      known = strlen(s->e[i].name) == strlen(nm) - 4 && !strncmp(s->e[i].name, nm, strlen(nm) - 4);
    if (!known)
    {
      char p[4096];
      snprintf(p, sizeof p, "%s/%s", s->dir, nm);
      unlink(p);
    }
  }
  closedir(d);
  trim_dir(s, -1);
  return true;
}

Slots *slots_create(const SlotsOptions *o, char err[SLOTS_ERR])
{
  if (!o || !o->cap)
    return fail(err, "no options, or a cap of 0"), NULL;
  Slots *s = calloc(1, sizeof *s);
  if (!s)
    return fail(err, "out of memory"), NULL;
  s->o = *o;
  if (o->dir)
  {
    if (!(s->dir = strdup(o->dir)) || (mkdir(s->dir, 0755) != 0 && errno != EEXIST) || !index_dir(s, err))
    {
      if (s->dir && err && !err[0])
        fail(err, "the folder cannot be made");
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

static bool same(const Entry *e, const int32_t *t, uint32_t n)
{
  return e->n == n && !memcmp(e->tokens, t, (uint64_t)n * sizeof *t);
}

bool slots_keeps(const Slots *s, uint32_t n) { return s && n >= s->o.min_tokens; }

bool slots_put(Slots *s, const int32_t *tokens, uint32_t n, void *state, uint64_t bytes, bool *kept,
               char err[SLOTS_ERR])
{
  if (kept)
    *kept = false;
  if (!s || !tokens || !state || !bytes)
    return free(state), fail(err, "no store, tokens or state");
  const bool past_ram = bytes > s->o.ram_bytes; /* RAM could not hold it alone: the folder's, or not kept */
  if (n < s->o.min_tokens || (past_ram && !s->dir))
    return free(state), true;
  for (uint32_t i = 0; i < s->n; i++) /* the same conversation kept already: this one in its place */
    if (same(&s->e[i], tokens, n))
    {
      if (!s->e[i].state)
        unlink_entry(s, s->e[i].name);
      drop(s, i);
      break;
    }
  Entry e = {malloc((uint64_t)n * sizeof *tokens), n, state, bytes, stamp(), {0}};
  if (!e.tokens || !grow(s))
  {
    free(e.tokens), free(state);
    return fail(err, "out of memory");
  }
  memcpy(e.tokens, tokens, (uint64_t)n * sizeof *tokens);
  snprintf(e.name, sizeof e.name, "slot-%016llx", (unsigned long long)fnv(tokens, (uint64_t)n * sizeof *tokens, FNV_SEED));
  s->e[s->n++] = e;
  if (past_ram) /* straight to the folder: the ones in RAM stay */
  {
    if (!to_dir(s, s->n - 1))
      return drop(s, s->n - 1), fail(err, "a conversation could not be written to the folder");
    trim_dir(s, s->n - 1);
  }
  if (!trim_ram(s))
    return fail(err, "a conversation could not be written to the folder");
  if (kept)
    *kept = true;
  return true;
}

int64_t slots_find(const Slots *s, const int32_t *prompt, uint32_t n, uint32_t beyond, uint32_t *len)
{
  int64_t best = -1;
  for (uint32_t i = 0; s && prompt && i < s->n; i++)
  {
    const Entry *e = &s->e[i];
    if (e->n > n || e->n <= beyond || (best >= 0 && e->n <= s->e[best].n))
      continue;
    if (!memcmp(e->tokens, prompt, (uint64_t)e->n * sizeof *prompt))
      best = i;
  }
  if (best >= 0 && len)
    *len = s->e[best].n;
  return best;
}

/* a file as a state's sink or source: its bytes in order, counted */
typedef struct
{
  FILE *f;
  uint64_t done;
} FilePieces;

static bool file_sink(void *ctx, const void *bytes, uint64_t n)
{
  FilePieces *p = ctx;
  if (fwrite(bytes, 1, n, p->f) != n)
    return false;
  p->done += n;
  return true;
}

static bool file_source(void *ctx, void *bytes, uint64_t n)
{
  FilePieces *p = ctx;
  if (fread(bytes, 1, n, p->f) != n)
    return false;
  p->done += n;
  return true;
}

/* RAM's state as a source */
typedef struct
{
  const uint8_t *p;
  uint64_t at, n;
} MemPieces;

static bool mem_source(void *ctx, void *bytes, uint64_t n)
{
  MemPieces *m = ctx;
  if (n > m->n - m->at)
    return false;
  memcpy(bytes, m->p + m->at, n), m->at += n;
  return true;
}

bool slots_put_stream(Slots *s, const int32_t *tokens, uint32_t n, uint64_t bytes, SlotsSave save, void *ctx,
                      bool *kept, char err[SLOTS_ERR])
{
  if (kept)
    *kept = false;
  if (!s || !tokens || !save || !bytes)
    return fail(err, "no store, tokens or save");
  if (n < s->o.min_tokens || !s->dir)
    return true;
  for (uint32_t i = 0; i < s->n; i++) /* the same conversation kept already: this one in its place */
    if (same(&s->e[i], tokens, n))
    {
      if (!s->e[i].state)
        unlink_entry(s, s->e[i].name);
      drop(s, i);
      break;
    }
  Entry e = {malloc((uint64_t)n * sizeof *tokens), n, NULL, bytes, stamp(), {0}};
  if (!e.tokens || !grow(s))
    return free(e.tokens), fail(err, "out of memory");
  memcpy(e.tokens, tokens, (uint64_t)n * sizeof *tokens);
  snprintf(e.name, sizeof e.name, "slot-%016llx", (unsigned long long)fnv(tokens, (uint64_t)n * sizeof *tokens, FNV_SEED));
  char bin[4096], tmp[4200];
  path(s, e.name, ".bin", bin, sizeof bin);
  snprintf(tmp, sizeof tmp, "%s.tmp", bin);
  FilePieces fp = {fopen(tmp, "wb"), 0};
  bool ok = fp.f && save(ctx, file_sink, &fp) && fp.done == bytes && fflush(fp.f) == 0 && fsync(fileno(fp.f)) == 0;
  if (fp.f && fclose(fp.f) != 0)
    ok = false;
  if (!ok || rename(tmp, bin) != 0)
  {
    unlink(tmp), free(e.tokens);
    return fail(err, "a conversation's state could not be written to the folder");
  }
  s->e[s->n++] = e;
  if (!write_meta(s, &s->e[s->n - 1]))
  {
    unlink_entry(s, s->e[s->n - 1].name), drop(s, s->n - 1);
    return fail(err, "a conversation's .meta could not be written");
  }
  trim_dir(s, s->n - 1);
  if (kept)
    *kept = true;
  return true;
}

bool slots_take_stream(Slots *s, int64_t id, SlotsLoad load, void *ctx, char err[SLOTS_ERR])
{
  if (!s || id < 0 || (uint64_t)id >= s->n || !load)
    return fail(err, "no such conversation, or no load");
  Entry *e = &s->e[id];
  bool ok;
  if (e->state) /* RAM's */
  {
    MemPieces m = {e->state, 0, e->bytes};
    ok = load(ctx, e->bytes, mem_source, &m) && m.at == e->bytes;
  }
  else /* the folder's: read in pieces, its files removed */
  {
    char p[4096];
    path(s, e->name, ".bin", p, sizeof p);
    FilePieces fp = {fopen(p, "rb"), 0};
    ok = fp.f && load(ctx, e->bytes, file_source, &fp) && fp.done == e->bytes;
    if (fp.f)
      fclose(fp.f);
    unlink_entry(s, e->name);
  }
  drop(s, (uint32_t)id);
  return ok || fail(err, "a kept conversation's state could not be taken back");
}

bool slots_flush(Slots *s, char err[SLOTS_ERR])
{
  if (!s || !s->dir)
    return fail(err, "no folder");
  int64_t last = -1;
  uint64_t newest = 0;
  for (uint32_t i = 0; i < s->n; i++)
  {
    if (!s->e[i].state)
      continue;
    if (!to_dir(s, i))
    {
      drop(s, i);
      return fail(err, "a conversation could not be written to the folder");
    }
    if (s->e[i].used >= newest)
      newest = s->e[i].used, last = i;
  }
  trim_dir(s, last); /* (the most recently used kept: the last a run set aside) */
  return true;
}

bool slots_room(Slots *s, uint64_t bytes, char err[SLOTS_ERR])
{
  if (!s || bytes > s->o.ram_bytes)
    return fail(err, "a state larger than RAM's room for kept conversations");
  while (slots_ram_bytes(s) + bytes > s->o.ram_bytes)
  {
    const int64_t i = lru(s, false, -1);
    if (!s->dir)
      drop(s, (uint32_t)i);
    else if (!to_dir(s, (uint32_t)i))
      return drop(s, (uint32_t)i), fail(err, "a conversation could not be written to the folder");
    else
      trim_dir(s, i);
  }
  return true;
}
