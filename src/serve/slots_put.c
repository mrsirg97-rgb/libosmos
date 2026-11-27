/* a kept conversation put, found and streamed (osmos/serve/slots.h, slots_impl.h): a state into RAM or the
   folder, the longest prefix found, a state streamed in and out in pieces, RAM's flushed, room made */
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

static bool same(const Entry *e, const int32_t *t, uint32_t n)
{
  return e->n == n && !memcmp(e->tokens, t, (uint64_t)n * sizeof *t);
}

static void slot_name(const Slots *s, const int32_t *tokens, uint32_t n, char out[32])
{
  uint64_t h = slots_fnv(&s->o.identity, sizeof s->o.identity, FNV_SEED);
  h = slots_fnv(tokens, (uint64_t)n * sizeof *tokens, h);
  snprintf(out, 32, "slot-%016llx", (unsigned long long)h);
}

bool slots_keeps(const Slots *s, uint32_t n) { return s && n >= s->o.min_tokens; }

bool slots_put(Slots *s, const int32_t *tokens, uint32_t n, void *state, uint64_t bytes, bool *kept,
               char err[OSMOS_ERR])
{
  if (kept)
    *kept = false;
  if (!s || !tokens || !state || !bytes)
    return free(state), slots_fail(err, "no store, tokens or state");
  const bool past_ram = bytes > s->o.ram_bytes; /* RAM could not hold it alone: the folder's, or not kept */
  if (n < s->o.min_tokens || (past_ram && !s->dir))
    return free(state), true;
  for (uint32_t i = 0; i < s->n; i++) /* the same conversation kept already: this one in its place */
    if (same(&s->e[i], tokens, n))
    {
      if (!s->e[i].state)
        slots_unlink_entry(s, s->e[i].name);
      slots_drop(s, i);
      break;
    }
  Entry e = {malloc((uint64_t)n * sizeof *tokens), n, state, bytes, slots_stamp(), {0}};
  if (!e.tokens || !slots_grow(s))
  {
    free(e.tokens), free(state);
    return slots_fail(err, "out of memory");
  }
  memcpy(e.tokens, tokens, (uint64_t)n * sizeof *tokens);
  slot_name(s, tokens, n, e.name);
  s->e[s->n++] = e;
  if (past_ram) /* straight to the folder: the ones in RAM stay */
  {
    if (!slots_to_dir(s, s->n - 1))
      return slots_drop(s, s->n - 1), slots_fail(err, "a conversation could not be written to the folder");
    slots_trim_dir(s, s->n - 1);
  }
  if (!slots_trim_ram(s))
    return slots_drop(s, s->n - 1), slots_fail(err, "a conversation could not be written to the folder");
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
                      bool *kept, char err[OSMOS_ERR])
{
  if (kept)
    *kept = false;
  if (!s || !tokens || !save || !bytes)
    return slots_fail(err, "no store, tokens or save");
  if (n < s->o.min_tokens || !s->dir)
    return true;
  for (uint32_t i = 0; i < s->n; i++) /* the same conversation kept already: this one in its place */
    if (same(&s->e[i], tokens, n))
    {
      if (!s->e[i].state)
        slots_unlink_entry(s, s->e[i].name);
      slots_drop(s, i);
      break;
    }
  Entry e = {malloc((uint64_t)n * sizeof *tokens), n, NULL, bytes, slots_stamp(), {0}};
  if (!e.tokens || !slots_grow(s))
    return free(e.tokens), slots_fail(err, "out of memory");
  memcpy(e.tokens, tokens, (uint64_t)n * sizeof *tokens);
  slot_name(s, tokens, n, e.name);
  char bin[4096], tmp[4200];
  slots_path(s, e.name, ".bin", bin, sizeof bin);
  snprintf(tmp, sizeof tmp, "%s.tmp", bin);
  FilePieces fp = {fopen(tmp, "wb"), 0};
  bool ok = fp.f && save(ctx, file_sink, &fp) && fp.done == bytes && fflush(fp.f) == 0 && fsync(fileno(fp.f)) == 0;
  if (fp.f && fclose(fp.f) != 0)
    ok = false;
  if (!ok || rename(tmp, bin) != 0)
  {
    unlink(tmp), free(e.tokens);
    return slots_fail(err, "a conversation's state could not be written to the folder");
  }
  s->e[s->n++] = e;
  if (!slots_write_meta(s, &s->e[s->n - 1]))
  {
    slots_unlink_entry(s, s->e[s->n - 1].name), slots_drop(s, s->n - 1);
    return slots_fail(err, "a conversation's .meta could not be written");
  }
  slots_trim_dir(s, s->n - 1);
  if (kept)
    *kept = true;
  return true;
}

bool slots_take_stream(Slots *s, int64_t id, SlotsLoad load, void *ctx, char err[OSMOS_ERR])
{
  if (!s || id < 0 || (uint64_t)id >= s->n || !load)
    return slots_fail(err, "no such conversation, or no load");
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
    slots_path(s, e->name, ".bin", p, sizeof p);
    FilePieces fp = {fopen(p, "rb"), 0};
    ok = fp.f && load(ctx, e->bytes, file_source, &fp) && fp.done == e->bytes;
    if (fp.f)
      fclose(fp.f);
    slots_unlink_entry(s, e->name);
  }
  slots_drop(s, (uint32_t)id);
  return ok || slots_fail(err, "a kept conversation's state could not be taken back");
}

bool slots_flush(Slots *s, char err[OSMOS_ERR])
{
  if (!s || !s->dir)
    return slots_fail(err, "no folder");
  int64_t last = -1;
  uint64_t newest = 0;
  for (uint32_t i = 0; i < s->n; i++)
  {
    if (!s->e[i].state)
      continue;
    if (!slots_to_dir(s, i))
    {
      slots_drop(s, i);
      return slots_fail(err, "a conversation could not be written to the folder");
    }
    if (s->e[i].used >= newest)
      newest = s->e[i].used, last = i;
  }
  slots_trim_dir(s, last); /* (the most recently used kept: the last a run set aside) */
  return true;
}

bool slots_room(Slots *s, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || bytes > s->o.ram_bytes)
    return slots_fail(err, "a state larger than RAM's room for kept conversations");
  while (slots_ram_bytes(s) + bytes > s->o.ram_bytes)
  {
    const int64_t i = slots_lru(s, false, -1);
    if (!s->dir)
      slots_drop(s, (uint32_t)i);
    else if (!slots_to_dir(s, (uint32_t)i))
      return slots_drop(s, (uint32_t)i), slots_fail(err, "a conversation could not be written to the folder");
    else
      slots_trim_dir(s, i);
  }
  return true;
}
