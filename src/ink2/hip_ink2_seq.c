/* an Inkling sequence (osmos/ink2/forward.h): its tails, rollback, its saves and marks */
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ink2_impl.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/state_parts.h"
#include "osmos/model/fnv.h"
#include "osmos/err.h"
#include "osmos/err_hip.h"

#define STATE_MAGIC 0x32474e494b4e49ull /* "INKLNG2" */
#define MARK_MAGIC 0x4b52414d324b4e49ull
#define STATE_VERSION 2u /* 2: the keys and values f16 */

void *ink2_tail_row(const Ink2Tail *t, uint32_t r) { return (uint8_t *)t->d + (uint64_t)r * t->row * t->esize; }

bool ink2_tail_room(Ink2Tail *t, uint32_t n)
{
  if (t->held + n <= t->cap || n > t->cap - 2 * t->keep) /* (a step past the tail's: ink2_tail_put keeps its last) */
    return true;
  /* held > 2 keep here (cap 2 keep + a step's most): the last keep rows and the front apart */
  if (hipMemcpyAsync(t->d, ink2_tail_row(t, t->held - t->keep), (uint64_t)t->keep * t->row * t->esize,
                     hipMemcpyDeviceToDevice, NULL) != hipSuccess)
    return false;
  t->held = t->keep;
  return true;
}

const void *ink2_tail_last(const Ink2Tail *t, uint32_t k) { return ink2_tail_row(t, t->held - k); }

void *ink2_tail_claim(Ink2Tail *t, uint32_t n)
{
  if (t->held + n > t->cap)
    return NULL;
  void *at = ink2_tail_row(t, t->held);
  t->held += n;
  return at;
}

bool ink2_tail_put(Ink2Tail *t, const float *x, uint32_t n)
{
  if (t->held + n > t->cap && n >= t->keep && t->esize == 4) /* a step past an F32 tail's room: its last keep rows
                                                                 alone, all a later step or a rollback reads */
  {
    if (hipMemcpyAsync(t->d, x + (uint64_t)(n - t->keep) * t->row, (uint64_t)t->keep * t->row * 4,
                       hipMemcpyDeviceToDevice, NULL) != hipSuccess)
      return false;
    t->held = t->keep;
    return true;
  }
  if (t->held + n > t->cap)
    return false;
  void *at = ink2_tail_row(t, t->held);
  const uint64_t vals = (uint64_t)n * t->row;
  if (t->esize == INK2_KV8_RECORD   ? !ink2_to_kv8(x, vals, at, NULL)
      : t->esize == INK2_KV5_RECORD ? !ink2_to_kv5(x, vals, at, NULL)
      : t->esize == 2             ? !ink2_to_f16(x, vals, at, NULL)
                                  : hipMemcpyAsync(at, x, vals * 4, hipMemcpyDeviceToDevice, NULL) != hipSuccess)
    return false;
  t->held += n;
  return true;
}

static bool tail_make(OsmosLedger *l, Ink2Tail *t, uint32_t row, uint32_t esize, uint32_t keep, uint32_t cap,
                      uint32_t zeros)
{
  *t = (Ink2Tail){NULL, row, esize, keep, cap, zeros};
  return ledger_malloc(l, &t->d, (uint64_t)cap * row * esize) == hipSuccess &&
         hipMemset(t->d, 0, (uint64_t)cap * row * esize) == hipSuccess;
}

/* what a sequence's next token reads of each kind: the conv's taps - 1, a window's window - 1 and the keys before
   them back to a tile's start (INK2_FA_TILE - 1 at most; fewer before them all) */
static uint32_t conv_need(const Ink2Forward *f) { return (uint32_t)f->w->geo.conv - 1; }
static uint32_t win_need(const Ink2Forward *f, int32_t pos)
{
  const uint32_t w = (uint32_t)f->w->geo.window - 1 + INK2_FA_TILE - 1;
  return (uint32_t)pos < w ? (uint32_t)pos : w;
}
static uint32_t conv_row(const Ink2Forward *f, int c)
{
  const Ink2Geometry *g = &f->w->geo;
  return c == INK2_CONV_K || c == INK2_CONV_V ? (uint32_t)(g->n_kv_heads * g->head_dim) : (uint32_t)g->n_embd;
}

Ink2Seq *ink2_seq_create(Ink2Forward *f, char err[OSMOS_ERR])
{
  if (!f)
    return osmos_fail(err, "no forward"), NULL;
  Ink2Seq *s = calloc(1, sizeof *s);
  if (!s)
    return osmos_fail(err, "out of memory"), NULL;
  s->f = f;
  const Ink2Geometry *g = &f->w->geo;
  const uint32_t kd = (uint32_t)(g->n_kv_heads * g->head_dim), kv = (uint32_t)g->n_kv_heads, T = f->max_tokens;
  /* a window keeps window - 1 + INK2_FA_TILE - 1 (its keys handed in from a tile's start) and a rollback's */
  const uint32_t ck = conv_need(f) + INK2_MAX_BACK, wk = (uint32_t)g->window - 1 + INK2_FA_TILE - 1 + INK2_MAX_BACK;
  bool ok = true;
  for (int l = 0; ok && l < g->n_slots; l++)
  {
    for (int c = 0; ok && c < INK2_CONVS; c++)
      ok = tail_make(f->ledger, &s->conv[l][c], conv_row(f, c), 4, ck, 2 * ck + INK2_TAIL_STEP, conv_need(f));
    if (ok && ink2_slot_sliding(g, l))
      ok = tail_make(f->ledger, &s->kw[l], kd, 2, wk, 2 * wk + T, 0) && tail_make(f->ledger, &s->vw[l], kd, 2, wk, 2 * wk + T, 0);
    else if (ok)
      ok = tail_make(f->ledger, &s->kw[l], kv, INK2_KV8_RECORD, f->max_ctx, f->max_ctx, 0) &&
           tail_make(f->ledger, &s->vw[l], kv, f->v8 ? INK2_KV8_RECORD : INK2_KV5_RECORD, f->max_ctx, f->max_ctx, 0);
  }
  if (ok && f->mtp)
    ok = ink2_mtp_seq_make(s);
  if (!ok)
    return osmos_fail(err, "GPU memory refused (a sequence's caches)"), ink2_seq_destroy(s), NULL;
  return s;
}

void ink2_seq_destroy(Ink2Seq *s)
{
  if (!s)
    return;
  hipDeviceSynchronize();
  OsmosLedger *led = s->f->ledger; /* (its memory its forward's ledger's) */
  for (int l = 0; l < INK2_MAX_LAYERS; l++)
  {
    for (int c = 0; c < INK2_CONVS; c++)
      if (s->conv[l][c].d)
        ledger_free(led, s->conv[l][c].d);
    if (s->kw[l].d)
      ledger_free(led, s->kw[l].d);
    if (s->vw[l].d)
      ledger_free(led, s->vw[l].d);
  }
  ink2_mtp_seq_free(s);
  free(s);
}

int32_t ink2_seq_position(const Ink2Seq *s) { return s ? s->pos : 0; }
uint32_t ink2_seq_reach(const Ink2Seq *s) { return s && !s->failed ? s->reach : 0; }
void ink2_seq_fail(Ink2Seq *s)
{
  if (s)
    s->failed = true;
}

bool ink2_seq_prompt(Ink2Seq *s, uint32_t end)
{
  if (!s || end > s->f->max_ctx)
    return false;
  s->prompt_end = end;
  return true;
}

bool ink2_seq_rollback(Ink2Seq *s, uint32_t n)
{
  if (!s || s->failed || n > s->reach)
    return false;
  const Ink2Geometry *g = &s->f->w->geo;
  for (int l = 0; l < g->n_slots; l++)
  {
    for (int c = 0; c < INK2_CONVS; c++)
      s->conv[l][c].held -= n;
    s->kw[l].held -= n, s->vw[l].held -= n;
  }
  s->pos -= (int32_t)n, s->reach -= n;
  if (s->mend > s->pos) /* the drafter's context ends with the sequence */
    s->mend = s->pos;
  return true;
}

uint64_t ink2_identity(const Ink2Forward *f)
{
  if (!f)
    return 0;
  const Ink2Geometry *g = &f->w->geo;
  uint64_t h = fnv_u64(STATE_VERSION, fnv_u64(STATE_MAGIC, FNV_SEED));
  const uint64_t dims[] = {(uint64_t)g->n_layers, (uint64_t)g->n_embd, (uint64_t)g->n_kv_heads,
                           (uint64_t)g->head_dim, (uint64_t)g->window, (uint64_t)g->conv};
  const uint64_t loop[] = {(uint64_t)g->n_slots, (uint64_t)g->loop.start, (uint64_t)g->loop.end,
                           (uint64_t)g->loop.passes};
  h = fnv(dims, sizeof dims, h);
  if (g->loop.passes) /* (an unlooped model's identity unchanged: its saves still load) */
    h = fnv(loop, sizeof loop, h);
  if (f->v8) /* a full layer's values Q8: another cache's bytes (Q5's identity unchanged) */
    h = fnv_u64(0x7638, h);
  for (int l = 0; l < g->n_slots; l++)
    h = fnv_u64(ink2_slot_sliding(g, l), h);
  const Gguf *m = f->w->model;
  for (int i = 0; i < m->n_files; i++) /* each file: fnv.h's fnv_file (its patched shards another identity) */
    h = fnv_file(m->paths[i], h);
  return h;
}

typedef struct Head
{
  uint64_t magic, identity;
  uint32_t version;
  int32_t pos;
} Head;

/* a state's parts at position pos, in order: each layer's conv tails' last taps - 1, each sliding layer's window,
   each full layer's rows (mark: no full layer's) */
typedef StatePart Part; /* dev: the tail's rows to read or write, its first */

#define MAX_PARTS (INK2_MAX_LAYERS * (INK2_CONVS + 2))

static int parts(const Ink2Seq *s, int32_t pos, bool mark, bool load, Part *p)
{
  const Ink2Forward *f = s->f;
  const Ink2Geometry *g = &f->w->geo;
  int n = 0;
  const uint32_t cn = conv_need(f), wn = win_need(f, pos);
  for (int l = 0; l < g->n_slots; l++)
  {
    for (int c = 0; c < INK2_CONVS; c++)
    {
      const Ink2Tail *t = &s->conv[l][c];
      /* a save the last cn rows; a load into the front (held then cn) */
      p[n++] = (Part){load ? t->d : (void *)ink2_tail_last(t, cn), (uint64_t)cn * t->row * t->esize};
    }
    const Ink2Tail *k = &s->kw[l], *v = &s->vw[l];
    if (ink2_slot_sliding(g, l))
    {
      p[n++] = (Part){load ? k->d : (void *)ink2_tail_last(k, wn), (uint64_t)wn * k->row * k->esize};
      p[n++] = (Part){load ? v->d : (void *)ink2_tail_last(v, wn), (uint64_t)wn * v->row * v->esize};
    }
    else if (!mark)
    {
      p[n++] = (Part){k->d, (uint64_t)pos * k->row * k->esize};
      p[n++] = (Part){v->d, (uint64_t)pos * v->row * v->esize};
    }
  }
  return n;
}

static uint64_t bytes_at(const Ink2Seq *s, int32_t pos, bool mark)
{
  Part p[MAX_PARTS];
  return state_parts_bytes(sizeof(Head), p, parts(s, pos, mark, false, p));
}

/* after a load or mark at pos: every tail holds what it was given */
bool ink2_seq_reset(Ink2Seq *s, char err[OSMOS_ERR])
{
  if (!s)
    return osmos_fail(err, "no sequence");
  const Ink2Geometry *g = &s->f->w->geo;
  bool ok = true; /* the conv tails' zeros (a new one's: tail_make's), every tail's rows its start's */
  for (int l = 0; l < g->n_slots; l++)
    for (int c = 0; ok && c < INK2_CONVS; c++)
    {
      Ink2Tail *t = &s->conv[l][c];
      ok = osmos_hip(hipMemset(t->d, 0, (uint64_t)t->cap * t->row * t->esize), err, "the sequence's reset");
    }
  for (int l = 0; l < g->n_slots; l++)
  {
    for (int c = 0; c < INK2_CONVS; c++)
      s->conv[l][c].held = conv_need(s->f);
    s->kw[l].held = s->vw[l].held = 0;
  }
  s->pos = 0, s->reach = 0, s->prompt_end = 0, s->failed = !ok;
  s->mfrom = s->mend = 0, s->tap_step = 0;
  return ok;
}

static void settle(Ink2Seq *s, int32_t pos, bool mark)
{
  const Ink2Geometry *g = &s->f->w->geo;
  for (int l = 0; l < g->n_slots; l++)
  {
    for (int c = 0; c < INK2_CONVS; c++)
      s->conv[l][c].held = conv_need(s->f);
    if (ink2_slot_sliding(g, l))
      s->kw[l].held = s->vw[l].held = win_need(s->f, pos);
    else if (!mark)
      s->kw[l].held = s->vw[l].held = (uint32_t)pos;
  }
  s->pos = pos, s->reach = 0, s->prompt_end = 0, s->failed = false;
  s->mfrom = s->mend = 0; /* the drafter's context not saved: seeded again from here */
  s->tap_step = 0;        /* no step's taps its own: a seed waits for its next step */
}

uint64_t ink2_seq_state_bytes(const Ink2Seq *s) { return s ? bytes_at(s, s->pos, false) : 0; }
uint64_t ink2_seq_mark_bytes(const Ink2Seq *s) { return s ? bytes_at(s, s->pos, true) : 0; }

bool ink2_seq_save_to(const Ink2Seq *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  if (!s || s->failed)
    return osmos_fail(err, "no sequence, or a failed one");
  if (hipDeviceSynchronize() != hipSuccess)
    return osmos_fail(err, "a GPU call failed (the save)");
  const Head h = {STATE_MAGIC, ink2_identity(s->f), STATE_VERSION, s->pos};
  Part p[MAX_PARTS];
  return !state_parts_save(&h, sizeof h, p, parts(s, s->pos, false, false, p), sink, ctx) ||
         osmos_fail(err, "a GPU copy or the sink failed (the save)");
}

bool ink2_seq_load_from(Ink2Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  if (!s)
    return osmos_fail(err, "no sequence");
  Head h;
  if (bytes < sizeof h || !src(ctx, &h, sizeof h))
    return settle(s, 0, false), osmos_fail(err, "a state too short, or its source failed");
  const char *bad = state_head_check(h.magic, STATE_MAGIC, h.version, STATE_VERSION, h.identity, ink2_identity(s->f));
  if ((bad = bad ? bad : state_pos_check(h.pos, s->f->max_ctx)))
    return settle(s, 0, false), osmos_fail(err, "%s", bad);
  if (bytes != bytes_at(s, h.pos, false))
    return settle(s, 0, false), osmos_fail(err, "a state not its position's size");
  Part p[MAX_PARTS];
  const bool ok = !state_parts_load(&h, sizeof h, p, parts(s, h.pos, false, true, p), src, ctx);
  settle(s, ok ? h.pos : 0, false);
  return ok || osmos_fail(err, "a damaged state, or its source or a GPU copy failed (the load)");
}

bool ink2_seq_mark_save(const Ink2Seq *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || s->failed || bytes != ink2_seq_mark_bytes(s))
    return osmos_fail(err, "no sequence, a failed one, or not its mark's size");
  if (hipDeviceSynchronize() != hipSuccess)
    return osmos_fail(err, "a GPU call failed (the mark)");
  const Head h = {MARK_MAGIC, ink2_identity(s->f), STATE_VERSION, s->pos};
  uint8_t *o = out;
  memcpy(o, &h, sizeof h), o += sizeof h;
  Part p[MAX_PARTS];
  const int n = parts(s, s->pos, true, false, p);
  if (state_parts_to_host(p, n, o))
    return osmos_fail(err, "a GPU copy failed (the mark)");
  for (int i = 0; i < n; i++)
    o += p[i].bytes;
  const uint64_t sum = fnv(out, (uint64_t)(o - (uint8_t *)out), FNV_SEED);
  memcpy(o, &sum, sizeof sum);
  return true;
}

bool ink2_seq_mark_load(Ink2Seq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || s->failed)
    return osmos_fail(err, "no sequence, or a failed one");
  Head h;
  if (bytes < sizeof h + 8)
    return osmos_fail(err, "a mark too short");
  memcpy(&h, in, sizeof h);
  const char *bad = state_head_check(h.magic, MARK_MAGIC, h.version, STATE_VERSION, h.identity, ink2_identity(s->f));
  if ((bad = bad ? bad : state_pos_check(h.pos, (uint32_t)s->pos))) /* (a mark at or below the sequence's position) */
    return osmos_fail(err, "%s", bad);
  if (bytes != bytes_at(s, h.pos, true))
    return osmos_fail(err, "a mark not its position's size");
  uint64_t sum;
  memcpy(&sum, (const uint8_t *)in + bytes - 8, 8);
  if (sum != fnv(in, bytes - 8, FNV_SEED))
    return osmos_fail(err, "a damaged mark");
  if (hipDeviceSynchronize() != hipSuccess)
    return osmos_fail(err, "a GPU call failed (the mark)");
  Part p[MAX_PARTS];
  if (state_parts_from_host(p, parts(s, h.pos, true, true, p), (const uint8_t *)in + sizeof h))
    return s->failed = true, osmos_fail(err, "a GPU copy failed (the mark's load)");
  /* a full layer's rows below the mark are the sequence's still: held the mark's position */
  const Ink2Geometry *g = &s->f->w->geo;
  for (int l = 0; l < g->n_slots; l++)
    if (!ink2_slot_sliding(g, l))
      s->kw[l].held = s->vw[l].held = (uint32_t)h.pos;
  settle(s, h.pos, true);
  return true;
}
