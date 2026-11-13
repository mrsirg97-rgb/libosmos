#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <hip/hip_runtime_api.h>

#include "drafter_impl.h"
#include "forward_impl.h"
#include "osmos/gpu/kvfmt.h"
#include "osmos/model/fnv.h"

/* a sequence's state as one byte string (forward.h, seq_save): a header, then by position each layer's window rows
   (with seq_rollback's depth before them), its compressed rows and indexer keys, its compressor's open positions, the
   drafter's rings, then a checksum. rows are copied a contiguous run at a time (a ring's positions wrap once at most) */

#define STATE_MAGIC 0x51425553u /* "SUBQ" */
#define STATE_LAST (ENGRAM_MAX_NGRAM + SEQ_MAX_BACK)

typedef struct Header
{
  uint32_t magic, version;
  uint64_t identity;
  int32_t pos, prompt_end, dec_from, n_last;
  int32_t last[STATE_LAST];
  uint32_t has_dring, pad;
} Header;

uint64_t forward_identity(const Forward *f)
{
  if (!f)
    return 0;
  const Geometry *g = f->g;
  uint64_t h = fnv_u64(SEQ_STATE_VERSION, FNV_SEED);
  const uint64_t dims[] = {(uint64_t)f->n_layers, (uint64_t)g->n_embd, (uint64_t)g->head_dim, (uint64_t)g->window,
                           (uint64_t)g->ix_head_dim, STATE_LAST, DRAFTER_BLOCKS};
  h = fnv(dims, sizeof dims, h);
  for (int l = 0; l < f->n_layers; l++)
    h = fnv_u64((uint64_t)g->compress[l] << 2 | (uint64_t)g->kv_source[l] << 1 | g->ix_source[l], h);
  /* the files read: the model's and an overlay's (w->paths), the Engram file's - with an overlay or without, another */
  const int n = f->w->n_files + (f->w->engram ? f->w->engram->n_files : 0);
  for (int i = 0; i < n; i++)
  {
    const char *p = i < f->w->n_files ? f->w->paths[i] : f->w->engram->paths[i - f->w->n_files];
    const char *base = strrchr(p, '/');
    base = base ? base + 1 : p;
    struct stat st;
    h = fnv(base, strlen(base), h);
    h = fnv_u64(stat(p, &st) == 0 ? (uint64_t)st.st_size : 0, h);
  }
  if (f->npu) /* the NPU's sums its own, its parts by its share (forward_npu): another's states */
  {
    uint32_t share;
    memcpy(&share, &f->npu_share, sizeof share);
    h = fnv_u64(0x4e50550000000000ull | share, h);
  }
  return h;
}

int32_t seq_position(const Seq *s) { return s ? s->pos : 0; }

/* the positions a state keeps for a ring of a layer: from back (seq_rollback's depth and the window before it) */
static int32_t ring_from(const Seq *s) { return s->pos - s->f->g->window - SEQ_MAX_BACK > 0 ? s->pos - s->f->g->window - SEQ_MAX_BACK : 0; }

/* a compressor's open positions, from the group seq_rollback's depth could reopen */
static int32_t pend_from(const Seq *s, int r)
{
  const int32_t back = s->pos - SEQ_MAX_BACK > 0 ? s->pos - SEQ_MAX_BACK : 0;
  return back / r * r;
}

/* each of the state's parts, in order: a device range of rows (at most two runs: a ring's wrap) and its bytes */
typedef struct Part
{
  uint8_t *dev;     /* the buffer */
  uint64_t row;     /* a row's bytes */
  uint64_t cap;     /* the buffer's rows (a ring's: positions at p % cap); 0: rows at their index */
  uint64_t from, n; /* positions (or indices) from, n of them */
} Part;

/* the state's parts for s at its position (n_parts out); false for more than the table holds */
#define MAX_PARTS (WEIGHTS_MAX_LAYERS * 5 + 1)
static int parts(const Seq *s, Part *p)
{
  const Forward *f = s->f;
  const Geometry *g = f->g;
  const uint64_t hd = (uint64_t)g->head_dim, F = sizeof(float);
  int n = 0;
  const int32_t rf = ring_from(s);
  for (int l = 0; l < f->n_layers; l++)
  {
    p[n++] = (Part){s->ring[l], kv_row_bytes(KV_FP8, (uint32_t)hd), f->slots, (uint64_t)rf, (uint64_t)(s->pos - rf)};
    if (!g->kv_source[l])
      continue;
    const int r = g->compress[l];
    const uint64_t rows = (uint64_t)s->pos / (uint64_t)r;
    p[n++] = (Part){s->ckv[l], kv_row_bytes(KV_FP4_E4M3, (uint32_t)hd), 0, 0, rows};
    if (g->ix_source[l])
      p[n++] = (Part){s->ik[l], kv_row_bytes(KV_FP4_POW2, (uint32_t)g->ix_head_dim), 0, 0, rows};
    if (r >= 2)
    {
      const uint64_t pend = FWD_PEND_ROWS(f->step, r), pf = (uint64_t)pend_from(s, r);
      p[n++] = (Part){(uint8_t *)s->pend_kvc[l], hd * F, pend, pf, (uint64_t)s->pos - pf};
      p[n++] = (Part){(uint8_t *)s->pend_sc[l], hd * F, pend, pf, (uint64_t)s->pos - pf};
    }
  }
  if (s->dring)
    p[n++] = (Part){s->dring, kv_row_bytes(KV_FP8, (uint32_t)hd), 0, 0, (uint64_t)DRAFTER_BLOCKS * g->window};
  return n;
}

uint64_t seq_state_bytes(const Seq *s)
{
  if (!s)
    return 0;
  Part p[MAX_PARTS];
  const int n = parts(s, p);
  uint64_t bytes = sizeof(Header) + sizeof(uint64_t);
  for (int i = 0; i < n; i++)
    bytes += p[i].n * p[i].row;
  return bytes;
}

/* a part's rows between the device and a sink or source, in pieces through bounce (STATE_CHUNK bytes): its positions
   in at most two contiguous runs, each in pieces of whole rows; sum the FNV-1a of every byte moved, in order */
static bool move_part(const Part *p, uint8_t *bounce, bool out, StateSink sink, StateSource src, void *ctx,
                      uint64_t *sum)
{
  const uint64_t per = STATE_CHUNK / p->row ? STATE_CHUNK / p->row : 1; /* (a row is a few KB: never past a piece) */
  for (uint64_t done = 0; done < p->n;)
  {
    const uint64_t at = p->cap ? (p->from + done) % p->cap : p->from + done;
    uint64_t run = p->n - done;
    if (p->cap && at + run > p->cap)
      run = p->cap - at;
    if (run > per)
      run = per;
    uint8_t *dev = p->dev + at * p->row;
    const uint64_t n = run * p->row;
    if (out)
    {
      if (hipMemcpy(bounce, dev, n, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
      *sum = fnv(bounce, n, *sum);
      if (!sink(ctx, bounce, n))
        return false;
    }
    else
    {
      if (!src(ctx, bounce, n))
        return false;
      *sum = fnv(bounce, n, *sum);
      if (hipMemcpy(dev, bounce, n, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    }
    done += run;
  }
  return true;
}

bool seq_save_to(const Seq *s, StateSink sink, void *ctx, char err[FORWARD_ERR])
{
  if (!s || !sink)
    return fwd_fail(err, "no sequence, or no sink");
  if (hipDeviceSynchronize() != hipSuccess) /* the stream's writes of its rows done */
    return fwd_fail(err, "a GPU call failed (before the save)");
  Header h = {STATE_MAGIC, SEQ_STATE_VERSION, forward_identity(s->f), s->pos, s->prompt_end, s->dec_from, s->n_last,
              {0}, s->dring != NULL, 0};
  memcpy(h.last, s->last, sizeof h.last);
  uint8_t *bounce = malloc(STATE_CHUNK);
  uint64_t sum = fnv(&h, sizeof h, FNV_SEED);
  bool ok = bounce && sink(ctx, &h, sizeof h);
  Part p[MAX_PARTS];
  const int n = parts(s, p);
  for (int i = 0; ok && i < n; i++)
    ok = move_part(&p[i], bounce, true, sink, NULL, ctx, &sum);
  ok = ok && sink(ctx, &sum, sizeof sum);
  free(bounce);
  return ok || fwd_fail(err, bounce ? "a GPU copy or the sink failed (the save)" : "out of memory (the save)");
}

/* a sequence at position 0, as seq_create made it (its rows are rewritten before they are read: a position's
   state written by its step) */
static void reset(Seq *s)
{
  s->pos = 0, s->prompt_end = 0, s->staged = 0, s->dec_from = 0, s->n_last = 0, s->broken = false;
  memset(s->last, 0, sizeof s->last);
}

static bool refuse(Seq *s, char *err, const char *why)
{
  reset(s);
  return fwd_fail(err, why);
}

bool seq_load_from(Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[FORWARD_ERR])
{
  if (!s || !src)
    return fwd_fail(err, "no sequence, or no source");
  const Forward *f = s->f;
  Header h;
  uint64_t sum = 0;
  if (bytes < sizeof h + sizeof sum)
    return refuse(s, err, "a state shorter than its header");
  if (!src(ctx, &h, sizeof h))
    return refuse(s, err, "the source failed (the state's header)");
  if (h.magic != STATE_MAGIC || h.version != SEQ_STATE_VERSION)
    return refuse(s, err, "not a sequence's state, or another version's");
  if (h.identity != forward_identity(f))
    return refuse(s, err, "a state of another model or geometry (its identity)");
  if (h.pos < 0 || (uint32_t)h.pos > f->max_ctx || h.prompt_end < 0 || h.prompt_end > h.pos || h.dec_from < 0 ||
      h.n_last < 0 || h.n_last > STATE_LAST || h.has_dring > 1)
    return refuse(s, err, "a state's position past this forward's context, or its fields out of range");
  /* the parts as the state laid them out: the sequence at its position (and a drafter's rings to fill) */
  if (h.has_dring && !s->dring &&
      !fwd_alloc(&s->dring, (uint64_t)DRAFTER_BLOCKS * f->g->window * kv_row_bytes(KV_FP8, (uint32_t)f->g->head_dim)))
    return refuse(s, err, "GPU memory refused (the drafter's rings)");
  s->pos = h.pos;
  uint8_t *keep = s->dring;
  if (!h.has_dring) /* (the sequence's own rings, if any, are not the state's: not read until drafter_seed) */
    s->dring = NULL;
  Part p[MAX_PARTS];
  const int n = parts(s, p);
  s->dring = keep;
  uint64_t want = sizeof h + sizeof sum;
  for (int i = 0; i < n; i++)
    want += p[i].n * p[i].row;
  if (want != bytes)
    return refuse(s, err, "a state whose size does not agree with its position");
  if (hipDeviceSynchronize() != hipSuccess) /* the sequence's rows not still being read by the stream */
    return refuse(s, err, "a GPU call failed (before the load)");
  uint8_t *bounce = malloc(STATE_CHUNK);
  if (!bounce)
    return refuse(s, err, "out of memory (the load)");
  uint64_t got = fnv(&h, sizeof h, FNV_SEED);
  bool ok = true;
  for (int i = 0; ok && i < n; i++)
    ok = move_part(&p[i], bounce, false, NULL, src, ctx, &got);
  free(bounce);
  if (!ok || !src(ctx, &sum, sizeof sum))
    return refuse(s, err, "a GPU copy or the source failed (the load)");
  if (got != sum) /* (its rows already on the GPU: the reset makes them a fresh sequence's, rewritten before read) */
    return refuse(s, err, "a damaged state (its checksum)");
  s->prompt_end = h.prompt_end, s->dec_from = h.dec_from, s->n_last = h.n_last, s->staged = 0;
  memcpy(s->last, h.last, sizeof s->last);
  s->broken = false;
  return true;
}
