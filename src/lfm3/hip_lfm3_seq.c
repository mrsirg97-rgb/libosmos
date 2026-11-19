/* an lfm3 sequence (osmos/lfm3/forward.h): its convolutions' state rings, its full layers' caches by position, its
   sliding layers' rings; rollback; the prompt's end; its state saved and loaded (state_parts.h's format) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/state_parts.h"
#include "osmos/model/fnv.h"
#include "lfm3_impl.h"

static uint64_t conv_bytes(const Lfm3Geometry *g) { return (uint64_t)(g->conv_taps - 1) * (uint64_t)g->n_embd * F; }
static uint64_t kv_row(const Lfm3Geometry *g) { return q4x_kv_row_bytes((uint32_t)(g->n_kv_heads * g->head_dim)); }

Lfm3Seq *lfm3_seq_create(Lfm3Forward *f, char err[LFM3_ERR])
{
  if (!f)
    return lfm3_fail(err, "no forward"), NULL;
  Lfm3Seq *s = calloc(1, sizeof *s);
  if (!s)
    return lfm3_fail(err, "out of memory"), NULL;
  s->f = f;
  const Lfm3Geometry *g = &f->w->geo;
  const uint64_t conv = LFM3_CONV_SLOTS * conv_bytes(g), row = kv_row(g);
  bool ok = true;
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    const uint64_t rows = g->kind[l] == LFM3_SLIDING ? f->ring : f->max_ctx;
    ok = g->kind[l] == LFM3_CONV ? lfm3_alloc(&s->conv[l], conv) && hipMemset(s->conv[l], 0, conv) == hipSuccess
                                 : lfm3_alloc(&s->kc[l], rows * row) && lfm3_alloc(&s->vc[l], rows * row);
    if (ok && f->lsa && g->kind[l] == LFM3_FULL)
    {
      const uint64_t blocks = f->max_ctx / LSA_WINDOW + 1;
      ok = lfm3_alloc(&s->ik[l], (uint64_t)f->max_ctx * LSA_DIM * F) &&
           lfm3_alloc(&s->cent[l], blocks * LSA_BUCKETS * LSA_DIM * F) &&
           lfm3_alloc(&s->mem[l], blocks * LSA_BUCKETS * LSA_CAP * 4);
    }
  }
  if (!ok)
    return lfm3_seq_destroy(s), lfm3_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

void lfm3_seq_destroy(Lfm3Seq *s)
{
  if (!s)
    return;
  for (int l = 0; l < LFM3_MAX_LAYERS; l++)
    hipFree(s->conv[l]), hipFree(s->kc[l]), hipFree(s->vc[l]), hipFree(s->ik[l]), hipFree(s->cent[l]),
        hipFree(s->mem[l]);
  free(s);
}

int32_t lfm3_seq_position(const Lfm3Seq *s) { return s ? s->pos : -1; }

bool lfm3_seq_prompt(Lfm3Seq *s, uint32_t end)
{
  if (!s || s->broken)
    return false;
  s->prompt_end = end;
  return true;
}

uint32_t lfm3_seq_stage(const Lfm3Seq *s) { return s ? s->stage : 0; }

uint32_t lfm3_seq_reach(const Lfm3Seq *s)
{
  if (!s || s->broken)
    return 0;
  const int64_t past_close = (int64_t)s->pos - (int64_t)s->closed * LSA_WINDOW; /* a rollback never un-closes */
  return !s->f->lsa || past_close >= (int64_t)s->back ? s->back : (uint32_t)(past_close > 0 ? past_close : 0);
}

/* a convolution's state the slot before (its ring kept each of the last tokens'); a full layer's rows past the
   position never read again; a sliding ring holds the window past the reach */
bool lfm3_seq_rollback(Lfm3Seq *s, uint32_t n)
{
  if (!s || s->broken || n > lfm3_seq_reach(s))
    return false;
  s->pos -= (int32_t)n, s->back -= n, s->cur = (s->cur + LFM3_CONV_SLOTS - n) % LFM3_CONV_SLOTS;
  return true;
}

void lfm3_seq_fail(Lfm3Seq *s)
{
  if (s)
    s->broken = true;
}

#define STATE_MAGIC 0x53334d4cu /* "LM3S" */
#define STATE_VERSION 2u /* 2: its YaRN stage */

uint64_t lfm3_identity(const Lfm3Forward *f)
{
  const Lfm3Geometry *g = &f->w->geo;
  uint64_t h = fnv_u64(STATE_VERSION, 1469598103934665603ull);
  h = fnv(g, sizeof *g, h), h = fnv_u64(f->ring, h), h = fnv_u64(f->lsa, h), h = fnv_u64(f->yarn_from, h), h = fnv_u64(f->stages, h);
  for (int i = 0; i < f->w->model->n_files; i++)
    h = fnv_file(f->w->model->paths[i], h);
  return h;
}

typedef struct Header
{
  uint32_t magic, version;
  uint64_t identity;
  int32_t pos;
  uint32_t closed; /* LSA's closed blocks (0 without it) */
  uint32_t stage;  /* its YaRN stage (its keys' rotation) */
} Header;

#define MAX_PARTS (5 * LFM3_MAX_LAYERS)

/* the parts at position pos, in the string's order: a convolution's state in slot `slot`; a full layer's K and V rows
   below pos (with LSA its index vectors below pos, its closed blocks' centroids and buckets); a sliding layer's whole
   ring */
static int parts(const Lfm3Seq *s, int32_t pos, uint32_t slot, uint32_t closed, StatePart *p)
{
  const Lfm3Forward *f = s->f;
  const Lfm3Geometry *g = &f->w->geo;
  const uint64_t row = kv_row(g), conv = conv_bytes(g);
  int n = 0;
  for (int l = 0; l < g->n_layers; l++)
    if (g->kind[l] == LFM3_CONV)
      p[n++] = (StatePart){(uint8_t *)s->conv[l] + slot * conv, conv};
    else
    {
      const uint64_t rows = g->kind[l] == LFM3_SLIDING ? f->ring : (uint64_t)pos;
      p[n++] = (StatePart){s->kc[l], rows * row}, p[n++] = (StatePart){s->vc[l], rows * row};
      if (f->lsa && g->kind[l] == LFM3_FULL)
        p[n++] = (StatePart){s->ik[l], (uint64_t)pos * LSA_DIM * F},
        p[n++] = (StatePart){s->cent[l], (uint64_t)closed * LSA_BUCKETS * LSA_DIM * F},
        p[n++] = (StatePart){s->mem[l], (uint64_t)closed * LSA_BUCKETS * LSA_CAP * 4};
    }
  return n;
}

uint64_t lfm3_seq_state_bytes(const Lfm3Seq *s)
{
  if (!s)
    return 0;
  StatePart p[MAX_PARTS];
  return state_parts_bytes(sizeof(Header), p, parts(s, s->pos, s->cur, s->closed, p));
}

bool lfm3_seq_save_to(const Lfm3Seq *s, StateSink sink, void *ctx, char err[LFM3_ERR])
{
  if (!s || !sink)
    return lfm3_fail(err, "no sequence or sink");
  if (s->broken)
    return lfm3_fail(err, "a sequence after a failed step (its state part written): load it, or make another");
  Header h = {STATE_MAGIC, STATE_VERSION, lfm3_identity(s->f), s->pos, s->closed, s->stage};
  StatePart p[MAX_PARTS];
  const char *why = state_parts_save(&h, sizeof h, p, parts(s, s->pos, s->cur, s->closed, p), sink, ctx);
  return !why || lfm3_fail(err, why);
}

static bool refuse(Lfm3Seq *s, char *err, const char *why)
{
  s->pos = 0, s->back = 0, s->prompt_end = 0, s->cur = 0, s->closed = 0, s->broken = false; /* empty */
  s->stage = 0, s->turned = 0;
  const Lfm3Geometry *g = &s->f->w->geo;
  for (int l = 0; l < g->n_layers; l++)
    if (g->kind[l] == LFM3_CONV)
      hipMemset(s->conv[l], 0, LFM3_CONV_SLOTS * conv_bytes(g)); /* a new sequence's: zeros (the conv's from 0) */
  return lfm3_fail(err, why);
}

bool lfm3_seq_load_from(Lfm3Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[LFM3_ERR])
{
  if (!s || !src)
    return lfm3_fail(err, "no sequence or source");
  Header h;
  if (bytes < sizeof h + sizeof(uint64_t) || !src(ctx, &h, sizeof h))
    return refuse(s, err, "a state shorter than its header, or its source failed");
  if (h.magic != STATE_MAGIC || h.version != STATE_VERSION || h.identity != lfm3_identity(s->f))
    return refuse(s, err, "a state of another format or model (its identity)");
  if (h.pos < 0 || (uint32_t)h.pos > s->f->max_ctx || h.closed > (uint32_t)h.pos / LSA_WINDOW ||
      (!s->f->lsa && h.closed) || h.stage >= s->f->stages)
    return refuse(s, err, "a state past this context, or its closed blocks past its position");
  StatePart p[MAX_PARTS];
  const int n = parts(s, h.pos, 0, h.closed, p);
  if (bytes != state_parts_bytes(sizeof h, p, n))
    return refuse(s, err, "a state not its position's size");
  const char *why = state_parts_load(&h, sizeof h, p, n, src, ctx);
  if (why)
    return refuse(s, err, why);
  s->pos = h.pos, s->back = 0, s->prompt_end = 0, s->cur = 0, s->closed = h.closed, s->broken = false;
  s->stage = h.stage, s->turned = 0;
  return true;
}
