/* a q35 sequence (osmos/q35/forward.h, DESIGN.md "0.9.0: q35 behind the engine"): its rings of DeltaNet states and
   conv histories, its attention caches, LSA's index - and what the engine asks of it: prompt, rollback, state, marks */
#include <hip/hip_runtime_api.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "osmos/gpu/state_parts.h"
#include "osmos/model/fnv.h"
#include "q35_impl.h"
#include "osmos/err.h"

#define STATE_MAGIC 0x53353351u /* "Q35S" */
#define MARK_MAGIC 0x4b353351u  /* "Q35K" */
#define STATE_VERSION 1u

typedef struct Header
{
  uint32_t magic, version;
  uint64_t identity;
  int32_t pos;
  uint32_t closed; /* LSA's closed blocks (every attention layer's the same), 0 without it */
} Header;

Q35SeqBytes q35_seq_bytes(const Q35Geometry *g)
{
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner;
  return (Q35SeqBytes){(uint64_t)g->ssm_v_heads * g->ssm_state * g->ssm_state * F, (uint64_t)(g->ssm_conv - 1) * ch * F,
                       q4x_kv_row_bytes((uint32_t)(g->n_kv_heads * g->head_dim))};
}

static bool reset(Q35Seq *s)
{
  const Q35Geometry *g = &s->f->w->geo;
  const Q35SeqBytes sb = q35_seq_bytes(g);
  s->cur = 0, s->back = 0; /* slot 0 the latest, zeroed: the others not read until a step writes them */
  bool ok = true;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = !g->recurrent[l] ||
         (hipMemset(s->ssm[l], 0, sb.ssm) == hipSuccess && hipMemset(s->conv[l], 0, sb.conv) == hipSuccess);
  s->pos = 0, s->prompt_end = 0;
  memset(s->closed, 0, sizeof s->closed);
  s->broken = !ok;
  return ok;
}

Q35Seq *q35_seq_create(Q35Forward *f, char err[OSMOS_ERR])
{
  if (!f)
    return osmos_fail(err, "no forward"), NULL;
  const Q35Geometry *g = &f->w->geo;
  Q35Seq *s = calloc(1, sizeof *s);
  if (!s)
    return osmos_fail(err, "out of memory"), NULL;
  s->f = f;
  const Q35SeqBytes sb = q35_seq_bytes(g);
  const uint64_t blocks = f->max_ctx / LSA_WINDOW + 1;
  bool ok = true;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] ? q35_alloc(&s->ssm[l], Q35_SLOTS * sb.ssm) && q35_alloc(&s->conv[l], Q35_SLOTS * sb.conv)
                         : q35_alloc(&s->kc[l], f->max_ctx * sb.kv_row) && q35_alloc(&s->vc[l], f->max_ctx * sb.kv_row) &&
                               (!f->lsa || (q35_alloc(&s->ik[l], (uint64_t)f->max_ctx * LSA_DIM * F) &&
                                            q35_alloc(&s->cent[l], blocks * LSA_BUCKETS * LSA_DIM * F) &&
                                            q35_alloc(&s->mem[l], blocks * LSA_BUCKETS * LSA_CAP * 4)));
  if (!ok || !reset(s))
    return q35_seq_destroy(s), osmos_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

void q35_seq_destroy(Q35Seq *s)
{
  if (!s)
    return;
  for (int l = 0; l < Q35_MAX_LAYERS; l++)
    hipFree(s->conv[l]), hipFree(s->ssm[l]), hipFree(s->kc[l]), hipFree(s->vc[l]), hipFree(s->ik[l]),
        hipFree(s->cent[l]), hipFree(s->mem[l]);
  free(s);
}

int32_t q35_seq_position(const Q35Seq *s) { return s ? s->pos : -1; }

bool q35_seq_prompt(Q35Seq *s, uint32_t end)
{
  if (!s || end > s->f->max_ctx)
    return false;
  s->prompt_end = end;
  return true;
}

void q35_seq_fail(Q35Seq *s)
{
  if (s)
    s->broken = true;
}

/* LSA's closed blocks: every attention layer's (the same count; 0 without LSA) */
static uint32_t closed_blocks(const Q35Seq *s)
{
  const Q35Geometry *g = &s->f->w->geo;
  for (int l = 0; s->f->lsa && l < g->n_layers; l++)
    if (!g->recurrent[l])
      return s->closed[l];
  return 0;
}

uint32_t q35_seq_reach(const Q35Seq *s)
{
  if (!s || s->broken)
    return 0;
  const int64_t past_close = (int64_t)s->pos - (int64_t)closed_blocks(s) * LSA_WINDOW; /* a rollback never un-closes */
  return past_close < (int64_t)s->back ? (uint32_t)(past_close > 0 ? past_close : 0) : s->back;
}

bool q35_seq_rollback(Q35Seq *s, uint32_t n)
{
  if (!s || s->broken || n > q35_seq_reach(s))
    return false;
  s->pos -= (int32_t)n, s->back -= n;
  s->cur = (s->cur + Q35_SLOTS - n) % Q35_SLOTS;
  return true;
}

uint64_t q35_identity(const Q35Forward *f)
{
  const Q35Geometry *g = &f->w->geo;
  uint64_t h = fnv_u64(STATE_VERSION, FNV_SEED);
  const uint64_t dims[] = {(uint64_t)g->n_layers,  (uint64_t)g->n_embd,     (uint64_t)g->ssm_v_heads,
                           (uint64_t)g->ssm_state, (uint64_t)g->ssm_groups, (uint64_t)g->ssm_inner,
                           (uint64_t)g->ssm_conv,  (uint64_t)g->n_kv_heads, (uint64_t)g->head_dim,
                           (uint64_t)f->lsa};
  h = fnv(dims, sizeof dims, h);
  for (int l = 0; l < g->n_layers; l++)
    h = fnv_u64(g->recurrent[l], h);
  h = fnv(&f->rope, sizeof f->rope, h); /* YaRN's or the plain rotation: another context's numbers another model's */
  const Gguf *m = f->w->model;
  for (int i = 0; i < m->n_files; i++) /* each file: fnv.h's fnv_file */
    h = fnv_file(m->paths[i], h);
  return h;
}

#define MAX_PARTS (5 * Q35_MAX_LAYERS)

/* the parts at position pos with closed blocks (the rings' current slot), in the string's order: each DeltaNet's state
   and conv history; each attention layer's cache rows below pos (keys, values), and with LSA its index vectors below
   pos, its closed blocks' centroids and buckets. mark: the DeltaNets' alone (the rest only grows below the position) */
static int parts(const Q35Seq *s, int32_t pos, uint32_t closed, bool mark, StatePart *p)
{
  const Q35Geometry *g = &s->f->w->geo;
  const Q35SeqBytes sb = q35_seq_bytes(g);
  int n = 0;
  for (int l = 0; l < g->n_layers; l++)
    if (g->recurrent[l])
      p[n++] = (StatePart){(uint8_t *)s->ssm[l] + s->cur * sb.ssm, sb.ssm},
      p[n++] = (StatePart){(uint8_t *)s->conv[l] + s->cur * sb.conv, sb.conv};
    else if (!mark)
    {
      p[n++] = (StatePart){s->kc[l], (uint64_t)pos * sb.kv_row}, p[n++] = (StatePart){s->vc[l], (uint64_t)pos * sb.kv_row};
      if (s->f->lsa)
        p[n++] = (StatePart){s->ik[l], (uint64_t)pos * LSA_DIM * F},
        p[n++] = (StatePart){s->cent[l], (uint64_t)closed * LSA_BUCKETS * LSA_DIM * F},
        p[n++] = (StatePart){s->mem[l], (uint64_t)closed * LSA_BUCKETS * LSA_CAP * 4};
    }
  return n;
}

static void settle(Q35Seq *s, const Header *h)
{
  const Q35Geometry *g = &s->f->w->geo;
  s->pos = h->pos, s->cur = 0, s->back = 0, s->prompt_end = 0, s->broken = false; /* loaded into slot 0, none behind */
  for (int l = 0; l < g->n_layers; l++)
    s->closed[l] = g->recurrent[l] ? 0 : h->closed;
}

uint64_t q35_seq_state_bytes(const Q35Seq *s)
{
  if (!s)
    return 0;
  StatePart p[MAX_PARTS];
  return state_parts_bytes(sizeof(Header), p, parts(s, s->pos, closed_blocks(s), false, p));
}

bool q35_seq_save_to(const Q35Seq *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  if (!s || !sink)
    return osmos_fail(err, "no sequence or sink");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its states part written): load it, or make another");
  Header h;
  memset(&h, 0, sizeof h);
  h.magic = STATE_MAGIC, h.version = STATE_VERSION, h.identity = q35_identity(s->f), h.pos = s->pos,
  h.closed = closed_blocks(s);
  StatePart p[MAX_PARTS];
  const char *why = state_parts_save(&h, sizeof h, p, parts(s, s->pos, h.closed, false, p), sink, ctx);
  return !why || osmos_fail(err, "%s", why);
}

bool q35_seq_reset(Q35Seq *s, char err[OSMOS_ERR])
{
  return !s ? osmos_fail(err, "no sequence") : reset(s) || osmos_fail(err, "the sequence's reset: a GPU call failed");
}

static bool refuse(Q35Seq *s, char *err, const char *why)
{
  reset(s);
  return osmos_fail(err, "%s", why);
}

bool q35_seq_load_from(Q35Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  if (!s || !src)
    return osmos_fail(err, "no sequence or source");
  Header h;
  if (bytes < sizeof h + sizeof(uint64_t) || !src(ctx, &h, sizeof h))
    return refuse(s, err, "a state shorter than its header, or its source failed");
  if (h.magic != STATE_MAGIC || h.version != STATE_VERSION || h.identity != q35_identity(s->f))
    return refuse(s, err, "a state of another format or model (its identity)");
  StatePart p[MAX_PARTS];
  s->cur = 0; /* the parts' DeltaNet slots: 0 */
  const int n = parts(s, h.pos, h.closed, false, p);
  if (h.pos < 0 || (uint32_t)h.pos > s->f->max_ctx || (uint64_t)h.closed * LSA_WINDOW > (uint64_t)h.pos ||
      bytes != state_parts_bytes(sizeof h, p, n))
    return refuse(s, err, "a state past this context, or not its position's size");
  const char *why = state_parts_load(&h, sizeof h, p, n, src, ctx);
  if (why) /* the parts part written: the reset makes the sequence empty */
    return refuse(s, err, why);
  settle(s, &h);
  return true;
}

uint64_t q35_seq_mark_bytes(const Q35Seq *s)
{
  if (!s)
    return 0;
  StatePart p[MAX_PARTS];
  return state_parts_bytes(sizeof(Header), p, parts(s, s->pos, 0, true, p)) - sizeof(uint64_t); /* no sum: in memory */
}

bool q35_seq_mark_save(const Q35Seq *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || !out || bytes != q35_seq_mark_bytes(s))
    return osmos_fail(err, "no sequence, or a mark's room not its size");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its states part written): no mark of it");
  if (hipDeviceSynchronize() != hipSuccess)
    return osmos_fail(err, "a GPU call failed (before the mark's save)");
  Header h;
  memset(&h, 0, sizeof h);
  h.magic = MARK_MAGIC, h.version = STATE_VERSION, h.identity = q35_identity(s->f), h.pos = s->pos,
  h.closed = closed_blocks(s);
  memcpy(out, &h, sizeof h);
  StatePart p[MAX_PARTS];
  const char *why = state_parts_to_host(p, parts(s, s->pos, 0, true, p), (uint8_t *)out + sizeof h);
  return !why || osmos_fail(err, "%s", why);
}

bool q35_seq_mark_load(Q35Seq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || !in || bytes != q35_seq_mark_bytes(s))
    return osmos_fail(err, "no sequence, or a mark not its size");
  Header h;
  memcpy(&h, in, sizeof h);
  if (h.magic != MARK_MAGIC || h.version != STATE_VERSION || h.identity != q35_identity(s->f) || h.pos < 0 ||
      h.pos > s->pos)
    return osmos_fail(err, "not this model's mark, or one past the sequence's position");
  if (h.closed != closed_blocks(s)) /* LSA's blocks closed since: their buckets of keys a mark's sequence lacks */
    return osmos_fail(err, "a mark below a block LSA has closed since");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step: its rows below the mark not known (load a state)");
  if (hipDeviceSynchronize() != hipSuccess) /* its states not still being read */
    return osmos_fail(err, "a GPU call failed (before the mark's load)");
  s->cur = 0;
  StatePart p[MAX_PARTS];
  if (state_parts_from_host(p, parts(s, h.pos, 0, true, p), (const uint8_t *)in + sizeof h))
  {
    reset(s);
    return osmos_fail(err, "a GPU copy failed (the mark's load): the sequence emptied");
  }
  settle(s, &h);
  return true;
}
