#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "osmos/gpu/state_parts.h"
#include "osmos/model/fnv.h"
#include "q4x_impl.h"
#include "osmos/err.h"

/* a sequence as one byte string (osmos/q4x/forward.h, q4x_seq_save): a header, the n-gram layer's conv history,
   each layer's own - a DeltaNet's state and conv history, an attention's cache rows below the position, keys then
   values, then its indexer's pooled keys below the position's blocks and its raw keys' ring - then a checksum */

#define STATE_MAGIC 0x53583451u /* "Q4XS" */
#define STATE_VERSION 4u /* 2: the attention cache in Q8_0; 3: QSA's pooled keys and ring; 4 (0.9.2): the MTP head's
                            cache, indexer and pending h (a forward with a head: its identity says so) */

typedef struct Header
{
  uint32_t magic, version;
  uint64_t identity;
  int32_t pos, n_last;
  int32_t last[Q4X_MAX_NGRAM];
  int32_t pend_pos, pad; /* the head's pending h's position (-1: none) */
} Header;

typedef StatePart Part;

#define MAX_PARTS (1 + 4 * Q4X_MAX_LAYERS + 5)

/* the parts of a sequence at position pos (its rings' latest slot), in the string's order: their count */
static int parts(const Q4xSeq *s, int32_t pos, Part *p)
{
  const Q4xGeometry *g = &s->f->w->geo;
  const Q4xSeqBytes sb = q4x_seq_bytes(g);
  int n = 0;
  p[n++] = (Part){(uint8_t *)s->ple_hist + s->cur * sb.ple, sb.ple};
  for (int l = 0; l < g->n_layers; l++)
    if (g->recurrent[l])
      p[n++] = (Part){(uint8_t *)s->ssm[l] + s->cur * sb.ssm, sb.ssm},
      p[n++] = (Part){(uint8_t *)s->conv[l] + s->cur * sb.conv, sb.conv};
    else
      p[n++] = (Part){s->kc[l], (uint64_t)pos * sb.kv_row}, p[n++] = (Part){s->vc[l], (uint64_t)pos * sb.kv_row},
      p[n++] = (Part){s->ixk[l], (uint64_t)(pos / g->ix_ratio) * sb.ix_block}, p[n++] = (Part){s->ixr[l], sb.ix_ring};
  /* the MTP head's (0.9.2): its cache, indexer and pending h - a state taken back drafts as the conversation never set
     aside did. without them the head's attention read the rows another conversation's seeding left (it has no lower
     bound: every position from 0) */
  if (s->mkc)
    p[n++] = (Part){s->mkc, (uint64_t)pos * sb.kv_row}, p[n++] = (Part){s->mvc, (uint64_t)pos * sb.kv_row},
    p[n++] = (Part){s->mixk, (uint64_t)(pos / g->ix_ratio) * sb.ix_block}, p[n++] = (Part){s->mixr, sb.ix_ring},
    p[n++] = (Part){s->pend, (uint64_t)g->hc * g->n_embd * sizeof(float)};
  return n;
}

static uint64_t bytes_at(const Q4xSeq *s, int32_t pos)
{
  Part p[MAX_PARTS];
  return state_parts_bytes(sizeof(Header), p, parts(s, pos, p));
}

uint64_t q4x_identity(const Q4xForward *f)
{
  if (!f)
    return 0;
  const Q4xGeometry *g = &f->w->geo;
  uint64_t h = fnv_u64(STATE_VERSION, FNV_SEED);
  const uint64_t dims[] = {(uint64_t)g->n_layers, (uint64_t)g->n_embd, (uint64_t)g->hc,
                           (uint64_t)g->ssm_v_heads, (uint64_t)g->ssm_state, (uint64_t)g->ssm_groups,
                           (uint64_t)g->ssm_inner, (uint64_t)g->ssm_conv, (uint64_t)g->n_kv_heads,
                           (uint64_t)g->head_dim, (uint64_t)g->ple_conv, (uint64_t)g->ple_ngram, Q4X_MAX_NGRAM,
                           (uint64_t)g->ix_ratio, (uint64_t)g->ix_heads, (uint64_t)g->ix_top_k};
  h = fnv(dims, sizeof dims, h);
  for (int l = 0; l < g->n_layers; l++)
    h = fnv_u64(g->recurrent[l], h);
  h = fnv(&f->rope, sizeof f->rope, h); /* YaRN's or the plain rotation: another context's numbers another model's */
  if (f->mtp) /* a head's cache in the state (parts()): its file's */
    for (int i = 0; i < f->mtp->model->n_files; i++)
      h = fnv_file(f->mtp->model->paths[i], fnv_u64(0x4d5450ull, h));
  if (f->npu) /* the NPU's sums its own, its parts by its share (q4x_forward_npu): another's states */
  {
    uint32_t share;
    memcpy(&share, &f->npu_share, sizeof share);
    h = fnv_u64(0x4e50550000000000ull | share, h);
  }
  const Gguf *files[2] = {f->w->model, f->w->overlay}; /* with an overlay or without: another model's states */
  for (int k = 0; k < 2 && files[k]; k++)
    for (int i = 0; i < files[k]->n_files; i++) /* each file: fnv.h's fnv_file */
      h = fnv_file(files[k]->paths[i], h);
  return h;
}

bool q4x_seq_reset(Q4xSeq *s)
{
  const Q4xGeometry *g = &s->f->w->geo;
  const Q4xSeqBytes sb = q4x_seq_bytes(g);
  s->cur = 0, s->back = 0; /* slot 0 the latest, zeroed: the others not read until a step writes them */
  bool ok = hipMemset(s->ple_hist, 0, sb.ple) == hipSuccess;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = !g->recurrent[l] ||
         (hipMemset(s->ssm[l], 0, sb.ssm) == hipSuccess && hipMemset(s->conv[l], 0, sb.conv) == hipSuccess);
  s->pos = 0, s->n_last = 0, s->prompt_end = 0;
  memset(s->last, 0, sizeof s->last);
  s->pend_pos = -1, s->step_n = 0; /* the head: nothing pending, no step to seed from */
  s->df_lo = s->df_pos = 0;         /* the drafter: no context */
  s->broken = !ok;                  /* a reset that failed leaves it refused */
  return ok;
}

uint64_t q4x_seq_state_bytes(const Q4xSeq *s) { return s ? bytes_at(s, s->pos) : 0; }

bool q4x_seq_save_to(const Q4xSeq *s, StateSink sink, void *ctx, char err[OSMOS_ERR])
{
  if (!s || !sink)
    return osmos_fail(err, "no sequence or sink");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its states part written): load it, or make another");
  Header h;
  memset(&h, 0, sizeof h);
  h.magic = STATE_MAGIC, h.version = STATE_VERSION, h.identity = q4x_identity(s->f), h.pos = s->pos;
  h.n_last = s->n_last < Q4X_MAX_NGRAM ? s->n_last : Q4X_MAX_NGRAM;
  h.pend_pos = s->mkc ? s->pend_pos : -1;
  memcpy(h.last, s->last, sizeof h.last); /* the latest Q4X_MAX_NGRAM: what the n-gram layer reads */
  Part p[MAX_PARTS];
  const char *why = state_parts_save(&h, sizeof h, p, parts(s, s->pos, p), sink, ctx);
  return !why || osmos_fail(err, "%s", why);
}

bool q4x_seq_reset_to(Q4xSeq *s, char err[OSMOS_ERR])
{
  return !s ? osmos_fail(err, "no sequence") : q4x_seq_reset(s) || osmos_fail(err, "the sequence's reset: a GPU call failed");
}

static bool refuse(Q4xSeq *s, char *err, const char *why)
{
  q4x_seq_reset(s);
  return osmos_fail(err, "%s", why);
}

bool q4x_seq_load_from(Q4xSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR])
{
  if (!s || !src)
    return osmos_fail(err, "no sequence or source");
  Header h;
  s->prompt_end = 0; /* the loaded conversation's prompt: told again (q4x_seq_prompt) */
  memset(s->last, 0, sizeof s->last);
  if (bytes < sizeof h + sizeof(uint64_t))
    return refuse(s, err, "a state shorter than its header");
  if (!src(ctx, &h, sizeof h))
    return refuse(s, err, "the source failed (the state's header)");
  if (h.magic != STATE_MAGIC || h.version != STATE_VERSION || h.identity != q4x_identity(s->f))
    return refuse(s, err, "a state of another format or model (its identity)");
  if (h.pos < 0 || (uint32_t)h.pos > s->f->max_ctx || h.n_last < 0 || h.n_last > Q4X_MAX_NGRAM ||
      h.pend_pos < -1 || h.pend_pos >= h.pos || bytes != bytes_at(s, h.pos))
    return refuse(s, err, "a state past this context, or not its position's size");
  Part p[MAX_PARTS];
  s->cur = 0, s->back = 0; /* loaded into slot 0, nothing behind it */
  const char *why = state_parts_load(&h, sizeof h, p, parts(s, h.pos, p), src, ctx);
  if (why) /* (the parts part written: the reset makes the sequence empty) */
    return refuse(s, err, why);
  s->pos = h.pos, s->n_last = h.n_last;
  s->pend_pos = h.pend_pos, s->step_n = 0; /* the head's cache the state's (parts()); its last step's taps not */
  s->df_lo = s->df_pos = s->pos;           /* the drafter's rings are not the state's: its context starts here */
  memcpy(s->last, h.last, sizeof h.last);
  s->broken = false;
  return true;
}

uint32_t q4x_seq_reach(const Q4xSeq *s) { return s && !s->broken ? s->back : 0; } /* (a broken one's rollback refused:
                                                                                       its reach none - 0.11.2) */

void q4x_seq_fail(Q4xSeq *s)
{
  if (s)
    s->broken = true;
}

bool q4x_seq_rollback(Q4xSeq *s, uint32_t n)
{
  if (!s || s->broken || n > s->back)
    return false;
  const int keep = (int)(sizeof s->last / sizeof *s->last);
  memmove(s->last, s->last + n, (size_t)(keep - (int)n) * sizeof *s->last);
  s->n_last = s->n_last > (int)n ? s->n_last - (int)n : 0;
  s->pos -= (int32_t)n, s->back -= n;
  if (s->df_pos > s->pos) /* the drafter's context past the position dropped (its rows written again when fed) */
    s->df_pos = s->pos, s->df_lo = s->df_lo < s->df_pos ? s->df_lo : s->df_pos;
  s->cur = (s->cur + Q4X_SLOTS - n) % Q4X_SLOTS;
  return true;
}

/* a mark (forward.h): a header, then the parts a later token changes - the n-gram conv history, each DeltaNet's state
   and conv history, each attention layer's raw keys' ring - in parts()'s order */
#define MARK_MAGIC 0x4b4d3451u /* "Q4MK" */

typedef struct MarkHeader
{
  uint32_t magic;
  int32_t pos, n_last;
  int32_t last[Q4X_MAX_NGRAM];
  int32_t pend_pos; /* the head's pending h's (-1: none) */
} MarkHeader;

static int mark_parts(const Q4xSeq *s, Part *p)
{
  const Q4xGeometry *g = &s->f->w->geo;
  const Q4xSeqBytes sb = q4x_seq_bytes(g);
  int n = 0;
  p[n++] = (Part){(uint8_t *)s->ple_hist + s->cur * sb.ple, sb.ple};
  for (int l = 0; l < g->n_layers; l++)
    if (g->recurrent[l])
      p[n++] = (Part){(uint8_t *)s->ssm[l] + s->cur * sb.ssm, sb.ssm},
      p[n++] = (Part){(uint8_t *)s->conv[l] + s->cur * sb.conv, sb.conv};
    else
      p[n++] = (Part){s->ixr[l], sb.ix_ring};
  if (s->mkc) /* the head's raw keys' ring and pending h: later tokens change them (its cache rows below the mark they
                 do not - the same conversation's) */
    p[n++] = (Part){s->mixr, sb.ix_ring}, p[n++] = (Part){s->pend, (uint64_t)g->hc * g->n_embd * sizeof(float)};
  return n;
}

uint64_t q4x_seq_mark_bytes(const Q4xSeq *s)
{
  if (!s)
    return 0;
  Part p[MAX_PARTS];
  return state_parts_bytes(sizeof(MarkHeader), p, mark_parts(s, p)) - sizeof(uint64_t); /* a mark has no sum */
}

bool q4x_seq_mark_save(const Q4xSeq *s, void *out, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || !out || bytes != q4x_seq_mark_bytes(s))
    return osmos_fail(err, "no sequence, or a mark's room not its size");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step (its states part written): no mark of it");
  if (hipDeviceSynchronize() != hipSuccess)
    return osmos_fail(err, "a GPU call failed (before the mark's save)");
  MarkHeader h;
  memset(&h, 0, sizeof h);
  h.magic = MARK_MAGIC, h.pos = s->pos, h.n_last = s->n_last < Q4X_MAX_NGRAM ? s->n_last : Q4X_MAX_NGRAM;
  h.pend_pos = s->mkc ? s->pend_pos : -1;
  memcpy(h.last, s->last, sizeof h.last);
  memcpy(out, &h, sizeof h);
  Part p[MAX_PARTS];
  const char *why = state_parts_to_host(p, mark_parts(s, p), (uint8_t *)out + sizeof h);
  return !why || osmos_fail(err, "%s", why);
}

bool q4x_seq_mark_load(Q4xSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR])
{
  if (!s || !in || bytes != q4x_seq_mark_bytes(s))
    return osmos_fail(err, "no sequence, or a mark not its size");
  MarkHeader h;
  memcpy(&h, in, sizeof h);
  if (h.magic != MARK_MAGIC || h.pos < 0 || h.pos > s->pos || h.n_last < 0 || h.n_last > Q4X_MAX_NGRAM)
    return osmos_fail(err, "not a mark, or one past the sequence's position");
  if (s->broken)
    return osmos_fail(err, "a sequence after a failed step: its rows below the mark not known (load a state)");
  if (hipDeviceSynchronize() != hipSuccess) /* its states not still being read */
    return osmos_fail(err, "a GPU call failed (before the mark's load)");
  s->cur = 0, s->back = 0; /* into slot 0, nothing behind it (as a load) */
  Part p[MAX_PARTS];
  if (state_parts_from_host(p, mark_parts(s, p), (const uint8_t *)in + sizeof h))
  {
    q4x_seq_reset(s);
    return osmos_fail(err, "a GPU copy failed (the mark's load): the sequence emptied");
  }
  s->pos = h.pos, s->n_last = h.n_last, s->prompt_end = 0;
  memcpy(s->last, h.last, sizeof h.last);
  s->pend_pos = h.pend_pos < h.pos ? h.pend_pos : -1, s->step_n = 0; /* the head's as at the mark (mark_parts) */
  s->df_lo = s->df_pos = s->pos; /* the drafter's context starts here */
  return true;
}
