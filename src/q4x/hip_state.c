#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "osmos/model/fnv.h"
#include "q4x_impl.h"

/* a sequence as one byte string (osmos/q4x/forward.h, q4x_seq_save): a header, the n-gram layer's conv history,
   each layer's own - a DeltaNet's state and conv history, an attention's cache rows below the position, keys then
   values, then its indexer's pooled keys below the position's blocks and its raw keys' ring - then a checksum */

#define STATE_MAGIC 0x53583451u /* "Q4XS" */
#define STATE_VERSION 3u /* 2: the attention cache in Q8_0; 3: QSA's pooled keys and ring */

typedef struct Header
{
  uint32_t magic, version;
  uint64_t identity;
  int32_t pos, n_last;
  int32_t last[Q4X_MAX_NGRAM];
} Header;

typedef struct Part
{
  void *dev;
  uint64_t bytes;
} Part;

#define MAX_PARTS (1 + 4 * Q4X_MAX_LAYERS)

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
  return n;
}

static uint64_t bytes_at(const Q4xSeq *s, int32_t pos)
{
  Part p[MAX_PARTS];
  uint64_t b = sizeof(Header) + sizeof(uint64_t);
  for (int i = 0, n = parts(s, pos, p); i < n; i++)
    b += p[i].bytes;
  return b;
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
  const Gguf *files[2] = {f->w->model, f->w->overlay}; /* with an overlay or without: another model's states */
  for (int k = 0; k < 2 && files[k]; k++)
    for (int i = 0; i < files[k]->n_files; i++) /* each file's name and size */
    {
      const char *base = strrchr(files[k]->paths[i], '/');
      base = base ? base + 1 : files[k]->paths[i];
      struct stat st;
      h = fnv(base, strlen(base), h);
      h = fnv_u64(stat(files[k]->paths[i], &st) == 0 ? (uint64_t)st.st_size : 0, h);
    }
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

/* a part between the device and a sink or source in pieces through bounce (STATE_CHUNK bytes); sum the FNV-1a of
   every byte moved, in order */
static bool move_part(const Part *p, uint8_t *bounce, bool out, StateSink sink, StateSource src, void *ctx,
                      uint64_t *sum)
{
  for (uint64_t done = 0; done < p->bytes;)
  {
    const uint64_t n = p->bytes - done < STATE_CHUNK ? p->bytes - done : STATE_CHUNK;
    uint8_t *dev = (uint8_t *)p->dev + done;
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
    done += n;
  }
  return true;
}

bool q4x_seq_save_to(const Q4xSeq *s, StateSink sink, void *ctx, char err[Q4X_FWD_ERR])
{
  if (!s || !sink)
    return q4x_fail(err, "no sequence or sink");
  if (s->broken)
    return q4x_fail(err, "a sequence after a failed step (its states part written): load it, or make another");
  if (hipDeviceSynchronize() != hipSuccess) /* every stream's writes of its state done (the side stream's too) */
    return q4x_fail(err, "a GPU call failed (before the state's save)");
  Header h;
  memset(&h, 0, sizeof h);
  h.magic = STATE_MAGIC, h.version = STATE_VERSION, h.identity = q4x_identity(s->f), h.pos = s->pos;
  h.n_last = s->n_last < Q4X_MAX_NGRAM ? s->n_last : Q4X_MAX_NGRAM;
  memcpy(h.last, s->last, sizeof h.last); /* the latest Q4X_MAX_NGRAM: what the n-gram layer reads */
  uint8_t *bounce = malloc(STATE_CHUNK);
  uint64_t sum = fnv(&h, sizeof h, FNV_SEED);
  bool ok = bounce && sink(ctx, &h, sizeof h);
  Part p[MAX_PARTS];
  for (int i = 0, n = parts(s, s->pos, p); ok && i < n; i++)
    ok = move_part(&p[i], bounce, true, sink, NULL, ctx, &sum);
  ok = ok && sink(ctx, &sum, sizeof sum);
  free(bounce);
  return ok || q4x_fail(err, bounce ? "a GPU copy or the sink failed (the state's save)" : "out of memory (the save)");
}

static bool refuse(Q4xSeq *s, char *err, const char *why)
{
  q4x_seq_reset(s);
  return q4x_fail(err, why);
}

bool q4x_seq_load_from(Q4xSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[Q4X_FWD_ERR])
{
  if (!s || !src)
    return q4x_fail(err, "no sequence or source");
  Header h;
  uint64_t sum = 0;
  s->prompt_end = 0; /* the loaded conversation's prompt: told again (q4x_seq_prompt) */
  memset(s->last, 0, sizeof s->last);
  if (bytes < sizeof h + sizeof sum)
    return refuse(s, err, "a state shorter than its header");
  if (!src(ctx, &h, sizeof h))
    return refuse(s, err, "the source failed (the state's header)");
  if (h.magic != STATE_MAGIC || h.version != STATE_VERSION || h.identity != q4x_identity(s->f))
    return refuse(s, err, "a state of another format or model (its identity)");
  if (h.pos < 0 || (uint32_t)h.pos > s->f->max_ctx || h.n_last < 0 || h.n_last > Q4X_MAX_NGRAM ||
      bytes != bytes_at(s, h.pos))
    return refuse(s, err, "a state past this context, or not its position's size");
  if (hipDeviceSynchronize() != hipSuccess) /* the sequence's state not still being read */
    return refuse(s, err, "a GPU call failed (before the state's load)");
  uint8_t *bounce = malloc(STATE_CHUNK);
  if (!bounce)
    return refuse(s, err, "out of memory (the state's load)");
  uint64_t got = fnv(&h, sizeof h, FNV_SEED);
  Part p[MAX_PARTS];
  s->cur = 0, s->back = 0; /* loaded into slot 0, nothing behind it */
  bool ok = true;
  for (int k = 0, n = parts(s, h.pos, p); ok && k < n; k++)
    ok = move_part(&p[k], bounce, false, NULL, src, ctx, &got);
  free(bounce);
  if (!ok || !src(ctx, &sum, sizeof sum))
    return refuse(s, err, "a GPU copy or the source failed (the state's load)");
  if (got != sum) /* (the parts already on the GPU: the reset makes the sequence empty) */
    return refuse(s, err, "a damaged state (its checksum)");
  s->pos = h.pos, s->n_last = h.n_last;
  s->pend_pos = -1, s->step_n = 0; /* the head's cache not saved: its seeding starts over (drafts weaker, no bit moved) */
  s->df_lo = s->df_pos = s->pos;    /* the drafter's rings neither: its context starts here */
  memcpy(s->last, h.last, sizeof h.last);
  s->broken = false;
  return true;
}

uint32_t q4x_seq_reach(const Q4xSeq *s) { return s ? s->back : 0; }

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
