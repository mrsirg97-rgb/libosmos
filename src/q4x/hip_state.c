#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "osmos/model/fnv.h"
#include "q4x_impl.h"

/* a sequence as one byte string (osmos/q4x/forward.h, q4x_seq_save): a header, the n-gram layer's conv history,
   each layer's own - a DeltaNet's state and conv history, an attention's cache rows below the position, keys then
   values - then a checksum */

#define STATE_MAGIC 0x53583451u /* "Q4XS" */
#define STATE_VERSION 2u /* 2: the attention cache in Q8_0 */

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

#define MAX_PARTS (1 + 2 * Q4X_MAX_LAYERS)

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
      p[n++] = (Part){s->kc[l], (uint64_t)pos * sb.kv_row}, p[n++] = (Part){s->vc[l], (uint64_t)pos * sb.kv_row};
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
                           (uint64_t)g->head_dim, (uint64_t)g->ple_conv, (uint64_t)g->ple_ngram, Q4X_MAX_NGRAM};
  h = fnv(dims, sizeof dims, h);
  for (int l = 0; l < g->n_layers; l++)
    h = fnv_u64(g->recurrent[l], h);
  const Gguf *m = f->w->model;
  for (int i = 0; i < m->n_files; i++) /* each file's name and size */
  {
    const char *base = strrchr(m->paths[i], '/');
    base = base ? base + 1 : m->paths[i];
    struct stat st;
    h = fnv(base, strlen(base), h);
    h = fnv_u64(stat(m->paths[i], &st) == 0 ? (uint64_t)st.st_size : 0, h);
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
  s->pos = 0, s->n_last = 0;
  memset(s->last, 0, sizeof s->last);
  s->pend_pos = -1, s->step_n = 0; /* the head: nothing pending, no step to seed from */
  return ok;
}

uint64_t q4x_seq_state_bytes(const Q4xSeq *s) { return s ? bytes_at(s, s->pos) : 0; }

bool q4x_seq_save(const Q4xSeq *s, void *out, uint64_t bytes, char err[Q4X_FWD_ERR])
{
  if (!s || !out || bytes != bytes_at(s, s->pos))
    return q4x_fail(err, "no sequence or buffer, or a buffer not the state's size (q4x_seq_state_bytes)");
  Header h;
  memset(&h, 0, sizeof h);
  h.magic = STATE_MAGIC, h.version = STATE_VERSION, h.identity = q4x_identity(s->f), h.pos = s->pos;
  h.n_last = s->n_last < Q4X_MAX_NGRAM ? s->n_last : Q4X_MAX_NGRAM;
  memcpy(h.last, s->last, sizeof h.last); /* the latest Q4X_MAX_NGRAM: what the n-gram layer reads */
  uint8_t *o = out, *at = o + sizeof h;
  memcpy(o, &h, sizeof h);
  Part p[MAX_PARTS];
  for (int i = 0, n = parts(s, s->pos, p); i < n; at += p[i].bytes, i++)
    if (p[i].bytes && hipMemcpy(at, p[i].dev, p[i].bytes, hipMemcpyDeviceToHost) != hipSuccess)
      return q4x_fail(err, "a GPU copy failed (the state's save)");
  const uint64_t sum = fnv(o, (uint64_t)(at - o), FNV_SEED);
  memcpy(at, &sum, sizeof sum);
  return true;
}

static bool refuse(Q4xSeq *s, char *err, const char *why)
{
  q4x_seq_reset(s);
  return q4x_fail(err, why);
}

bool q4x_seq_load(Q4xSeq *s, const void *in, uint64_t bytes, char err[Q4X_FWD_ERR])
{
  if (!s || !in)
    return q4x_fail(err, "no sequence or state");
  Header h;
  if (bytes < sizeof h + sizeof(uint64_t))
    return refuse(s, err, "a state shorter than its header");
  memcpy(&h, in, sizeof h);
  memset(s->last, 0, sizeof s->last);
  if (h.magic != STATE_MAGIC || h.version != STATE_VERSION || h.identity != q4x_identity(s->f))
    return refuse(s, err, "a state of another format or model (its identity)");
  if (h.pos < 0 || (uint32_t)h.pos > s->f->max_ctx || h.n_last < 0 || h.n_last > Q4X_MAX_NGRAM ||
      bytes != bytes_at(s, h.pos))
    return refuse(s, err, "a state past this context, or not its position's size");
  const uint8_t *i = in;
  uint64_t sum;
  memcpy(&sum, i + bytes - sizeof sum, sizeof sum);
  if (sum != fnv(i, bytes - sizeof sum, FNV_SEED))
    return refuse(s, err, "a damaged state (its checksum)");
  const uint8_t *at = i + sizeof h;
  Part p[MAX_PARTS];
  s->cur = 0, s->back = 0; /* loaded into slot 0, nothing behind it */
  for (int k = 0, n = parts(s, h.pos, p); k < n; at += p[k].bytes, k++)
    if (p[k].bytes && hipMemcpy(p[k].dev, at, p[k].bytes, hipMemcpyHostToDevice) != hipSuccess)
      return refuse(s, err, "a GPU copy failed (the state's load)");
  s->pos = h.pos, s->n_last = h.n_last;
  s->pend_pos = -1, s->step_n = 0; /* the head's cache not saved: its seeding starts over (drafts weaker, no bit moved) */
  memcpy(s->last, h.last, sizeof h.last);
  return true;
}

uint32_t q4x_seq_reach(const Q4xSeq *s) { return s ? s->back : 0; }

bool q4x_seq_rollback(Q4xSeq *s, uint32_t n)
{
  if (!s || n > s->back)
    return false;
  const int keep = (int)(sizeof s->last / sizeof *s->last);
  memmove(s->last, s->last + n, (size_t)(keep - (int)n) * sizeof *s->last);
  s->n_last = s->n_last > (int)n ? s->n_last - (int)n : 0;
  s->pos -= (int32_t)n, s->back -= n;
  s->cur = (s->cur + Q4X_SLOTS - n) % Q4X_SLOTS;
  return true;
}
