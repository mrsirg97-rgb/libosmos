/* LFM3's geometry and weights (osmos/lfm3/forward.h): the metadata read through gguf_meta's checks, every tensor in
   a role (its name, shape and type), none left over */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gguf_meta.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/lfm3/forward.h"
#include "osmos/model/qtype.h"
#include "osmos/err.h"

#define ARCH "lfm3"
#define DIM_MAX (1 << 24)
#define CTX_MAX (1 << 30)
#define UNROTATED 1e30f /* rope_freqs' factor on a pair the full layers leave unrotated (the zero tail) */
_Static_assert(OSMOS_ERR == OSMOS_ERR && LFM3_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");

bool lfm3_geometry(const Gguf *model, Lfm3Geometry *g, char err[OSMOS_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int ctx = 0;
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 1, LFM3_MAX_LAYERS, &g->n_layers))
    return false;
  const int L = g->n_layers;
  int64_t kv[LFM3_MAX_LAYERS];
  bool sliding[LFM3_MAX_LAYERS];
  size_t n_kv = 0;
  bool ok = meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
            meta_uint_in(&m, ARCH ".feed_forward_length", 1, DIM_MAX, &g->n_ff) &&
            meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
            meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
            meta_uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
            meta_uint_is(&m, ARCH ".attention.key_length", LFM3_HEAD_DIM) && (g->head_dim = LFM3_HEAD_DIM) &&
            meta_uint_is(&m, ARCH ".attention.value_length", LFM3_HEAD_DIM) &&
            meta_uint_is(&m, ARCH ".rope.dimension_count", LFM3_HEAD_DIM) &&
            meta_real(&m, ARCH ".rope.freq_base", &g->rope_base) &&
            meta_real(&m, ARCH ".rope.freq_base_swa", &g->rope_base_swa) &&
            meta_uint_in(&m, ARCH ".attention.sliding_window", 1, CTX_MAX, &g->window) &&
            meta_uint_in(&m, ARCH ".shortconv.l_cache", 1, DIM_MAX, &g->conv_taps) &&
            (g->conv_taps <= Q4X_CONV_TAPS_MAX ||
             meta_fail(&m, ARCH ".shortconv.l_cache", "more taps than the convolution's kernel (Q4X_CONV_TAPS_MAX)")) &&
            meta_ints(&m, ARCH ".attention.head_count_kv", (size_t)L, (size_t)L, kv, &n_kv) &&
            meta_flags(&m, ARCH ".attention.sliding_window_pattern", (size_t)L, sliding);
  for (int l = 0; ok && l < L; l++)
  {
    if (kv[l] == 0)
    {
      ok = !sliding[l] || meta_fail(&m, ARCH ".attention.sliding_window_pattern", "a convolution layer's a window");
      g->kind[l] = LFM3_CONV;
      continue;
    }
    if (!g->n_kv_heads)
      g->n_kv_heads = (int)kv[l];
    ok = (kv[l] == g->n_kv_heads ||
          meta_fail(&m, ARCH ".attention.head_count_kv", "the attention layers' key heads not one count")) &&
         (g->n_heads % g->n_kv_heads == 0 || meta_fail(&m, ARCH ".attention.head_count_kv", "not a heads' divisor")) &&
         (g->n_heads / g->n_kv_heads <= Q4X_GQA_MAX ||
          meta_fail(&m, ARCH ".attention.head_count_kv", "more query heads a key head than the attention's tile"));
    g->kind[l] = sliding[l] ? LFM3_SLIDING : LFM3_FULL;
  }
  ok = ok && (g->n_kv_heads || meta_fail(&m, ARCH ".attention.head_count_kv", "no attention layer"));
  g->native_ctx = (uint32_t)ctx;
  return ok;
}

typedef struct Find
{
  const Gguf *g;
  char *err;
  uint32_t matrix_type; /* the file's matrices' one type: the first matrix's */
  size_t found;         /* the tensors a role took: every one of the file's, at the end */
} Find;

/* a tensor by name, its ggml dims (cols first; d1 0: a vector) as asked; a matrix (matrix) the file's one matrix type
   and one the matvec takes, else F32 */
static bool find(Find *f, const char *name, uint64_t d0, uint64_t d1, bool matrix, const GgufTensor **t)
{
  if (!(*t = gguf_tensor(f->g, name)))
    return osmos_fail(f->err, "%s: %s", name, "not in the model");
  const GgufTensor *x = *t;
  if (x->dims[0] != d0 || (d1 ? x->n_dims != 2 || x->dims[1] != d1 : x->n_dims != 1))
  {
    char why[96];
    snprintf(why, sizeof why, "shape %llu x %llu, the geometry's %llu x %llu", (unsigned long long)x->dims[0],
             (unsigned long long)(x->n_dims > 1 ? x->dims[1] : 0), (unsigned long long)d0, (unsigned long long)d1);
    return osmos_fail(f->err, "%s: %s", name, why);
  }
  f->found++;
  if (!matrix)
    return x->type == QTYPE_F32 || osmos_fail(f->err, "%s: %s", name, "not F32 (a norm, a convolution's taps, the rope's factors)");
  const QTypeInfo *q = qtype_info(x->type);
  if (!q || !(q->paths & QTYPE_MV))
    return osmos_fail(f->err, "%s: %s", name, "a type the matvec does not take");
  if (!f->matrix_type)
    f->matrix_type = x->type;
  return x->type == f->matrix_type || osmos_fail(f->err, "%s: %s", name, "of another type than the file's other matrices");
}

/* rope_freqs: the rotated pairs (a factor 1) first, then the unrotated (at least UNROTATED) */
static bool rope_pairs(Find *f, const GgufTensor *t, int *pairs)
{
  float v[LFM3_HEAD_DIM / 2];
  const int fd = open(f->g->paths[t->file], O_RDONLY);
  const bool got = fd >= 0 && pread(fd, v, sizeof v, (off_t)t->offset) == (ssize_t)sizeof v;
  if (fd >= 0)
    close(fd);
  if (!got)
    return osmos_fail(f->err, "%s: %s", t->name, "not read");
  int p = 0;
  while (p < LFM3_HEAD_DIM / 2 && v[p] == 1.0f)
    p++;
  for (int i = p; i < LFM3_HEAD_DIM / 2; i++)
    if (!(v[i] >= UNROTATED))
      return osmos_fail(f->err, "%s: %s", t->name, "a factor neither 1 (rotated) nor the zero tail's (unrotated)");
  *pairs = p;
  return p > 0 || osmos_fail(f->err, "%s: %s", t->name, "no pair rotated");
}

Lfm3Weights *lfm3_weights_create(const Gguf *model, char err[OSMOS_ERR])
{
  Lfm3Weights *w = calloc(1, sizeof *w);
  if (!w)
    return osmos_fail(err, "%s: %s", ARCH, "out of memory"), NULL;
  w->model = model;
  Lfm3Geometry *g = &w->geo;
  Find f = {model, err, 0, 0};
  bool ok = lfm3_geometry(model, g, err);
  const uint64_t D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff, hd = (uint64_t)g->head_dim;
  const uint64_t H = (uint64_t)g->n_heads, q = H * hd, kv = (uint64_t)g->n_kv_heads * hd;
  const GgufTensor *e = ok ? gguf_tensor(model, "token_embd.weight") : NULL;
  ok = ok && (e || osmos_fail(err, "%s: %s", "token_embd.weight", "not in the model")) && e->n_dims == 2 && e->dims[0] == D &&
       (g->n_vocab = (int)e->dims[1]) > 0 && (e->type == QTYPE_BF16 || osmos_fail(err, "%s: %s", e->name, "the tied head not BF16"));
  if (ok)
    w->embd = e, f.found++;
  ok = ok && find(&f, "output_norm.weight", D, 0, false, &w->out_norm) &&
       find(&f, "rope_freqs.weight", hd / 2, 0, false, &w->rope_freqs) && rope_pairs(&f, w->rope_freqs, &g->rope_pairs);
  char n[96];
#define T(name, d0, d1, matrix, field) (snprintf(n, sizeof n, "blk.%d." name, l), find(&f, n, d0, d1, matrix, &L->field))
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    Lfm3Layer *L = &w->layer[l];
    ok = T("attn_norm.weight", D, 0, false, attn_norm) && T("post_attention_norm.weight", D, 0, false, post_attn_norm) &&
         T("ffn_norm.weight", D, 0, false, ffn_norm) && T("post_ffw_norm.weight", D, 0, false, post_ffn_norm) &&
         T("ffn_gate.weight", D, ff, true, ffn_gate) && T("ffn_up.weight", D, ff, true, ffn_up) &&
         T("ffn_down.weight", ff, D, true, ffn_down);
    if (ok && g->kind[l] == LFM3_CONV)
      ok = T("shortconv.in_proj.weight", D, 3 * D, true, in_proj) &&
           T("shortconv.conv.weight", (uint64_t)g->conv_taps, D, false, conv) &&
           T("shortconv.out_proj.weight", D, D, true, out_proj);
    else if (ok)
      ok = T("attn_q.weight", D, q, true, q) && T("attn_k.weight", D, kv, true, k) && T("attn_v.weight", D, kv, true, v) &&
           T("attn_q_norm.weight", hd, 0, false, q_norm) && T("attn_k_norm.weight", hd, 0, false, k_norm) &&
           T("attn_gate.weight", D, H, true, gate) && T("attn_output.weight", q, D, true, out);
  }
#undef T
  if (ok && f.found != model->n_tensors)
    for (size_t i = 0; ok && i < model->n_tensors; i++)
    {
      const GgufTensor *t = &model->tensors[i];
      bool taken = t == w->embd || t == w->out_norm || t == w->rope_freqs;
      for (int l = 0; !taken && l < g->n_layers; l++)
      {
        const GgufTensor *const *r = (const GgufTensor *const *)&w->layer[l];
        for (size_t k = 0; !taken && k < sizeof(Lfm3Layer) / sizeof(*r); k++)
          taken = r[k] == t;
      }
      ok = taken || osmos_fail(err, "%s: %s", t->name, "a tensor no role takes");
    }
  if (!ok)
    return free(w), NULL;
  return w;
}

void lfm3_weights_destroy(Lfm3Weights *w) { free(w); }
