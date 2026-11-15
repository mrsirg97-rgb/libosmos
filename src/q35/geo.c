#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gguf_meta.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/model/qtype.h"
#include "osmos/q35/forward.h"

#define ARCH "qwen35"
#define DIM_MAX (1 << 24)
#define CTX_MAX (1 << 30)
_Static_assert(Q35_ERR == GGUF_META_ERR && Q35_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");

/* the kernels' fixed shapes are q4x's (q4x_ops.h): a head 256, a DeltaNet state 128, a conv's taps 8 at most */
bool q35_geometry(const Gguf *model, Q35Geometry *g, char err[Q35_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int ctx = 0;
  int mtp = 0; /* block_count holds the MTP head's layers too (nextn_predict_layers), after the model's */
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 1, Q35_MAX_LAYERS, &g->n_layers) ||
      (meta_present(&m, ARCH ".nextn_predict_layers") &&
       !meta_uint_in(&m, ARCH ".nextn_predict_layers", 0, g->n_layers - 1, &mtp)))
    return false;
  g->n_layers -= mtp;
  const bool ok =
      meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
      meta_uint_in(&m, ARCH ".feed_forward_length", 1, DIM_MAX, &g->n_ff) &&
      meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
      meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
      meta_uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
      meta_uint_in(&m, ARCH ".attention.head_count_kv", 1, g->n_heads, &g->n_kv_heads) &&
      (g->n_heads % g->n_kv_heads == 0 || meta_fail(&m, ARCH ".attention.head_count_kv", "not a heads' divisor")) &&
      (g->n_heads / g->n_kv_heads <= Q4X_GQA_MAX ||
       meta_fail(&m, ARCH ".attention.head_count_kv", "more query heads a key head than the attention's tile")) &&
      meta_uint_is(&m, ARCH ".attention.key_length", Q4X_HEAD_DIM) && (g->head_dim = Q4X_HEAD_DIM) &&
      meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
      meta_uint_in(&m, ARCH ".rope.dimension_count", 2, g->head_dim, &g->rope_dim) &&
      meta_real(&m, ARCH ".rope.freq_base", &g->rope_base) &&
      meta_uint_in(&m, ARCH ".full_attention_interval", 1, g->n_layers, &g->attn_interval) &&
      meta_uint_in(&m, ARCH ".ssm.conv_kernel", 1, Q4X_CONV_TAPS_MAX, &g->ssm_conv) &&
      meta_uint_is(&m, ARCH ".ssm.state_size", Q4X_SSM_STATE) && (g->ssm_state = Q4X_SSM_STATE) &&
      meta_uint_in(&m, ARCH ".ssm.group_count", 1, DIM_MAX, &g->ssm_groups) &&
      meta_uint_in(&m, ARCH ".ssm.time_step_rank", 1, DIM_MAX, &g->ssm_v_heads) &&
      (g->ssm_v_heads % g->ssm_groups == 0 || meta_fail(&m, ARCH ".ssm.group_count", "not the value heads' divisor")) &&
      meta_uint_is(&m, ARCH ".ssm.inner_size", (uint64_t)g->ssm_v_heads * (uint64_t)g->ssm_state) &&
      (g->ssm_inner = g->ssm_v_heads * g->ssm_state);
  if (!ok)
    return false;
  /* text alone: MRoPE's sections all one position, the plain rotation of the first rope_dim dims */
  for (int l = 0; l < g->n_layers; l++)
    g->recurrent[l] = (l + 1) % g->attn_interval != 0;
  g->native_ctx = (uint32_t)ctx;
  return true;
}

static bool fail(char *err, const char *name, const char *why)
{
  if (err)
    snprintf(err, Q35_ERR, "%s: %s", name, why);
  return false;
}

/* a tensor by name, its ggml dims (cols first) as asked; with a scale (s), an NVFP4 matrix's .scale read beside it -
   1 for any other type */
static bool find(const Gguf *g, const char *name, uint64_t d0, uint64_t d1, const GgufTensor **t, float *s, char *err)
{
  if (!(*t = gguf_tensor(g, name)))
    return fail(err, name, "not in the model");
  const GgufTensor *x = *t;
  if (x->dims[0] != d0 || (d1 ? x->n_dims != 2 || x->dims[1] != d1 : x->n_dims != 1))
  {
    char why[96];
    snprintf(why, sizeof why, "shape %llu x %llu, the geometry's %llu x %llu", (unsigned long long)x->dims[0],
             (unsigned long long)(x->n_dims > 1 ? x->dims[1] : 0), (unsigned long long)d0, (unsigned long long)d1);
    return fail(err, name, why);
  }
  if (!s)
    return x->type == QTYPE_F32 || fail(err, name, "not F32 (a norm, a gate's bias, a conv)");
  *s = 1.0f;
  if (x->type != QTYPE_NVFP4)
    return true;
  char sn[160];
  snprintf(sn, sizeof sn, "%.*s.scale", (int)(strlen(name) - strlen(".weight")), name);
  const GgufTensor *st = gguf_tensor(g, sn);
  if (!st || st->type != QTYPE_F32 || st->bytes != 4)
    return fail(err, name, "NVFP4 without its one F32 .scale beside it");
  const int fd = open(g->paths[st->file], O_RDONLY);
  const bool read_ok = fd >= 0 && pread(fd, s, 4, (off_t)st->offset) == 4;
  if (fd >= 0)
    close(fd);
  return read_ok || fail(err, sn, "not read");
}

Q35Weights *q35_weights_create(const Gguf *model, char err[Q35_ERR])
{
  Q35Weights *w = calloc(1, sizeof *w);
  if (!w)
    return fail(err, "q35", "out of memory"), NULL;
  w->model = model;
  Q35Geometry *g = &w->geo;
  bool ok = q35_geometry(model, g, err);
  const uint64_t D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff, hd = (uint64_t)g->head_dim;
  const uint64_t q = (uint64_t)g->n_heads * hd, kv = (uint64_t)g->n_kv_heads * hd, vh = (uint64_t)g->ssm_v_heads;
  const uint64_t in = (uint64_t)g->ssm_inner, ch = 2 * (uint64_t)g->ssm_groups * (uint64_t)g->ssm_state + in;
  const GgufTensor *e = ok ? gguf_tensor(model, "token_embd.weight") : NULL;
  ok = ok && (e || fail(err, "token_embd.weight", "not in the model")) && e->n_dims == 2 && e->dims[0] == D &&
       (g->n_vocab = (int)e->dims[1]) > 0 && find(model, "token_embd.weight", D, e->dims[1], &w->embd, &(float){0}, err) &&
       find(model, "output_norm.weight", D, 0, &w->out_norm, NULL, err) &&
       find(model, "output.weight", D, (uint64_t)g->n_vocab, &w->output, &w->s_output, err);
  char n[96];
#define T(name, d0, d1, field, scale)                                                                                \
  (snprintf(n, sizeof n, "blk.%d." name, l), find(model, n, d0, d1, &L->field, scale, err))
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    Q35Layer *L = &w->layer[l];
    ok = T("attn_norm.weight", D, 0, attn_norm, NULL) && T("post_attention_norm.weight", D, 0, post_norm, NULL) &&
         T("ffn_gate.weight", D, ff, ffn_gate, &L->s_ffn_gate) && T("ffn_up.weight", D, ff, ffn_up, &L->s_ffn_up) &&
         T("ffn_down.weight", ff, D, ffn_down, &L->s_ffn_down);
    if (ok && g->recurrent[l])
      ok = T("attn_qkv.weight", D, ch, qkv, &L->s_qkv) && T("attn_gate.weight", D, in, z, &L->s_z) &&
           T("ssm_alpha.weight", D, vh, alpha, &L->s_alpha) && T("ssm_beta.weight", D, vh, beta, &L->s_beta) &&
           T("ssm_conv1d.weight", (uint64_t)g->ssm_conv, ch, conv, NULL) && T("ssm_dt.bias", vh, 0, dt, NULL) &&
           T("ssm_a", vh, 0, a, NULL) && T("ssm_norm.weight", (uint64_t)g->ssm_state, 0, ssm_norm, NULL) &&
           T("ssm_out.weight", in, D, ssm_out, &L->s_ssm_out);
    else if (ok)
      ok = T("attn_q.weight", D, 2 * q, q, &L->s_q) && T("attn_k.weight", D, kv, k, &L->s_k) &&
           T("attn_v.weight", D, kv, v, &L->s_v) && T("attn_q_norm.weight", hd, 0, q_norm, NULL) &&
           T("attn_k_norm.weight", hd, 0, k_norm, NULL) && T("attn_output.weight", q, D, out, &L->s_out);
  }
#undef T
  if (!ok)
    return free(w), NULL;
  return w;
}

void q35_weights_destroy(Q35Weights *w) { free(w); }
