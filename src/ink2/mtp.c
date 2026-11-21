#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf_meta.h"
#include "osmos/ink2/mtp.h"
#include "osmos/model/qtype.h"

/* the drafter's geometry and tensors (osmos/ink2/mtp.h), each checked */

#define ARCH "inkling2_mtp"
#define T_F32 0u
#define T_BF16 30u

static bool fail(char *err, const char *what, const char *why)
{
  snprintf(err, INK2_ERR, "inkling2_mtp: %s: %s", what, why);
  return false;
}

/* a type the matvec reads (qtype.h's QTYPE_MV), its rows whole blocks */
static bool is_mat(const GgufTensor *t)
{
  const QTypeInfo *q = qtype_info(t->type);
  return q && (q->paths & QTYPE_MV) && t->dims[0] % q->values == 0;
}

/* a tensor of name, dims (ne, innermost first) and a type the role reads */
static const GgufTensor *want(const Gguf *g, const char *name, uint64_t d0, uint64_t d1, bool mat, char *err)
{
  const GgufTensor *t = NULL;
  for (size_t i = 0; i < g->n_tensors && !t; i++)
    if (!strcmp(g->tensors[i].name, name))
      t = &g->tensors[i];
  if (!t)
    return fail(err, name, "missing"), NULL;
  const bool dims = t->dims[0] == d0 && (d1 ? t->n_dims == 2 && t->dims[1] == d1 : t->n_dims == 1);
  if (!dims)
    return fail(err, name, "not the geometry's shape"), NULL;
  if (mat ? !is_mat(t) : t->type != T_F32)
    return fail(err, name, mat ? "a type the matvec does not read" : "not F32"), NULL;
  return t;
}

Ink2Mtp *ink2_mtp_create(const Gguf *g, const Ink2Weights *target, char err[INK2_ERR])
{
  if (!g || !target)
    return fail(err, "create", "no drafter or no target"), NULL;
  Ink2Mtp *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "create", "out of memory"), NULL;
  GgufMeta meta = {g, err, NULL, 0};
  int heads_dim = 0;
  int64_t local[INK2_MTP_MAX_LAYERS], taps[INK2_MTP_MAX_TAPS];
  size_t n_local = 0, n_taps = 0;
  bool ok = meta_text_is(&meta, "general.architecture", ARCH) &&
            meta_uint_in(&meta, ARCH ".block_count", 1, INK2_MTP_MAX_LAYERS, &m->n_layers) &&
            meta_uint_in(&meta, ARCH ".embedding_length", 1, 1 << 16, &m->n_embd) &&
            meta_uint_in(&meta, ARCH ".feed_forward_length", 1, 1 << 20, &m->ff) &&
            meta_uint_in(&meta, ARCH ".attention.head_count", 1, 1024, &m->n_heads) &&
            meta_uint_in(&meta, ARCH ".attention.head_count_kv", 1, m->n_heads, &m->n_kv_heads) &&
            meta_uint_in(&meta, ARCH ".attention.key_length", 1, 1024, &heads_dim) &&
            meta_real(&meta, ARCH ".attention.layer_norm_rms_epsilon", &m->eps) &&
            meta_real(&meta, ARCH ".rope.freq_base", &m->rope_base) &&
            meta_uint_in(&meta, ARCH ".attention.sliding_window", 1, 1 << 24, &m->window) &&
            meta_uint_in(&meta, ARCH ".block_size", 2, INK2_MTP_BLOCK, &m->block) &&
            meta_uint_in(&meta, ARCH ".markov_rank", 1, 1 << 16, &m->rank) &&
            meta_ints(&meta, ARCH ".local_layers", 0, INK2_MTP_MAX_LAYERS, local, &n_local) &&
            meta_ints(&meta, ARCH ".target_layers", 1, INK2_MTP_MAX_TAPS, taps, &n_taps);
  if (!ok)
    return ink2_mtp_destroy(m), NULL;
  const Ink2Geometry *tg = &target->geo;
  if (m->n_embd != tg->n_embd)
    ok = fail(err, ARCH ".embedding_length", "not the target's width");
  else if (heads_dim != INK2_MTP_HEAD || m->n_heads % m->n_kv_heads)
    ok = fail(err, ARCH ".attention.key_length", "heads not of 128 (the kernels'), or not a KV heads' multiple");
  else if (m->block * (m->n_heads / m->n_kv_heads) > INK2_MTP_ATTN_ROWS)
    ok = fail(err, ARCH ".block_size", "a block's queries times a KV head's heads past the attention's 48 rows");
  else if (m->rank != INK2_MTP_RANK)
    ok = fail(err, ARCH ".markov_rank", "not 256 (the kernels')");
  m->n_taps = (int)n_taps;
  for (size_t i = 0; ok && i < n_taps; i++)
  {
    m->targets[i] = (int32_t)taps[i];
    const bool last = i + 1 == n_taps;
    if ((last && taps[i] != tg->n_slots) || (!last && (taps[i] < 0 || taps[i] >= tg->n_slots)) ||
        (i && taps[i] <= taps[i - 1]))
      ok = fail(err, ARCH ".target_layers", "not the target's steps ascending, the last its n_slots");
  }
  for (size_t i = 0; ok && i < n_local; i++)
    if (local[i] < 0 || local[i] >= m->n_layers)
      ok = fail(err, ARCH ".local_layers", "a layer not the drafter's");
    else
      m->local[local[i]] = true;
  if (!ok)
    return ink2_mtp_destroy(m), NULL;
  m->model = g;
  const uint64_t C = (uint64_t)m->n_embd, Q = (uint64_t)m->n_heads * INK2_MTP_HEAD,
                 KV = (uint64_t)m->n_kv_heads * INK2_MTP_HEAD, F = (uint64_t)m->ff, R = (uint64_t)m->rank;
  char n[128];
#define W(field, suffix, d0, d1, mat)                                                                                  \
  (snprintf(n, sizeof n, "layers.%d." suffix, l), (L->field = want(g, n, d0, d1, mat, err)) != NULL)
  for (int l = 0; ok && l < m->n_layers; l++)
  {
    Ink2MtpLayer *L = &m->layer[l];
    ok = W(in_norm, "input_layernorm.weight", C, 0, false) && W(post_norm, "post_attention_layernorm.weight", C, 0, false) &&
         W(dflare_norm, "dflare_norm.weight", C, 0, false) && W(q_norm, "self_attn.q_norm.weight", INK2_MTP_HEAD, 0, false) &&
         W(k_norm, "self_attn.k_norm.weight", INK2_MTP_HEAD, 0, false) &&
         W(k_norm_ctx, "self_attn.k_norm_ctx.weight", INK2_MTP_HEAD, 0, false) && W(q, "self_attn.q_proj.weight", C, Q, true) &&
         W(k, "self_attn.k_proj.weight", C, KV, true) && W(v, "self_attn.v_proj.weight", C, KV, true) &&
         W(k_ctx, "self_attn.k_proj_ctx.weight", C, KV, true) && W(v_ctx, "self_attn.v_proj_ctx.weight", C, KV, true) &&
         W(o, "self_attn.o_proj.weight", Q, C, true) && W(gate, "mlp.gate_proj.weight", C, F, true) &&
         W(up, "mlp.up_proj.weight", C, F, true) && W(down, "mlp.down_proj.weight", F, C, true);
  }
#undef W
  const uint64_t V = target->output->dims[1];
  ok = ok && (m->fuse = want(g, "fuse_weight", n_taps, (uint64_t)m->n_layers + 1, false, err)) &&
       (m->mask = want(g, "mask_embed", C, 0, false, err)) && (m->norm = want(g, "norm.weight", C, 0, false, err)) &&
       (m->m_dflare = want(g, "markov_head.dflare_norm.weight", C, 0, false, err)) &&
       (m->m_hidden_w = want(g, "markov_head.hidden_out.weight", R, C, false, err)) &&
       (m->m_hidden_b = want(g, "markov_head.hidden_out.bias", C, 0, false, err)) &&
       (m->m_rnn_w = want(g, "markov_head.rnn_proj.weight", 2 * R + 2 * C, 3 * R, false, err)) &&
       (m->m_rnn_b = want(g, "markov_head.rnn_proj.bias", 3 * R, 0, false, err));
  if (ok)
  {
    const GgufTensor *w1 = NULL;
    for (size_t i = 0; i < g->n_tensors && !w1; i++)
      if (!strcmp(g->tensors[i].name, "markov_head.markov_w1.weight"))
        w1 = &g->tensors[i];
    if (!w1 || w1->type != T_BF16 || w1->n_dims != 2 || w1->dims[0] != R || w1->dims[1] != V)
      ok = fail(err, "markov_head.markov_w1.weight", "missing, not BF16, or not [the head's vocabulary][rank]");
    m->m_w1 = w1;
  }
  if (!ok)
    return ink2_mtp_destroy(m), NULL;
  return m;
}

void ink2_mtp_destroy(Ink2Mtp *m) { free(m); }
