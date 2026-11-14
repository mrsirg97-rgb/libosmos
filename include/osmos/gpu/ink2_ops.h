#ifndef OSMOS_GPU_INK2_OPS_H
#define OSMOS_GPU_INK2_OPS_H

#include <stdbool.h>
#include <stdint.h>

/* Inkling-2-Small's own steps (DESIGN.md, "the forward, op by op"), F32 on the GPU, queued on `stream`: each held to
   transformers' own module by tools/convert/ink2_cases.py's files (test_ink2_ops_gpu) */

#define INK2_MAX_TAPS 8   /* a short conv's taps at most (Inkling's 4) */
#define INK2_MAX_RES 16   /* attn_res's states a token at most (its blocks, then the partial one: Inkling's <= 9) */
#define INK2_MAX_QUERIES 3 /* attn_res's queries at once (a full layer's q, k, v) */

bool ink2_softcap(float *x, uint64_t n, float cap, void *stream);
bool ink2_swiglu(const float *g, const float *u, uint32_t n, float gcap, float ucap, uint32_t n_tok, float *h,
                 void *stream);
bool ink2_conv(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float *w, const float *hist,
               float scale, float *y, void *stream);
bool ink2_conv_dyn(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads, const float *w,
                   const float *dyn, const float *hist, float scale, float *y, void *stream);
bool ink2_attn_res(const float *query, uint32_t queries, const float *blocks, uint32_t n_blocks, uint32_t b_stride,
                   const float *partial, uint32_t n_tok, uint32_t dim, float eps, float scale, float *out, void *stream);
bool ink2_scale(float *x, uint64_t n, float s, void *stream);
bool ink2_axpy(float *y, const float *x, uint64_t n, float a, void *stream);
#define INK2_MAX_HEAD 128   /* attention's head dim at most (Inkling's 128) */
#define INK2_MAX_EXTENT 1024 /* the relative bias's extent at most (Inkling's 1,024) */
#define INK2_MAX_REL 32      /* a head's relative states at most (Inkling's 16) */
#define INK2_FA_MAX_GROUP 8  /* ink2_fa's query heads a KV head at most (Inkling's 6): its waves */
typedef struct Ink2Tau
{
  uint32_t pos0;     /* the step's first token's position */
  float alpha, floor; /* 1 + alpha ln(max(1, (pos + 1) / floor)); alpha 0: none (a sliding layer's) */
} Ink2Tau;
bool ink2_clip_rows(float *x, uint32_t n_tok, uint32_t per_tok, uint32_t dim, const float *w, float clip, Ink2Tau tau,
                    void *stream);
typedef struct Ink2Attn
{
  uint32_t heads, kv_heads, head_dim, d_rel, extent;
  uint32_t window; /* the keys a token sees, itself the last (0: every one before it) */
  float scaling;
} Ink2Attn;
bool ink2_attn(const Ink2Attn *a, const float *q, const float *rel, const float *sink, const float *gate,
               uint32_t n_tok, const float *k, const float *v, uint32_t n_keys, const float *rel_proj, float *out,
               void *stream);
bool ink2_fa(const Ink2Attn *a, const float *q, const float *rel, const float *sink, const float *gate, uint32_t n_tok,
             const float *k, const float *v, uint32_t n_keys, const float *rel_proj, float *out, void *stream);

#endif

/*
ink2_softcap
  x[i] = cap tanh(x[i] / cap), in place: the logits' (final_logit_softcapping, 30).

ink2_swiglu
  h = gcap tanh(g / gcap) sigmoid(g) . ucap tanh(u / ucap) for n_tok rows of n (the experts' softcapped SwiGLU: 4 and
  25) - sigmoid of the raw gate, not the capped one.

ink2_conv
  the static short conv: y[t] = (sum_j w[c, j] x'[t + j] + x[t]) scale over channels c (rows of ch), x' the taps - 1
  rows of hist (the sequence's last inputs, the oldest first; zeros for a new one) then x - tap taps - 1 the current
  token. scale 1 / sqrt(taps + 1). w [ch][taps] (GGUF ne taps, ch).

ink2_conv_dyn
  a full layer's k / v conv: y[t] = (sum_j (w[j, c] + dyn[t, h(c) taps + j]) x'[t + taps - 1 - j] + x[t]) scale - tap 0
  the current token, w [taps][ch] (ne ch, taps), dyn [n_tok][heads taps] (a head's taps a token, h(c) = c / (ch /
  heads)). scale 1 / sqrt(2 taps + 1).

ink2_attn_res
  the attention residual: a token's states (n_blocks rows of blocks [n_tok][b_stride][dim], then partial [n_tok][dim])
  each RMS-normalized (no weight, eps); query q (query [queries][dim]) weights them softmax(q . s scale) and out[q][t]
  = their weighted sum ([queries][n_tok][dim]). n_blocks + 1 <= INK2_MAX_RES, queries <= INK2_MAX_QUERIES, n_blocks <=
  b_stride (the buffer's blocks a token: the block states so far, its room for more).

ink2_scale, ink2_axpy
  x *= s; y += a x - an NVFP4 expert's scale on its gate and up rows, a routed expert's weighted sum.

ink2_clip_rows
  attention's rows before it, in place: n_tok tokens of per_tok rows of dim (row j of a token its head j's) - each
  times w[j][dim] (NULL: none), then L2-clipped to clip (x min(1, clip / (|x| + 1e-6)); 0: not), then times tau(pos)
  (a full layer's q and relative states). q (w q_clip, clip qk_l2_clip, tau), k (no w, clip, no tau), the relative
  states (w rel_clip, clip rel_l2_clip, tau); v is ops_rms with no weight.

ink2_attn
  the attention of n_tok tokens - q [n_tok][heads][head_dim], rel [n_tok][heads][d_rel] (clipped, tau's), sink
  [n_tok][heads] - over n_keys keys (k, v [n_keys][kv_heads][head_dim]: the step's own the last n_tok; GQA, heads /
  kv_heads a key head): token t is key n_keys - n_tok + t, and sees the keys at distance 0 .. window - 1 before it
  (window 0: all). logits q . k scaling + (rel . rel_proj[:, d] at distance d < extent, else 0), a sink logit of
  its own (value 0), softmax; out [n_tok][heads head_dim] the values' sum, times sigmoid(gate) (gate NULL: ungated).
  rel_proj [d_rel][extent].

ink2_fa
  ink2_attn for a prompt (the same arguments and result): a block a (16-token tile, KV head), a wave a query head of
  its group (heads / kv_heads <= INK2_FA_MAX_GROUP), each tile of 16 keys loaded into LDS once for the group's waves -
  q . k and p . v on WMMA (f16 in, f32 sums), the relative bias, mask, sink and online softmax in f32.
*/
