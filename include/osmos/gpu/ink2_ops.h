#ifndef OSMOS_GPU_INK2_OPS_H
#define OSMOS_GPU_INK2_OPS_H

#include <stdbool.h>
#include <stdint.h>

/* Inkling-2-Small's own steps (DESIGN.md, "the forward, op by op"), F32 on the GPU, queued on `stream`: each held to
   transformers' own module by tools/convert/ink2_cases.py's files (test_ink2_ops_gpu) */

#define INK2_MAX_TAPS 8   /* a short conv's taps at most (Inkling's 4) */
#define INK2_MAX_RES 16   /* attn_res's states a token at most (its blocks, then the partial one: Inkling's <= 9) */
#define INK2_MAX_QUERIES 3 /* attn_res's queries at once (a full layer's q, k, v) */
#define INK2_RES_SPLIT_TOKENS 64 /* attn_res's steps of at most these split (its scratch's) */
#define INK2_RES_SCRATCH (INK2_RES_SPLIT_TOKENS * (1 + INK2_MAX_QUERIES) * INK2_MAX_RES) /* its floats */

bool ink2_softcap(float *x, uint64_t n, float cap, void *stream);
bool ink2_swiglu(const float *g, const float *u, uint32_t n, float gcap, float ucap, uint32_t n_tok, float *h,
                 void *stream);
bool ink2_swiglu_rows(const float *g, const float *u, const float *sg, const float *su, uint32_t n, float gcap,
                      float ucap, uint32_t rows, float *h, void *stream);
bool ink2_moe_sum(float *y, const float *x, const float *a, const int32_t *row, uint32_t k, uint32_t n_tok,
                  uint32_t dim, void *stream);
bool ink2_gram(const float *x, uint32_t n_tok, uint32_t dim, float *h, void *stream);
bool ink2_conv(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float *w, const float *hist,
               float res, float scale, float *y, void *stream);
bool ink2_conv_dyn(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads, const float *w,
                   const float *dyn, const float *hist, float res, float scale, float *y, void *stream);
bool ink2_conv_keep(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float *w, const float *hist,
                    float res, float scale, const float *add, float *y, float *keep, void *stream);
bool ink2_conv_dyn_keep(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads, const float *w,
                        const float *dyn, const float *hist, float res, float scale, float *y, float *keep,
                        void *stream);
bool ink2_attn_res(const float *query, uint32_t queries, const float *blocks, uint32_t n_blocks, uint32_t b_stride,
                   const float *partial, uint32_t n_tok, uint32_t dim, float eps, float scale, float *out, float *scratch,
                   void *stream);
bool ink2_scale(float *x, uint64_t n, float s, void *stream);
bool ink2_axpy(float *y, const float *x, uint64_t n, float a, void *stream);
#define INK2_MAX_HEAD 128   /* attention's head dim at most (Inkling's 128) */
#define INK2_MAX_EXTENT 1024 /* the relative bias's extent at most (Inkling's 1,024) */
#define INK2_MAX_REL 32      /* a head's relative states at most (Inkling's 16) */
#define INK2_FA_MAX_GROUP 8  /* ink2_fa's query heads a KV head at most (Inkling's 6): its waves */
#define INK2_FA_TILE 16      /* ink2_fa's keys a tile (WMMA's 16): a window's keys handed in from a multiple of it */
#define INK2_FA_UNIT 256     /* its keys a unit (absolute): each its own softmax, folded in order - a decode's in parallel */
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
             const float *k, const float *v, uint32_t n_keys, uint32_t pos0, const float *rel_proj, float *out,
             float *scratch, void *stream);
bool ink2_fa16(const Ink2Attn *a, const float *q, const float *rel, const float *sink, const float *gate,
               uint32_t n_tok, const uint16_t *k, const uint16_t *v, uint32_t n_keys, uint32_t pos0,
               const float *rel_proj, float *out, float *scratch, void *stream);
uint64_t ink2_fa_scratch(uint32_t heads, uint32_t max_keys);
bool ink2_to_f16(const float *x, uint64_t n, uint16_t *y, void *stream);
#define INK2_KV8_BLOCK 32 /* a Q8 record's values a scale (Q8_0's) */
#define INK2_KV8_RECORD (INK2_MAX_HEAD + INK2_MAX_HEAD / INK2_KV8_BLOCK * 2) /* a head's row: its int8s, f16 scales */
bool ink2_fa8(const Ink2Attn *a, const float *q, const float *rel, const float *sink, const float *gate,
              uint32_t n_tok, const uint8_t *k, const uint8_t *v, uint32_t n_keys, uint32_t pos0, const float *rel_proj,
              float *out, float *scratch, void *stream);
bool ink2_to_kv8(const float *x, uint64_t rows, uint8_t *y, void *stream);
#define INK2_KV5_RECORD (INK2_MAX_HEAD / 2 + INK2_MAX_HEAD / 8 + INK2_MAX_HEAD / INK2_KV8_BLOCK * 2) /* nibbles, fifth
                                                                                                    bits, scales */
bool ink2_fa85(const Ink2Attn *a, const float *q, const float *rel, const float *sink, const float *gate,
               uint32_t n_tok, const uint8_t *k, const uint8_t *v, uint32_t n_keys, uint32_t pos0,
               const float *rel_proj, float *out, float *scratch, void *stream);
bool ink2_to_kv5(const float *x, uint64_t rows, uint8_t *y, void *stream);
typedef struct Ink2PrepKv
{
  const float *x, *w, *dyn, *hist; /* the projection's rows, the conv's weights (dyn NULL: the static conv's), the
                                      conv tail's last taps - 1 rows (NULL: zeros) */
  float *y, *keep;                 /* the conv's rows clipped (k) or normed (v); x's rows copied (NULL: none) */
  void *store;                     /* the cache's next n_tok rows */
} Ink2PrepKv;
typedef struct Ink2Prep
{
  uint32_t n_tok, heads, kv_heads, head_dim, d_rel, taps;
  float *q, *rel;           /* clipped in place */
  const float *q_w, *rel_w; /* their clips' weights */
  float q_clip, rel_clip, k_clip, eps, conv_res, conv_scale;
  Ink2Tau tau;  /* q's and rel's */
  bool records; /* the stores keys Q8, values Q5 records (a full layer's); else f16 (a window's) */
  Ink2PrepKv k, v;
} Ink2Prep;
bool ink2_attn_prep(const Ink2Prep *p, void *stream);

/* the drafter's (kernels/ink2_mtp.hip; src/ink2/hip_mtp.c) */
bool ink2_mtp_mix(const float *taps, uint32_t n_taps, const float *alpha, uint32_t n, uint32_t dim, float *out,
                  void *stream);
bool ink2_mtp_kv_put(uint16_t *ring, uint32_t ring_rows, const float *x, uint32_t n, uint32_t row, int32_t pos0,
                     void *stream);
uint64_t ink2_mtp_attn_scratch(uint32_t n_q, uint32_t heads, uint32_t max_keys);
bool ink2_mtp_attn(const float *q, uint32_t n_q, int32_t pos0, const float *kb, const float *vb, const uint16_t *kr,
                   const uint16_t *vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                   uint32_t window, float *scratch, float *out, void *stream);
bool ink2_mtp_markov_in(const float *state, const uint16_t *w1, const int32_t *prev, uint32_t rank, float *vec,
                        void *stream);
bool ink2_mtp_markov_gru(const float *p, const float *inv, const float *bias, uint32_t rank, float *state,
                         float *latent, void *stream);
bool ink2_mtp_markov_logits(const float *base, const float *corr, const float *fold_b, float cap, uint32_t n, float *out,
                            void *stream);
bool ink2_transpose_bf16(const float *in, uint32_t rows, uint32_t cols, uint16_t *out, void *stream);

#endif

/*
ink2_softcap
  x[i] = cap tanh(x[i] / cap), in place: the logits' (final_logit_softcapping, 30).

ink2_swiglu
  h = gcap tanh(g / gcap) sigmoid(g) . ucap tanh(u / ucap) for n_tok rows of n (the experts' softcapped SwiGLU: 4 and
  25) - sigmoid of the raw gate, not the capped one.

ink2_swiglu_rows
  ink2_swiglu over rows of n with row r's gate times sg[r] and its up times su[r] first (each pick's expert's scales:
  ink2_scale then ink2_swiglu, the same bits).

ink2_moe_sum
  y[t] = the sum of a[t k + s] x[row[t k + s]] over a token's k slots in order, from zero - ink2_axpy after a zeroed
  y, slot after slot: the same bits.

ink2_gram
  h [dim][dim] += x^T x over n_tok rows of x [n_tok][dim]: a calibration's sums for GPTQ (ink2_run's HESS=).

ink2_conv
  the static short conv: y[t] = (sum_j w[c, j] x'[t + j] + res x[t]) scale over channels c (rows of ch), x' the taps - 1
  rows of hist (the sequence's last inputs, the oldest first; zeros for a new one) then x - tap taps - 1 the current
  token. the checkpoint's weights: res 1, scale 1 / sqrt(taps + 1); llama.cpp's (unsloth's GGUFs) fold both into them -
  (w + 1 at the current tap) / sqrt(taps + 1) - res 0, scale 1. w [ch][taps] (GGUF ne taps, ch).

ink2_conv_dyn
  a full layer's k / v conv: y[t] = (sum_j (w[j, c] + dyn[t, h(c) taps + j]) x'[t + taps - 1 - j] + res x[t]) scale - tap 0
  the current token, w [taps][ch] (ne ch, taps), dyn [n_tok][heads taps] (a head's taps a token, h(c) = c / (ch /
  heads)). the checkpoint's: res 1, scale 1 / sqrt(2 taps + 1); llama.cpp's fold them into the static weights ((w + 1
  at tap 0) / 3, stored [ch][taps] with tap 0 last) and the per-token ones (dyn / 3): res 0, scale 1.

ink2_conv_keep, ink2_conv_dyn_keep
  ink2_conv's and ink2_conv_dyn's y, and x's n_tok rows copied to keep as they are read (a tail's next rows: not
  hist's; NULL none) - the copy's launch saved. ink2_conv_keep's add (NULL: none): y = add + the conv, ink2_axpy's
  sum of the two (a residual's; add may be y) - the axpy's launch saved.

ink2_attn_res
  the attention residual: a token's states (n_blocks rows of blocks [n_tok][b_stride][dim], then partial [n_tok][dim])
  each RMS-normalized (no weight, eps); query q (query [queries][dim]) weights them softmax(q . s scale) and out[q][t]
  = their weighted sum ([queries][n_tok][dim]). n_blocks + 1 <= INK2_MAX_RES, queries <= INK2_MAX_QUERIES, n_blocks <=
  b_stride (the buffer's blocks a token: the block states so far, its room for more). scratch (INK2_RES_SCRATCH floats, or NULL): a step of
  at most INK2_RES_SPLIT_TOKENS a block a sum, not a block a token - the same bits.

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
  q . k and p . v on WMMA (f16 in, f32 sums), the relative bias, mask, sink and online softmax in f32. a window's
  first tile at a multiple of INK2_FA_TILE: a query's keys always the same tiles (the rounding of p to f16 against
  the running max is a tile's), so its bits are the same in any block - and in any step, given keys whose index mod
  INK2_FA_TILE is their position's (a sequence's window handed in from such a position).

units: a query's keys cut at absolute multiples of INK2_FA_UNIT (pos0: key 0's position, a multiple of
  INK2_FA_TILE), each unit's online softmax from nothing, folded into the sink's start in order. given scratch
  (ink2_fa_scratch(heads, the most keys) bytes) and at most INK2_FA_TILE tokens - a decode's, a draft block's - a block
  a unit and KV head, then a fold kernel: the units in parallel, the prompt's bits (the same units, the same fold).

ink2_fa16, ink2_to_f16
  ink2_fa over keys and values stored as f16 (a cache's: ink2_to_f16 of the F32 rows, IEEE half bits in uint16_t) -
  the values ink2_fa rounds them to as it loads them, so the same bits; half the memory and the reads.

ink2_fa8, ink2_to_kv8
  ink2_fa over keys and values stored as Q8 records (a full layer's cache): ink2_to_kv8 of F32 rows of INK2_MAX_HEAD
  (a key's KV heads), each its INK2_MAX_HEAD int8s then a f16 scale an INK2_KV8_BLOCK of them (Q8_0's: d = amax /
  127, q = round(x / d)). a value loaded is q d rounded to f16 - ink2_fa16's bits on those values; 0.53 of f16's
  memory and reads. not f16's values: a cache of them its own bits.

ink2_fa85, ink2_to_kv5
  ink2_fa8's keys, its values Q5 records (ink2_to_kv5): a row's low nibbles (value i at byte i / 2, odd i the high
  one), its fifth bits (byte INK2_MAX_HEAD / 2 + i / 8, bit i % 8), a f16 scale an INK2_KV8_BLOCK - d = amax / 15, q
  = round(x / d) stored q + 16. a value loaded (q - 16) d rounded to f16 - ink2_fa16's bits on those values; values
  0.65 of Q8's memory.

ink2_attn_prep
  a layer's attention inputs from its projections, one launch: q's and rel's ink2_clip_rows (q_w, q_clip / rel_w,
  rel_clip, tau); k's conv (ink2_conv_keep, or ink2_conv_dyn_keep given dyn: conv_res, conv_scale), its
  ink2_clip_rows (k_clip, no weight, no tau) and its store; v's conv, ops_rms (no weight, eps) and its store - the
  stores ink2_to_f16, or ink2_to_kv8 (k) and ink2_to_kv5 (v) when records. each value those ops' bits. head_dim
  INK2_MAX_HEAD; q [n_tok][heads][head_dim], rel [n_tok][heads][d_rel], k and v [n_tok][kv_heads head_dim].

ink2_mtp_mix
  out[t] = sum over k of alpha[k] taps[t][k] (taps [n][n_taps][dim]): a drafter layer's mixture of the target's taps
  (softmax(fuse_weight)'s row), before its dflare norm.

ink2_mtp_kv_put
  n rows of x (row values each, F32) into an f16 ring of ring_rows: x's row t at (pos0 + t) % ring_rows.

ink2_mtp_attn
  the drafter's attention for a block of n_q queries at positions pos0 .. (q [n_q][heads][128], RoPE'd), against the
  context's keys and values in the ring for positions ctx_from .. pos0 - 1 and the block's own (kb, vb [n_q][kv_heads]
  [128] F32): a context key visible to query i when within `window` of it back (0: every one), every block key
  visible (not causal). scores q.k / sqrt(128), each chunk of 256 keys its softmax's partial (scratch:
  ink2_mtp_attn_scratch), merged in order. out [n_q][heads][128].

ink2_mtp_markov_in, _gru, _logits
  the Markov head's step (TokenSpeed's sample_block): vec = [state | markov_w1[*prev]] (w1 BF16 [V][rank]); with p =
  rnn_proj's [state | prev] columns . vec and inv its [hidden | features] columns' product (the step's own row): g =
  sigmoid(p + inv + b), state = g state + (1 - g) tanh(.), latent = tanh(.) (gate, candidate, output thirds); out =
  cap tanh((base + corr + fold_b) / cap) - the target head's logits, the fold's correction (lm_head . hidden_out:
  corr = fold . latent) and bias, softcapped.

ink2_transpose_bf16
  out [cols][rows] = in [rows][cols] rounded to BF16 (nearest, ties to even): the fold's layout for the matvec.
*/
