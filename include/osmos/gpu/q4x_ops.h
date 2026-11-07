#ifndef OSMOS_GPU_Q4X_OPS_H
#define OSMOS_GPU_Q4X_OPS_H

#include <stdbool.h>
#include <stdint.h>

/* an op's output also as mm's f16 activations (mm.h's x16: tiled, rows of `cols`), the very values mm_x16 would make
   of it - so the matmul after it skips the conversion; NULL: none. an f16 overflow sets *overflow, as mm_x16's */
typedef struct Q4xX16
{
  void *x16;
  int32_t *overflow;
  uint32_t cols;
  uint32_t row0; /* the op's first row in x16: a step's sequence's (q4x_step), its tokens' first */
} Q4xX16;

/* a context's most keys: four times the model's native 262,144 (YaRN's factor 4, the model card's for 1M; q4x_rope)
   - no kernel limit (QSA's picks a token's 2,051 keys at most), a context's: its Q8_0 cache ~13 KB a token a sequence
   (1M: ~13.6 GB), its pooled keys ~1.5 KB */
#define OPS_Q4X_MAX_KEYS 1048576

/* Qwen3.8-Flash-Next's own steps (DESIGN.md, "Qwen's forward, op by op"), F32 on the GPU, queued on `stream` */

bool q4x_hc_norm(const float *h, const float *w, uint32_t n_tok, uint32_t hc, uint32_t dim, float eps, float *xn,
                 const Q4xX16 *side, void *stream);
bool q4x_scale_silu(float *x, uint64_t n, float scale, const Q4xX16 *side, void *stream);
bool q4x_hc_gated_mean(const float *xn, const float *gate, uint32_t n_tok, uint32_t hc, uint32_t dim, float *out,
                       const Q4xX16 *side, void *stream);
bool q4x_hc_combine_norm(float *h, const float *out, const float *inject, uint32_t n_tok, uint32_t hc, uint32_t dim,
                         float inject_scale, const float *w, float eps, float *xn, const Q4xX16 *side, void *stream);
bool q4x_hc_combine(float *h, const float *out, const float *inject, uint32_t n_tok, uint32_t hc, uint32_t dim,
                    float inject_scale, void *stream);

bool q4x_conv(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t dilation, const float *w,
              float *ring, uint32_t slots, uint32_t cur, float *y, void *stream);
bool q4x_ssm_gates(const float *alpha, const float *beta, const float *dt, const float *a, uint32_t n_tok,
                   uint32_t heads, float *g, float *b, void *stream);
bool q4x_delta(const float *qkv, const float *g, const float *b, uint32_t n_tok, uint32_t k_heads, uint32_t v_heads,
               uint32_t dim, float eps, float *qk, float *ring, uint32_t slots, uint32_t cur, float *out, void *stream);
bool q4x_gated_norm(const float *x, const float *w, const float *z, uint32_t n_tok, uint32_t heads, uint32_t dim,
                    float eps, float *y, const Q4xX16 *side, void *stream);

bool q4x_top(const float *logits, uint32_t n, uint32_t rows, uint32_t stride, int32_t *id, float *p, void *stream);
bool q4x_join(const float *e, const float *h, uint32_t n_tok, uint32_t hc, uint32_t dim, float *out, void *stream);

/* a rotation (q4x_rope): its dims (a head's first), its angles' step, and YaRN's - none: ext_factor 0, freq_scale and
   mscale 1 (the plain rotation's bits) */
typedef struct Q4xRope
{
  uint32_t dim;
  float theta_scale;      /* base^(-2 / dim): theta_i a running product of it */
  float freq_scale;       /* YaRN's 1 / factor */
  float ext_factor;       /* YaRN's extrapolation mix */
  float corr_lo, corr_hi; /* its ramp: the correction dims */
  float mscale;           /* a rotated pair's magnitude: 1 + 0.1 ln(factor) */
} Q4xRope;
bool q4x_rope(uint32_t dim, float base, uint32_t native_ctx, uint32_t ctx, Q4xRope *rope);
bool q4x_qk(const float *qfull, const float *k, uint32_t n_tok, uint32_t heads, uint32_t kv_heads, uint32_t dim,
            const Q4xRope *rope, const float *q_norm, const float *k_norm, float eps, int32_t pos0, float *q,
            float *gate, float *k_out, void *stream);
#define Q4X_KV_BLOCK 32       /* the cache's Q8_0: 32 values a block */
#define Q4X_KV_BLOCK_BYTES 34 /* its f16 scale, then 32 int8 */
uint64_t q4x_kv_row_bytes(uint32_t values);
bool q4x_kv_put(const float *k, const float *v, uint32_t n_tok, uint32_t values, void *kc, void *vc, void *stream);
/* QSA's picks (q4x_ix_pick): a token's blocks of `ratio` keys, `blocks` at most - sel[t][i] ascending, npick[t] of
   them; npick[t] the token's visible blocks every one (none read from sel) */
typedef struct Q4xPick
{
  const int32_t *sel, *npick;
  uint32_t ratio, blocks;
} Q4xPick;
uint64_t q4x_attn_scratch(uint32_t heads, uint32_t max_ctx);
bool q4x_attn(const float *q, const void *kc, const void *vc, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
              uint32_t dim, uint32_t pos0, const float *gate, const Q4xPick *pick, float *scratch, float *out,
              const Q4xX16 *side, void *stream);

#define Q4X_IX_DIM 128 /* an indexer head's dim (qwen4exp.attention.indexer.key_length) */
bool q4x_ix_q(const float *x, uint32_t n_tok, uint32_t heads, const Q4xRope *rope, const float *norm, float eps,
              int32_t pos0, float *q, void *stream);
bool q4x_ix_pool(const float *raw, uint32_t n_tok, int32_t pos0, uint32_t ratio, float *ring, uint32_t ring_rows,
                 const Q4xRope *rope, const float *norm, float eps, float *pooled, void *stream);
uint64_t q4x_ix_scratch(uint32_t max_ctx, uint32_t ratio);
bool q4x_ix_pick(const float *q, const float *pooled, uint32_t n_tok, uint32_t heads, int32_t pos0, uint32_t ratio,
                 uint32_t blocks, float *scratch, uint64_t scratch_floats, int32_t *sel, int32_t *npick, void *stream);

bool q4x_route(const float *logits, uint32_t n_tok, uint32_t n_exp, uint32_t k, int32_t *ids, float *w,
               int32_t *rows, void *stream);
#define Q4X_GROUP_TENSORS 3 /* an MoE's expert tensors: gate, up, down */
typedef struct Q4xExpertTiles
{
  const void *base[Q4X_GROUP_TENSORS]; /* each tensor's bytes: its experts one after another */
  uint64_t bytes[Q4X_GROUP_TENSORS];   /* an expert's */
  const void **ptr[Q4X_GROUP_TENSORS]; /* out: each tensor's entries' matrices */
  int32_t *cnt;                        /* out: an entry's tokens (0: past the live ones) */
  int32_t *gidx, *hidx;                /* out: a slot's token (the gate's and up's rows), its place (down's); -1 pad */
  uint32_t *xg, *yg, *xd, *yd;         /* out: an entry's x16 and y offsets, gate / up then down */
  int32_t *where;                      /* out: a pick's place */
  int32_t *off;                        /* scratch: an expert's first place, first slot and count (3 n_exp) */
} Q4xExpertTiles;

#define Q4X_TILE_RUNS 4 /* an expert part's tiles: 1, 2, 4 or 8 (16 to 128 tokens) - a launch each */
typedef struct Q4xExpertBounds
{
  uint32_t first[Q4X_TILE_RUNS], count[Q4X_TILE_RUNS]; /* each run's entries: where they start, their bound */
  uint32_t entries, slots;                             /* the tables' sizes */
} Q4xExpertBounds;

void q4x_expert_bounds(uint32_t n, uint32_t n_exp, Q4xExpertBounds *b);
bool q4x_expert_tiles(const int32_t *ids, uint32_t n, uint32_t k, uint32_t n_exp, uint32_t dim, uint32_t ff,
                      const Q4xExpertTiles *t, void *stream);
bool q4x_gather(const float *src, const int32_t *rows, uint32_t n, uint32_t len, float *dst, void *stream);
bool q4x_swiglu(const float *g, const float *u, uint64_t n, float *y, const Q4xX16 *side, void *stream);
bool q4x_swiglu_rows(const float *g, const float *u, uint32_t ff, const int32_t *idx, uint32_t slots, void *x16,
                     int32_t *overflow, void *stream);
bool q4x_moe_sum(const float *y, const int32_t *where, const float *w, uint32_t n_tok, uint32_t k, uint32_t dim,
                 const float *shared, const float *shared_logit, float *out, void *stream);

bool q4x_rows(uint32_t type, const void *table, uint64_t row_bytes, uint32_t row_values, const int32_t *rows,
              uint32_t n, float *out, void *stream);
bool q4x_ple_gate(const float *key, const float *query, const float *value, uint32_t n_tok, uint32_t hc, uint32_t dim,
                  float *gate, float *gated, void *stream);
bool q4x_add2(float *h, const float *a, const float *b, uint64_t n, void *stream);
bool q4x_streams(const float *x, uint32_t n_tok, uint32_t hc, uint32_t dim, float *h, void *stream);

#endif

/*
the residual h: n_tok tokens of hc streams of dim, h[(t hc + s) dim + d]. false, nothing queued, for an empty shape,
hc past 8 or a launch refused.

(q4x_hc_norm, q4x_scale_silu, q4x_hc_gated_mean, q4x_gated_norm, q4x_swiglu, q4x_attn: `side` their output's f16
copy too, Q4xX16's - or NULL)

q4x_hc_norm
  xn[t][s][d] = h[t][s][d] / sqrt(mean over d of h[t][s][]^2 + eps) * w[s dim + d]: an RMS norm a stream, its
  weight a stream's own.

q4x_scale_silu
  x[i] = silu(x[i] * scale), in place: the mixer's rank after its down matrix (scale 1 / hc).

q4x_hc_gated_mean
  out[t][d] = (sum over s, in order, of xn[t][s][d] sigmoid(gate[t][s][d])) / hc: the streams' gated mean, the
  block's input.

q4x_hc_combine
  h[t][s][d] += 2 sigmoid(inject[t hc + s] inject_scale) out[t][d]: the block's output into every stream, each by its
  own weight (2 sigmoid: 1 at an inject of 0, a plain residual add).

q4x_hc_combine_norm
  q4x_hc_combine, then q4x_hc_norm of the h it leaves, one launch: the same bits as the two (a value its combine's,
  the norm's sums in its order) - a block's output into the residual and the next mixer's norm of it.

the Gated DeltaNet (a layer's, its tokens in order). its states kept in a ring of `slots` (a state each): read from
slot cur; the state after token t written to slot (cur + 1 + t) % slots for the step's last min(n_tok, slots)
tokens (its last one, in place, when slots is 1) - a step of n tokens leaves the sequence's state in slot
(cur + n) % slots and the states before it behind, back to slots - 1 tokens: a rollback a slot index.

q4x_conv
  y[t][c] = silu(sum over k < taps, in order, of w[c taps + k] x'[t + k dilation][c]), x' the last (taps - 1)
  dilation inputs before (hist[i][c], oldest first) and then x: a depthwise causal conv, its current token the last
  tap (the DeltaNet's dilation 1; the n-gram layer's its n-gram size). hist (a ring's slot: (taps - 1) dilation x ch
  floats) read from slot cur; after token t, x''s last (taps - 1) dilation up to it - the conv state - into its slot.

q4x_ssm_gates
  b[t][h] = sigmoid(beta[t][h]); g[t][h] = softplus(alpha[t][h] + dt[h]) a[h] (a = -exp(A_log): g <= 0, a decay).

q4x_delta
  qkv a token's conv output: q (k_heads x dim), k (k_heads x dim), v (v_heads x dim). q and k each l2-normed a head
  (x / sqrt(sum x^2 + eps)); value head h reads key group h % k_heads; its state S (dim x dim, state[h][j][i] = S[i][j])
  a token at a time: S *= exp(g); delta_j = (v_j - sum_i S[i][j] k_i) b; S[i][j] += k_i delta_j; out[t][h][j] =
  sum_i S[i][j] q_i / sqrt(dim). S (a ring's slot: v_heads x dim x dim floats) read from slot cur; after token t,
  into its slot. qk: scratch, n_tok x 2 k_heads x dim floats (the normed q and k, made first, all tokens at once).

q4x_gated_norm
  y[t][h][d] = x[t][h][d] / sqrt(mean over d of x^2 + eps) w[d] sigmoid(z[t][h][d]): the DeltaNet's output norm a
  head, gated by z.

the MTP head's:

q4x_top
  rows rows of n logits (stride floats apart): id[r] row r's greatest (the lowest among equals, as ops_argmax), p[r]
  its softmax probability - 1 / sum over i of exp(logits[i] - max), the sum in a fixed order (a draft's confidence;
  p NULL: not wanted - a greedy pick's).

q4x_join
  out[t][s] = [e[t] ; h[t][s]] (2 dim floats a stream): a token's normed embedding beside each stream's normed state,
  the head's eh_proj input (llama.cpp's concat of e, repeated a stream, and h).

the sparse attention (a layer's; one sequence's tokens at positions pos0, pos0 + 1, ...):

q4x_rope
  the rotation of a context of ctx positions: plain at or below the model's native_ctx; past it the model card's
  static YaRN at factor ctx / native_ctx (llama.cpp's: ext_factor 1, beta_fast 32 and beta_slow 1 its correction
  dims, ggml_rope_yarn_corr_dims' floor and ceil, mscale 1 + 0.1 ln(factor)) - a pair's angle theta's interpolation
  (freq_scale theta) and theta mixed by the ramp 1 - clamp((i - lo) / max(0.001, hi - lo), 0, 1), its cos and sin
  times mscale. false for a dim not even, no native context, or ctx past 64 times it.

q4x_qk
  qfull a token's attn_q: each head's q (dim) then its gate (dim), head after head. q[t][h] = rope(rms_norm(q) q_norm),
  gate[t][h] its gate; k_out[t][g] = rope(rms_norm(k[t][g]) k_norm) (k_out may be a cache's row pos0). rope: the first
  rope->dim of a head, pairs (i, i + dim / 2), angle (pos0 + t) theta_i, theta_i = theta_scale^i as ggml builds it (a
  running product; YaRN's as q4x_rope says); IMRoPE's sections all the token's position (text) - this rotation. heads 0: the
  keys alone (qfull, q and gate unread: the MTP head's cache seeding).

q4x_kv_row_bytes, q4x_kv_put
  the attention's cache in Q8_0 (ggml's: a block of 32, d = max |x| / 127 as f16, q = x / d rounded to nearest):
  a row of `values` (a token's kv_heads x dim) q4x_kv_row_bytes of it; q4x_kv_put quantizes n_tok rows of k and of v
  (F32, values apart) into kc and vc - a row its own, so a token's the same whatever its batch. a quarter of F32's
  bytes, the cache a long context reads whole at each decode step.

q4x_attn
  out[t][h][d] = sigmoid(gate[t][h][d]) sum over p in K(t) of softmax_p(q[t][h] . kc[p][g] / sqrt(dim)) vc[p][g][d],
  K(t) the token's keys: with no pick every position up to its own, pos0 + t; with one its key list - its picked
  blocks' positions (ratio of them a block, in block order), then its tail (the positions past its visible blocks,
  ratio nv .. pos0 + t, nv = (pos0 + t + 1) / ratio) - every block picked, the list 0 .. pos0 + t: the dense bits.
  g = h / (heads / kv_heads) (llama.cpp's GQA: a KV head's heads consecutive). kc, vc: the cache, Q8_0 rows of
  kv_heads x dim (q4x_kv_put's), rows 0 .. pos0 + n_tok - 1 written; at most OPS_Q4X_MAX_KEYS keys. on the matrix cores: q, k, v and the softmax's
  weights as f16, the sums F32; a token's key list in chunks of 128, a row's chunks merged in order - a row's bits
  its own whatever its batch (its q, its keys, its position alone). scratch: q4x_attn_scratch(heads, max_ctx) floats
  (a step of more tokens taken in pieces).

QSA's indexer (llama.cpp's build_qsa_sel; DESIGN.md, "QSA"): a layer's keys pooled in blocks of `ratio` positions,
a token's keys its top `blocks` of them by score and its tail.

q4x_ix_q
  x[t][h] (heads of Q4X_IX_DIM: index_q_proj's) to q[t][h] = rope(rms_norm(x[t][h]) norm) at position pos0 + t - the
  rotation q4x_qk's (its first rope->dim, the same angles: YaRN's too).

q4x_ix_pool
  raw[t] (Q4X_IX_DIM: index_k_proj's, neither normed nor rotated) the raw keys of positions pos0 + t: each block j
  the step completes (ratio j + ratio - 1 in pos0 .. pos0 + n_tok - 1) pooled, pooled[j] = rope(rms_norm(m) norm) at
  position ratio j, m its members' raw keys summed in order times 1 / ratio - a member before pos0 from the ring
  (ring[p % ring_rows]: the sequence's last ring_rows positions' raw keys); then the step's last ring_rows positions'
  raw keys into the ring. ring_rows at least ratio - 1 plus a rollback's reach: a partial block's members held.

q4x_ix_scratch, q4x_ix_pick
  token t at position p = pos0 + t sees nv = (p + 1) / ratio blocks (those complete at or before it): with nv <=
  blocks, every one (npick[t] = nv, sel unwritten); else its scores, block j's the sum over heads in order of
  relu(q[t][h] . pooled[j]) / sqrt(Q4X_IX_DIM) (ggml_lightning_indexer), its top `blocks` by score - the lower block
  among equals - in sel[t] (row t `blocks` apart) ascending, npick[t] = blocks. scratch: scratch_floats of it, at least
  q4x_ix_scratch(max_ctx, ratio) (a token's scores; a step's tokens taken in pieces). a token's pick its own: its q,
  the pooled keys, its position.

the MoE (a token's k experts of n_exp: entry e = t k + j its j-th):

q4x_route
  softmax over a token's n_exp logits; its k most probable (the greater first, the lower id among equals) into ids
  [t][j]; their weights w[t][j] = p / (sum of the k p's, at least 6.103515625e-5); rows[e] = t (the token an entry
  reads).

q4x_expert_bounds, q4x_expert_tiles
  a step's n picks (ids: n_tok x k, token t's slots t k ..) as mm_group's tables, so each expert picked is read once
  for up to 128 of its tokens: the picks sorted by expert, then pick (stable) - where[pick] its place; each expert's
  slots its places packed (from its first, rounded up to whole tiles of MM_TILE: gidx the token, hidx the place, -1
  past them); its places in parts of at most 128, an entry each, in the run of its tiles (1 to 16 tokens: run 0, to
  32: 1, to 64: 2, to 128: 3), each run's entries in expert order from bounds.first[r]: cnt[e] its tokens, xg / xd its
  first slot times dim / ff (halves: its tiles of the x16), yg / yd its first place times ff / dim (y's rows in place
  order), each tensor's matrix base[i] + expert bytes[i]. a run's entries past its live ones cnt 0, to its bound
  (q4x_expert_bounds: min(n, n_exp), min(n / 17, n_exp), min(n / 33, n_exp), n / 65), the slots past the live ones
  -1, to n + 15 min(n, n_exp) - the launches' and the tables'. a launch a run (mm_group, max_cnt 16 << r): a part's
  slab decoded once for its tiles. which picks share a part changes no bit: mm's tokens are their own.

q4x_gather
  dst[e][] = src[rows[e]][] for e < n, len floats a row: an entry's own copy of its token's input.

q4x_swiglu
  y[i] = silu(g[i]) u[i].

q4x_swiglu_rows
  the experts' activation straight into the down's f16 (x16, slots rows of ff, mm's tiles): slot s's column c
  silu(g[r][c]) u[r][c] for its row r = idx[s] (-1: zero), rounded as mm_x16_rows rounds - q4x_swiglu then
  mm_x16_rows's bytes, without the F32 between them (overflow set past f16's range). ff a multiple of 16.

q4x_moe_sum
  out[t][d] = (sum over j, in order, of w[t][j] y[where[t k + j]][d]) + sigmoid(shared_logit[t]) shared[t][d]: the experts'
  weighted sum and the shared expert, gated.

the n-gram table (PLE):

q4x_rows
  out[i][] = row rows[i] of the table, decoded (row_values of the type's values, row_bytes apart): a token's heads'
  rows, one after another (its embedding: heads x row_values) - or a token's row of token_embd. IQ4_NL, Q8_0, Q6_K.

q4x_ple_gate
  key, query: [t][s][dim], each normed a stream already. s_ts = (sum over d, in order, of key[t][s][d] query[t][s][d])
  / sqrt(dim); gate[t][s] = sigmoid(sgn(s_ts) sqrt(max(|s_ts|, 1e-6))); gated[t][s][d] = value[t][d] gate[t][s].

q4x_add2
  h[i] += a[i] + b[i] (a + b first, as llama.cpp adds them).

q4x_streams
  h[t][s][d] = x[t][d] for every stream s: the residual a token starts with, its embedding in each.
*/
