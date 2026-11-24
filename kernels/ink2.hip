// Inkling-2-Small's own steps (include/osmos/gpu/ink2_ops.h), F32
#include <type_traits>
#include <cstdlib>

#include "block.h"
#include "launch.h"
#include "mm.h"
#include "moe_tiles.h"
#include "router.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

namespace {

using launches::kThreads;
using launches::grid;
using launches::st;
using launches::ok;

// the experts' tables' scales on the GPU (a whole membrane's MoE): pick m of expert e at place q = where[m] - the place's
// gate and up scales, the pick's weight times the down's scale (moe's host products: the same bits), its picks counted
// pick m's scales: its expert's gate and up scales at its place, its weight times the down's, the layer's count
__device__ inline void moe_scale_pick(uint32_t m, const int32_t* ids, const float* w, const int32_t* where,
                                      const float* scale, uint32_t n_exp, float* sg, float* su, float* a,
                                      uint32_t* picks) {
  const int32_t e = ids[m], q = where[m];
  sg[q] = scale[e], su[q] = scale[n_exp + e], a[m] = w[m] * scale[2 * n_exp + e];
  atomicAdd(picks + e, 1u);
}

__global__ void moe_scales_kernel(const int32_t* ids, const float* w, const int32_t* where, const float* scale,
                                  uint32_t n_exp, uint32_t n, float* sg, float* su, float* a, uint32_t* picks) {
  const uint32_t m = blockIdx.x * kThreads + threadIdx.x;
  if (m < n) moe_scale_pick(m, ids, w, where, scale, n_exp, sg, su, a, picks);
}

// a decode's or a verify's MoE preparation in one block, not three launches: each token's picks (router.h's), the
// picks as mm_group's tables (moe_tiles.h's), their scales (moe_scale_pick) - the same functions the three kernels
// run, so the same bits. its picks the block's LDS between them (at most kPlaceInline: the tiles' inline placing)
__global__ __launch_bounds__(moe_tiles::kGroupThreads) void moe_prep_kernel(
    const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float route_scale, uint32_t n_tok,
    int32_t* idx, float* w, uint32_t dim, uint32_t ff, Q4xExpertTiles t, Q4xExpertBounds bd, const float* scale,
    float* sg, float* su, float* a, uint32_t* picks) {
  __shared__ float sc[OPS_MAX_EXPERTS], keys[OPS_MAX_EXPERTS];
  __shared__ int32_t ids[moe_tiles::kPlaceInline];
  for (uint32_t tok = 0; tok < n_tok; tok++) {
    router::token<OPS_SCORE_SIGMOID>(logits, bias, n_exp, k, route_scale, tok, moe_tiles::kGroupThreads, sc, keys,
                                     idx, w, ids);
    __syncthreads();  // (the next token's scores over this one's LDS)
  }
  const uint32_t P = n_tok * k;
  moe_tiles::tiles_block(ids, P, k, n_exp, dim, ff, t, bd);
  __syncthreads();  // its places (t.where) and the router's weights, the block's writes
  for (uint32_t m = threadIdx.x; m < P; m += moe_tiles::kGroupThreads)
    moe_scale_pick(m, ids, w, t.where, scale, n_exp, sg, su, a, picks);
}

// a loop's control: x - tanh(forget) x + tanh(gain) m, upstream's order (inkling2.cpp's control: sub, then add)
__global__ void loop_ctrl_kernel(const float* x, const float* m, const float* gain, const float* forget, uint64_t n,
                                 float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  out[i] = (x[i] - tanhf(forget[i]) * x[i]) + tanhf(gain[i]) * m[i];
}

struct LoopCtrlParts {
  Ink2LoopCtrlPart p[INK2_LOOP_CTRL_PARTS];
};
__global__ void loop_ctrl_many_kernel(LoopCtrlParts ps, uint64_t n) {  // part blockIdx.y: loop_ctrl_kernel's expression
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  const Ink2LoopCtrlPart& p = ps.p[blockIdx.y];
  p.out[i] = (p.x[i] - tanhf(p.forget[i]) * p.x[i]) + tanhf(p.gain[i]) * p.m[i];
}

__global__ void softcap_kernel(float* x, uint64_t n, float cap) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

__device__ inline float swiglu_of(float gv, float uv, float gcap, float ucap) {
  return gcap * tanhf(gv / gcap) * sigmoid(gv) * (ucap * tanhf(uv / ucap));
}

__global__ void swiglu_kernel(const float* g, const float* u, uint64_t n, float gcap, float ucap, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  h[i] = swiglu_of(g[i], u[i], gcap, ucap);
}

// the same with each row's gate and up scaled first (an NVFP4 expert's scale: ink2_scale then ink2_swiglu's bits)
__global__ void swiglu_rows_kernel(const float* g, const float* u, const float* sg, const float* su, uint32_t n,
                                   uint64_t all, float gcap, float ucap, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= all) return;
  const uint64_t r = i / n;
  h[i] = swiglu_of(g[i] * sg[r], u[i] * su[r], gcap, ucap);
}

// the same straight into the down's f16 (mm_x16_rows of its rows): slot t row idx[t]'s (-1: zero) - a thread
// kX16Run adjacent columns of a slot in x16_rows_kernel's tiled order, each value swiglu_rows_kernel's, then x16_put's
// rounding and flag
constexpr uint32_t kX16Run = 8;
__global__ void swiglu_x16_kernel(const float* g, const float* u, const float* sg, const float* su, uint32_t n,
                                  float gcap, float ucap, const int32_t* idx, uint32_t slots, _Float16* x16,
                                  int32_t* overflow) {
  const uint64_t o = ((uint64_t) blockIdx.x * kThreads + threadIdx.x) * kX16Run;
  const uint64_t tile = (uint64_t) mm::kTile * n;
  const uint32_t in = (uint32_t) (o % tile), frag = in % mm::kFrag;
  const uint32_t t = (uint32_t) (o / tile) * mm::kTile + frag / mm::kTile;
  if (t >= slots) return;
  const uint32_t c = in / mm::kFrag * mm::kTile + frag % mm::kTile;
  const int32_t r = idx[t];
  bool over = false;
  _Float16 h[kX16Run];
  for (uint32_t j = 0; j < kX16Run; j++) {
    const uint64_t i = (uint64_t) r * n + c + j;
    const float v = r >= 0 ? swiglu_of(g[i] * sg[r], u[i] * su[r], gcap, ucap) : 0.0f;
    over |= !(fabsf(v) <= mm::kF16Max), h[j] = (_Float16) v;
  }
  if (over) atomicOr(overflow, 1);
  __builtin_memcpy(x16 + o, h, sizeof h);
}

// token t's experts at c summed in its slot order from zero: ink2_axpy's sum, one after another
__device__ inline float moe_at(const float* x, const float* a, const int32_t* row, uint32_t k, uint64_t t, uint32_t c,
                               uint32_t dim) {
  float v = 0.0f;
#pragma unroll 8
  for (uint32_t s = 0; s < k; s++) v += a[t * k + s] * x[(uint64_t) row[t * k + s] * dim + c];
  return v;
}

__global__ void moe_sum_kernel(float* y, const float* x, const float* a, const int32_t* row, uint32_t k, uint64_t all,
                               uint32_t dim) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= all) return;
  y[i] = moe_at(x, a, row, k, i / dim, (uint32_t) (i % dim), dim);
}

// h[i][j] += sum over t of x[t][i] x[t][j] (a calibration's X^T X, F32): a 16 x 16 tile of h a block, the tokens in
// chunks of 16 through LDS
constexpr int kGram = 16;
__global__ void gram_kernel(const float* x, uint32_t n_tok, uint32_t dim, float* h) {
  __shared__ float a[kGram][kGram + 1], b[kGram][kGram + 1];
  const uint32_t i = blockIdx.y * kGram + threadIdx.y, j = blockIdx.x * kGram + threadIdx.x;
  float acc = 0.0f;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += kGram) {
    const uint32_t t = t0 + threadIdx.y, ci = blockIdx.y * kGram + threadIdx.x, cj = blockIdx.x * kGram + threadIdx.x;
    a[threadIdx.y][threadIdx.x] = t < n_tok && ci < dim ? x[(uint64_t) t * dim + ci] : 0.0f;
    b[threadIdx.y][threadIdx.x] = t < n_tok && cj < dim ? x[(uint64_t) t * dim + cj] : 0.0f;
    __syncthreads();
    for (int k = 0; k < kGram; k++) acc = fmaf(a[k][threadIdx.y], b[k][threadIdx.x], acc);
    __syncthreads();
  }
  if (i < dim && j < dim) h[(uint64_t) i * dim + j] += acc;
}

// x' = hist (taps - 1 rows) then x: token t's taps x'[t .. t + taps - 1], the current last
__device__ inline float tap(const float* x, const float* hist, uint32_t at, uint32_t h, uint32_t ch, uint32_t c) {
  return at < h ? (hist ? hist[(uint64_t) at * ch + c] : 0.0f) : x[(uint64_t) (at - h) * ch + c];
}

// the static conv's value at token t, channel c
__device__ inline float conv_at(const float* x, uint32_t t, uint32_t c, uint32_t ch, uint32_t taps, const float* w,
                                const float* hist, float res, float scale) {
  const uint32_t h = taps - 1;
  float sum = 0;
  for (uint32_t j = 0; j < taps; j++) sum += w[(uint64_t) c * taps + j] * tap(x, hist, t + j, h, ch, c);
  return (sum + res * x[(uint64_t) t * ch + c]) * scale;
}

// the dynamic conv's (a full layer's k / v)
__device__ inline float conv_dyn_at(const float* x, uint32_t t, uint32_t c, uint32_t ch, uint32_t taps,
                                    uint32_t heads, const float* w, const float* dyn, const float* hist, float res,
                                    float scale) {
  const uint32_t h = taps - 1, head = c / (ch / heads);
  const float* d = dyn + (uint64_t) t * heads * taps + (uint64_t) head * taps;
  float sum = 0;
  for (uint32_t j = 0; j < taps; j++)  // tap j: j tokens back
    sum += (w[(uint64_t) j * ch + c] + d[j]) * tap(x, hist, t + h - j, h, ch, c);
  return (sum + res * x[(uint64_t) t * ch + c]) * scale;
}

// add (not NULL): y the conv plus add's value (ink2_axpy's sum; add may be y)
__global__ void conv_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                            const float* hist, float res, float scale, const float* add, float* y, float* keep) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  if (keep) keep[i] = x[i];
  const float out = conv_at(x, (uint32_t) (i / ch), (uint32_t) (i % ch), ch, taps, w, hist, res, scale);
  y[i] = add ? add[i] + out : out;
}

// conv_kernel on the experts' sums, each tap's value moe_at's where it reads one (the step's tokens): ink2_moe_sum
// then ink2_conv_keep's bits
__global__ void moe_conv_kernel(const float* x, const float* a, const int32_t* row, uint32_t k, uint32_t n_tok,
                                uint32_t ch, uint32_t taps, const float* w, const float* hist, float res, float scale,
                                const float* add, float* y, float* keep) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = taps - 1;
  float sum = 0, xt = 0;
  for (uint32_t j = 0; j < taps; j++) {
    const uint32_t at = t + j;
    const float v = at < h ? (hist ? hist[(uint64_t) at * ch + c] : 0.0f) : moe_at(x, a, row, k, at - h, c, ch);
    sum += w[(uint64_t) c * taps + j] * v;
    xt = v;  // (the last tap: token t's own)
  }
  if (keep) keep[i] = xt;
  const float out = (sum + res * xt) * scale;
  y[i] = add ? add[i] + out : out;
}

__global__ void conv_dyn_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                                const float* w, const float* dyn, const float* hist, float res, float scale, float* y,
                                float* keep) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  if (keep) keep[i] = x[i];
  y[i] = conv_dyn_at(x, (uint32_t) (i / ch), (uint32_t) (i % ch), ch, taps, heads, w, dyn, hist, res, scale);
}

// a block a token: its states' inverse RMS, each query's scores and softmax, then the weighted sum
__global__ void attn_res_kernel(const float* query, uint32_t queries, const float* blocks, uint32_t nb,
                                uint32_t bs, const float* partial, uint32_t dim, float eps, float scale,
                                uint32_t n_tok, float* out) {
  __shared__ float lds[kThreads];
  __shared__ float inv[INK2_MAX_RES], p[INK2_MAX_QUERIES][INK2_MAX_RES];
  const uint32_t t = blockIdx.x, ns = nb + 1;
  auto state = [&](uint32_t s) {
    return s < nb ? blocks + ((uint64_t) t * bs + s) * dim : partial + (uint64_t) t * dim;
  };
  for (uint32_t s = 0; s < ns; s++) {
    const float* v = state(s);
    float sq = 0;
    for (uint32_t c = threadIdx.x; c < dim; c += kThreads) sq += v[c] * v[c];
    sq = block_sum<kThreads>(sq, lds);
    if (!threadIdx.x) inv[s] = rsqrtf(sq / (float) dim + eps);
  }
  __syncthreads();
  for (uint32_t q = 0; q < queries; q++)
    for (uint32_t s = 0; s < ns; s++) {
      const float* v = state(s);
      float d = 0;
      for (uint32_t c = threadIdx.x; c < dim; c += kThreads) d += query[(uint64_t) q * dim + c] * v[c];
      d = block_sum<kThreads>(d, lds);
      if (!threadIdx.x) p[q][s] = d * inv[s] * scale;
    }
  __syncthreads();
  if (threadIdx.x < queries) {
    float* r = p[threadIdx.x];
    float m = r[0], z = 0;
    for (uint32_t s = 1; s < ns; s++) m = fmaxf(m, r[s]);
    for (uint32_t s = 0; s < ns; s++) z += (r[s] = expf(r[s] - m));
    for (uint32_t s = 0; s < ns; s++) r[s] = r[s] / z * inv[s];  // (the state's own normalization folded in)
  }
  __syncthreads();
  for (uint32_t c = threadIdx.x; c < dim; c += kThreads)
    for (uint32_t q = 0; q < queries; q++) {
      float o = 0;
      for (uint32_t s = 0; s < ns; s++) o += p[q][s] * state(s)[c];
      out[((uint64_t) q * n_tok + t) * dim + c] = o;
    }
}

// attn_res split for a few tokens (one block a token is one CU's serial reductions): a block each state's square
// sum and each query's dot with it - attn_res_kernel's own sums - into scratch [n_tok][1 + queries][ns], then the
// softmax and the weighted sum a block a kThreads of dim, each value's arithmetic attn_res_kernel's: the same bits
constexpr uint32_t kResRun = 8;  // a 6144-wide state's 24 values a thread in 3 runs
__global__ void attn_res_dots_kernel(const float* query, uint32_t queries, const float* blocks, uint32_t nb,
                                     uint32_t bs, const float* partial, uint32_t dim, float* scratch) {
  __shared__ float lds[kThreads];
  const uint32_t s = blockIdx.x, y = blockIdx.y, t = blockIdx.z, ns = nb + 1;
  const float* v = s < nb ? blocks + ((uint64_t) t * bs + s) * dim : partial + (uint64_t) t * dim;
  float d = 0;  // (unrolled: a thread's loads in flight together, its adds in order)
  if (!y)
#pragma unroll kResRun
    for (uint32_t c = threadIdx.x; c < dim; c += kThreads) d += v[c] * v[c];
  else
#pragma unroll kResRun
    for (uint32_t c = threadIdx.x; c < dim; c += kThreads) d += query[(uint64_t) (y - 1) * dim + c] * v[c];
  d = block_sum<kThreads>(d, lds);
  if (!threadIdx.x) scratch[((uint64_t) t * (1 + queries) + y) * ns + s] = d;
}

__global__ void attn_res_mix_kernel(uint32_t queries, const float* blocks, uint32_t nb, uint32_t bs,
                                    const float* partial, uint32_t dim, float eps, float scale, uint32_t n_tok,
                                    const float* scratch, float* out) {
  __shared__ float inv[INK2_MAX_RES], p[INK2_MAX_QUERIES][INK2_MAX_RES];
  const uint32_t t = blockIdx.y, ns = nb + 1;
  const float* sc = scratch + (uint64_t) t * (1 + queries) * ns;
  if (threadIdx.x < ns) inv[threadIdx.x] = rsqrtf(sc[threadIdx.x] / (float) dim + eps);
  __syncthreads();
  if (threadIdx.x < queries * ns) {
    const uint32_t q = threadIdx.x / ns, s = threadIdx.x % ns;
    p[q][s] = sc[(1 + q) * ns + s] * inv[s] * scale;
  }
  __syncthreads();
  if (threadIdx.x < queries) {
    float* r = p[threadIdx.x];
    float m = r[0], z = 0;
    for (uint32_t s = 1; s < ns; s++) m = fmaxf(m, r[s]);
    for (uint32_t s = 0; s < ns; s++) z += (r[s] = expf(r[s] - m));
    for (uint32_t s = 0; s < ns; s++) r[s] = r[s] / z * inv[s];
  }
  __syncthreads();
  const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= dim) return;
  for (uint32_t q = 0; q < queries; q++) {
    float o = 0;
#pragma unroll kResRun
    for (uint32_t s = 0; s < ns; s++)
      o += p[q][s] * (s < nb ? blocks + ((uint64_t) t * bs + s) * dim : partial + (uint64_t) t * dim)[c];
    out[((uint64_t) q * n_tok + t) * dim + c] = o;
  }
}

// token t's row j, by a block: times w, the L2 clip, tau - a thread its values d = threadIdx.x + k kThreads
__device__ inline void clip_row(float* row, uint32_t t, uint32_t j, uint32_t dim, const float* w, float clip,
                                Ink2Tau tau, float* lds) {
  float sq = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    const float v = w ? row[d] * w[(uint64_t) j * dim + d] : row[d];
    row[d] = v, sq += v * v;
  }
  float f = 1.0f;
  if (clip > 0) f = fminf(1.0f, clip / (sqrtf(block_sum<kThreads>(sq, lds)) + 1e-6f));
  if (tau.alpha > 0) f *= 1.0f + tau.alpha * logf(fmaxf(1.0f, (float) (tau.pos0 + t + 1) / tau.floor));
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) row[d] *= f;
}

// a block a row
__global__ void clip_rows_kernel(float* x, uint32_t per_tok, uint32_t dim, const float* w, float clip, Ink2Tau tau) {
  __shared__ float lds[kThreads];
  const uint32_t r = blockIdx.x;
  clip_row(x + (uint64_t) r * dim, r / per_tok, r % per_tok, dim, w, clip, tau, lds);
}

constexpr uint32_t kChunk = INK2_MAX_HEAD;  // keys a pass: a thread a key, then a thread a value dim

// a block a (token, head), head_dim threads: the relative bias's row (rel . rel_proj, extent values), then the keys
// in chunks - a thread a key's logit, the chunk's max and sum folded into the running ones (online softmax), a
// thread a dim's weighted sum of the chunk's values
__global__ void attn_kernel(Ink2Attn a, const float* q, const float* rel, const float* sink, const float* gate,
                            uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, const float* rel_proj,
                            float* out) {
  __shared__ float qs[INK2_MAX_HEAD], rl[INK2_MAX_EXTENT], p[kChunk], red[kChunk];
  const uint32_t t = blockIdx.x / a.heads, h = blockIdx.x % a.heads, kvh = h / (a.heads / a.kv_heads);
  const uint32_t D = a.head_dim, i = threadIdx.x, qi = n_keys - n_tok + t;
  qs[i] = q[((uint64_t) t * a.heads + h) * D + i];
  const float* rr = rel + ((uint64_t) t * a.heads + h) * a.d_rel;
  for (uint32_t e = i; e < a.extent; e += D) {
    float b = 0;
    for (uint32_t r = 0; r < a.d_rel; r++) b += rr[r] * rel_proj[(uint64_t) r * a.extent + e];
    rl[e] = b;
  }
  __syncthreads();
  const uint32_t lo = a.window && qi + 1 > a.window ? qi + 1 - a.window : 0;
  float m = sink[(uint64_t) t * a.heads + h], z = 1.0f, acc = 0;  // the sink: its logit, weight 1 at its max, value 0
  for (uint32_t s0 = lo; s0 <= qi; s0 += kChunk) {
    const uint32_t s = s0 + i, n = qi + 1 - s0 < kChunk ? qi + 1 - s0 : kChunk;
    float l = -INFINITY;
    if (i < n) {
      const float* kr = k + ((uint64_t) s * a.kv_heads + kvh) * D;
      float d = 0;
      for (uint32_t c = 0; c < D; c++) d += qs[c] * kr[c];
      const uint32_t dist = qi - s;
      l = d * a.scaling + (dist < a.extent ? rl[dist] : 0.0f);
    }
    const float cm = block_max<kChunk>(l, red), nm = fmaxf(m, cm), f = expf(m - nm);
    const float e = i < n ? expf(l - nm) : 0.0f;
    p[i] = e;
    const float cs = block_sum<kChunk>(e, red);
    z = z * f + cs, acc *= f, m = nm;
    for (uint32_t j = 0; j < n; j++) acc += p[j] * v[((uint64_t) (s0 + j) * a.kv_heads + kvh) * D + i];
    __syncthreads();
  }
  const uint64_t o = (uint64_t) t * a.heads * D + (uint64_t) h * D + i;
  out[o] = acc / z * (gate ? sigmoid(gate[o]) : 1.0f);
}

// the prompt's attention on WMMA (16 x 16 x 16, f16 in, f32 sums): a block a 16-token tile and KV head, a wave a
// query head of its group; each 16-key tile's K and V^T in LDS once for the group. a lane holds rows 2 j + half
// (j < 8) at column lane % 16 of a tile's scores and of each 16-dim slice of O (q4x_attn.hip's layout)
constexpr int kFaTile = INK2_FA_TILE;
constexpr int kFaHead = INK2_MAX_HEAD;
constexpr int kFaRow = kFaHead + 8;  // f16s a row in LDS, then a pad
constexpr int kFaKey = kFaTile + 8;
constexpr int kFaRowTiles = 4;
// a split decode's waves past which a call's blocks keep one wave (more waves then slower: the GPU full, and only a
// one-wave block reads Q8 keys straight into registers), tuned on gfx1151 at Mini's shape
constexpr uint32_t kFaDecodeWaves = 256;
constexpr uint32_t kFaUnit = INK2_FA_UNIT;  // a unit's keys (absolute positions): each its own softmax, folded in order
typedef _Float16 fa_h16 __attribute__((ext_vector_type(16)));
typedef float fa_f8 __attribute__((ext_vector_type(8)));

// a row's 16 lanes' max and sum, each a butterfly over lane ^ 1, 2, 4, 8 - by DPP's row_xmask (within a row of 16:
// the same partners as __shfl_xor's, so the same values in the same order; not ds_bpermute's trip through LDS)
template <int O>
__device__ inline float fa_xor(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 + O, 0xF, 0xF, false));
}
// the max a v_max_f32 with its DPP operand: one instruction a step, not fmaxf's move, canonicalize and max (the
// scores never NaN - a finite value or -inf - so the canonicalize kept nothing: the same values)
template <int O>
__device__ inline float fa_max_xor(float v) {
  float r;
  asm volatile("v_max_f32_dpp %0, %1, %1 row_xmask:%2 row_mask:0xf bank_mask:0xf bound_ctrl:1" : "=v"(r) : "v"(v), "i"(O));
  return r;
}
__device__ inline float fa_row_max(float v) {
  v = fa_max_xor<1>(v), v = fa_max_xor<2>(v), v = fa_max_xor<4>(v);
  return fa_max_xor<8>(v);
}
__device__ inline float fa_row_sum(float v) {
  v += fa_xor<1>(v), v += fa_xor<2>(v), v += fa_xor<4>(v);
  return v + fa_xor<8>(v);
}

// a unit's online softmax folded into the running one (the sink's start): M the max, each side scaled to it - the
// prompt's kernel and a split decode's fold the same units in the same order with this, so the same bits
__device__ inline void fa_fold_coef(float& gm, float& gl, float m, float l, float& a, float& b) {
  const float M = fmaxf(gm, m);
  a = expf(gm - M), b = m == -INFINITY ? 0.0f : expf(m - M);
  gl = gl * a + l * b, gm = M;
}

// a full layer's cache as Q8 records (ink2_ops.h's INK2_KV8_RECORD): the kernel's KV a byte of them
struct Kv8 {
  uint8_t b;
};
typedef _Float16 fa_h8 __attribute__((ext_vector_type(8)));

// a record's 8 values from d as loaded (its int8s, its block's f16 scale), then as values: q times the scale, rounded
// to f16 (ink2_fa8's values) - loaded and turned into values apart, so a tile's loads are in flight under the last's work
struct Kv8Raw {
  uint2 q;
  _Float16 s;
};
__device__ inline Kv8Raw kv8_raw(const uint8_t* rec, uint32_t d) {
  return {*(const uint2*) (rec + d), *(const _Float16*) (rec + INK2_MAX_HEAD + d / INK2_KV8_BLOCK * 2)};
}
// (the product in f16: an int8 and an f16 exact in it, their F32 product exact - 19 significant bits - so the f16
// multiply's one rounding is the F32 product's to f16: the same bits, two values a packed multiply)
typedef _Float16 fa_h2 __attribute__((ext_vector_type(2)));
__device__ inline fa_h8 kv8_vals(const Kv8Raw& r) {
  const fa_h2 s2 = {r.s, r.s};
  const int8_t* q = (const int8_t*) &r.q;
  fa_h8 out;
  for (int e = 0; e < 8; e += 2) {
    const fa_h2 p = fa_h2{(_Float16) q[e], (_Float16) q[e + 1]} * s2;
    out[e] = p[0], out[e + 1] = p[1];
  }
  return out;
}

// keys Q8 records, values Q5 (ink2_fa85): the kernel's KV a byte of them
struct Kv85 {
  uint8_t b;
};

// a Q5 record's 8 values from d (a multiple of 8) as loaded (nibbles, fifth bits, scale), then as values: (q - 16)
// times its block's f16 scale, rounded to f16
struct Kv5Raw {
  uint32_t nib, hi;
  _Float16 s;
};
__device__ inline Kv5Raw kv5_raw(const uint8_t* rec, uint32_t d) {
  constexpr uint32_t H = INK2_MAX_HEAD;
  return {*(const uint32_t*) (rec + d / 2), rec[H / 2 + d / 8],
          *(const _Float16*) (rec + H / 2 + H / 8 + d / INK2_KV8_BLOCK * 2)};
}
__device__ inline fa_h8 kv5_vals(const Kv5Raw& r) {  // (kv8_vals' f16 product: q - 16 exact in f16 too)
  const fa_h2 s2 = {r.s, r.s};
  fa_h8 out;
  for (int e = 0; e < 8; e += 2) {
    const int q0 = (int) ((r.nib >> (4 * e) & 15u) | (r.hi >> e & 1u) << 4),
              q1 = (int) ((r.nib >> (4 * e + 4) & 15u) | (r.hi >> (e + 1) & 1u) << 4);
    const fa_h2 p = fa_h2{(_Float16) (q0 - 16), (_Float16) (q1 - 16)} * s2;
    out[e] = p[0], out[e + 1] = p[1];
  }
  return out;
}

// G waves; Q's A fragments in registers (a lane its row's 8 x 16 halves), the next tile's K and V prefetched into
// registers while this one's WMMAs run - LDS ~16 KB a block (K, V^T, P), several blocks a CU
// KV: the keys' and values' type in memory - F32 (rounded to f16 as loaded) or f16 (a cache's: the same values).
// a query's keys in units of kFaUnit at absolute positions (pos0 key 0's): each unit's online softmax from nothing,
// folded into the running one in order. SPLIT: a block a unit (blockIdx.z, from unit u0), its partial (m, l, o)
// written to part for ink2_fa_fold - a decode's units in parallel, the prompt's fold's bits
template <int G, int RT, class KV, bool SPLIT, int W = G * RT>
__global__ __launch_bounds__(W * 32) void fa_kernel(Ink2Attn a, const float* q, const float* rel, const float* sink,
                                                     const float* gate, uint32_t n_tok, const KV* k,
                                                     const KV* v, uint32_t n_keys, uint32_t pos0,
                                                     const float* rel_proj, float* out, uint32_t u0, float* part) {
  // W waves: the prompt's G RT (a row tile's heads); a split decode's as many as its (token, head) rows fill - a wave
  // without rows still holds its registers, so a decode's token runs one wave, not G (more blocks resident)
  static_assert(SPLIT || W == G * RT, "the prompt's waves its row tiles' heads");
  constexpr int kNth = W * 32, kItems = (kFaTile * kFaHead + kNth - 1) / kNth;
  // a one-wave decode over a full layer's Q8 records: its keys straight from each lane's own record into the scores'
  // B fragments (a lane its key's, as LDS would hand them) - no K tile in LDS, so more blocks resident
  constexpr bool kKD = sizeof(KV) == 1 && SPLIT && W == 1;
  __shared__ __attribute__((aligned(32))) _Float16 ks[kKD ? 8 : kFaTile * kFaRow], vt[kFaHead * kFaKey];
  __shared__ __attribute__((aligned(32))) _Float16 ps[W][kFaTile * kFaKey];
  // a block RT row tiles of 16 tokens; a wave (row tile, query head) - every K / V tile shared by the block's waves
  const uint32_t wave = threadIdx.x / 32, lane = threadIdx.x % 32, half = lane >> 4, col = lane % kFaTile;
  const uint32_t tb = blockIdx.x * kFaTile * RT, t0 = tb + wave / G * kFaTile, kvh = blockIdx.y, H = a.heads;
  const uint32_t h = kvh * G + wave % G, base = n_keys - n_tok;  // token t is key base + t
  // a row's token and query head: the prompt's a wave's head, its rows 16 tokens; a split decode's rows (token, head)
  // pairs, wave w's the w * 16th on (a token's G heads in G rows: one wave for a decode's token, not G of one row each)
  // - each row's sums its own, so the same bits either way; a wave past the pairs only loads the tiles
  auto rtok = [&](uint32_t r) -> uint32_t { return SPLIT ? (wave * kFaTile + r) / G : t0 + r; };
  auto rhead = [&](uint32_t r) -> uint32_t { return SPLIT ? kvh * G + (wave * kFaTile + r) % G : h; };
  const bool active = !SPLIT || wave * kFaTile < n_tok * G;
  fa_h16 qa[kFaHead / kFaTile];
  {
    const uint32_t t = rtok(col);
    const float* qr = q + ((uint64_t) t * H + rhead(col)) * kFaHead;
    for (int g = 0; g < kFaHead / kFaTile; g++)
      for (int e = 0; e < kFaTile; e++) qa[g][e] = (_Float16) (t < n_tok ? qr[g * kFaTile + e] : 0.0f);
  }
  // the running softmax: the sink a logit of its own, its value zero (gm, gl, go); the unit's from nothing (m, l, o)
  float m[8], l[8];
  // the running fold's (gm, gl) a row's in LDS: read at a unit's end alone (every 16 tiles), not 16 registers a lane
  // (a row's 16 lanes the same values; its lane 0 writes) - fewer spills
  __shared__ float gml[W][kFaTile][2];
  auto gm = [&](int j) -> float& { return gml[wave][2 * j + half][0]; };
  auto gl = [&](int j) -> float& { return gml[wave][2 * j + half][1]; };
  // the unit's (m, l, o) in registers; the prompt's running fold's o in its out rows (each lane its own elements: zero
  // first, a unit's end its read and write, the end its division) - not 64 more registers a wave, so fewer spills
  fa_f8 o[kFaHead / kFaTile] = {};
  auto out_at = [&](uint32_t t, int nt) -> uint64_t { return ((uint64_t) t * H + h) * kFaHead + nt * kFaTile + col; };
  if constexpr (!SPLIT)
    for (int j = 0; j < 8; j++)
      if (rtok(2 * j + half) < n_tok)
        for (int nt = 0; nt < kFaHead / kFaTile; nt++) out[out_at(rtok(2 * j + half), nt)] = 0.0f;
  for (int j = 0; j < 8; j++) {
    const uint32_t t = rtok(2 * j + half);
    gm(j) = t < n_tok ? sink[(uint64_t) t * H + rhead(2 * j + half)] : 0.0f, gl(j) = 1.0f, m[j] = -INFINITY, l[j] = 0.0f;
  }
  // the block's key range (its first row tile's lowest to its last's highest); a wave's own rows masked within it
  const uint32_t last = tb + kFaTile * RT - 1 < n_tok - 1 ? tb + kFaTile * RT - 1 : n_tok - 1;
  const uint32_t qlo = base + tb, qhi = base + last;
  // the first key tile at a multiple of kFaTile: a query's keys cut into the same tiles whatever block it is in
  // (the tiles before its window wholly masked: no-ops), so its bits are its own - given keys whose index is their
  // position mod kFaTile (INK2_FA_TILE: the engine's windows start so)
  uint32_t klo = a.window && qlo + 1 > a.window ? (qlo + 1 - a.window) / kFaTile * kFaTile : 0, khi = qhi;
  if constexpr (SPLIT) {  // this block's unit alone
    const uint32_t u = u0 + blockIdx.z, ua = u * kFaUnit;  // the unit's keys as indices (it may start before key 0)
    const uint32_t us = ua > pos0 ? ua - pos0 : 0, ue = ua + kFaUnit - 1 - pos0;
    klo = klo > us ? klo : us, khi = khi < ue ? khi : ue;
  }
  // a tile's K and V into registers (keys past the last zero): F32 a value a thread at a time; f16 (a cache's) 8
  // halves a 16-byte load - the same values either way
  constexpr bool kQ8 = sizeof(KV) == 1, kHalf = sizeof(KV) <= 2;  // (kHalf: 8 f16s a load - a Q8 record's too)
  constexpr bool kV5 = std::is_same<KV, Kv85>::value;             // (the values Q5 records: ink2_fa85)
  constexpr uint32_t kVRec = kV5 ? INK2_KV5_RECORD : INK2_KV8_RECORD;
  constexpr int kVec = 8, kVecs = kFaTile * kFaHead / kVec, kVItems = (kVecs + kNth - 1) / kNth;
  float kr[kHalf ? 1 : kItems], vr[kHalf ? 1 : kItems];
  fa_h8 kh[kHalf && !kQ8 ? kVItems : 1], vh[kHalf && !kQ8 ? kVItems : 1];
  Kv8Raw k8[kQ8 ? kVItems : 1], v8[kQ8 && !kV5 ? kVItems : 1];  // (a Q8 / Q5 tile as loaded: its values at the store)
  Kv5Raw v5[kV5 ? kVItems : 1];
  uint2 kq[kKD ? kFaHead / kVec : 1];  // (K-direct: the lane's key's record - its int8s 8 a piece, a scale a block)
  _Float16 ksc[kKD ? kFaHead / INK2_KV8_BLOCK : 1];
  // K-direct: the lane's key's record, loaded after this tile's scores (its only reader), in flight under the rest
  auto fetch_k = [&](uint32_t kb) {
    const bool in = kb + col < n_keys;
    const uint8_t* rec = (const uint8_t*) k + ((uint64_t) (kb + col) * a.kv_heads + kvh) * INK2_KV8_RECORD;
    for (int i = 0; i < kFaHead / kVec; i++) kq[i] = in ? *(const uint2*) (rec + i * kVec) : uint2{0, 0};
    for (int b = 0; b < kFaHead / INK2_KV8_BLOCK; b++)
      ksc[b] = in ? *(const _Float16*) (rec + INK2_MAX_HEAD + 2 * b) : (_Float16) 0.0f;
  };
  auto fetch = [&](uint32_t kb) {
    if constexpr (kQ8)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i % kFaTile, d = i / kFaTile * kVec;  // (a wave's threads its keys: V^T's stores apart)
        const bool in = i < (uint32_t) kVecs && kb + r < n_keys;
        const uint64_t at = (uint64_t) (kb + r) * a.kv_heads + kvh;
        if constexpr (!kKD) k8[it] = in ? kv8_raw((const uint8_t*) k + at * INK2_KV8_RECORD, d) : Kv8Raw{};
        if constexpr (kV5)
          v5[it] = in ? kv5_raw((const uint8_t*) v + at * kVRec, d) : Kv5Raw{};
        else
          v8[it] = in ? kv8_raw((const uint8_t*) v + at * kVRec, d) : Kv8Raw{};
      }
    else if constexpr (kHalf)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i % kFaTile, d = i / kFaTile * kVec;  // (a wave's threads its keys: V^T's stores apart)
        const bool in = i < (uint32_t) kVecs && kb + r < n_keys;
        const uint64_t at = ((uint64_t) (kb + r) * a.kv_heads + kvh) * kFaHead + d;
        kh[it] = in ? *(const fa_h8*) (k + at) : fa_h8{}, vh[it] = in ? *(const fa_h8*) (v + at) : fa_h8{};
      }
    else
      for (int it = 0; it < kItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / kFaHead, d = i % kFaHead, key = kb + r;
        const bool in = i < kFaTile * kFaHead && key < n_keys;
        const uint64_t at = ((uint64_t) key * a.kv_heads + kvh) * kFaHead + d;
        kr[it] = in ? (float) k[at] : 0.0f, vr[it] = in ? (float) v[at] : 0.0f;
      }
  };
  if (klo <= khi) {
    fetch(klo);
    if constexpr (kKD) fetch_k(klo);
  }
  for (uint32_t kb = klo; kb <= khi; kb += kFaTile) {
    __syncthreads();  // the last tile's K, V read by every wave
    if constexpr (kHalf)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i % kFaTile, d = i / kFaTile * kVec;  // (a wave's threads its keys: V^T's stores apart)
        if (i >= (uint32_t) kVecs) continue;
        fa_h8 kv, vv;  // a Q8 / Q5 tile's values now (a missing key's raw zeros: 0 x 0, f16's +0)
        if constexpr (kQ8) {
          if constexpr (!kKD) kv = kv8_vals(k8[it]);
          if constexpr (kV5)
            vv = kv5_vals(v5[it]);
          else
            vv = kv8_vals(v8[it]);
        } else
          kv = kh[it], vv = vh[it];
        if constexpr (!kKD) *(fa_h8*) (ks + r * kFaRow + d) = kv;
        for (int e = 0; e < kVec; e++) vt[(d + e) * kFaKey + r] = vv[e];
      }

    else
      for (int it = 0; it < kItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / kFaHead, d = i % kFaHead;
        if (i < kFaTile * kFaHead) ks[r * kFaRow + d] = (_Float16) kr[it], vt[d * kFaKey + r] = (_Float16) vr[it];
      }
    __syncthreads();
    if (kb + kFaTile <= khi) fetch(kb + kFaTile);  // the next tile's, in flight under this one's WMMAs
    if (!active) continue;                         // (its loads done: no row of its own)
    fa_f8 sacc = {};
    for (int g = 0; g < kFaHead / kFaTile; g++) {
      fa_h16 kb16;
      if constexpr (kKD) {  // the lane's key's B fragment: dims g 16 .. g 16 + 15, kv8_vals' values
        constexpr int kPer = INK2_KV8_BLOCK / kVec;  // (pieces a scale)
        const fa_h8 lo = kv8_vals({kq[2 * g], ksc[2 * g / kPer]}), hi = kv8_vals({kq[2 * g + 1], ksc[(2 * g + 1) / kPer]});
        for (int e = 0; e < kVec; e++) kb16[e] = lo[e], kb16[kVec + e] = hi[e];
      } else
        kb16 = *(const fa_h16*) (ks + col * kFaRow + g * kFaTile);
      sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[g], kb16, sacc);
    }
    if constexpr (kKD)
      if (kb + kFaTile <= khi) fetch_k(kb + kFaTile);
    const uint32_t key = kb + col;
    float alpha[8];
#pragma unroll  // (the rows' registers named: not m0's indirect moves and a branch a row)
    for (int j = 0; j < 8; j++) {
      const uint32_t t = rtok(2 * j + half), qi = base + t;
      const bool valid = t < n_tok && key <= qi && (!a.window || qi - key < a.window);
      float s = -INFINITY;
      if (valid) {
        const uint32_t dist = qi - key;
        s = sacc[j] * a.scaling;
        if (dist < a.extent) {
          const float* rr = rel + ((uint64_t) t * H + rhead(2 * j + half)) * a.d_rel;
          float b = 0;
#pragma unroll  // (fa_launch's INK2_FA_REL: the 16 terms' loads in flight together, the sum in their order)
          for (uint32_t r = 0; r < INK2_FA_REL; r++) b += rr[r] * rel_proj[(uint64_t) r * a.extent + dist];
          s += b;
        }
      }
      const float mn = fmaxf(m[j], fa_row_max(s));
      const float pv = s == -INFINITY ? 0.0f : expf(s - mn);
      alpha[j] = m[j] == mn ? 1.0f : expf(m[j] - mn);  // (a unit's start: -inf both, nothing held)
      l[j] = l[j] * alpha[j] + fa_row_sum(pv), m[j] = mn;
      ps[wave][(2 * j + half) * kFaKey + col] = (_Float16) pv;
    }
    __builtin_amdgcn_wave_barrier();
    const fa_h16 pa = *(const fa_h16*) (ps[wave] + col * kFaKey);
    // a row whose own key is in this tile (not its last): its p . v over the tile's values up to its own alone - the
    // WMMA's sum moves with a value whose p is 0, and the keys past a query are the prompt's next tokens in one
    // step, none in a step that ends at it (a decode's). cut c: the rows at key kb + c (the wave's rows the valid)
    const uint32_t tlo = rtok(0), thi = rtok(kFaTile - 1) < n_tok - 1 ? rtok(kFaTile - 1) : n_tok - 1;
    const int64_t c_lo = (int64_t) (base + tlo) - kb, c_last = (int64_t) (base + thi) - kb;
    const int cmin = c_lo > 0 ? (int) c_lo : 0, cmax = c_last < kFaTile - 2 ? (int) c_last : kFaTile - 2;
#pragma unroll
    for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
      for (int j = 0; j < 8; j++) o[nt][j] *= alpha[j];
      const fa_h16 bv = *(const fa_h16*) (vt + (nt * kFaTile + col) * kFaKey);
      const fa_f8 prev = o[nt];
      o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bv, prev);
      for (int c = cmin; c <= cmax; c++) {
        fa_h16 bc = bv;
        for (int e = c + 1; e < kFaTile; e++) bc[e] = (_Float16) 0.0f;
        const fa_f8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bc, prev);
        for (int j = 0; j < 8; j++)
          if ((int64_t) (base + rtok(2 * j + half)) == (int64_t) kb + c) o[nt][j] = d[j];
      }
    }
    __builtin_amdgcn_wave_barrier();
    if constexpr (!SPLIT)
      if ((pos0 + kb + kFaTile) % kFaUnit == 0 || kb + kFaTile > khi) {  // a unit's end: folded, the next from nothing
        for (int j = 0; j < 8; j++) {
          float ca, cb, fgm = gm(j), fgl = gl(j);
          fa_fold_coef(fgm, fgl, m[j], l[j], ca, cb);
          __builtin_amdgcn_wave_barrier();
          if (col == 0) gm(j) = fgm, gl(j) = fgl;
          const uint32_t t = rtok(2 * j + half);
          for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
            if (t < n_tok) out[out_at(t, nt)] = out[out_at(t, nt)] * ca + o[nt][j] * cb;
            o[nt][j] = 0.0f;
          }
          m[j] = -INFINITY, l[j] = 0.0f;
        }
      }
  }
  for (int j = 0; j < 8 && active; j++) {
    const uint32_t t = rtok(2 * j + half);
    if (t >= n_tok) continue;
    if constexpr (SPLIT) {  // the unit's partial: m, l, then o (ink2_fa_fold's)
      float* pp = part + (((uint64_t) blockIdx.z * n_tok + t) * H + rhead(2 * j + half)) * (kFaHead + 2);
      if (col == 0) pp[0] = m[j], pp[1] = l[j];
      for (int nt = 0; nt < kFaHead / kFaTile; nt++) pp[2 + nt * kFaTile + col] = o[nt][j];
    } else
      for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
        const uint64_t i = out_at(t, nt);
        out[i] = out[i] / gl(j) * (gate ? sigmoid(gate[i]) : 1.0f);
      }
  }
}

// a split decode's units folded in order into the sink's start (fa_kernel's fold): a block a (token, head), a
// thread a value dim
__global__ void fa_fold_kernel(const float* part, uint32_t units, uint32_t n_tok, uint32_t H, const float* sink,
                               const float* gate, float* out, _Float16* x16, int32_t* overflow) {
  const uint32_t t = blockIdx.x / H, h = blockIdx.x % H, d = threadIdx.x, cols = H * kFaHead;
  if (t >= n_tok) return mm::x16_put(x16, t, h * kFaHead + d, cols, 0.0f, overflow);  // a pad row of x16's tile
  float gm = sink[(uint64_t) t * H + h], gl = 1.0f, go = 0.0f;
  for (uint32_t u = 0; u < units; u++) {
    const float* pp = part + (((uint64_t) u * n_tok + t) * H + h) * (kFaHead + 2);
    float ca, cb;
    fa_fold_coef(gm, gl, pp[0], pp[1], ca, cb);
    go = go * ca + pp[2 + d] * cb;
  }
  const uint64_t i = ((uint64_t) t * H + h) * kFaHead + d;
  const float o = go / gl * (gate ? sigmoid(gate[i]) : 1.0f);
  out[i] = o;
  if (x16) mm::x16_put(x16, t, h * kFaHead + d, cols, o, overflow);
}

}  // namespace

template <class KV>
static bool fa_launch(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                      uint32_t n_tok, const KV* k, const KV* v, uint32_t n_keys, uint32_t pos0, const float* rel_proj,
                      float* out, float* scratch, void* stream) {
  if (!a || !q || !rel || !sink || !k || !v || !rel_proj || !out || !n_tok || n_tok > n_keys ||
      a->head_dim != INK2_MAX_HEAD || !a->kv_heads || a->heads % a->kv_heads ||
      a->heads / a->kv_heads > INK2_FA_MAX_GROUP || a->d_rel != INK2_FA_REL || !a->extent ||
      a->extent > INK2_MAX_EXTENT || pos0 % kFaTile)
    return false;
  const uint32_t G = a->heads / a->kv_heads;
  if (a->x16 && (!scratch || n_tok > (uint32_t) kFaTile || !a->overflow)) return false;  // a split decode's alone
  if (scratch && n_tok <= (uint32_t) kFaTile) {  // a decode's: a block a unit and KV head, then the fold
    const uint32_t qlo = n_keys - n_tok, qhi = n_keys - 1;
    const uint32_t klo = a->window && qlo + 1 > a->window ? (qlo + 1 - a->window) / kFaTile * kFaTile : 0;
    const uint32_t u0 = (pos0 + klo) / kFaUnit, units = (pos0 + qhi) / kFaUnit - u0 + 1;
    const dim3 grid(1, a->kv_heads, units);
    const uint32_t rows = a->x16 ? (n_tok + mm::kTile - 1) / mm::kTile * mm::kTile : n_tok;  // x16's: a whole tile
    // the waves the (token, head) rows fill (n_tok G of them, 16 a wave), then more while the call's blocks leave
    // the GPU idle: a decode's blocks are latency-bound (waiting on memory at one wave), and a
    // block's waves past its rows only load its tiles - each row's sums its own, so the same bits at any W
    const uint32_t need = (n_tok * G + kFaTile - 1) / kFaTile, blocks = a->kv_heads * units;
    uint32_t W = need <= 1 ? 1 : need <= 2 ? 2 : need <= 4 ? 4 : 8;
    while (W < 8 && blocks * W * 2 <= kFaDecodeWaves) W *= 2;
#define FSW(g, w)                                                                                                    \
  if (G == g && W == w) {                                                                                            \
    fa_kernel<g, 1, KV, true, w><<<grid, w * 32, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys, pos0,  \
                                                                  rel_proj, out, u0, scratch);                       \
    fa_fold_kernel<<<rows * a->heads, kFaHead, 0, st(stream)>>>(scratch, units, n_tok, a->heads, sink, gate, out,    \
                                                               (_Float16*) a->x16, a->overflow);                    \
    return ok();                                                                                                     \
  }
#define FS(g) FSW(g, 1) FSW(g, 2) FSW(g, 4) FSW(g, 8)
    FS(1) FS(2) FS(3) FS(4) FS(6) FS(8)
#undef FS
#undef FSW
    return false;
  }
  // a block's waves G R: Mini's 3 heads a KV head 8 row tiles (24 waves, Small's 6 x 4) - every key tile decoded once
  // for twice the rows; the tiles a query's keys are cut into are the same in any block, so R changes no bit
  const uint32_t R = G == 3 ? 8u : kFaRowTiles;
  const dim3 grid_fa((n_tok + kFaTile * R - 1) / (kFaTile * R), a->kv_heads);
#define FA(g, r)                                                                                                    \
  if (G == g && R == r) {                                                                                           \
    fa_kernel<g, r, KV, false><<<grid_fa, g * r * 32, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys,  \
                                                                       pos0, rel_proj, out, 0, nullptr);            \
    return ok();                                                                                                    \
  }
  FA(1, 1) FA(2, 1) FA(3, 1) FA(4, 1) FA(6, 1) FA(8, 1) FA(1, 2) FA(2, 2) FA(3, 2) FA(4, 2) FA(6, 2) FA(8, 2) FA(1, 4)
  FA(2, 4) FA(3, 4) FA(4, 4) FA(6, 4) FA(3, 8)
#undef FA
  return false;
}

extern "C" uint64_t ink2_fa_scratch(uint32_t heads, uint32_t max_keys) {
  return ((uint64_t) max_keys / kFaUnit + 2) * kFaTile * heads * (kFaHead + 2) * sizeof(float);
}

extern "C" bool ink2_fa(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                        uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, uint32_t pos0,
                        const float* rel_proj, float* out, float* scratch, void* stream) {
  return fa_launch<float>(a, q, rel, sink, gate, n_tok, k, v, n_keys, pos0, rel_proj, out, scratch, stream);
}

extern "C" bool ink2_fa16(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                          uint32_t n_tok, const uint16_t* k, const uint16_t* v, uint32_t n_keys, uint32_t pos0,
                          const float* rel_proj, float* out, float* scratch, void* stream) {
  return fa_launch<_Float16>(a, q, rel, sink, gate, n_tok, (const _Float16*) k, (const _Float16*) v, n_keys, pos0,
                             rel_proj, out, scratch, stream);
}

extern "C" bool ink2_fa8(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                         uint32_t n_tok, const uint8_t* k, const uint8_t* v, uint32_t n_keys, uint32_t pos0,
                         const float* rel_proj, float* out, float* scratch, void* stream) {
  return fa_launch<Kv8>(a, q, rel, sink, gate, n_tok, (const Kv8*) k, (const Kv8*) v, n_keys, pos0, rel_proj, out,
                        scratch, stream);
}

extern "C" bool ink2_fa85(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                          uint32_t n_tok, const uint8_t* k, const uint8_t* v, uint32_t n_keys, uint32_t pos0,
                          const float* rel_proj, float* out, float* scratch, void* stream) {
  return fa_launch<Kv85>(a, q, rel, sink, gate, n_tok, (const Kv85*) k, (const Kv85*) v, n_keys, pos0, rel_proj, out,
                         scratch, stream);
}

// a thread a block of a row: its amax, d = amax / 15, q = round(x / d) as x * (1 / d), stored q + 16 - its nibbles'
// 16 bytes, its fifth bits' 4 and its scale its own
__device__ inline void kv5_block(const float* row, uint32_t b, uint8_t* rec) {
  constexpr uint32_t H = INK2_MAX_HEAD;
  const float* xb = row + b * INK2_KV8_BLOCK;
  float amax = 0;
  for (int e = 0; e < INK2_KV8_BLOCK; e++) amax = fmaxf(amax, fabsf(xb[e]));
  const float d = amax / 15.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
  uint8_t nib[INK2_KV8_BLOCK / 2] = {}, hi[INK2_KV8_BLOCK / 8] = {};
  for (int e = 0; e < INK2_KV8_BLOCK; e++) {
    const int q = (int) roundf(xb[e] * id) + 16;
    nib[e / 2] |= (uint8_t) ((q & 15) << (e % 2 * 4));
    hi[e / 8] |= (uint8_t) ((q >> 4 & 1) << (e % 8));
  }
  for (int j = 0; j < INK2_KV8_BLOCK / 2; j++) rec[b * INK2_KV8_BLOCK / 2 + j] = nib[j];
  for (int j = 0; j < INK2_KV8_BLOCK / 8; j++) rec[H / 2 + b * INK2_KV8_BLOCK / 8 + j] = hi[j];
  *(_Float16*) (rec + H / 2 + H / 8 + 2 * b) = (_Float16) d;
}

__global__ void to_kv5_kernel(const float* x, uint64_t blocks, uint8_t* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= blocks) return;
  constexpr uint32_t per = INK2_MAX_HEAD / INK2_KV8_BLOCK;
  const uint64_t r = i / per;
  kv5_block(x + r * INK2_MAX_HEAD, (uint32_t) (i % per), y + r * INK2_KV5_RECORD);
}

extern "C" bool ink2_to_kv5(const float* x, uint64_t rows, uint8_t* y, void* stream) {
  if (!x || !y || !rows) return false;
  const uint64_t blocks = rows * (INK2_MAX_HEAD / INK2_KV8_BLOCK);
  to_kv5_kernel<<<grid(blocks), kThreads, 0, st(stream)>>>(x, blocks, y);
  return ok();
}

// a thread a block of a row: its amax, d = amax / 127, q = round(x / d) as x * (1 / d); d's f16 after the row's int8s
__device__ inline void kv8_block(const float* row, uint32_t b, uint8_t* rec) {
  const float d = q8_0_block<INK2_KV8_BLOCK>(row + b * INK2_KV8_BLOCK, 1, rec + b * INK2_KV8_BLOCK);
  *(_Float16*) (rec + INK2_MAX_HEAD + 2 * b) = (_Float16) d;
}

__global__ void to_kv8_kernel(const float* x, uint64_t blocks, uint8_t* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= blocks) return;
  constexpr uint32_t per = INK2_MAX_HEAD / INK2_KV8_BLOCK;
  const uint64_t r = i / per;
  kv8_block(x + r * INK2_MAX_HEAD, (uint32_t) (i % per), y + r * INK2_KV8_RECORD);
}

extern "C" bool ink2_to_kv8(const float* x, uint64_t rows, uint8_t* y, void* stream) {
  if (!x || !y || !rows) return false;
  const uint64_t blocks = rows * (INK2_MAX_HEAD / INK2_KV8_BLOCK);
  to_kv8_kernel<<<grid(blocks), kThreads, 0, st(stream)>>>(x, blocks, y);
  return ok();
}

__global__ void to_f16_kernel(const float* x, uint64_t n, _Float16* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] = (_Float16) x[i];
}

extern "C" bool ink2_to_f16(const float* x, uint64_t n, uint16_t* y, void* stream) {
  if (!x || !y || !n) return false;
  to_f16_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, (_Float16*) y);
  return ok();
}

// a block a row of a token's (q's heads, rel's, then k's and v's KV heads): q and rel clipped; a key row its conv,
// clip and store, a value row its conv, norm (rms_kernel's: its threads, its block_sum) and store - in LDS between
constexpr uint32_t kPrepBlocks = INK2_MAX_HEAD / INK2_KV8_BLOCK;  // a record's scale blocks: a thread each
__global__ void attn_prep_kernel(Ink2Prep p) {
  __shared__ float lds[kThreads], row[INK2_MAX_HEAD];
  const uint32_t H = p.heads, KV = p.kv_heads, D = INK2_MAX_HEAD, per = 2 * H + 2 * KV;
  const uint32_t t = blockIdx.x / per, r = blockIdx.x % per;
  if (r < H) return clip_row(p.q + ((uint64_t) t * H + r) * D, t, r, D, p.q_w, p.q_clip, p.tau, lds);
  if (r < 2 * H) {
    const uint32_t j = r - H;
    return clip_row(p.rel + ((uint64_t) t * H + j) * p.d_rel, t, j, p.d_rel, p.rel_w, p.rel_clip, p.tau, lds);
  }
  const bool key = r < 2 * H + KV;
  const uint32_t kvh = r - 2 * H - (key ? 0 : KV), ch = KV * D;
  const Ink2PrepKv& s = key ? p.k : p.v;
  for (uint32_t d = threadIdx.x; d < D; d += kThreads) {
    const uint32_t c = kvh * D + d;
    if (s.keep) s.keep[(uint64_t) t * ch + c] = s.x[(uint64_t) t * ch + c];
    row[d] = s.dyn ? conv_dyn_at(s.x, t, c, ch, p.taps, KV, s.w, s.dyn, s.hist, p.conv_res, p.conv_scale)
                   : conv_at(s.x, t, c, ch, p.taps, s.w, s.hist, p.conv_res, p.conv_scale);
  }
  if (key) {
    clip_row(row, t, kvh, D, nullptr, p.k_clip, Ink2Tau{0, 0, 0}, lds);
  } else {
    float ss = 0;
    for (uint32_t d = threadIdx.x; d < D; d += kThreads) ss += row[d] * row[d];
    const float rr = 1.0f / sqrtf(block_sum<kThreads>(ss, lds) / (float) D + p.eps);
    for (uint32_t d = threadIdx.x; d < D; d += kThreads) row[d] = row[d] * rr;
  }
  __syncthreads();
  const uint64_t at = (uint64_t) t * KV + kvh;
  for (uint32_t d = threadIdx.x; d < D; d += kThreads) s.y[at * D + d] = row[d];
  if (!p.records)
    for (uint32_t d = threadIdx.x; d < D; d += kThreads) ((_Float16*) s.store)[at * D + d] = (_Float16) row[d];
  else if (threadIdx.x < kPrepBlocks && (key || p.v8))
    kv8_block(row, threadIdx.x, (uint8_t*) s.store + at * INK2_KV8_RECORD);
  else if (threadIdx.x < kPrepBlocks)
    kv5_block(row, threadIdx.x, (uint8_t*) s.store + at * INK2_KV5_RECORD);
}

extern "C" bool ink2_attn_prep(const Ink2Prep* p, void* stream) {
  if (!p || !p->n_tok || !p->q || !p->rel || !p->heads || !p->kv_heads || p->head_dim != INK2_MAX_HEAD ||
      !p->d_rel || p->taps < 2 || p->taps > INK2_MAX_TAPS || (p->tau.alpha > 0 && !(p->tau.floor > 0)))
    return false;
  for (const Ink2PrepKv* s : {&p->k, &p->v})
    if (!s->x || !s->w || !s->y || !s->store) return false;
  attn_prep_kernel<<<p->n_tok * (2 * p->heads + 2 * p->kv_heads), kThreads, 0, st(stream)>>>(*p);
  return ok();
}

extern "C" bool ink2_clip_rows(float* x, uint32_t n_tok, uint32_t per_tok, uint32_t dim, const float* w, float clip,
                               Ink2Tau tau, void* stream) {
  if (!x || !n_tok || !per_tok || !dim || (tau.alpha > 0 && !(tau.floor > 0))) return false;
  clip_rows_kernel<<<n_tok * per_tok, kThreads, 0, st(stream)>>>(x, per_tok, dim, w, clip, tau);
  return ok();
}

extern "C" bool ink2_attn(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                          uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, const float* rel_proj,
                          float* out, void* stream) {
  if (!a || !q || !rel || !sink || !k || !v || !rel_proj || !out || !n_tok || n_tok > n_keys ||
      a->head_dim != INK2_MAX_HEAD || !a->kv_heads || a->heads % a->kv_heads || !a->d_rel ||
      a->d_rel > INK2_MAX_REL || !a->extent || a->extent > INK2_MAX_EXTENT)
    return false;
  attn_kernel<<<n_tok * a->heads, a->head_dim, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys, rel_proj,
                                                                out);
  return ok();
}

extern "C" bool ink2_moe_prep(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float route_scale,
                              uint32_t n_tok, int32_t* idx, float* w, uint32_t dim, uint32_t ff,
                              const Q4xExpertTiles* t, const float* scale, float* sg, float* su, float* a,
                              uint32_t* picks, void* stream) {
  const uint32_t P = n_tok * k;
  if (!logits || !bias || !n_exp || n_exp > OPS_MAX_EXPERTS || !k || k > OPS_MAX_TOPK || k > n_exp || !n_tok ||
      P > moe_tiles::kPlaceInline || !idx || !w || !dim || !ff || !t || !t->cnt || !t->gidx || !t->hidx || !t->xg ||
      !t->yg || !t->xd || !t->yd || !t->where || !t->off || !scale || !sg || !su || !a || !picks)
    return false;
  for (int j = 0; j < Q4X_GROUP_TENSORS; j++)
    if ((!t->tab[j] && (!t->base[j] || !t->bytes[j])) || !t->ptr[j]) return false;
  Q4xExpertBounds bd;
  q4x_expert_bounds(P, n_exp, &bd);
  moe_prep_kernel<<<1, moe_tiles::kGroupThreads, 0, st(stream)>>>(logits, bias, n_exp, k, route_scale, n_tok, idx, w,
                                                                  dim, ff, *t, bd, scale, sg, su, a, picks);
  return ok();
}

extern "C" bool ink2_moe_scales(const int32_t* ids, const float* w, const int32_t* where, const float* scale,
                                uint32_t n_exp, uint32_t n, float* sg, float* su, float* a, uint32_t* picks, void* stream) {
  if (!ids || !w || !where || !scale || !n_exp || !n || !sg || !su || !a || !picks) return false;
  moe_scales_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, st(stream)>>>(ids, w, where, scale, n_exp, n, sg, su,
                                                                                a, picks);
  return ok();
}

extern "C" bool ink2_loop_ctrl(const float* x, const float* m, const float* gain, const float* forget, uint64_t n,
                               float* out, void* stream) {
  if (!x || !m || !gain || !forget || !out || !n) return false;
  loop_ctrl_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, m, gain, forget, n, out);
  return ok();
}

extern "C" bool ink2_loop_ctrl_many(const Ink2LoopCtrlPart* parts, uint32_t count, uint64_t n, void* stream) {
  if (!parts || !count || count > INK2_LOOP_CTRL_PARTS || !n) return false;
  LoopCtrlParts ps = {};
  for (uint32_t c = 0; c < count; c++) {
    const Ink2LoopCtrlPart& p = parts[c];
    if (!p.x || !p.m || !p.gain || !p.forget || !p.out) return false;
    ps.p[c] = p;
  }
  loop_ctrl_many_kernel<<<dim3(grid(n), count), kThreads, 0, st(stream)>>>(ps, n);
  return ok();
}

extern "C" bool ink2_softcap(float* x, uint64_t n, float cap, void* stream) {
  if (!x || !n || !(cap > 0)) return false;
  softcap_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, cap);
  return ok();
}

extern "C" bool ink2_swiglu(const float* g, const float* u, uint32_t n, float gcap, float ucap, uint32_t n_tok,
                            float* h, void* stream) {
  if (!g || !u || !h || !n || !n_tok || !(gcap > 0) || !(ucap > 0)) return false;
  const uint64_t all = (uint64_t) n * n_tok;
  swiglu_kernel<<<grid(all), kThreads, 0, st(stream)>>>(g, u, all, gcap, ucap, h);
  return ok();
}

extern "C" bool ink2_swiglu_rows(const float* g, const float* u, const float* sg, const float* su, uint32_t n,
                                 float gcap, float ucap, uint32_t rows, float* h, void* stream) {
  if (!g || !u || !sg || !su || !h || !n || !rows || !(gcap > 0) || !(ucap > 0)) return false;
  const uint64_t all = (uint64_t) n * rows;
  swiglu_rows_kernel<<<grid(all), kThreads, 0, st(stream)>>>(g, u, sg, su, n, all, gcap, ucap, h);
  return ok();
}

extern "C" bool ink2_swiglu_x16(const float* g, const float* u, const float* sg, const float* su, uint32_t n,
                                float gcap, float ucap, const int32_t* idx, uint32_t slots, void* x16,
                                int32_t* overflow, void* stream) {
  if (!g || !u || !sg || !su || !idx || !x16 || !overflow || !n || n % mm::kTile || !slots || !(gcap > 0) ||
      !(ucap > 0))
    return false;
  const uint64_t runs = (uint64_t) (slots + mm::kTile - 1) / mm::kTile * mm::kTile * n / kX16Run;
  swiglu_x16_kernel<<<grid(runs), kThreads, 0, st(stream)>>>(g, u, sg, su, n, gcap, ucap, idx, slots,
                                                             (_Float16*) x16, overflow);
  return ok();
}

extern "C" bool ink2_moe_sum(float* y, const float* x, const float* a, const int32_t* row, uint32_t k, uint32_t n_tok,
                             uint32_t dim, void* stream) {
  if (!y || !x || !a || !row || !k || !n_tok || !dim) return false;
  const uint64_t all = (uint64_t) n_tok * dim;
  moe_sum_kernel<<<grid(all), kThreads, 0, st(stream)>>>(y, x, a, row, k, all, dim);
  return ok();
}

extern "C" bool ink2_gram(const float* x, uint32_t n_tok, uint32_t dim, float* h, void* stream) {
  if (!x || !h || !n_tok || !dim) return false;
  const dim3 g((dim + kGram - 1) / kGram, (dim + kGram - 1) / kGram), b(kGram, kGram);
  gram_kernel<<<g, b, 0, st(stream)>>>(x, n_tok, dim, h);
  return ok();
}

extern "C" bool ink2_conv_keep(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                               const float* hist, float res, float scale, const float* add, float* y, float* keep,
                               void* stream) {
  if (!x || !w || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS) return false;
  conv_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, w, hist, res, scale, add,
                                                                         y, keep);
  return ok();
}

extern "C" bool ink2_moe_conv(const float* x, const float* a, const int32_t* row, uint32_t k, uint32_t n_tok,
                              uint32_t ch, uint32_t taps, const float* w, const float* hist, float res, float scale,
                              const float* add, float* y, float* keep, void* stream) {
  if (!x || !a || !row || !k || !w || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS) return false;
  moe_conv_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, a, row, k, n_tok, ch, taps, w, hist,
                                                                             res, scale, add, y, keep);
  return ok();
}

extern "C" bool ink2_conv(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                          const float* hist, float res, float scale, float* y, void* stream) {
  return ink2_conv_keep(x, n_tok, ch, taps, w, hist, res, scale, nullptr, y, nullptr, stream);
}

extern "C" bool ink2_conv_dyn_keep(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                                   const float* w, const float* dyn, const float* hist, float res, float scale,
                                   float* y, float* keep, void* stream) {
  if (!x || !w || !dyn || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS || !heads || ch % heads)
    return false;
  conv_dyn_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, heads, w, dyn, hist,
                                                                             res, scale, y, keep);
  return ok();
}

extern "C" bool ink2_conv_dyn(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                              const float* w, const float* dyn, const float* hist, float res, float scale, float* y,
                              void* stream) {
  return ink2_conv_dyn_keep(x, n_tok, ch, taps, heads, w, dyn, hist, res, scale, y, nullptr, stream);
}

extern "C" bool ink2_attn_res(const float* query, uint32_t queries, const float* blocks, uint32_t n_blocks,
                              uint32_t b_stride, const float* partial, uint32_t n_tok, uint32_t dim, float eps,
                              float scale, float* out, float* scratch, void* stream) {
  if (!query || !partial || !out || (n_blocks && !blocks) || !queries || queries > INK2_MAX_QUERIES ||
      n_blocks + 1 > INK2_MAX_RES || n_blocks > b_stride || !n_tok || !dim)
    return false;
  if (scratch && n_tok <= INK2_RES_SPLIT_TOKENS) {
    attn_res_dots_kernel<<<dim3(n_blocks + 1, 1 + queries, n_tok), kThreads, 0, st(stream)>>>(
        query, queries, blocks, n_blocks, b_stride, partial, dim, scratch);
    attn_res_mix_kernel<<<dim3((dim + kThreads - 1) / kThreads, n_tok), kThreads, 0, st(stream)>>>(
        queries, blocks, n_blocks, b_stride, partial, dim, eps, scale, n_tok, scratch, out);
    return ok();
  }
  attn_res_kernel<<<n_tok, kThreads, 0, st(stream)>>>(query, queries, blocks, n_blocks, b_stride, partial, dim, eps,
                                                      scale, n_tok, out);
  return ok();
}

__global__ void ink2_scale_kernel(float* x, uint64_t n, float s) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] *= s;
}

__global__ void ink2_axpy_kernel(float* y, const float* x, uint64_t n, float a) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

extern "C" bool ink2_scale(float* x, uint64_t n, float s, void* stream) {
  if (!x || !n) return false;
  ink2_scale_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, s);
  return ok();
}

extern "C" bool ink2_axpy(float* y, const float* x, uint64_t n, float a, void* stream) {
  if (!y || !x || !n) return false;
  ink2_axpy_kernel<<<grid(n), kThreads, 0, st(stream)>>>(y, x, n, a);
  return ok();
}
