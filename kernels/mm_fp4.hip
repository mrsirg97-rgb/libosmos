// NVFP4 weights times NVFP4 activations on the 5090's FP4 matrix cores (mm_fp4: W4A4, a prompt's matmul). the
// activations quantized as ModelOpt's (an E4M3 scale a 16 under the matrix's calibrated input scale, E2M1 to nearest,
// ties to even), packed in k order (a byte two values, the low nibble first); the weights ggml 40's blocks as the
// GGUF holds them (quant.h: 4 UE4M3 scales, then sub-block s's value j at byte 4 + 8s + j's low nibble, j + 8 its
// high). mma.sync kind::mxf4nvf4 block_scale scale_vec::4X m16n8k64 (E2M1 x E2M1, a UE4M3 scale a 16 on each side),
// its fragments measured on the device (a probe a nibble): A's register r element i row g + 8 (r % 2), k 32 (r / 2) +
// 8q + i; B's register r element i column g, k 32 r + 8q + i; A's scales lane q 0 row g, q 1 row g + 8, B's lane q 0
// column g, byte b block b. the tensors' scales (the weight's and the input's) on the sums. sm_120a only: elsewhere
// the calls refuse (false) and the caller keeps W4A16
#include <hip/hip_runtime.h>

#include <cstdint>

#include "launch.h"
#include "osmos/gpu/mm.h"
#include "round.h"

namespace {

constexpr int kBlockBytes = 36;

// a 16's E4M3 scale (amax / 6 / gs, to nearest, saturating at 448), as nvfp4.py's act_quant
__device__ inline uint32_t e4m3_bits(float v) {
  const float a = fminf(v, 448.0f);
  if (!(a > 0.0f)) return 0;
  int e;
  const float m = frexpf(a, &e);  // a = m 2^e, m in [0.5, 1)
  // normal E4M3: 1.xxx 2^(E - 7), E 1 .. 15; subnormal 0.xxx 2^-6
  int E = e - 1 + 7;
  if (E >= 1) {
    float f = m * 2.0f;                             // 1.xxx
    float q = rintf((f - 1.0f) * 8.0f);             // 3 mantissa bits, ties to even (rintf: the current mode, nearest)
    if (q >= 8.0f) q = 0.0f, E++;
    if (E > 15 || (E == 15 && q > 6.0f)) return 0x7E;  // 448, the largest finite
    return (uint32_t) (E << 3) | (uint32_t) q;
  }
  const float q = rintf(a / 0.001953125f);            // subnormal: m 2^-9
  return q >= 8.0f ? 0x08u : (uint32_t) q;
}

// E2M1's code of x (|x| <= 6): nearest of 0, .5, 1, 1.5, 2, 3, 4, 6, ties to the even index (nvfp4.py's e2m1_idx)
__device__ inline uint32_t e2m1_code(float x) {
  const float a = fabsf(x);
  uint32_t i;
  if (a <= 0.25f) i = 0;
  else if (a < 0.75f) i = 1;
  else if (a <= 1.25f) i = 2;
  else if (a < 1.75f) i = 3;
  else if (a <= 2.5f) i = 4;
  else if (a < 3.5f) i = 5;
  else if (a <= 5.0f) i = 6;
  else i = 7;
  return i ? (x < 0.0f ? 8u : 0u) | i : 0u;
}

__device__ inline float load(const float* x, uint64_t i) { return x[i]; }
__device__ inline float load(const uint16_t* x, uint64_t i) { return __uint_as_float((uint32_t) x[i] << 16); }

// a uniform in [0, 1) of (seed, i): murmur3's finalizer
__device__ inline float uniform01(uint32_t seed, uint64_t i) {
  uint32_t h = (uint32_t) i * 0x9E3779B1u ^ (uint32_t) (i >> 32) * 0x85EBCA77u ^ seed;
  h ^= h >> 16, h *= 0x85EBCA6Bu, h ^= h >> 13, h *= 0xC2B2AE35u, h ^= h >> 16;
  return (float) (h >> 8) * (1.0f / 16777216.0f);
}

// E2M1's code of x (|x| <= 6) rounded stochastically: up to the grid point above with probability its distance from
// the one below over the step (an unbiased rounding: a gradient's, NVIDIA's NVFP4 training recipe)
__device__ inline uint32_t e2m1_code_sr(float x, float u) {
  constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float a = fabsf(x);
  uint32_t i = 0;
  while (i < 7 && a >= kGrid[i + 1]) i++;
  if (i < 7 && u < (a - kGrid[i]) / (kGrid[i + 1] - kGrid[i])) i++;
  return i ? (x < 0.0f ? 8u : 0u) | i : 0u;
}

// a block a token, a thread a 16: its scale, its codes packed two a byte (sr: stochastic, seeded)
template <typename T>
__global__ void act_kernel(const T* __restrict__ x, uint32_t xs, uint32_t cols, float gs_host,
                           const float* __restrict__ gs_dev, uint32_t sr, uint8_t* __restrict__ q,
                           uint8_t* __restrict__ sc) {
  const uint32_t t = blockIdx.x;
  const float gs = gs_dev ? *gs_dev : gs_host;
  // the stream salted with the scale when the device made it (a gradient's amax): each call its own stream, a
  // function of its values, not of the order the calls ran in (a pipelined backward's two threads); never 0, which
  // means nearest
  sr = sr && gs_dev ? (sr ^ __float_as_uint(gs) * 0x9E3779B1u) | 1u : sr;
  for (uint32_t b = threadIdx.x; b < cols / 16; b += blockDim.x) {
    const uint64_t at = (uint64_t) t * xs + b * 16;
    float v[16], amax = 0.0f;
    for (int j = 0; j < 16; j++) v[j] = load(x, at + j), amax = fmaxf(amax, fabsf(v[j]));
    const uint32_t s = e4m3_bits(amax / 6.0f / gs);
    const float sf = rnd::e4m3_value(s) * gs;
    sc[(uint64_t) t * (cols / 16) + b] = (uint8_t) s;
    uint8_t* o = q + (uint64_t) t * (cols / 2) + b * 8;
    for (int j = 0; j < 16; j += 2) {
      const float a = sf > 0.0f ? fminf(fmaxf(v[j] / sf, -6.0f), 6.0f) : 0.0f;
      const float c = sf > 0.0f ? fminf(fmaxf(v[j + 1] / sf, -6.0f), 6.0f) : 0.0f;
      const uint64_t k = (uint64_t) t * cols + b * 16 + j;
      o[j / 2] = (uint8_t) (sr ? e2m1_code_sr(a, uniform01(sr, k)) | e2m1_code_sr(c, uniform01(sr, k + 1)) << 4
                               : e2m1_code(a) | e2m1_code(c) << 4);
    }
  }
}

// a tensor's scale from its amax (amax / 6 / 448: its largest 16 at E4M3's top), on the device
__global__ void amax_kernel(const uint16_t* __restrict__ x, uint64_t n, uint32_t* __restrict__ bits) {
  float m = 0.0f;
  for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t) gridDim.x * blockDim.x)
    m = fmaxf(m, fabsf(load(x, i)));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if (threadIdx.x % 32 == 0) atomicMax(bits, __float_as_uint(m));  // (non-negative floats order as their bits)
}
__global__ void gs_kernel(const uint32_t* __restrict__ bits, float* __restrict__ gs) {
  const float m = __uint_as_float(*bits);
  *gs = m > 0.0f ? m / (6.0f * 448.0f) : 1.0f;
}

// 4 bytes' low (hi false) or high nibbles, in order, into 16 bits
__device__ inline uint32_t nibbles(uint32_t w, bool hi) {
  uint32_t x = (hi ? w >> 4 : w) & 0x0F0F0F0Fu;
  x = (x | x >> 4) & 0x00FF00FFu;
  return (x | x >> 8) & 0xFFFFu;
}

// a block 128 tokens x 128 rows of W, a K block (64) at a time through LDS; 8 warps 4 x 2, a warp 32 tokens (2 m
// tiles) x 64 rows (8 n tiles): 16 mma a K block. a staged row (a token's or W's) 9 words: its 64 values in k order (a
// byte two, the low nibble first: W's repacked from ggml's sub-block order as staged), then the K block's 4 scales -
// a fragment's loads 9 words apart, every bank once
constexpr int kBM = 128, kBN = 128, kRow = 9, kTileThreads = 256;

// W's layouts: ggml 40's blocks (kGgml: the GGUF's); ModelOpt's (kModelOpt: packed [rows, cols/2] in k order, its E4M3
// scales [rows, cols/16] at ws: the training engine's train/fp4.py); a MoE's routed experts at 2 bits (kTwoBit:
// mm_2b_grouped's - codes [E, rows, cols/4], 4 a byte in k order, the lowest bits first, each one of the 4 values of
// an E2M1-subset grid, expanded to its E2M1 nibble as staged; scales [E, rows, cols/16]; the tokens sorted by expert)
enum WLayout { kGgml, kModelOpt, kTwoBit };

// a MoE's grouping: E experts, expert e's tokens offsets[e] .. offsets[e + 1] (sorted by expert), the tile list's
// starts (tile_start[e]: the token tiles of the experts before e; tile_start[E] all of them), a tensor scale an
// expert (gsw), the grid (lut: code c's E2M1 nibble at bits 4c)
struct Experts {
  const uint32_t* offsets;
  const uint32_t* tile_start;
  const float* gsw;
  uint32_t e, lut;
};

template <int W>
__global__ __launch_bounds__(kTileThreads) void mm_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                                          uint32_t rows, uint32_t cols,
                                                          const uint8_t* __restrict__ aq,
                                                          const uint8_t* __restrict__ as, uint32_t n, float scale_host,
                                                          const float* __restrict__ scale_dev, bool bf16,
                                                          void* __restrict__ y, uint32_t ys, Experts ex) {
  float scale = scale_dev ? scale_host * *scale_dev : scale_host;
  __shared__ uint32_t sa[kBM * kRow], sb[kBN * kRow];
  const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32, g = lane / 4, q = lane % 4;
  const uint32_t wm = warp % 4, wn = warp / 4;  // the warp's 32 tokens, its 64 rows
  const uint32_t r0 = blockIdx.x * kBN, nb = cols / 64;
  uint32_t t0 = blockIdx.y * kBM;
  if (W == kTwoBit) {   // this block's expert: the last whose tiles start at or before it (a tile past them all: none)
    if (blockIdx.y >= ex.tile_start[ex.e]) return;
    uint32_t lo = 0, hi = ex.e;
    while (hi - lo > 1) {
      const uint32_t mid = (lo + hi) / 2;
      (ex.tile_start[mid] <= blockIdx.y ? lo : hi) = mid;
    }
    t0 = ex.offsets[lo] + (blockIdx.y - ex.tile_start[lo]) * kBM;
    n = ex.offsets[lo + 1];   // (the expert's end: a token at or past it is another expert's)
    w += (uint64_t) lo * rows * (cols / 4), ws += (uint64_t) lo * rows * (cols / 16), scale *= ex.gsw[lo];
  }
  const uint64_t row_bytes = (uint64_t) nb * kBlockBytes;
  float d[2][8][4] = {};
  for (uint32_t kb = 0; kb < nb; kb++) {
    // stage A: a token's 32 bytes of the K block (8 words) and its 4 scales; B: a row's 4 sub-blocks repacked
    for (uint32_t i = tid; i < kBM * 8; i += kTileThreads) {
      const uint32_t r = i / 8, c = i % 8, t = t0 + r < n ? t0 + r : n - 1;
      sa[r * kRow + c] = *(const uint32_t*) (aq + (uint64_t) t * (cols / 2) + (uint64_t) kb * 32 + 4 * c);
    }
    for (uint32_t r = tid; r < kBM; r += kTileThreads) {
      const uint32_t t = t0 + r < n ? t0 + r : n - 1;
      sa[r * kRow + 8] = *(const uint32_t*) (as + (uint64_t) t * (cols / 16) + (uint64_t) kb * 4);
    }
    for (uint32_t i = tid; W == kModelOpt && i < kBN * 8; i += kTileThreads) {
      const uint32_t r = i / 8, c = i % 8, row = r0 + r < rows ? r0 + r : rows - 1;
      sb[r * kRow + c] = *(const uint32_t*) (w + (uint64_t) row * (cols / 2) + (uint64_t) kb * 32 + 4 * c);
      if (c == 0) sb[r * kRow + 8] = *(const uint32_t*) (ws + (uint64_t) row * (cols / 16) + (uint64_t) kb * 4);
    }
    for (uint32_t i = tid; W == kTwoBit && i < kBN * 4; i += kTileThreads) {   // a word of 16 codes: 2 of nibbles
      const uint32_t r = i / 4, c = i % 4, row = r0 + r < rows ? r0 + r : rows - 1;
      const uint32_t cw = *(const uint32_t*) (w + (uint64_t) row * (cols / 4) + (uint64_t) kb * 16 + 4 * c);
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        lo |= (ex.lut >> (4 * ((cw >> (2 * j)) & 3u)) & 0xFu) << (4 * j);
        hi |= (ex.lut >> (4 * ((cw >> (2 * j + 16)) & 3u)) & 0xFu) << (4 * j);
      }
      sb[r * kRow + 2 * c] = lo, sb[r * kRow + 2 * c + 1] = hi;
      if (c == 0) sb[r * kRow + 8] = *(const uint32_t*) (ws + (uint64_t) row * (cols / 16) + (uint64_t) kb * 4);
    }
    for (uint32_t i = tid; W == kGgml && i < kBN * 4; i += kTileThreads) {
      const uint32_t r = i / 4, sub = i % 4, row = r0 + r < rows ? r0 + r : rows - 1;
      const uint8_t* blk = w + (uint64_t) row * row_bytes + (uint64_t) kb * kBlockBytes;
      const uint32_t* p = (const uint32_t*) (blk + 4 + 8 * sub);
      const uint32_t w0 = p[0], w1 = p[1];
      sb[r * kRow + 2 * sub] = nibbles(w0, false) | nibbles(w1, false) << 16;     // values 16 sub + 0 .. 7
      sb[r * kRow + 2 * sub + 1] = nibbles(w0, true) | nibbles(w1, true) << 16;   // 16 sub + 8 .. 15
      if (sub == 0) sb[r * kRow + 8] = *(const uint32_t*) blk;
    }
    __syncthreads();
    uint32_t a[2][4], scale_a[2];
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      const uint32_t ra = wm * 32 + mt * 16 + g;  // rows g and g + 8 of the m tile
      // register r: row ra + 8 (r % 2), words 4 (r / 2) + q (k 32 (r / 2) + 8q ..)
      a[mt][0] = sa[ra * kRow + q], a[mt][1] = sa[(ra + 8) * kRow + q];
      a[mt][2] = sa[ra * kRow + 4 + q], a[mt][3] = sa[(ra + 8) * kRow + 4 + q];
      scale_a[mt] = q == 0 ? sa[ra * kRow + 8] : q == 1 ? sa[(ra + 8) * kRow + 8] : 0u;
    }
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      const uint32_t rb = wn * 64 + nt * 8 + g;
      const uint32_t b0 = sb[rb * kRow + q], b1 = sb[rb * kRow + 4 + q];  // k 8q .., 32 + 8q ..
      const uint32_t scale_b = q == 0 ? sb[rb * kRow + 8] : 0u;
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
#if defined(__CUDACC__)
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
            : "+f"(d[mt][nt][0]), "+f"(d[mt][nt][1]), "+f"(d[mt][nt][2]), "+f"(d[mt][nt][3])
            : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b0), "r"(b1), "r"(scale_a[mt]),
              "r"(scale_b));
#else
        (void) b0, (void) b1, (void) scale_b;  // the FP4 MMA is sm_120a's: on HIP mm_fp4_ok() refuses before a launch
#endif
      }
    }
    __syncthreads();
  }
  // D: element e token (m tile's) g + 8 (e / 2), row (n tile's) 2q + e % 2
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const uint32_t t = t0 + wm * 32 + mt * 16 + g + 8 * (e / 2), row = r0 + wn * 64 + nt * 8 + 2 * q + e % 2;
        if (t < n && row < rows) {
          const float v = d[mt][nt][e] * scale;
          if (bf16) {
            const uint32_t u = __float_as_uint(v);  // to nearest, ties to even (quiet NaN preserved)
            ((uint16_t*) y)[(uint64_t) t * ys + row] =
                (u & 0x7FFFFFFFu) > 0x7F800000u ? (uint16_t) ((u >> 16) | 0x0040u)
                                                : (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
          } else {
            ((float*) y)[(uint64_t) t * ys + row] = v;
          }
        }
      }
}

// the experts' token tiles, one thread: tile_start[e] the tiles before expert e, tile_start[E] all (kBM tokens a tile)
__global__ void tiles_kernel(const uint32_t* __restrict__ offsets, uint32_t e, uint32_t* __restrict__ tile_start) {
  uint32_t s = 0;
  for (uint32_t i = 0; i < e; i++) {
    tile_start[i] = s;
    s += (offsets[i + 1] - offsets[i] + kBM - 1) / kBM;
  }
  tile_start[e] = s;
}

// W2A16: a MoE's routed experts at 2 bits (mm_2b_grouped's codes, scales and grouping) times BF16 activations on the
// BF16 matrix cores (mma m16n8k16, F32 sums). each code decoded as staged to its grid value times its block's UE4M3
// scale - exact in BF16 (a grid value's 2 significant bits times E4M3's 4) - the expert's tensor scale on the sums.
// TRANS false: y[t, r] = sum_k x[t, k] W[r, k] (the forward, x [n, cols]); TRANS true: y[t, k] = sum_r x[t, r] W[r, k]
// (the input's gradient, x [n, rows]): W's blocks run along k, so the transposed product decodes each element under
// its own block's scale, which BF16 can carry and a block-scaled FP4 operand cannot. a block 128 tokens x 128 outputs,
// a K step 64 through LDS (rows 72 BF16 apart: a fragment's loads on distinct banks); 8 warps 4 x 2, a warp 32 x 64
constexpr int kBK = 64, kLd = kBK + 8;

__device__ inline uint16_t bf16_rn(float v) {
  const uint32_t u = __float_as_uint(v);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t) ((u >> 16) | 0x0040u);
  return (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// E2M1's value of a nibble (sign bit 8): 0, .5, 1, 1.5, 2, 3, 4, 6 (no indexed array: clang spills those)
__device__ inline float e2m1_value(uint32_t nib) {
  const uint32_t m = nib & 7u, e = m >> 1;
  const float v = e ? (1.0f + 0.5f * (float) (m & 1u)) * (float) (1u << (e - 1)) : 0.5f * (float) m;
  return nib & 8u ? -v : v;
}

template <bool TRANS>
__global__ __launch_bounds__(kTileThreads) void w2a16_kernel(const uint8_t* __restrict__ codes,
                                                             const uint8_t* __restrict__ ws, uint32_t rows,
                                                             uint32_t cols, const uint16_t* __restrict__ x,
                                                             uint32_t xs, bool bf16, void* __restrict__ y,
                                                             uint32_t ys, Experts ex) {
  __shared__ __align__(16) uint16_t sa[kBM * kLd], sb[kBN * kLd];
  const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32, g = lane / 4, q = lane % 4;
  const uint32_t wm = warp % 4, wn = warp / 4;
  const uint32_t nout = TRANS ? cols : rows, kdim = TRANS ? rows : cols, n0 = blockIdx.x * kBN;
  if (blockIdx.y >= ex.tile_start[ex.e]) return;
  uint32_t lo = 0, hi = ex.e;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    (ex.tile_start[mid] <= blockIdx.y ? lo : hi) = mid;
  }
  const uint32_t t0 = ex.offsets[lo] + (blockIdx.y - ex.tile_start[lo]) * kBM, tend = ex.offsets[lo + 1];
  const uint8_t* wc = codes + (uint64_t) lo * rows * (cols / 4);
  const uint8_t* wsc = ws + (uint64_t) lo * rows * (cols / 16);
  const float gsw = ex.gsw[lo];
  float gv[4];
#pragma unroll
  for (int c = 0; c < 4; c++) gv[c] = e2m1_value(ex.lut >> (4 * c) & 0xFu);
  float d[2][8][4] = {};
  for (uint32_t kb = 0; kb < kdim / kBK; kb++) {
    for (uint32_t i = tid; i < kBM * (kBK / 8); i += kTileThreads) {   // A: 8 BF16 (16 bytes) a load
      const uint32_t r = i / (kBK / 8), c8 = i % (kBK / 8), t = t0 + r < tend ? t0 + r : tend - 1;
      *(uint4*) &sa[r * kLd + c8 * 8] = *(const uint4*) (x + (uint64_t) t * xs + (uint64_t) kb * kBK + c8 * 8);
    }
    if (!TRANS) {   // B[n][k]: W's row n0 + n, k kb*64 ..: a word of 16 codes and their block's scale an item
      for (uint32_t i = tid; i < kBN * (kBK / 16); i += kTileThreads) {
        const uint32_t n = i / (kBK / 16), c = i % (kBK / 16), row = n0 + n < rows ? n0 + n : rows - 1;
        const uint32_t cw = *(const uint32_t*) (wc + (uint64_t) row * (cols / 4) + (uint64_t) kb * (kBK / 4) + 4 * c);
        const float s = rnd::e4m3_value(wsc[(uint64_t) row * (cols / 16) + (uint64_t) kb * (kBK / 16) + c]);
#pragma unroll
        for (int j = 0; j < 16; j++) {
          const uint32_t code = cw >> (2 * j) & 3u;
          const float v = code == 0 ? gv[0] : code == 1 ? gv[1] : code == 2 ? gv[2] : gv[3];
          sb[n * kLd + 16 * c + j] = bf16_rn(v * s);
        }
      }
    } else {   // B[n][r]: W's column n0 + n, row kb*64 + r: a row's word of 16 codes (16 columns) an item
      for (uint32_t i = tid; i < kBK * (kBN / 16); i += kTileThreads) {
        const uint32_t r = i / (kBN / 16), c = i % (kBN / 16), row = kb * kBK + r;
        const uint32_t col0 = n0 + 16 * c < cols ? n0 + 16 * c : cols - 16;
        const uint32_t cw = *(const uint32_t*) (wc + (uint64_t) row * (cols / 4) + col0 / 4);
        const float s = rnd::e4m3_value(wsc[(uint64_t) row * (cols / 16) + col0 / 16]);
#pragma unroll
        for (int j = 0; j < 16; j++) {
          const uint32_t code = cw >> (2 * j) & 3u;
          const float v = code == 0 ? gv[0] : code == 1 ? gv[1] : code == 2 ? gv[2] : gv[3];
          sb[(16 * c + j) * kLd + r] = bf16_rn(v * s);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kBK / 16; kk++) {
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {   // A: rows g, g + 8 of the m tile; k 2q (regs 0, 1) and 2q + 8 (2, 3)
        const uint32_t ra = wm * 32 + mt * 16 + g, k = kk * 16 + 2 * q;
        a[mt][0] = *(const uint32_t*) &sa[ra * kLd + k];
        a[mt][1] = *(const uint32_t*) &sa[(ra + 8) * kLd + k];
        a[mt][2] = *(const uint32_t*) &sa[ra * kLd + k + 8];
        a[mt][3] = *(const uint32_t*) &sa[(ra + 8) * kLd + k + 8];
      }
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {   // B: column g of the n tile, k 2q and 2q + 8
        const uint32_t rb = wn * 64 + nt * 8 + g, k = kk * 16 + 2 * q;
        const uint32_t b0 = *(const uint32_t*) &sb[rb * kLd + k], b1 = *(const uint32_t*) &sb[rb * kLd + k + 8];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
#if defined(__CUDACC__)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
              "{%0,%1,%2,%3};"
              : "+f"(d[mt][nt][0]), "+f"(d[mt][nt][1]), "+f"(d[mt][nt][2]), "+f"(d[mt][nt][3])
              : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b0), "r"(b1));
#else
          (void) b0, (void) b1;  // CUDA's BF16 MMA: on HIP mm_fp4_ok() refuses before a launch
#endif
        }
      }
    }
    __syncthreads();
  }
  // D: element e row (token) g + 8 (e / 2) of the m tile, column 2q + e % 2 of the n tile
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const uint32_t t = t0 + wm * 32 + mt * 16 + g + 8 * (e / 2), o = n0 + wn * 64 + nt * 8 + 2 * q + e % 2;
        if (t < tend && o < nout) {
          const float v = d[mt][nt][e] * gsw;
          if (bf16)
            ((uint16_t*) y)[(uint64_t) t * ys + o] = bf16_rn(v);
          else
            ((float*) y)[(uint64_t) t * ys + o] = v;
        }
      }
}

}  // namespace

extern "C" bool mm_fp4_ok(void) {
#if defined(__CUDACC__)
  int dev = 0, major = 0, minor = 0;
  return hipGetDevice(&dev) == hipSuccess &&
         hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, dev) == hipSuccess &&
         hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, dev) == hipSuccess && major == 12 &&
         minor == 0;
#else
  return false;
#endif
}

extern "C" bool mm_fp4_quant(const void* x, uint32_t bf16, uint32_t xs, uint32_t n, uint32_t cols, float gs,
                             const float* gs_dev, uint32_t sr_seed, void* q, void* sc, void* stream) {
  if (!mm_fp4_ok() || !x || !q || !sc || !n || cols % 64 || (!gs_dev && !(gs > 0.0f))) return false;
  const hipStream_t st = launches::st(stream);
  if (bf16)
    act_kernel<uint16_t><<<n, 128, 0, st>>>((const uint16_t*) x, xs, cols, gs, gs_dev, sr_seed, (uint8_t*) q, (uint8_t*) sc);
  else
    act_kernel<float><<<n, 128, 0, st>>>((const float*) x, xs, cols, gs, gs_dev, sr_seed, (uint8_t*) q, (uint8_t*) sc);
  return launches::ok();
}

extern "C" bool mm_fp4_act(const float* x, uint32_t xs, uint32_t n, uint32_t cols, float gs, void* q, void* sc,
                           void* stream) {
  return mm_fp4_quant(x, 0, xs, n, cols, gs, nullptr, 0, q, sc, stream);
}

extern "C" bool mm_fp4_amax(const void* x_bf16, uint64_t n, uint32_t* bits, float* gs, void* stream) {
  if (!mm_fp4_ok() || !x_bf16 || !n || !bits || !gs) return false;
  const hipStream_t st = launches::st(stream);
  if (hipMemsetAsync(bits, 0, 4, st) != hipSuccess) return false;
  const uint64_t blocks = (n + 255) / 256;
  amax_kernel<<<blocks < 1024 ? blocks : 1024, 256, 0, st>>>((const uint16_t*) x_bf16, n, bits);
  gs_kernel<<<1, 1, 0, st>>>(bits, gs);
  return launches::ok();
}

extern "C" bool mm_fp4(const void* w, uint32_t rows, uint32_t cols, const void* q, const void* sc, uint32_t n,
                       float scale, float* y, uint32_t ys, void* stream) {
  if (!mm_fp4_ok() || !w || !q || !sc || !y || !n || cols % 64 || rows % 8) return false;
  const dim3 grid((rows + kBN - 1) / kBN, (n + kBM - 1) / kBM);
  mm_kernel<kGgml><<<grid, kTileThreads, 0, launches::st(stream)>>>((const uint8_t*) w, nullptr, rows, cols,
                                                                   (const uint8_t*) q, (const uint8_t*) sc, n, scale,
                                                                   nullptr, false, y, ys, Experts{});
  return launches::ok();
}

extern "C" bool mm_fp4_mo(const void* packed, const void* wscale, uint32_t rows, uint32_t cols, const void* q,
                          const void* sc, uint32_t n, float scale, const float* scale_dev, uint32_t bf16, void* y,
                          uint32_t ys, void* stream) {
  if (!mm_fp4_ok() || !packed || !wscale || !q || !sc || !y || !n || cols % 64 || rows % 8) return false;
  const dim3 grid((rows + kBN - 1) / kBN, (n + kBM - 1) / kBM);
  mm_kernel<kModelOpt><<<grid, kTileThreads, 0, launches::st(stream)>>>((const uint8_t*) packed, (const uint8_t*) wscale,
                                                                    rows, cols, (const uint8_t*) q, (const uint8_t*) sc,
                                                                    n, scale, scale_dev, bf16 != 0, y, ys, Experts{});
  return launches::ok();
}

extern "C" bool mm_2b_grouped(const void* codes, const void* wscale, const float* gsw, uint32_t e, uint32_t lut,
                              uint32_t rows, uint32_t cols, const uint32_t* offsets, uint32_t* tile_start,
                              const void* q, const void* sc, uint32_t n, float scale, const float* scale_dev,
                              uint32_t bf16, void* y, uint32_t ys, void* stream) {
  if (!mm_fp4_ok() || !codes || !wscale || !gsw || !e || !offsets || !tile_start || !q || !sc || !y || !n ||
      cols % 64 || rows % 8)
    return false;
  const hipStream_t st = launches::st(stream);
  tiles_kernel<<<1, 1, 0, st>>>(offsets, e, tile_start);
  // a grid of every tile there could be (each expert adds at most one partial tile): the ones past the list return
  const dim3 grid((rows + kBN - 1) / kBN, (n + kBM - 1) / kBM + e);
  mm_kernel<kTwoBit><<<grid, kTileThreads, 0, st>>>((const uint8_t*) codes, (const uint8_t*) wscale, rows, cols,
                                                    (const uint8_t*) q, (const uint8_t*) sc, n, scale, scale_dev,
                                                    bf16 != 0, y, ys, Experts{offsets, tile_start, gsw, e, lut});
  return launches::ok();
}

extern "C" bool mm_w2a16_grouped(const void* codes, const void* wscale, const float* gsw, uint32_t e, uint32_t lut,
                                 uint32_t rows, uint32_t cols, uint32_t trans, const uint32_t* offsets,
                                 uint32_t* tile_start, const void* x, uint32_t xs, uint32_t n, uint32_t bf16, void* y,
                                 uint32_t ys, void* stream) {
  const uint32_t kdim = trans ? rows : cols, nout = trans ? cols : rows;
  if (!mm_fp4_ok() || !codes || !wscale || !gsw || !e || !offsets || !tile_start || !x || !y || !n || kdim % kBK ||
      nout % 16 || cols % 16 || xs % 8)
    return false;
  const hipStream_t st = launches::st(stream);
  tiles_kernel<<<1, 1, 0, st>>>(offsets, e, tile_start);
  const dim3 grid((nout + kBN - 1) / kBN, (n + kBM - 1) / kBM + e);
  const Experts ex{offsets, tile_start, gsw, e, lut};
  if (trans)
    w2a16_kernel<true><<<grid, kTileThreads, 0, st>>>((const uint8_t*) codes, (const uint8_t*) wscale, rows, cols,
                                                      (const uint16_t*) x, xs, bf16 != 0, y, ys, ex);
  else
    w2a16_kernel<false><<<grid, kTileThreads, 0, st>>>((const uint8_t*) codes, (const uint8_t*) wscale, rows, cols,
                                                       (const uint16_t*) x, xs, bf16 != 0, y, ys, ex);
  return launches::ok();
}
