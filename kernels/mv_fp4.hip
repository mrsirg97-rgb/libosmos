// NVFP4's matvec for a few tokens (mv_fp4): ggml 40's 36-byte blocks (quant.h: 4 UE4M3 scales, a 16 each, then 32
// bytes of E2M1 - sub-block s's value j at byte 4 + 8s + j's low nibble, j + 8 its high one), R rows a warp. a warp's
// 32 blocks of a row (1,152 bytes, 16-byte aligned: a row whole blocks of 4) read as 72 uint4s into its LDS, then each
// lane its own block from there (a lane's 9 words: stride 9, every bank once) - a lane reading its block itself takes
// one 36-byte stride a load, a ninth of each line used, measured 290 GB/s on a 5090 against this kernel's. x read once
// a block for all R rows, the tensor's own scale on the sum. its sums are not mv_strided's order (a lane's
// blocks, then the warp's tree): its own bits, every token's alone (q35's decode: DESIGN.md, "a second GPU")
#include <hip/hip_runtime.h>

#include <cstdint>

#include "launch.h"
#include "port.h"
#include "round.h"

namespace {

constexpr int kBlock = 64, kBlockWords = 9, kWarps = 8;
// E2M1's values doubled (ggml's kvalues_mxfp4), the block scale halved: every product exact. looked up from LDS,
// not __constant__: a warp's 32 codes differ, and CUDA's constant cache serves one address at a time (the lookup
// serialized up to 32 ways); 16 floats in 16 banks, any mix of codes one read
constexpr float kHalfE2m1[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

template <int R, int NT>
__global__ __launch_bounds__(32 * kWarps) void fp4_kernel(const uint8_t* __restrict__ w, uint32_t rows, uint32_t cols,
                                                          const float* __restrict__ x, uint32_t xs, float* __restrict__ y,
                                                          uint32_t ys, float scale) {
  __shared__ float tab[16];
  if (threadIdx.x < 16) tab[threadIdx.x] = kHalfE2m1[threadIdx.x];
  __syncthreads();
  const uint32_t lane = threadIdx.x % 32, r0 = (blockIdx.x * kWarps + threadIdx.x / 32) * R;
  if (r0 >= rows) return;
  const uint32_t nb = cols / kBlock;
  const uint64_t row_bytes = (uint64_t) nb * kBlockWords * 4;
  float acc[R][NT] = {};
  // a chunk of 32 blocks a row: 72 uint4s, 3 a lane at most. the next chunk's loads issued into registers before this
  // chunk's sums (each lane's slots unrolled: constant indices, the registers' own), staged through LDS after them
  constexpr int kQuads = 32 * kBlockWords / 4, kPer = (kQuads + 31) / 32;
  __shared__ uint4 stage[kWarps][R][kQuads];
  uint4(*my)[kQuads] = stage[threadIdx.x / 32];
  uint4 nx[R][kPer];
#define FP4_FETCH(c0)                                                                                                  \
  {                                                                                                                    \
    const uint32_t quads = (nb - (c0) < 32 ? nb - (c0) : 32) * kBlockWords / 4;                                      \
    _Pragma("unroll") for (int r = 0; r < R; r++) {                                                                  \
      const uint4* p = (const uint4*) (w + (uint64_t) (r0 + r < rows ? r0 + r : r0) * row_bytes +                     \
                                       (uint64_t) (c0) * kBlockWords * 4);                                            \
      _Pragma("unroll") for (int k = 0; k < kPer; k++) if (lane + 32u * k < quads) nx[r][k] = p[lane + 32 * k];      \
    }                                                                                                                  \
  }
  FP4_FETCH(0u);
  for (uint32_t b0 = 0; b0 < nb; b0 += 32) {
    const uint32_t b = b0 + lane, quads = (nb - b0 < 32 ? nb - b0 : 32) * kBlockWords / 4;
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int k = 0; k < kPer; k++)
        if (lane + 32u * k < quads) my[r][lane + 32 * k] = nx[r][k];
    osmos_wave_barrier();
    if (b0 + 32 < nb) FP4_FETCH(b0 + 32);
    uint32_t q[R][kBlockWords];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const uint32_t* sw = (const uint32_t*) my[r] + (b < nb ? lane : 0) * kBlockWords;
#pragma unroll
      for (int k = 0; k < kBlockWords; k++) q[r][k] = sw[k];
    }
    osmos_wave_barrier();
    if (b >= nb) continue;
#pragma unroll
    for (int s = 0; s < 4; s++) {  // sub-block s: values 16 s .. 16 s + 15 of the block
      float xv[NT][16];
#pragma unroll
      for (int t = 0; t < NT; t++) {
        const float4* xp = (const float4*) (x + (uint64_t) t * xs + (uint64_t) b * kBlock + 16 * s);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          const float4 v = xp[k];
          xv[t][4 * k] = v.x, xv[t][4 * k + 1] = v.y, xv[t][4 * k + 2] = v.z, xv[t][4 * k + 3] = v.w;
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        const float d = rnd::ue4m3_value(q[r][0] >> (8 * s) & 0xFFu) * 0.5f;
        const uint32_t lo = q[r][1 + 2 * s], hi = q[r][2 + 2 * s];  // bytes 4 + 8s .. 4 + 8s + 7
        float part[NT] = {};
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t byte = (j < 4 ? lo >> (8 * j) : hi >> (8 * (j - 4))) & 0xFFu;
          const float a = tab[byte & 0xF], c = tab[byte >> 4];
#pragma unroll
          for (int t = 0; t < NT; t++) part[t] = __fmaf_rn(a, xv[t][j], __fmaf_rn(c, xv[t][j + 8], part[t]));
        }
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = __fmaf_rn(d, part[t], acc[r][t]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int t = 0; t < NT; t++) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
      if (lane == 0 && r0 + r < rows) y[(uint64_t) t * ys + r0 + r] = v * scale;
    }
}

// RDNA's (gfx1151): no stage - a lane its own block's 9 words straight from memory (plain loads, U blocks' worth
// before any sum), E2M1 by v_perm: codes 0..7's doubled values' f16 high bytes from an 8-entry table, a negative
// code's sign or'd, the f16 to F32 (each the table's value: its -0 for code 8 a +0's sums - a part never -0). fp4_kernel's
// sums in its order: its bits. FFN gate / up 17408 x 5120, one token, low-power: fp4_kernel 41 GB/s (R 4, spilled), 139
// (R 2); this 228 (fp4exp)
constexpr uint32_t kTabLo = 0x42403C00u, kTabHi = 0x4A484644u;  // 0, 1, 2, 3 | 4, 6, 8, 12 as f16 high bytes
__device__ inline void e2m1x4(uint32_t codes, float* v) {  // bytes' low nibbles 0..3 as 4 values
  const uint32_t hb = osmos_perm(kTabHi, kTabLo, codes & 0x07070707u) | (codes & 0x08080808u) << 4;
  const uint32_t h01 = osmos_perm(0, hb, 0x010C000Cu), h23 = osmos_perm(0, hb, 0x030C020Cu);  // (0, hb_i) halves
  v[0] = (float) __builtin_bit_cast(_Float16, (uint16_t) h01), v[1] = (float) __builtin_bit_cast(_Float16, (uint16_t) (h01 >> 16));
  v[2] = (float) __builtin_bit_cast(_Float16, (uint16_t) h23), v[3] = (float) __builtin_bit_cast(_Float16, (uint16_t) (h23 >> 16));
}

template <int R, int NT, int U>
__global__ __launch_bounds__(32 * kWarps) void direct_kernel(const uint8_t* __restrict__ w, uint32_t rows,
                                                             uint32_t cols, const float* __restrict__ x, uint32_t xs,
                                                             float* __restrict__ y, uint32_t ys, float scale) {
  const uint32_t lane = threadIdx.x % 32, r0 = (blockIdx.x * kWarps + threadIdx.x / 32) * R;
  if (r0 >= rows) return;
  const uint32_t nb = cols / kBlock;
  const uint64_t row_bytes = (uint64_t) nb * kBlockWords * 4;
  float acc[R][NT] = {};
  for (uint32_t b = lane; b < nb; b += 32 * U) {
    uint32_t q[U][R][kBlockWords];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < R; r++) {
        const uint32_t bb = b + 32 * u < nb ? b + 32 * u : b;
        const uint32_t* p =
            (const uint32_t*) (w + (uint64_t) (r0 + r < rows ? r0 + r : r0) * row_bytes + (uint64_t) bb * kBlockWords * 4);
#pragma unroll
        for (int k = 0; k < kBlockWords; k++) q[u][r][k] = p[k];
      }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint32_t bu = b + 32 * u;
      if (bu >= nb) break;
#pragma unroll
      for (int s = 0; s < 4; s++) {  // sub-block s: values 16 s .. 16 s + 15 of the block
        float xv[NT][16];
#pragma unroll
        for (int t = 0; t < NT; t++) {
          const float4* xp = (const float4*) (x + (uint64_t) t * xs + (uint64_t) bu * kBlock + 16 * s);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const float4 v = xp[k];
            xv[t][4 * k] = v.x, xv[t][4 * k + 1] = v.y, xv[t][4 * k + 2] = v.z, xv[t][4 * k + 3] = v.w;
          }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
          const float d = rnd::ue4m3_value(q[u][r][0] >> (8 * s) & 0xFFu) * 0.5f;
          const uint32_t lo = q[u][r][1 + 2 * s], hi = q[u][r][2 + 2 * s];
          float av[8], cv[8];
          e2m1x4(lo, av), e2m1x4(hi, av + 4), e2m1x4(lo >> 4, cv), e2m1x4(hi >> 4, cv + 4);
          float part[NT] = {};
#pragma unroll
          for (int j = 0; j < 8; j++)
#pragma unroll
            for (int t = 0; t < NT; t++) part[t] = __fmaf_rn(av[j], xv[t][j], __fmaf_rn(cv[j], xv[t][j + 8], part[t]));
#pragma unroll
          for (int t = 0; t < NT; t++) acc[r][t] = __fmaf_rn(d, part[t], acc[r][t]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int t = 0; t < NT; t++) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
      if (lane == 0 && r0 + r < rows) y[(uint64_t) t * ys + r0 + r] = v * scale;
    }
}

template <int R, int NT, int U>
bool run_direct(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys,
                float scale, void* stream) {
  const uint32_t warps = (rows + R - 1) / R;
  direct_kernel<R, NT, U><<<(warps + kWarps - 1) / kWarps, 32 * kWarps, 0, (hipStream_t) stream>>>(
      (const uint8_t*) w, rows, cols, x, xs, y, ys, scale);
  return hipGetLastError() == hipSuccess;
}

template <int R, int NT>
bool run(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, float scale,
         void* stream) {
  const uint32_t warps = (rows + R - 1) / R;
  fp4_kernel<R, NT><<<(warps + kWarps - 1) / kWarps, 32 * kWarps, 0, (hipStream_t) stream>>>(
      (const uint8_t*) w, rows, cols, x, xs, y, ys, scale);
  return hipGetLastError() == hipSuccess;
}

// several matrices, each over its own tokens (a MoE's picked experts: mv_group_tok's table) - a row's blocks over G
// lanes (a short row's: an expert's 1,536 values are 24 blocks), a warp 32 / G rows, a lane its blocks sub, sub + G,
// .. (2 in flight), then the G-lane tree. a token's sums its own at any count 1 .. 4 (its bits whatever the step).
// grid: (the rows' blocks, the table's entries); an entry of count 0 skipped
template <int G, int NT>
__device__ void tok_rows(const uint8_t* __restrict__ w, uint32_t rows, uint32_t cols, const float* __restrict__ x,
                         uint32_t xs, float* __restrict__ y, uint32_t ys) {
  constexpr int U = 2;
  const uint32_t lane = threadIdx.x % 32, sub = lane % G;
  const uint32_t r = (blockIdx.x * kWarps + threadIdx.x / 32) * (32 / G) + lane / G;
  const uint32_t nb = cols / kBlock;
  const uint64_t row_bytes = (uint64_t) nb * kBlockWords * 4;
  float acc[NT] = {};
  const bool live = r < rows;
  for (uint32_t b = sub; live && b < nb; b += G * U) {
    uint32_t q[U][kBlockWords];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint32_t bb = b + G * u < nb ? b + G * u : b;
      const uint32_t* p = (const uint32_t*) (w + (uint64_t) r * row_bytes + (uint64_t) bb * kBlockWords * 4);
#pragma unroll
      for (int k = 0; k < kBlockWords; k++) q[u][k] = p[k];
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint32_t bu = b + G * u;
      if (bu >= nb) break;
#pragma unroll
      for (int s = 0; s < 4; s++) {
        float xv[NT][16];
#pragma unroll
        for (int t = 0; t < NT; t++) {
          const float4* xp = (const float4*) (x + (uint64_t) t * xs + (uint64_t) bu * kBlock + 16 * s);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const float4 v = xp[k];
            xv[t][4 * k] = v.x, xv[t][4 * k + 1] = v.y, xv[t][4 * k + 2] = v.z, xv[t][4 * k + 3] = v.w;
          }
        }
        const float d = rnd::ue4m3_value(q[u][0] >> (8 * s) & 0xFFu) * 0.5f;
        const uint32_t lo = q[u][1 + 2 * s], hi = q[u][2 + 2 * s];
        float av[8], cv[8];
        e2m1x4(lo, av), e2m1x4(hi, av + 4), e2m1x4(lo >> 4, cv), e2m1x4(hi >> 4, cv + 4);
        float part[NT] = {};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
          for (int t = 0; t < NT; t++) part[t] = __fmaf_rn(av[j], xv[t][j], __fmaf_rn(cv[j], xv[t][j + 8], part[t]));
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] = __fmaf_rn(d, part[t], acc[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < NT; t++) {
    float v = acc[t];
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    if (sub == 0 && live) y[(uint64_t) t * ys + r] = v;
  }
}

template <int G>
__global__ __launch_bounds__(32 * kWarps) void tok_kernel(const void* const* w, const int32_t* cnt,
                                                          const int32_t* xrow, uint32_t rows, uint32_t cols,
                                                          const float* x, uint32_t xs, float* y, uint32_t ys) {
  const uint32_t e = blockIdx.y;
  const int32_t c = cnt[e];
  const uint8_t* m = (const uint8_t*) w[e];
  const float* xe = x + (uint64_t) xrow[e] * xs;
  float* ye = y + (uint64_t) xrow[e] * ys;
  switch (c) {  // (uniform a block: an entry a grid row)
    case 1: tok_rows<G, 1>(m, rows, cols, xe, xs, ye, ys); break;
    case 2: tok_rows<G, 2>(m, rows, cols, xe, xs, ye, ys); break;
    case 3: tok_rows<G, 3>(m, rows, cols, xe, xs, ye, ys); break;
    case 4: tok_rows<G, 4>(m, rows, cols, xe, xs, ye, ys); break;
    default: break;
  }
}

}  // namespace

// y[t * ys + r] = scale * sum_c W[r, c] x[t * xs + c], n 1 .. 4 tokens; cols whole 64-value blocks, x 16-byte aligned
extern "C" bool mv_fp4(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, uint32_t n, float* y,
                       uint32_t ys, float scale, void* stream) {
  if (!w || !x || !y || !rows || !cols || cols % kBlock || xs % 4 || n < 1 || n > 4) return false;
  // RDNA (fp4bench SWEEP=1, the FFN's gate / up, low-power, GB/s): one token direct R 2 U 2 222 (fp4_kernel R 4 41,
  // R 2 139); two R 4 U 1 211 (fp4_kernel R 2 96); three R 2 U 1 163 (fp4_kernel R 1 100); four fp4_kernel R 1's 84
  // (direct's best 74: NT 4's registers). every one fp4_kernel's bits
  if constexpr (!osmos_cuda) switch (n) {
      case 1: return run_direct<2, 1, 2>(w, rows, cols, x, xs, y, ys, scale, stream);
      case 2: return run_direct<4, 2, 1>(w, rows, cols, x, xs, y, ys, scale, stream);
      case 3: return run_direct<2, 3, 1>(w, rows, cols, x, xs, y, ys, scale, stream);
      default: return run<1, 4>(w, rows, cols, x, xs, y, ys, scale, stream);
    }
  switch (n) {  // the 5090's: the LDS stage (a lane's own 36-byte stride measured slower there)
    case 1: return run<4, 1>(w, rows, cols, x, xs, y, ys, scale, stream);
    case 2: return run<2, 2>(w, rows, cols, x, xs, y, ys, scale, stream);
    case 3: return run<1, 3>(w, rows, cols, x, xs, y, ys, scale, stream);
    default: return run<1, 4>(w, rows, cols, x, xs, y, ys, scale, stream);
  }
}

// the routed experts' decode (Hy4-flash's latent experts: 1,536 x 1,536): mv_group_tok's table over tok_kernel - a
// row's G its length's (every count the same G: a token's bits whatever the step)
extern "C" bool mv_fp4_tok(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows,
                           uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, void* stream) {
  if (!w || !cnt || !xrow || !x || !y || !n || n > 65535 || !rows || !cols || cols % kBlock || xs % 4) return false;
  const uint32_t nb = cols / kBlock;
  const hipStream_t s = (hipStream_t) stream;
#define TOK(G)                                                                                              \
  tok_kernel<G><<<dim3((rows + kWarps * (32 / G) - 1) / (kWarps * (32 / G)), n), 32 * kWarps, 0, s>>>(w, cnt, xrow, \
                                                                                                    rows, cols, x, xs, y, ys)
  if (nb <= 24)
    TOK(8);
  else if (nb <= 48)
    TOK(16);
  else
    TOK(32);
#undef TOK
  return hipGetLastError() == hipSuccess;
}
