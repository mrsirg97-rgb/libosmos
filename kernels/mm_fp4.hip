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

// a block a token, a thread a 16: its scale, its codes packed two a byte
__global__ void act_kernel(const float* __restrict__ x, uint32_t xs, uint32_t cols, float gs,
                           uint8_t* __restrict__ q, uint8_t* __restrict__ sc) {
  const uint32_t t = blockIdx.x;
  for (uint32_t b = threadIdx.x; b < cols / 16; b += blockDim.x) {
    const float* v = x + (uint64_t) t * xs + b * 16;
    float amax = 0.0f;
    for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
    const uint32_t s = e4m3_bits(amax / 6.0f / gs);
    const float sf = rnd::e4m3_value(s) * gs;
    sc[(uint64_t) t * (cols / 16) + b] = (uint8_t) s;
    uint8_t* o = q + (uint64_t) t * (cols / 2) + b * 8;
    for (int j = 0; j < 16; j += 2) {
      const float a = sf > 0.0f ? fminf(fmaxf(v[j] / sf, -6.0f), 6.0f) : 0.0f;
      const float c = sf > 0.0f ? fminf(fmaxf(v[j + 1] / sf, -6.0f), 6.0f) : 0.0f;
      o[j / 2] = (uint8_t) (e2m1_code(a) | e2m1_code(c) << 4);
    }
  }
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

// W ggml 40's blocks (GGML: the GGUF's), or ModelOpt's (packed [rows, cols/2] in k order, its E4M3 scales [rows,
// cols/16] at ws: the training engine's train/fp4.py)
template <bool GGML>
__global__ __launch_bounds__(kTileThreads) void mm_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                                          uint32_t rows, uint32_t cols,
                                                          const uint8_t* __restrict__ aq,
                                                          const uint8_t* __restrict__ as, uint32_t n, float scale,
                                                          float* __restrict__ y, uint32_t ys) {
  __shared__ uint32_t sa[kBM * kRow], sb[kBN * kRow];
  const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32, g = lane / 4, q = lane % 4;
  const uint32_t wm = warp % 4, wn = warp / 4;  // the warp's 32 tokens, its 64 rows
  const uint32_t t0 = blockIdx.y * kBM, r0 = blockIdx.x * kBN, nb = cols / 64;
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
    for (uint32_t i = tid; !GGML && i < kBN * 8; i += kTileThreads) {
      const uint32_t r = i / 8, c = i % 8, row = r0 + r < rows ? r0 + r : rows - 1;
      sb[r * kRow + c] = *(const uint32_t*) (w + (uint64_t) row * (cols / 2) + (uint64_t) kb * 32 + 4 * c);
      if (c == 0) sb[r * kRow + 8] = *(const uint32_t*) (ws + (uint64_t) row * (cols / 16) + (uint64_t) kb * 4);
    }
    for (uint32_t i = tid; GGML && i < kBN * 4; i += kTileThreads) {
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
        if (t < n && row < rows) y[(uint64_t) t * ys + row] = d[mt][nt][e] * scale;
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

extern "C" bool mm_fp4_act(const float* x, uint32_t xs, uint32_t n, uint32_t cols, float gs, void* q, void* sc,
                           void* stream) {
  if (!mm_fp4_ok() || !x || !q || !sc || !n || cols % 64 || !(gs > 0.0f)) return false;
  act_kernel<<<n, 128, 0, launches::st(stream)>>>(x, xs, cols, gs, (uint8_t*) q, (uint8_t*) sc);
  return launches::ok();
}

extern "C" bool mm_fp4(const void* w, uint32_t rows, uint32_t cols, const void* q, const void* sc, uint32_t n,
                       float scale, float* y, uint32_t ys, void* stream) {
  if (!mm_fp4_ok() || !w || !q || !sc || !y || !n || cols % 64 || rows % 8) return false;
  const dim3 grid((rows + kBN - 1) / kBN, (n + kBM - 1) / kBM);
  mm_kernel<true><<<grid, kTileThreads, 0, launches::st(stream)>>>((const uint8_t*) w, nullptr, rows, cols,
                                                                   (const uint8_t*) q, (const uint8_t*) sc, n, scale, y,
                                                                   ys);
  return launches::ok();
}

extern "C" bool mm_fp4_mo(const void* packed, const void* wscale, uint32_t rows, uint32_t cols, const void* q,
                          const void* sc, uint32_t n, float scale, float* y, uint32_t ys, void* stream) {
  if (!mm_fp4_ok() || !packed || !wscale || !q || !sc || !y || !n || cols % 64 || rows % 8) return false;
  const dim3 grid((rows + kBN - 1) / kBN, (n + kBM - 1) / kBM);
  mm_kernel<false><<<grid, kTileThreads, 0, launches::st(stream)>>>((const uint8_t*) packed, (const uint8_t*) wscale,
                                                                    rows, cols, (const uint8_t*) q, (const uint8_t*) sc,
                                                                    n, scale, y, ys);
  return launches::ok();
}
