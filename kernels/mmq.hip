// mmq (include/osmos/gpu/mmq.h): the experts' matmul on the int8 matrix cores - the activations int8 a 32-value
// group, Q4_K's 4-bit codes the A operand as they are, the scales a float epilogue a group
#pragma clang fp contract(off) /* each product and sum its own rounding: the host reference's */
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" {
#include "osmos/gpu/mmq.h"
#include "osmos/model/qtype.h"
}

namespace {

constexpr int kThreads = 256, kWave = 32, kTile = 16;
constexpr int kWaves = 4;                    // the K split: a wave's 256-value blocks p, p + 4, ...
constexpr int kBlockValues = 256, kBlockBytes = 144, kGroups = kBlockValues / MMQ_GROUP;  // Q4_K's
constexpr float kQ8Max = 127.0f;
typedef int i8v __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef uint32_t u4 __attribute__((ext_vector_type(4)));
static_assert(MMQ_GROUP == 32, "a group a wave's lanes (mmq_x8_rows' reductions)");

inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }
__device__ inline float f16(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }

// a thread a value, a group a wave's 32 lanes: its max |x| and the sum of its codes by shuffles
__global__ void x8_rows_kernel(const float* x, uint32_t xs, const int32_t* idx, uint32_t slots, uint32_t cols,
                               int8_t* x8, float* dx, float* sx) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) slots * cols) return;  // whole groups: slots * cols a multiple of 32, a block of 256
  const uint32_t s = (uint32_t) (i / cols), c = (uint32_t) (i % cols);
  const int32_t r = idx[s];
  const float v = r >= 0 ? x[(uint64_t) r * xs + c] : 0.0f;
  float a = fabsf(v);
  for (int o = kWave / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, kWave));
  const float d = a / kQ8Max;
  const int q = a == 0.0f ? 0 : (int) fminf(fmaxf(rintf(v / d), -kQ8Max), kQ8Max);
  int sum = q;
  for (int o = kWave / 2; o > 0; o >>= 1) sum += __shfl_xor(sum, o, kWave);
  x8[mmq_x8_at(s, c, cols)] = (int8_t) q;
  if (c % MMQ_GROUP == 0) {
    const uint64_t g = mmq_dx_at(s, c / MMQ_GROUP, cols / MMQ_GROUP);
    dx[g] = a == 0.0f ? 0.0f : d;
    sx[g] = (a == 0.0f ? 0.0f : d) * (float) sum;
  }
}

// a block a 16-row tile x CT token tiles of one matrix: wave p its 256-value blocks p, p + kWaves, ...; a block's 16
// rows into the wave's LDS (whole rows' bytes a lane's 16 at a time), their (d sc, dmin m) a table beside them, then
// a group at a time: two iu8 WMMAs (its 32 values in two steps of 16 - lane l's A row l % 16's codes as bytes, its B
// token l % 16's codes) then the group's epilogue; the waves' partial sums in wave order
constexpr int kRowChunks = kBlockBytes / 16, kWaveChunks = kTile * kRowChunks;  // a block's 16 rows: 144 chunks
template <int CT>
__global__ __launch_bounds__(kWave* kWaves) void q4k_kernel(const uint8_t* const* __restrict__ wt,
                                                            const int32_t* __restrict__ cnt, uint32_t cols,
                                                            const int8_t* __restrict__ x8, const float* __restrict__ dx,
                                                            const float* __restrict__ sx,
                                                            const uint32_t* __restrict__ soff, float* __restrict__ ya,
                                                            const uint32_t* __restrict__ yoff, uint32_t ys) {
  constexpr int kRawBytes = kWaves * kTile * kBlockBytes, kPartBytes = kWaves * CT * 8 * kWave * 4;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kRawBytes > kPartBytes ? kRawBytes : kPartBytes];
  __shared__ float scl[kWaves][kGroups][2][kTile];
  uint8_t (*raw)[kTile * kBlockBytes] = (uint8_t (*)[kTile * kBlockBytes]) lds;  // the blocks', then the sums'
  float (*part)[CT][8][kWave] = (float (*)[CT][8][kWave]) lds;
  const uint32_t m = blockIdx.y, n = (uint32_t) cnt[m], row0 = blockIdx.x * kTile, c0 = blockIdx.z * kTile * CT;
  if (c0 >= n) return;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  const uint32_t nb = cols / kBlockValues, groups = cols / MMQ_GROUP;
  const uint64_t row_bytes = (uint64_t) nb * kBlockBytes;
  const uint8_t* w = wt[m] + (uint64_t) row0 * row_bytes;
  const uint32_t s0 = soff[m] + c0;  // the block's first slot: a multiple of 16
  float* y = ya + yoff[m];
  const int tiles = (int) ((n - c0 + kTile - 1) / kTile) < CT ? (int) ((n - c0 + kTile - 1) / kTile) : CT;
  const int8_t* xb[CT];
  const float *dxb[CT], *sxb[CT];
  for (int ct = 0; ct < CT; ct++) {
    xb[ct] = x8 + mmq_x8_at(s0 + ct * kTile + col, 0, cols);
    dxb[ct] = dx + mmq_dx_at(s0 + ct * kTile + col, 0, groups);
    sxb[ct] = sx + mmq_dx_at(s0 + ct * kTile + col, 0, groups);
  }
  uint8_t* lr = raw[wave];
  float acc[CT][8] = {};
  for (uint32_t b = wave; b < nb; b += kWaves) {
    for (int i = lane; i < kWaveChunks; i += kWave) {
      const int r = i / kRowChunks, k = i % kRowChunks;
      *(u4*) (lr + r * kBlockBytes + k * 16) = *(const u4*) (w + r * row_bytes + (uint64_t) b * kBlockBytes + k * 16);
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    {  // lane: row col's groups half * 4 .. + 4 - Q4_K's 6-bit (sc, m), times d and dmin
      const uint8_t* blk = lr + col * kBlockBytes;
      const float d = f16(blk[0] | blk[1] << 8), dmin = f16(blk[2] | blk[3] << 8);
      const uint8_t* s = blk + 4;
      for (int gi = 0; gi < kGroups / 2; gi++) {
        const int g = half * (kGroups / 2) + gi;
        const int sc = g < 4 ? s[g] & 0x3F : (s[g + 4] & 0x0F) | ((s[g - 4] >> 2) & 0x30);
        const int mn = g < 4 ? s[g + 4] & 0x3F : (s[g + 4] >> 4) | ((s[g] >> 2) & 0x30);
        scl[wave][g][0][col] = __fmul_rn(d, (float) sc);
        scl[wave][g][1][col] = __fmul_rn(dmin, (float) mn);
      }
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    const uint8_t* qa = lr + col * kBlockBytes + 16;
    for (int gp = 0; gp < kGroups / 2; gp++) {
      const u4 raw0 = *(const u4*) (qa + gp * 32), raw1 = *(const u4*) (qa + gp * 32 + 16);
      for (int e = 0; e < 2; e++) {
        const int g = 2 * gp + e;
        const uint32_t sh = 4u * (uint32_t) e;
        const u4 a0 = (raw0 >> sh) & 0x0F0F0F0Fu, a1 = (raw1 >> sh) & 0x0F0F0F0Fu;
        const uint32_t gg = b * kGroups + g;
        for (int ct = 0; ct < CT; ct++)
          if (ct < tiles) {
            const int8_t* xg = xb[ct] + (uint64_t) gg * (kTile * MMQ_GROUP);
            const u4 b0 = *(const u4*) xg, b1 = *(const u4*) (xg + 16);
            i8v ia = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, __builtin_bit_cast(i4v, a0), true,
                                                                __builtin_bit_cast(i4v, b0), i8v{}, false);
            ia = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, __builtin_bit_cast(i4v, a1), true,
                                                            __builtin_bit_cast(i4v, b1), ia, false);
            const float dxt = dxb[ct][gg * kTile], nsx = -sxb[ct][gg * kTile];
            for (int j = 0; j < 8; j++) {
              const int r = 2 * j + half;
              const float v = __fmul_rn(scl[wave][g][0][r], dxt);
              acc[ct][j] = __fmaf_rn(v, (float) ia[j], acc[ct][j]);
              acc[ct][j] = __fmaf_rn(scl[wave][g][1][r], nsx, acc[ct][j]);
            }
          }
      }
    }
    __builtin_amdgcn_wave_barrier();  // the block read before the next one's written
  }
  __syncthreads();  // every wave's blocks read before the sums take their place
  for (int ct = 0; ct < CT; ct++)
    if (ct < tiles)
      for (int j = 0; j < 8; j++) part[wave][ct][j][lane] = acc[ct][j];
  __syncthreads();
  if (wave) return;
  for (int ct = 0; ct < CT; ct++) {
    const uint32_t t = c0 + ct * kTile + col;
    if (ct < tiles && t < n)
      for (int j = 0; j < 8; j++) {
        float s = part[0][ct][j][lane];
        for (int p = 1; p < kWaves; p++) s = __fadd_rn(s, part[p][ct][j][lane]);
        y[(uint64_t) t * ys + row0 + 2 * j + half] = s;
      }
  }
}

template <int CT>
bool q4k_launch(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
                uint32_t cols, const int8_t* x8, const float* dx, const float* sx, const uint32_t* soff, float* y,
                const uint32_t* yoff, uint32_t ys, hipStream_t s) {
  const uint32_t z = (max_cnt + kTile * CT - 1) / (kTile * CT);
  q4k_kernel<CT><<<dim3(rows / kTile, n_mat, z), kWave * kWaves, 0, s>>>((const uint8_t* const*) w, cnt, cols, x8, dx,
                                                                       sx, soff, y, yoff, ys);
  return ok();
}

}  // namespace

extern "C" bool mmq_known(uint32_t type) { return type == QTYPE_Q4_K; }

extern "C" bool mmq_x8_rows(const float* x, uint32_t xs, const int32_t* idx, uint32_t slots, uint32_t cols,
                            int8_t* x8, float* dx, float* sx, void* stream) {
  if (!x || !idx || !x8 || !dx || !sx || !slots || !cols || cols % MMQ_GROUP || xs < cols) return false;
  const uint64_t work = (uint64_t) slots * cols;
  x8_rows_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, st(stream)>>>(x, xs, idx, slots, cols,
                                                                                            x8, dx, sx);
  return ok();
}

extern "C" bool mmq_group(uint32_t type, const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt,
                          uint32_t rows, uint32_t cols, const int8_t* x8, const float* dx, const float* sx,
                          const uint32_t* soff, float* y, const uint32_t* yoff, uint32_t ys, void* stream) {
  if (!mmq_known(type) || !w || !cnt || !x8 || !dx || !sx || !soff || !y || !yoff || !n_mat || !max_cnt ||
      !rows || rows % kTile || !cols || cols % kBlockValues || ys < rows)
    return false;
  const hipStream_t s = st(stream);
  if (max_cnt <= kTile) return q4k_launch<1>(w, cnt, n_mat, max_cnt, rows, cols, x8, dx, sx, soff, y, yoff, ys, s);
  if (max_cnt <= 2 * kTile) return q4k_launch<2>(w, cnt, n_mat, max_cnt, rows, cols, x8, dx, sx, soff, y, yoff, ys, s);
  return q4k_launch<MMQ_MAX_TILES>(w, cnt, n_mat, max_cnt, rows, cols, x8, dx, sx, soff, y, yoff, ys, s);
}
