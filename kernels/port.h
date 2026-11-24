// the GPU's own instructions behind one name each (DESIGN.md, "a second GPU: CUDA"): on HIP (gfx1151) each is the
// builtin itself; on CUDA (sm_120, clang's CUDA mode) the same result by CUDA's means. the
// WMMA tiles keep RDNA's wave32 layout lane for lane - a lane l < 16 holds A's row l and B's column l (16 values),
// lanes 16 .. 31 the same again; D's element j of lane l is row 2j + l / 16, column l % 16 - so a kernel's loads and
// stores are the same code on both
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#if defined(__HIP__)

#define osmos_sleep(n) __builtin_amdgcn_s_sleep(n)
// an F32 kept in a register as it is: no fused v_fma_mixlo_f16 across a product and its f16 conversion
#define osmos_keep(x) asm("" : "+v"(x))
#define osmos_keep3(a, b, c) asm("" : "+v"(a), "+v"(b), "+v"(c))
__device__ inline void osmos_wave_barrier() { __builtin_amdgcn_wave_barrier(); }
__device__ inline void osmos_wave_fence() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
__device__ inline uint32_t osmos_perm(uint32_t a, uint32_t b, uint32_t sel) { return __builtin_amdgcn_perm(a, b, sel); }
__device__ inline int osmos_sudot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
__device__ inline uint32_t osmos_load_acquire_system(const uint32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline uint64_t osmos_wall_clock() { return wall_clock64(); }

typedef _Float16 osmos_h16 __attribute__((ext_vector_type(16)));
typedef float osmos_f8 __attribute__((ext_vector_type(8)));
typedef int osmos_i4 __attribute__((ext_vector_type(4)));
typedef int osmos_i8 __attribute__((ext_vector_type(8)));
__device__ inline osmos_f8 osmos_wmma_f16(osmos_h16 a, osmos_h16 b, osmos_f8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
// A unsigned, B signed (mmq's: Q4_K's 4-bit codes against the input's int8)
__device__ inline osmos_i8 osmos_wmma_u8s8(osmos_i4 a, osmos_i4 b, osmos_i8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, true, b, c, false);
}

// a 16 x 16 x 16 f16 tile in the backend's own fragments (mm.h): A's 16 rows at `rows` (row r at rows + r stride, 16
// of K each), B's 16 columns in an x16 fragment (column c's 16 of K at frag + 16 c), D an osmos_f8 a lane whose
// element j is row osmos_acc_row(j), column osmos_acc_col(j). here RDNA's: a lane row / column lane % 16 whole
constexpr bool osmos_cuda = false;
typedef osmos_h16 osmos_fa;
typedef osmos_h16 osmos_fb;
__device__ inline osmos_fa osmos_load_a(const _Float16* rows, int stride) {
  return *(const osmos_h16*) (rows + (threadIdx.x % 32 % 16) * stride);
}
__device__ inline osmos_fb osmos_load_b(const _Float16* frag) { return *(const osmos_h16*) (frag + threadIdx.x % 32 % 16 * 16); }
__device__ inline osmos_f8 osmos_mma(osmos_fa a, osmos_fb b, osmos_f8 c) { return osmos_wmma_f16(a, b, c); }
__device__ inline int osmos_acc_row(int j) { return 2 * j + threadIdx.x % 32 / 16; }
__device__ inline int osmos_acc_col(int) { return threadIdx.x % 32 % 16; }
// B's columns `stride` apart (a column's 16 of K contiguous): column c at frag + c stride
__device__ inline osmos_fb osmos_load_b_s(const _Float16* frag, int stride) {
  return *(const osmos_h16*) (frag + threadIdx.x % 32 % 16 * stride);
}
// a lane's rows: its D elements' rows, each a slot (a row's running state a slot's); element j in slot osmos_slot(j),
// the slot's row osmos_slot_row(s) - here a lane 8 rows, one element each; a row's 16 columns on the 16 lanes of a half
constexpr int osmos_slots = 8;
__device__ constexpr int osmos_slot(int j) { return j; }
__device__ inline int osmos_slot_row(int s) { return 2 * s + threadIdx.x % 32 / 16; }
__device__ inline float osmos_row_max(float v) {  // over the lanes holding the row's other columns
  for (int o = 1; o < 16; o <<= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ inline float osmos_row_sum(float v) {
  for (int o = 1; o < 16; o <<= 1) v += __shfl_xor(v, o, 32);
  return v;
}

#else  // CUDA

#define osmos_sleep(n) __nanosleep(64u * (n))  // s_sleep's unit: 64 clocks (a back-off, not a time)
#define osmos_keep(x) asm volatile("" : "+f"(x))
#define osmos_keep3(a, b, c) asm volatile("" : "+f"(a), "+f"(b), "+f"(c))
__device__ inline void osmos_wave_barrier() { __syncwarp(); }
__device__ inline void osmos_wave_fence() { __syncwarp(); }  // __syncwarp orders the warp's shared memory too

// v_perm_b32: each byte of the result chosen by sel's byte from {a, b} as bytes 7..4 = a, 3..0 = b; 8 .. 11 a sign
// (bytes 1, 3, 5, 7's top bit spread), 12 zero, 13 .. 255 0xFF
__device__ inline uint32_t osmos_perm(uint32_t a, uint32_t b, uint32_t sel) {
  const uint64_t v = (uint64_t) a << 32 | b;
  uint32_t r = 0;
  for (int i = 0; i < 4; i++) {
    const uint32_t s = sel >> 8 * i & 0xFFu;
    uint32_t byte;
    if (s < 8) byte = (uint32_t) (v >> 8 * s) & 0xFFu;
    else if (s < 12) byte = (v >> (16 * (s - 8) + 15) & 1) ? 0xFFu : 0u;
    else byte = s == 12 ? 0u : 0xFFu;
    r |= byte << 8 * i;
  }
  return r;
}
__device__ inline int osmos_sudot4(int a, int b, int c) { return __dp4a(a, b, c); }
__device__ inline uint32_t osmos_load_acquire_system(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ inline uint64_t osmos_wall_clock() {  // ns (the host's hipDeviceAttributeWallClockRate: 1,000,000 kHz)
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

typedef _Float16 osmos_h16 __attribute__((ext_vector_type(16)));
typedef float osmos_f8 __attribute__((ext_vector_type(8)));
typedef int osmos_i4 __attribute__((ext_vector_type(4)));
typedef int osmos_i8 __attribute__((ext_vector_type(8)));

// RDNA's tile, lane for lane: lane l's D[j] = C[j] + sum_k A[2j + l/16][k] B[k][l % 16], k ascending, each product
// fused. A's row m lives in lane m; B's column l % 16 in lane l itself (lanes 16 .. 31 repeat 0 .. 15)
__device__ inline osmos_f8 osmos_wmma_f16(osmos_h16 a, osmos_h16 b, osmos_f8 c) {
  const int lane = threadIdx.x & 31, half = lane >> 4;
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  uint32_t aw[8];
  for (int w = 0; w < 8; w++) aw[w] = __builtin_bit_cast(uint32_t, h2{a[2 * w], a[2 * w + 1]});
  osmos_f8 d = c;
  for (int j = 0; j < 8; j++) {
    const int m = 2 * j + half;
    float acc = d[j];
    for (int w = 0; w < 8; w++) {
      const h2 x = __builtin_bit_cast(h2, __shfl_sync(0xFFFFFFFFu, aw[w], m));
      acc = __fmaf_rn((float) x[0], (float) b[2 * w], acc);
      acc = __fmaf_rn((float) x[1], (float) b[2 * w + 1], acc);
    }
    d[j] = acc;
  }
  return d;
}
__device__ inline int osmos_dot4_u8s8(uint32_t a, uint32_t b, int c) {
  for (int i = 0; i < 4; i++) c += (int) (a >> 8 * i & 0xFFu) * (int) (int8_t) (b >> 8 * i & 0xFFu);
  return c;
}
__device__ inline osmos_i8 osmos_wmma_u8s8(osmos_i4 a, osmos_i4 b, osmos_i8 c) {
  const int lane = threadIdx.x & 31, half = lane >> 4;
  osmos_i8 d = c;
  for (int j = 0; j < 8; j++) {
    const int m = 2 * j + half;
    int acc = d[j];
    for (int w = 0; w < 4; w++) acc = osmos_dot4_u8s8((uint32_t) __shfl_sync(0xFFFFFFFFu, a[w], m), (uint32_t) b[w], acc);
    d[j] = acc;
  }
  return d;
}

// the tile in mma.sync's fragments (m16n8k16, row.col, f16 -> f32), two n8 halves a 16 x 16: lane l, g = l / 4,
// q = l % 4 - A's pairs (row g | g + 8, K 2q | 2q + 8), B's (K 2q | 2q + 8, column g | 8 + g), D's (row g | g + 8,
// column 2q, 2q + 1 | 8 + those). every pair two adjacent f16s: one 32-bit load
constexpr bool osmos_cuda = true;
struct osmos_fa { uint32_t r[4]; };
struct osmos_fb { uint32_t r[4]; };  // [n half][K 2q, 2q + 8]
__device__ inline uint32_t osmos_pair(const _Float16* p) { return *(const uint32_t*) p; }
__device__ inline osmos_fa osmos_load_a(const _Float16* rows, int stride) {
  const int g = threadIdx.x % 32 / 4, q = threadIdx.x % 4;
  return {{osmos_pair(rows + g * stride + 2 * q), osmos_pair(rows + (g + 8) * stride + 2 * q),
           osmos_pair(rows + g * stride + 2 * q + 8), osmos_pair(rows + (g + 8) * stride + 2 * q + 8)}};
}
__device__ inline osmos_fb osmos_load_b(const _Float16* frag) {
  const int g = threadIdx.x % 32 / 4, q = threadIdx.x % 4;
  return {{osmos_pair(frag + g * 16 + 2 * q), osmos_pair(frag + g * 16 + 2 * q + 8),
           osmos_pair(frag + (8 + g) * 16 + 2 * q), osmos_pair(frag + (8 + g) * 16 + 2 * q + 8)}};
}
__device__ inline osmos_f8 osmos_mma(osmos_fa a, osmos_fb b, osmos_f8 c) {
  float d[8];
  for (int e = 0; e < 8; e++) d[e] = c[e];
  for (int h = 0; h < 2; h++)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[4 * h]), "+f"(d[4 * h + 1]), "+f"(d[4 * h + 2]), "+f"(d[4 * h + 3])
        : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]), "r"(b.r[2 * h]), "r"(b.r[2 * h + 1]));
  osmos_f8 r;
  for (int e = 0; e < 8; e++) r[e] = d[e];
  return r;
}
__device__ inline int osmos_acc_row(int j) { return threadIdx.x % 32 / 4 + 8 * (j % 4 / 2); }
__device__ inline int osmos_acc_col(int j) { return j / 4 * 8 + 2 * (threadIdx.x % 4) + j % 2; }
__device__ inline osmos_fb osmos_load_b_s(const _Float16* frag, int stride) {
  const int g = threadIdx.x % 32 / 4, q = threadIdx.x % 4;
  return {{osmos_pair(frag + g * stride + 2 * q), osmos_pair(frag + g * stride + 2 * q + 8),
           osmos_pair(frag + (8 + g) * stride + 2 * q), osmos_pair(frag + (8 + g) * stride + 2 * q + 8)}};
}
// a lane 2 rows (g, g + 8), 4 elements each (elements 0, 1, 4, 5 row g; 2, 3, 6, 7 row g + 8); a row's 16 columns on
// the 4 lanes of a quad, 4 each
constexpr int osmos_slots = 2;
__device__ constexpr int osmos_slot(int j) { return j % 4 / 2; }
__device__ inline int osmos_slot_row(int s) { return threadIdx.x % 32 / 4 + 8 * s; }
__device__ inline float osmos_row_max(float v) {
  for (int o = 1; o < 4; o <<= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}
__device__ inline float osmos_row_sum(float v) {
  for (int o = 1; o < 4; o <<= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  return v;
}

#endif
