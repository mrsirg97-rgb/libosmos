// q4x_delta (the Gated DeltaNet's rule, a head's columns over several blocks) against the one-block-a-head kernel it
// replaces, copied here as it was, on the same normed q and k (q4x_delta's own): every output and the state after the
// step the same bytes - at Qwen3.8's shape (16 key heads, 48 value heads, a state of 128) over 1 to 2,048 tokens,
// the ring's slots kept for the last tokens. needs a GPU; skipped (77) without one.
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr int kHead = Q4X_SSM_STATE;
// kernels/q4x_ssm.hip's delta_kernel before the column split
constexpr int kHalfRows = kHead / 2;
// the q / k rows in LDS: the high half a float on (a lane reads its half's row, its pair the other's - 64 floats apart
// they were one bank, a 2-way conflict each read; one apart each half a broadcast)
constexpr int kQkLds = kHead + 1;
__device__ inline uint32_t qk_at(uint32_t i) { return i + (i >= (uint32_t) kHalfRows); }
__global__ __launch_bounds__(2 * kHead) void delta_ref(const float* qkv, const float* qk, const float* g,
                                                          const float* b, uint32_t n_tok, uint32_t k_heads,
                                                          uint32_t v_heads, float* ring, uint32_t slots, uint32_t cur,
                                                          uint32_t keep, float* out) {
  __shared__ float qs[2][kQkLds], ks[2][kQkLds];
  const uint32_t h = blockIdx.x, j = threadIdx.x >> 1, half = threadIdx.x & 1, kg = h % k_heads;
  const uint32_t r0 = half * kHalfRows, l0 = qk_at(r0);
  const uint64_t ch = 2ull * k_heads * kHead + (uint64_t) v_heads * kHead, qkr = 2ull * k_heads * kHead;
  const uint64_t slot = (uint64_t) v_heads * kHead * kHead, col = ((uint64_t) h * kHead + j) * kHead + r0;
  float s[kHalfRows];
#pragma unroll
  for (int i = 0; i < kHalfRows; i++) s[i] = ring[cur * slot + col + i];
  const float scale = 1.0f / sqrtf((float) kHead);
  // a thread's share of the token's loads: its column's q (the low half's thread) or k (the high's); v both
  auto load = [&](uint32_t t, float* x, float* v, float* gg, float* bb) {
    *x = qk[t * qkr + (uint64_t) (half ? k_heads + kg : kg) * kHead + j];
    *v = qkv[t * ch + 2ull * k_heads * kHead + (uint64_t) h * kHead + j];
    *gg = g[(uint64_t) t * v_heads + h], *bb = b[(uint64_t) t * v_heads + h];
  };
  float nx, nv, ng, nb;
  load(0, &nx, &nv, &ng, &nb);
  for (uint32_t t = 0; t < n_tok; t++) {
    const int buf = t & 1;
    (half ? ks : qs)[buf][qk_at(j)] = nx;
    const float v = nv, decay = expf(ng), beta = nb;
    if (t + 1 < n_tok) load(t + 1, &nx, &nv, &ng, &nb);
    __syncthreads();  // this token's q, k in; the buffer before it read by every thread (it was, before the last)
    const float *qb = qs[buf] + l0, *kb = ks[buf] + l0;
    float part = 0;
#pragma unroll
    for (int i = 0; i < kHalfRows; i++) s[i] *= decay, part += s[i] * kb[i];
    const float other = __shfl_xor(part, 1);
    const float dot = half ? other + part : part + other;  // low + high in both threads
    const float delta = (v - dot) * beta;
    float po = 0;
#pragma unroll
    for (int i = 0; i < kHalfRows; i++) s[i] += kb[i] * delta, po += s[i] * qb[i];
    const float oo = __shfl_xor(po, 1);
    if (!half) out[((uint64_t) t * v_heads + h) * kHead + j] = (po + oo) * scale;
    if (t >= n_tok - keep) {
      float* to = ring + (cur + 1 + t) % slots * slot + col;
#pragma unroll
      for (int i = 0; i < kHalfRows; i++) to[i] = s[i];
    }
  }
}


static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / 8388608.0f - 1.0f;
}
static float* dev(size_t n, float scale, float off = 0.0f) {
  std::vector<float> h(n);
  for (float& v : h) v = off + scale * uniform();
  float* d = nullptr;
  CHECK(hipMalloc(&d, n * 4) == hipSuccess && hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
static std::vector<uint8_t> bytes(const float* d, size_t n) {
  std::vector<uint8_t> h(n * 4);
  CHECK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

static void run(uint32_t n, uint32_t slots) {
  constexpr uint32_t kh = 16, vh = 48;
  const uint64_t ch = 2ull * kh * kHead + (uint64_t) vh * kHead, state = (uint64_t) vh * kHead * kHead;
  float *qkv = dev(n * ch, 1.0f), *g = dev((size_t) n * vh, 0.5f, -0.6f), *b = dev((size_t) n * vh, 0.5f, 0.5f);
  float *ring0 = dev(slots * state, 0.1f), *ra = nullptr, *rb = nullptr, *qk = nullptr, *oa = nullptr, *ob = nullptr;
  CHECK(hipMalloc(&ra, slots * state * 4) == hipSuccess && hipMalloc(&rb, slots * state * 4) == hipSuccess);
  CHECK(hipMalloc(&qk, (size_t) n * 2 * kh * kHead * 4) == hipSuccess);
  CHECK(hipMalloc(&oa, (size_t) n * vh * kHead * 4) == hipSuccess && hipMalloc(&ob, (size_t) n * vh * kHead * 4) == hipSuccess);
  CHECK(hipMemcpy(ra, ring0, slots * state * 4, hipMemcpyDeviceToDevice) == hipSuccess);
  CHECK(hipMemcpy(rb, ring0, slots * state * 4, hipMemcpyDeviceToDevice) == hipSuccess);
  const uint32_t cur = slots > 1 ? 1 : 0, keep = n < slots ? n : slots;
  CHECK(q4x_delta(qkv, g, b, n, kh, vh, kHead, 1e-6f, qk, ra, slots, cur, oa, nullptr));
  delta_ref<<<vh, 2 * kHead>>>(qkv, qk, g, b, n, kh, vh, rb, slots, cur, keep, ob);
  CHECK(hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess);
  CHECK(bytes(oa, (size_t) n * vh * kHead) == bytes(ob, (size_t) n * vh * kHead));
  CHECK(bytes(ra, slots * state) == bytes(rb, slots * state));
  std::printf("  %4u tokens, %u slots: the one-block-a-head kernel's outputs and states\n", n, slots);
  for (float* p : {qkv, g, b, ring0, ra, rb, qk, oa, ob}) CHECK(hipFree(p) == hipSuccess);
}

int main() {
  std::printf("[test_q4x_delta_split_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_q4x_delta_split_gpu] skipped: no GPU\n");
    return 77;
  }
  run(1, 1);
  run(5, 4);     // a decode's verify: the ring's slots for each token
  run(37, 4);
  run(2048, 1);  // a prompt's step
  std::printf("[test_q4x_delta_split_gpu] ok\n");
  return 0;
}
