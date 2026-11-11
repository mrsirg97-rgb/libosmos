// ops_router against the kernel it replaced (a block of 256 threads, a tree a pick - below, as it was), bit for bit:
// the picks and their weights, over random logits, logits with exact ties (the lower expert first), every expert the
// same, expert counts not a multiple of a wave's, k 1 .. OPS_MAX_TOPK. needs a gfx1151; skipped (77) without one
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

namespace {

constexpr int kThreads = 256;               // the reference's block (kernels/ops.hip's kThreads when it was written)
constexpr float kSoftplusLinear = 20.0f;    // kernels/ops.hip's
constexpr float kRouteEps = 1e-20f;         // kernels/ops.hip's

__global__ void reference(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float scale,
                          int32_t* idx, float* w) {
  __shared__ float sc[OPS_MAX_EXPERTS], key[OPS_MAX_EXPERTS];
  __shared__ bool taken[OPS_MAX_EXPERTS];
  __shared__ float bk[kThreads];
  __shared__ uint32_t bi[kThreads];
  __shared__ int32_t pick[OPS_MAX_TOPK];
  const uint32_t t = blockIdx.x, none = n_exp;
  for (uint32_t e = threadIdx.x; e < n_exp; e += kThreads) {
    const float l = logits[(uint64_t) t * n_exp + e];
    sc[e] = sqrtf(l > kSoftplusLinear ? l : log1pf(expf(l)));
    key[e] = sc[e] + bias[e];
    taken[e] = false;
  }
  __syncthreads();
  for (uint32_t s = 0; s < k; s++) {
    float best = 0;
    uint32_t at = none;
    for (uint32_t e = threadIdx.x; e < n_exp; e += kThreads)
      if (!taken[e] && (at == none || key[e] > best)) best = key[e], at = e;
    bk[threadIdx.x] = best, bi[threadIdx.x] = at;
    __syncthreads();
    for (uint32_t h = kThreads / 2; h > 0; h >>= 1) {
      if (threadIdx.x < h) {
        const uint32_t o = threadIdx.x + h;
        const bool take = bi[o] != none && (bi[threadIdx.x] == none || bk[o] > bk[threadIdx.x] ||
                                            (bk[o] == bk[threadIdx.x] && bi[o] < bi[threadIdx.x]));
        if (take) bk[threadIdx.x] = bk[o], bi[threadIdx.x] = bi[o];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) pick[s] = (int32_t) bi[0], taken[bi[0]] = true;
    __syncthreads();
  }
  if (threadIdx.x != 0) return;
  float wt[OPS_MAX_TOPK], sum = 0;
  for (uint32_t s = 0; s < k; s++) sum += (wt[s] = sc[pick[s]]);
  for (uint32_t s = 0; s < k; s++) {
    idx[t * k + s] = pick[s];
    w[t * k + s] = wt[s] / (sum + kRouteEps) * scale;
  }
}

uint32_t rng = 1017;
float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

// logits for n tokens of e experts: kind 0 random, 1 a few values only (ties everywhere), 2 every expert the same
void check(uint32_t n, uint32_t e, uint32_t k, int kind) {
  std::vector<float> l((size_t) n * e), b(e);
  for (float& v : l) {
    rng = rng * 1664525u + 1013904223u;
    v = kind == 2 ? 0.25f : kind == 1 ? (float) (rng >> 29) : uniform() * 8;
  }
  for (float& v : b) v = kind ? 0.0f : uniform() * 0.1f;
  float *dl, *db, *wa, *wb;
  int32_t *ia, *ib;
  CHECK(hipMalloc(&dl, l.size() * 4) == hipSuccess && hipMalloc(&db, e * 4) == hipSuccess &&
        hipMalloc(&wa, n * k * 4) == hipSuccess && hipMalloc(&wb, n * k * 4) == hipSuccess &&
        hipMalloc(&ia, n * k * 4) == hipSuccess && hipMalloc(&ib, n * k * 4) == hipSuccess);
  CHECK(hipMemcpy(dl, l.data(), l.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(db, b.data(), e * 4, hipMemcpyHostToDevice) == hipSuccess);
  reference<<<n, kThreads>>>(dl, db, e, k, 1.5f, ia, wa);
  CHECK(hipGetLastError() == hipSuccess && ops_router(dl, db, e, k, 1.5f, n, ib, wb, nullptr));
  std::vector<int32_t> xa(n * k), xb(n * k);
  std::vector<float> va(n * k), vb(n * k);
  CHECK(hipMemcpy(xa.data(), ia, n * k * 4, hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(xb.data(), ib, n * k * 4, hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(va.data(), wa, n * k * 4, hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(vb.data(), wb, n * k * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(xa == xb);
  for (size_t i = 0; i < va.size(); i++) CHECK(std::memcmp(&va[i], &vb[i], 4) == 0);
  for (void* p : {(void*) dl, (void*) db, (void*) wa, (void*) wb, (void*) ia, (void*) ib})
    CHECK(hipFree(p) == hipSuccess);
}

// logits no pick is right for - NaNs, every one -inf, +inf among them: the picks are garbage (the step is refused
// elsewhere: its activations' f16 flag) but every id is an expert's - what reads them next indexes by them (Qwen's
// tiles on the GPU, with no host between). both routers (block.h's wave_pick)
void hostile(uint32_t e, uint32_t k) {
  constexpr uint32_t kTok = 6;
  std::vector<float> l((size_t) kTok * e), b(e, 0.0f);
  for (float& v : l) v = uniform() * 8;
  for (uint32_t i = 0; i < e; i++) l[0 * e + i] = NAN;                // every logit a NaN
  for (uint32_t i = 0; i < e; i += 3) l[1 * e + i] = NAN;             // NaNs among them
  for (uint32_t i = 0; i < e; i++) l[2 * e + i] = -INFINITY;          // none greater than another
  l[3 * e + e / 2] = INFINITY, l[3 * e + e - 1] = INFINITY;           // +inf (a softmax's inf - inf: NaNs)
  l[4 * e + 0] = NAN;                                                 // one NaN, the first
  float *dl, *db, *w;
  int32_t *ids, *rows;
  CHECK(hipMalloc(&dl, l.size() * 4) == hipSuccess && hipMalloc(&db, e * 4) == hipSuccess &&
        hipMalloc(&w, kTok * k * 4) == hipSuccess && hipMalloc(&ids, kTok * k * 4) == hipSuccess &&
        hipMalloc(&rows, kTok * k * 4) == hipSuccess);
  CHECK(hipMemcpy(dl, l.data(), l.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(db, b.data(), e * 4, hipMemcpyHostToDevice) == hipSuccess);
  std::vector<int32_t> got(kTok * k);
  for (int which = 0; which < 2; which++) {
    CHECK(hipMemset(ids, 0xFF, kTok * k * 4) == hipSuccess);
    CHECK(which ? q4x_route(dl, kTok, e, k, ids, w, rows, nullptr) : ops_router(dl, db, e, k, 1.5f, kTok, ids, w, nullptr));
    CHECK(hipMemcpy(got.data(), ids, kTok * k * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t i = 0; i < kTok * k; i++) {
      if (got[i] < 0 || (uint32_t) got[i] >= e)
        std::fprintf(stderr, "  %s, %u experts, k %u: token %u's pick %u is %d\n", which ? "q4x_route" : "ops_router", e,
                     k, i / k, i % k, got[i]);
      CHECK(got[i] >= 0 && (uint32_t) got[i] < e);
    }
  }
  for (void* p : {(void*) dl, (void*) db, (void*) w, (void*) ids, (void*) rows}) CHECK(hipFree(p) == hipSuccess);
}

}  // namespace

int main() {
  std::printf("[test_router_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_router_gpu] skipped: no GPU\n");
    return 77;
  }
  uint32_t cases = 0;
  for (uint32_t e : {384u, 256u, 300u, 33u, 7u})
    for (uint32_t k = 1; k <= OPS_MAX_TOPK && k <= e; k++)
      for (int kind = 0; kind < 3; kind++) check(kind ? 5 : 64, e, k, kind), cases++;
  std::printf("  %u cases (experts 7 .. 384, k 1 .. %d, random / tied / all equal logits): the picks and weights the"
              " reference's bits\n", cases, OPS_MAX_TOPK);
  for (uint32_t e : {512u, 384u, 33u})
    for (uint32_t k : {1u, 6u, 10u}) hostile(e, k);
  std::printf("  NaN, -inf and +inf logits, both routers: every pick an expert's id\n");
  std::printf("[test_router_gpu] ok\n");
  return 0;
}
