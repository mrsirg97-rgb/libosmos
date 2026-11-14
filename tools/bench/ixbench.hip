// ixbench: the indexer's kernels at a long context's shapes (0.7.3) - ops_index_score against ops_index_score_fp4
// and (0.7.4) ops_index_score_dot4 (128 tokens x 32 heads x 128, the forward's chunk at 1M: FWD_SCORE_BYTES / 4 MB),
// and ops_topk under a 2,048-block mask against ops_topk_rows over the same blocks listed, at 256K, 512K and 1M rows;
// synthetic FP4 rows. each timed as the median of 5 after a warm-up; the scores' bits compared (the plain kernel's).
//   ixbench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}

template <typename F>
static double time_ms(F f) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a) == hipSuccess && hipEventCreate(&b) == hipSuccess);
  f();
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<double> t;
  for (int i = 0; i < 5; i++) {
    CHECK(hipEventRecord(a) == hipSuccess);
    f();
    CHECK(hipEventRecord(b) == hipSuccess && hipEventSynchronize(b) == hipSuccess);
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess);
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  return t[2];
}

int main() {
  constexpr uint32_t T = 128, H = 32, HD = 128, BLOCK = 8, KEEP = 2048, K = 512;
  const uint32_t sizes[] = {262144, 524288, 1048576};
  const uint32_t RMAX = sizes[2], rb = (uint32_t) kv_row_bytes(KV_FP4_POW2, HD);
  std::vector<float> qh((size_t) T * H * HD), wh((size_t) T * H);
  for (float& v : qh) v = uniform();
  for (float& v : wh) v = uniform();
  float *q, *k, *w, *sa, *sb;
  int32_t *pos, *sel, *n_sel;
  uint8_t *kp, *qp;
  uint32_t *bits, *list, *n_list;
  const uint32_t words = (RMAX / BLOCK + 31) / 32;
  CHECK(hipMalloc(&q, qh.size() * 4) == hipSuccess && hipMalloc(&w, wh.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&k, (size_t) RMAX * HD * 4) == hipSuccess && hipMalloc(&kp, (size_t) RMAX * rb) == hipSuccess);
  CHECK(hipMalloc(&qp, (size_t) T * H * rb) == hipSuccess && hipMalloc(&pos, T * 4) == hipSuccess);
  CHECK(hipMalloc(&sa, (size_t) T * RMAX * 4) == hipSuccess && hipMalloc(&sb, (size_t) T * RMAX * 4) == hipSuccess);
  CHECK(hipMalloc(&sel, (size_t) T * OPS_MAX_SELECT * 4) == hipSuccess && hipMalloc(&n_sel, T * 4) == hipSuccess);
  CHECK(hipMalloc(&bits, (size_t) T * words * 4) == hipSuccess && hipMalloc(&list, (size_t) T * KEEP * 4) == hipSuccess);
  CHECK(hipMalloc(&n_list, T * 4) == hipSuccess);
  {  // the keys: uniform, in chunks (host memory)
    std::vector<float> kh((size_t) 65536 * HD);
    for (uint32_t r0 = 0; r0 < RMAX; r0 += 65536) {
      for (float& v : kh) v = uniform();
      CHECK(hipMemcpy(k + (size_t) r0 * HD, kh.data(), kh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    }
  }
  CHECK(hipMemcpy(q, qh.data(), qh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(ops_fp4_round(k, (uint64_t) RMAX * HD, KV_FP4_POW2_BLOCK, false, kp, HD, nullptr));
  CHECK(ops_fp4_round(q, qh.size(), KV_FP4_POW2_BLOCK, false, qp, HD, nullptr));
  CHECK(hipFree(k) == hipSuccess);
  std::printf("ixbench: %u tokens x %u heads x %u, FP4 keys\n", T, H, HD);
  for (uint32_t R : sizes) {
    std::vector<int32_t> ph(T);
    for (uint32_t t = 0; t < T; t++) ph[t] = (int32_t) (R - T + t);  // ratio 1: a token sees ~R rows
    CHECK(hipMemcpy(pos, ph.data(), T * 4, hipMemcpyHostToDevice) == hipSuccess);
    const double plain = time_ms([&] {
      CHECK(ops_index_score(q, kp, KV_FP4_POW2, w, pos, 1, R, H, HD, T, sa, R, nullptr));
    });
    const double fma = time_ms([&] {
      CHECK(ops_index_score_fp4(q, kp, w, pos, 1, R, H, HD, T, sb, R, nullptr, nullptr));
    });
    std::vector<float> fa((size_t) T * R), fb2((size_t) T * R);
    CHECK(hipMemcpy(fa.data(), sa, fa.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(fb2.data(), sb, fb2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    const auto differ = [&] {
      uint64_t n = 0;
      for (uint32_t t = 0; t < T; t++)
        for (uint32_t c = 0; c < (uint32_t) ph[t] + 1; c++)
          n += std::memcmp(&fa[(size_t) t * R + c], &fb2[(size_t) t * R + c], 4) != 0;
      return (unsigned long long) n;
    };
    const unsigned long long fdiff = differ();
    std::printf("  %7u rows: score plain %8.2f ms  fused (fp4) %8.2f ms (%.2fx, %llu bits differ)\n", R, plain, fma,
                plain / fma, fdiff);
    // 0.7.4: the int8 dots' (ops_index_score_dot4), its fallbacks counted on one more call
    CHECK(hipMemset(sb, 0, (size_t) T * R * 4) == hipSuccess);
    const double d4 = time_ms([&] {
      CHECK(ops_index_score_dot4(q, kp, w, pos, 1, R, H, HD, T, sb, R, nullptr, nullptr));
    });
    uint32_t* fbd;
    uint32_t nfb = 0;
    CHECK(hipMalloc(&fbd, 4) == hipSuccess && hipMemset(fbd, 0, 4) == hipSuccess);
    CHECK(ops_index_score_dot4(q, kp, w, pos, 1, R, H, HD, T, sb, R, fbd, nullptr));
    CHECK(hipMemcpy(&nfb, fbd, 4, hipMemcpyDeviceToHost) == hipSuccess && hipFree(fbd) == hipSuccess);
    CHECK(hipMemcpy(fb2.data(), sb, fb2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    std::printf("  %7u rows: score int8 dots (dot4) %8.2f ms (%.2fx plain, %.2fx fused; %llu bits differ, %u fallbacks)\n",
                R, d4, plain / d4, fma / d4, differ(), nfb);
    // the mask: KEEP random blocks a token (its last always), the list from the bits
    const uint32_t nb = R / BLOCK, wr = (nb + 31) / 32;
    std::vector<uint32_t> bh((size_t) T * wr, 0);
    for (uint32_t t = 0; t < T; t++) {
      uint32_t* bt = &bh[(size_t) t * wr];
      bt[(nb - 1) / 32] |= 1u << ((nb - 1) % 32);
      for (uint32_t kept = 1; kept < KEEP;) {
        rng = rng * 1664525u + 1013904223u;
        const uint32_t j = (rng >> 4) % nb;
        if (!((bt[j / 32] >> (j % 32)) & 1u)) bt[j / 32] |= 1u << (j % 32), kept++;
      }
    }
    CHECK(hipMemcpy(bits, bh.data(), bh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    const double lst = time_ms([&] { CHECK(ops_cand_list(bits, wr, T, list, KEEP, n_list, nullptr)); });
    const double tk = time_ms([&] {
      CHECK(ops_topk(sa, R, pos, 1, K, bits, BLOCK, wr, T, sel, OPS_MAX_SELECT, n_sel, nullptr));
    });
    const double tkr = time_ms([&] {
      CHECK(ops_topk_rows(sa, R, pos, 1, K, list, n_list, KEEP, BLOCK, T, sel, OPS_MAX_SELECT, n_sel, nullptr));
    });
    float* bm;
    CHECK(hipMalloc(&bm, (size_t) T * nb * 4) == hipSuccess);
    const double cb = time_ms([&] {
      CHECK(ops_cand_blocks(sa, R, pos, 1, BLOCK, KEEP, T, bits, wr, nullptr));
    });
    std::vector<uint32_t> c1((size_t) T * wr), c2(c1.size());
    CHECK(hipMemcpy(c1.data(), bits, c1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    const double cbm = time_ms([&] {
      CHECK(ops_cand_blocks_max(sa, R, pos, 1, BLOCK, KEEP, T, bits, wr, bm, nb, nullptr));
    });
    CHECK(hipMemcpy(c2.data(), bits, c2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipFree(bm) == hipSuccess);
    std::printf("  %7u rows: candidate blocks %6.2f ms, their keys folded once %6.2f ms (%.1fx, the bits %s)\n", R, cb,
                cbm, cb / cbm, c1 == c2 ? "the same" : "DIFFER");
    CHECK(hipMemcpy(bits, bh.data(), bh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);  // the random mask again
    const double tkp = time_ms([&] {
      CHECK(ops_topk(sa, R, pos, 1, K, nullptr, BLOCK, wr, T, sel, OPS_MAX_SELECT, n_sel, nullptr));
    });
    std::printf("  %7u rows: top-k plain %6.2f  masked %6.2f  over the list %6.2f (+ the list %.2f) ms\n", R, tkp, tk, tkr,
                lst);
  }
  return 0;
}
