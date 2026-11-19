// lsa_read_batch_gpu (osmos/gpu/lsa_ops.h): a prompt's reads in a batch against a host read with the same F32 scores
// (each product and sum rounded on its own, dims in order) and lsa_read's rules otherwise - the lists exactly: the
// buckets of blocks lsa_block made from random index vectors; tokens past 1,024 visible buckets (the radix select)
// and under (every bucket); a block's centroids copied onto another's (ties: the lower bucket); sub-batches smaller
// than the batch the same lists. skipped (77) without a GPU
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/lsa_ops.h"
#include "osmos/lsa.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

static uint32_t rng = 97;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 24) * 2.0f - 1.0f;
}

static void normalize(float* v) {
  double s = 0;
  for (int d = 0; d < LSA_DIM; d++) s += (double) v[d] * v[d];
  const float r = (float) (1.0 / std::sqrt(s));
  for (int d = 0; d < LSA_DIM; d++) v[d] *= r;
}

// the host's read with F32 scores: lsa_read's rules
static uint32_t host_read(const float* iq, uint32_t p, const std::vector<float>& cent, const std::vector<int32_t>& mem,
                          uint32_t blocks, uint32_t* out) {
  const uint32_t win0 = p + 1 > LSA_WINDOW ? p + 1 - LSA_WINDOW : 0;
  uint32_t vis = p / LSA_WINDOW;
  vis = (vis < blocks ? vis : blocks) * LSA_BUCKETS;
  std::vector<std::pair<float, uint32_t>> pk(vis);
  for (uint32_t b = 0; b < vis; b++) {
    float s = 0.0f;
    for (int d = 0; d < LSA_DIM; d++) {
      const float m = cent[(size_t) b * LSA_DIM + d] * iq[d];
      s = s + m;
    }
    pk[b] = {s, b};
  }
  std::stable_sort(pk.begin(), pk.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  uint32_t n = 0;
  for (uint32_t i = 0; i < vis && i < LSA_TOP_P && n < LSA_TOP_K; i++)
    for (int k = 0; k < LSA_CAP && n < LSA_TOP_K; k++) {
      const int32_t m = mem[(size_t) pk[i].second * LSA_CAP + k];
      if (m >= LSA_SINK && (uint32_t) m < win0) out[n++] = (uint32_t) m;
    }
  for (uint32_t q = win0; q <= p; q++) out[n++] = q;
  const uint32_t sinks = LSA_SINK < win0 ? LSA_SINK : win0 < p + 1 ? win0 : p + 1;
  for (uint32_t q = 0; q < sinks; q++) out[n++] = q;
  return n;
}

int main() {
  std::printf("[test_lsa_batch_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_lsa_batch_gpu] skipped: no GPU\n");
    return 77;
  }
  const uint32_t blocks = 40, max_b = blocks * LSA_BUCKETS;
  std::vector<float> cent((size_t) max_b * LSA_DIM), ik((size_t) LSA_WINDOW * LSA_DIM);
  std::vector<int32_t> mem((size_t) max_b * LSA_CAP);
  for (uint32_t b = 0; b < blocks; b++) {
    for (auto& x : ik) x = uniform();
    for (uint32_t i = 0; i < LSA_WINDOW; i++) normalize(&ik[(size_t) i * LSA_DIM]);
    CHECK(lsa_block(ik.data(), b, &cent[(size_t) b * LSA_BUCKETS * LSA_DIM], &mem[(size_t) b * LSA_BUCKETS * LSA_CAP]));
  }
  // ties: block 7's centroids those of block 3 (each a twin, the lower bucket first)
  std::memcpy(&cent[(size_t) 7 * LSA_BUCKETS * LSA_DIM], &cent[(size_t) 3 * LSA_BUCKETS * LSA_DIM],
              LSA_BUCKETS * LSA_DIM * 4);
  float *dc, *diq, *scr;
  int32_t *dm, *dn;
  uint32_t* dl;
  CHECK(hipMalloc((void**) &dc, cent.size() * 4) == hipSuccess && hipMalloc((void**) &dm, mem.size() * 4) == hipSuccess);
  CHECK(hipMemcpy(dc, cent.data(), cent.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dm, mem.data(), mem.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  struct Case {
    uint32_t p0, n, sub;
  } cases[] = {{40 * LSA_WINDOW - 200, 300, 300}, {40 * LSA_WINDOW - 200, 300, 37}, {600, 900, 128}, {3, 40, 40}};
  for (const Case& c : cases) {
    std::vector<float> iq((size_t) c.n * LSA_DIM);
    for (auto& x : iq) x = uniform();
    for (uint32_t t = 0; t < c.n; t++) normalize(&iq[(size_t) t * LSA_DIM]);
    CHECK(hipMalloc((void**) &diq, iq.size() * 4) == hipSuccess);
    CHECK(hipMalloc((void**) &scr, lsa_batch_scratch(c.sub, max_b) * 4) == hipSuccess);
    CHECK(hipMalloc((void**) &dl, (size_t) c.n * LSA_READ_MAX * 4) == hipSuccess);
    CHECK(hipMalloc((void**) &dn, (size_t) c.n * 4) == hipSuccess);
    CHECK(hipMemcpy(diq, iq.data(), iq.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(lsa_read_batch_gpu(diq, c.n, c.p0, dc, dm, blocks, scr, c.sub, max_b, dl, dn, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    std::vector<uint32_t> got((size_t) c.n * LSA_READ_MAX), want(LSA_READ_MAX);
    std::vector<int32_t> cnt(c.n);
    CHECK(hipMemcpy(got.data(), dl, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(cnt.data(), dn, cnt.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    uint32_t most = 0, picked = 0;
    for (uint32_t t = 0; t < c.n; t++) {
      const uint32_t n = host_read(&iq[(size_t) t * LSA_DIM], c.p0 + t, cent, mem, blocks, want.data());
      if ((uint32_t) cnt[t] != n || std::memcmp(&got[(size_t) t * LSA_READ_MAX], want.data(), n * 4)) {
        std::fprintf(stderr, "  token %u (position %u): count %d, the host's %u\n", t, c.p0 + t, cnt[t], n);
        CHECK(false);
      }
      most = std::max(most, n), picked += (c.p0 + t) / LSA_WINDOW * LSA_BUCKETS > LSA_TOP_P;
    }
    std::printf("  %u tokens at %u (sub-batches of %u): the host's lists exactly (the longest %u; %u past %u buckets)\n",
                c.n, c.p0, c.sub, most, picked, LSA_TOP_P);
    for (void* x : {(void*) diq, (void*) scr, (void*) dl, (void*) dn}) CHECK(hipFree(x) == hipSuccess);
  }
  CHECK(hipFree(dc) == hipSuccess && hipFree(dm) == hipSuccess);
  std::printf("[test_lsa_batch_gpu] ok\n");
  return 0;
}
