// latent sparse attention's read on the GPU (osmos/gpu/lsa_ops.h) against the host's (osmos/lsa.h): the same list,
// position for position - on tools/convert/lsa_cases.py's cases (OSMOS_LSA_CASES: -vibes' keys under its maps, the
// buckets the host's lsa_block makes of them), then on random buckets as many as 1M and 10M tokens close (every
// read checked; its time printed). skipped (77) without a GPU
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
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

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / 16777216.0f - 0.5f;
}

struct Dev {
  float *cent = nullptr, *iq = nullptr;
  int32_t* mem = nullptr;
  uint32_t *out = nullptr, *n = nullptr;
};

// every query of qs (position, index vector) read on both, the lists compared; the GPU's mean time a read
static double compare(const std::vector<float>& cent, const std::vector<int32_t>& mem, uint32_t blocks,
                      const std::vector<std::pair<uint32_t, std::vector<float>>>& qs, const char* what) {
  Dev d;
  CHECK(hipMalloc((void**) &d.cent, cent.size() * 4) == hipSuccess && hipMalloc((void**) &d.mem, mem.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &d.iq, LSA_DIM * 4) == hipSuccess && hipMalloc((void**) &d.out, LSA_READ_MAX * 4) == hipSuccess &&
        hipMalloc((void**) &d.n, 4) == hipSuccess);
  CHECK(hipMemcpy(d.cent, cent.data(), cent.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(d.mem, mem.data(), mem.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  LsaScratch* s = lsa_scratch_create(blocks * LSA_BUCKETS);
  CHECK(s);
  std::vector<uint32_t> host(LSA_READ_MAX), gpu(LSA_READ_MAX);
  double ms = 0;
  for (const auto& q : qs) {
    const uint32_t nh = lsa_read(q.second.data(), q.first, cent.data(), mem.data(), blocks, host.data());
    CHECK(hipMemcpy(d.iq, q.second.data(), LSA_DIM * 4, hipMemcpyHostToDevice) == hipSuccess);
    uint32_t ng = 0;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(lsa_read_gpu(d.iq, q.first, d.cent, d.mem, blocks, s, d.out, d.n, nullptr) &&
          hipDeviceSynchronize() == hipSuccess);
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(hipMemcpy(&ng, d.n, 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(gpu.data(), d.out, ng * 4, hipMemcpyDeviceToHost) == hipSuccess);
    if (ng != nh || std::memcmp(gpu.data(), host.data(), nh * 4)) {
      uint32_t i = 0;
      while (i < nh && i < ng && gpu[i] == host[i]) i++;
      std::printf("  %s: p %u: the GPU's read %u positions, the host's %u, first difference at %u\n", what, q.first, ng,
                  nh, i);
      CHECK(false);
    }
  }
  lsa_scratch_destroy(s);
  hipFree(d.cent), hipFree(d.mem), hipFree(d.iq), hipFree(d.out), hipFree(d.n);
  return ms / qs.size();
}

static void cases(const char* dir) {
  DIR* dd = opendir(dir);
  for (struct dirent* e; dd && (e = readdir(dd));) {
    const std::string name = e->d_name;
    if (name.size() < 5 || name.substr(name.size() - 4) != ".bin") continue;
    FILE* f = std::fopen((std::string(dir) + "/" + name).c_str(), "rb");
    CHECK(f);
    std::fseek(f, 0, SEEK_END);
    std::vector<unsigned char> b((size_t) std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    CHECK(std::fread(b.data(), 1, b.size(), f) == b.size());
    std::fclose(f);
    uint32_t T, Q, G;
    std::memcpy(&T, &b[4], 4), std::memcpy(&Q, &b[8], 4), std::memcpy(&G, &b[12], 4);
    const float* ik = (const float*) &b[16];
    std::vector<float> cent((size_t) G * LSA_BUCKETS * LSA_DIM);
    std::vector<int32_t> mem((size_t) G * LSA_BUCKETS * LSA_CAP);
    for (uint32_t g = 0; g < G; g++)
      CHECK(lsa_block(ik + (size_t) g * LSA_WINDOW * LSA_DIM, g, &cent[(size_t) g * LSA_BUCKETS * LSA_DIM],
                      &mem[(size_t) g * LSA_BUCKETS * LSA_CAP]));
    size_t off = 16 + (size_t) T * LSA_DIM * 4 + LSA_IN * 4 + LSA_DIM * 4 + (size_t) G * LSA_BUCKETS * LSA_DIM * 4 +
                 (size_t) G * LSA_BUCKETS * LSA_CAP * 4;
    std::vector<std::pair<uint32_t, std::vector<float>>> qs;
    for (uint32_t i = 0; i < Q; i++) {
      uint32_t p, nr;
      std::memcpy(&p, &b[off], 4);
      std::vector<float> iq(LSA_DIM);
      std::memcpy(iq.data(), &b[off + 4], LSA_DIM * 4);
      std::memcpy(&nr, &b[off + 4 + LSA_DIM * 4], 4);
      off += 8 + LSA_DIM * 4 + (size_t) nr * 4;
      qs.push_back({p, iq});
    }
    const double ms = compare(cent, mem, G, qs, name.c_str());
    std::printf("  %s: %u reads (T %u, %u blocks) the host's lists, position for position; %.3f ms a read\n",
                name.c_str(), Q, T, G, ms);
  }
  if (dd) closedir(dd);
}

// random normed centroids, members a block's own (8 a bucket, some short), queries at the context's end and before
static void random_scale(uint32_t blocks, const char* what) {
  std::vector<float> cent((size_t) blocks * LSA_BUCKETS * LSA_DIM);
  std::vector<int32_t> mem((size_t) blocks * LSA_BUCKETS * LSA_CAP, -1);
  for (size_t b = 0; b < (size_t) blocks * LSA_BUCKETS; b++) {
    double n = 0;
    for (int d = 0; d < LSA_DIM; d++) {
      const float v = uniform();
      cent[b * LSA_DIM + d] = v, n += (double) v * v;
    }
    for (int d = 0; d < LSA_DIM; d++) cent[b * LSA_DIM + d] /= (float) std::sqrt(n);
    const uint32_t g = (uint32_t) (b / LSA_BUCKETS), k = (uint32_t) (b % LSA_BUCKETS);
    for (int s = 0; s < 8 - (int) (k % 3); s++) mem[b * LSA_CAP + s] = (int32_t) (g * LSA_WINDOW + k * 8 + s);
  }
  std::vector<std::pair<uint32_t, std::vector<float>>> qs;
  const uint32_t T = blocks * LSA_WINDOW + 300;
  for (uint32_t p : {T - 1, T / 2, (uint32_t) LSA_WINDOW * 3 + 7, 2000u}) {
    std::vector<float> iq(LSA_DIM);
    double n = 0;
    for (auto& v : iq) v = uniform(), n += (double) v * v;
    for (auto& v : iq) v /= (float) std::sqrt(n);
    qs.push_back({p, iq});
  }
  // exact ties: two buckets of one centroid, the lower one first
  std::memcpy(&cent[(size_t) 5 * LSA_DIM], &cent[(size_t) 3 * LSA_DIM], LSA_DIM * 4);
  const double ms = compare(cent, mem, blocks, qs, what);
  std::printf("  %s: %u blocks (%.1fM tokens, %u buckets): 4 reads the host's lists; %.2f ms a read\n", what, blocks,
              blocks * (double) LSA_WINDOW / 1e6, blocks * LSA_BUCKETS, ms);
}

// the index on the GPU against lsa_index: a key's input as the forward holds it, a query's G heads a KV head summed -
// bits for bits, BF16 and F32. Qwen's (4 KV heads of 256, its first 64 rotary; 6 a group) and LFM3's (4 of 128, the
// zero tail's 32..63 and 96..127 free; 4 a group)
static void index_check(const char* dir, uint32_t hd, LsaDims dims, uint32_t G, const char* what) {
  const uint32_t heads = 4, free = (dims.hi0 - dims.lo0) + (dims.hi1 - dims.lo1), in = heads * free;
  auto dim_of = [&](uint32_t j) { return j < dims.hi0 - dims.lo0 ? dims.lo0 + j : dims.lo1 + (j - (dims.hi0 - dims.lo0)); };
  std::vector<float> W((size_t) in * LSA_DIM), key(heads * hd), qry(heads * G * hd), x(in), host(LSA_DIM), gpu(LSA_DIM);
  for (auto& v : W) v = uniform() * 0.1f;
  for (auto& v : key) v = uniform();
  for (auto& v : qry) v = uniform();
  if (dir && in == LSA_IN) {  // a real key's input and map (Qwen's), when the cases are here
    DIR* dd = opendir(dir);
    for (struct dirent* e; dd && (e = readdir(dd));) {
      const std::string n = e->d_name;
      if (n.size() < 5 || n.substr(n.size() - 4) != ".bin") continue;
      FILE* f = std::fopen((std::string(dir) + "/" + n).c_str(), "rb");
      FILE* fw = std::fopen((std::string(dir) + "/" + n + ".W").c_str(), "rb");
      uint32_t T = 0;
      CHECK(f && fw && std::fseek(f, 4, SEEK_SET) == 0 && std::fread(&T, 4, 1, f) == 1);
      CHECK(std::fseek(f, 16 + (long) T * LSA_DIM * 4, SEEK_SET) == 0 && std::fread(x.data(), 4, in, f) == in);
      CHECK(std::fread(W.data(), 4, W.size(), fw) == W.size());
      std::fclose(f), std::fclose(fw);
      for (uint32_t i = 0; i < in; i++) key[i / free * hd + dim_of(i % free)] = x[i];
      break;
    }
    if (dd) closedir(dd);
  }
  float *dW, *dk, *dq, *dout;
  CHECK(hipMalloc((void**) &dW, W.size() * 4) == hipSuccess && hipMalloc((void**) &dk, key.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &dq, qry.size() * 4) == hipSuccess && hipMalloc((void**) &dout, LSA_DIM * 4) == hipSuccess);
  CHECK(hipMemcpy(dW, W.data(), W.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dk, key.data(), key.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dq, qry.data(), qry.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  for (int q = 0; q < 2; q++) {
    for (uint32_t i = 0; i < in; i++) {
      const uint32_t h = i / free, j = dim_of(i % free);
      float v = 0.0f;
      for (uint32_t g = 0; g < (q ? G : 1); g++) v += q ? qry[(h * G + g) * hd + j] : key[h * hd + j];
      x[i] = v;
    }
    lsa_index(x.data(), in, W.data(), !q, host.data());
    CHECK(lsa_index_gpu(q ? dq : dk, 1, q ? heads * G * hd : heads * hd, heads, q ? G : 1, hd, &dims, dW, !q, dout,
                        nullptr));
    CHECK(hipMemcpy(gpu.data(), dout, LSA_DIM * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(!std::memcmp(gpu.data(), host.data(), LSA_DIM * 4));
  }
  std::printf("  the index, %s: a key (BF16, %s) and a query (%u heads summed) lsa_index's bits\n", what,
              dir && in == LSA_IN ? "-vibes' own" : "random", G);
  hipFree(dW), hipFree(dk), hipFree(dq), hipFree(dout);
}

int main() {
  std::printf("[test_lsa_gpu] start\n");
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || !n) {
    std::printf("[test_lsa_gpu] skipped: no GPU\n");
    return 77;
  }
  const char* dir = std::getenv("OSMOS_LSA_CASES");
  index_check(dir, 256, LsaDims{64, 256, 0, 0}, 6, "Qwen's (past its rotary prefix)");
  index_check(dir, 128, LsaDims{32, 64, 96, 128}, 4, "LFM3's (its zero tail)");
  { const LsaDims over{0, 256, 0, 0}; float* p = nullptr; CHECK(!lsa_index_gpu(p, 1, 1024, 4, 1, 256, &over, p, true, p, nullptr)); }
  if (dir)
    cases(dir);
  else
    std::printf("  lsa.py's cases: no OSMOS_LSA_CASES, not read\n");
  random_scale(3, "3 blocks (all buckets read)");
  random_scale(2048, "1M");
  random_scale(20480, "10M");
  std::printf("[test_lsa_gpu] ok\n");
  return 0;
}
