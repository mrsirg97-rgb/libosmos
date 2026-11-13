// q4x_layercmp <first shard> <dump dir> <n> [step]: where Qwen3.8-Flash-Next's forward parts from llama.cpp's on a
// long text - each layer's residual (l_last) against oracle/qwen_dump's of the same ids (<dump>/ids.txt, its
// l_last-<layer>.bin), its error a layer's own: every layer's output replaced by llama.cpp's before the next reads it
// (teacher forcing; the caches built from llama.cpp's states). the error by row position (0-2K, 2-4K, 4-8K, 8K on):
// a layer whose late rows part and early ones do not is where a long context's arithmetic is not llama.cpp's - the
// full-attention layers' QSA (every 4th), DESIGN.md "0.6.10". TEACH=0: no forcing (the error as it accumulates).
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/q4x_layercmp.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/q4x_layercmp
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/gpu.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr int kBuckets = 4;
static const uint32_t kEdge[kBuckets + 1] = {0, 2048, 4096, 8192, 1u << 31};

struct Dump {  // a QDMP file mapped: its values, rows of `width`
  const float* v = nullptr;
  uint64_t rows = 0, width = 0;
};

struct Cmp {
  std::string dir;
  uint32_t at = 0;  // the step's first row
  bool teach = true;
  std::vector<Dump> ref;
  std::vector<double> num, den;  // [layer][bucket]
};

static Dump map_dump(const std::string& path) {
  Dump d;
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return d;
  struct stat st;
  CHECK(fstat(fd, &st) == 0);
  void* m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  CHECK(m != MAP_FAILED);
  close(fd);
  const char* p = (const char*) m;
  CHECK(!std::memcmp(p, "QDMP", 4));
  uint32_t dims;
  int64_t ne[4];
  std::memcpy(&dims, p + 4, 4), std::memcpy(ne, p + 8, 32);
  uint64_t all = 1;
  for (int i = 0; i < 4; i++) all *= (uint64_t) (ne[i] ? ne[i] : 1);
  d.v = (const float*) (p + 40);
  d.rows = (uint64_t) (dims >= 2 ? ne[dims - 1] : 1);
  d.width = all / d.rows;
  return d;
}

static void probe(const char* step, int layer, float* dev, uint64_t n, void* arg) {
  Cmp* c = (Cmp*) arg;
  if (std::strcmp(step, "l_last") || layer < 0) return;
  if ((size_t) layer >= c->ref.size()) c->ref.resize(layer + 1);
  Dump& r = c->ref[layer];
  if (!r.v) r = map_dump(c->dir + "/l_last-" + std::to_string(layer) + ".bin");
  CHECK(r.v && n % r.width == 0);
  const uint64_t k = n / r.width;
  CHECK(c->at + k <= r.rows);
  std::vector<float> ours(n);
  CHECK(hipMemcpy(ours.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  const float* theirs = r.v + (uint64_t) c->at * r.width;
  if (c->num.size() < (size_t) (layer + 1) * kBuckets) c->num.resize((layer + 1) * kBuckets), c->den.resize((layer + 1) * kBuckets);
  for (uint64_t t = 0; t < k; t++) {
    const uint32_t row = c->at + (uint32_t) t;
    int b = 0;
    while (row >= kEdge[b + 1]) b++;
    double nu = 0, de = 0;
    for (uint64_t i = 0; i < r.width; i++) {
      const double e = (double) ours[t * r.width + i] - theirs[t * r.width + i];
      nu += e * e, de += (double) theirs[t * r.width + i] * theirs[t * r.width + i];
    }
    c->num[layer * kBuckets + b] += nu, c->den[layer * kBuckets + b] += de;
  }
  if (c->teach) CHECK(hipMemcpy(dev, theirs, n * 4, hipMemcpyHostToDevice) == hipSuccess);
}

int main(int argc, char** argv) {
  if (argc < 4) return std::fprintf(stderr, "usage: q4x_layercmp <first shard> <dump dir> <n> [step]\n"), 2;
  CHECK(gpu_sleep_on_waits());
  Cmp c;
  c.dir = argv[2];
  c.teach = !std::getenv("TEACH") || std::atoi(std::getenv("TEACH")) != 0;
  const uint32_t n = (uint32_t) std::atoi(argv[3]), step = argc > 4 ? (uint32_t) std::atoi(argv[4]) : 2048;
  std::vector<int32_t> ids;
  FILE* f = std::fopen((c.dir + "/ids.txt").c_str(), "r");
  for (int32_t x; f && std::fscanf(f, "%d,", &x) == 1;) ids.push_back(x);
  if (f) std::fclose(f);
  CHECK(ids.size() >= n);
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(argv[1], gerr);
  Q4xWeights* w = g ? q4x_weights_create(g, nullptr, werr) : nullptr;
  if (!w) return std::fprintf(stderr, "q4x_layercmp: %s%s\n", gerr, werr), 1;
  const Q4xOptions qo = {n, step, false, nullptr, 0, nullptr};
  Q4xForward* fw = q4x_forward_create_with(w, nullptr, &qo, ferr);
  Q4xSeq* s = fw ? q4x_seq_create(fw, ferr) : nullptr;
  if (!s) return std::fprintf(stderr, "q4x_layercmp: %s\n", ferr), 1;
  q4x_forward_probe(fw, probe, &c);
  for (uint32_t at = 0; at < n; at += step) {
    const uint32_t k = n - at < step ? n - at : step;
    c.at = at;
    if (!q4x_batch(fw, s, ids.data() + at, k, 0, ferr)) return std::fprintf(stderr, "q4x_layercmp: %s\n", ferr), 1;
    std::fprintf(stderr, "  %u of %u\n", at + k, n);
  }
  std::printf("layer  %s   rel L2 by rows: 0-2K      2-4K      4-8K      8K+   (late / early)\n", c.teach ? "own" : "acc");
  for (size_t l = 0; l * kBuckets < c.num.size(); l++) {
    double e[kBuckets];
    for (int b = 0; b < kBuckets; b++)
      e[b] = c.den[l * kBuckets + b] > 0 ? std::sqrt(c.num[l * kBuckets + b] / c.den[l * kBuckets + b]) : 0;
    std::printf("%5zu %s %21.3e %9.3e %9.3e %9.3e   %5.2f\n", l, l % 4 == 3 ? "attn" : "    ", e[0], e[1], e[2], e[3],
                e[0] > 0 ? e[3] / e[0] : 0);
  }
  q4x_seq_destroy(s);
  q4x_forward_destroy(fw);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
