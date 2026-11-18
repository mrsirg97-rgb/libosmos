// Hy4-flash's iHC and Engram's ends (osmos/gpu/hy4_ops.h) against the reference's own code (oracle/hy4_pieces.py:
// modeling_hy_v4_flash.py's decoder layer with its attention and MLP stubbed, its head's collapse, its Engram
// layer - in F32 on the GGUF's own weights): one layer replayed step by step - Engram's rows to its residual, the
// attention's input and iHC's post, the MLP's input, the layer's output - and the head's collapse, each within a
// stated error of the reference's (the summation order alone differs: a few ulps of the largest value). Engram's
// projection (a matmul, not under test) on the host in double. ~/osmos-oracle/hy4_pieces.bin; skipped (77) without.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

// F32's summation order over a 12,288-long dot moves a result by a few ulps of the largest term; 4e-6 of an array's
// largest value is ~30 ulps of it - room for the order, none for a wrong term (one of 3,072 weighs ~3e-4)
constexpr double kTol = 4e-6;

struct Arr {
  std::vector<uint64_t> dims;
  std::vector<float> v;
};
static std::map<std::string, Arr> fx;

static bool load(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t head[2];
  CHECK(std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "HY4P", 4) == 0);
  CHECK(std::fread(head, 4, 2, f) == 2 && head[0] == 1);
  for (uint32_t i = 0; i < head[1]; i++) {
    uint32_t len = 0, nd = 0;
    CHECK(std::fread(&len, 4, 1, f) == 1 && len < 64);
    std::string name(len, '\0');
    CHECK(std::fread(name.data(), 1, len, f) == len && std::fread(&nd, 4, 1, f) == 1 && nd <= 4);
    Arr a;
    a.dims.resize(nd);
    CHECK(std::fread(a.dims.data(), 8, nd, f) == nd);
    uint64_t n = 1;
    for (uint64_t d : a.dims) n *= d;
    a.v.resize(n);
    CHECK(std::fread(a.v.data(), 4, n, f) == n);
    fx[name] = std::move(a);
  }
  std::fclose(f);
  return true;
}

static const std::vector<float>& A(const char* name) {
  CHECK(fx.count(name));
  return fx[name].v;
}

static float* up(const std::vector<float>& v) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, v.size() * sizeof(float)) == hipSuccess);
  CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

static float* scratch(size_t n) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, n * sizeof(float)) == hipSuccess);
  return d;
}

static std::vector<float> down(const float* d, size_t n) {
  std::vector<float> v(n);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(v.data(), d, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

// the largest error against the reference, as a fraction of the reference's largest value
static void match(const char* what, const std::vector<float>& got, const char* ref) {
  const std::vector<float>& want = A(ref);
  CHECK(got.size() == want.size());
  double err = 0, top = 0;
  size_t at = 0;
  for (size_t i = 0; i < got.size(); i++) {
    CHECK(std::isfinite(got[i]));
    const double e = std::fabs((double) got[i] - want[i]);
    if (e > err) err = e, at = i;
    top = std::fmax(top, std::fabs((double) want[i]));
  }
  std::printf("  %-36s %zu values: max error %.3g of %.4g (%.2g; at %zu: %.9g, the reference's %.9g)\n", what,
              got.size(), err, top, err / top, at, got[at], want[at]);
  CHECK(err <= kTol * top);
}

int main() {
  char path[512];
  std::snprintf(path, sizeof path, "%s/osmos-oracle/hy4_pieces.bin", getenv("HOME") ? getenv("HOME") : "");
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0 || !load(path)) {
    std::printf("[test_hy4_hc_gpu] skipped: no GPU or no %s (oracle/hy4_pieces.py)\n", path);
    return 77;
  }
  const uint32_t T = (uint32_t) fx["h"].dims[0], hc = (uint32_t) fx["h"].dims[1], D = (uint32_t) fx["h"].dims[2];
  const uint32_t cols = (uint32_t) fx["eg_rows"].dims[1], rd = (uint32_t) fx["eg_rows"].dims[2], E = cols * rd;
  const float eps = 1e-5f, hc_eps = 1e-6f, mag = 2.0f;  // config.json: rms_norm_eps, hc_eps, hc_magnitude
  std::printf("[test_hy4_hc_gpu] %u tokens, %u streams of %u; Engram %u rows of %u\n", T, hc, D, cols, rd);

  // refusals: nothing queued for a shape the ops do not run
  CHECK(!hy4_hc_pre(nullptr, nullptr, nullptr, nullptr, 8, 4, D, eps, hc_eps, mag, nullptr, T, nullptr, nullptr, 0));
  CHECK(!hy4_hc_post(nullptr, nullptr, nullptr, HY4_HC_MAX + 1, D, T, 0));

  // Engram: the rows to the projection's input, the projection (host), the residual into the streams
  float* rows = up(A("eg_rows"));
  float* lnorm = up(A("eg_lookup_norm"));
  float* emb = scratch((size_t) T * E);
  CHECK(hy4_engram_rows(rows, lnorm, cols, rd, T, emb, 0));
  const std::vector<float> e = down(emb, (size_t) T * E);
  const std::vector<float>& W = A("eg_kv");
  const size_t KV = (size_t) (hc + 1) * D;
  std::vector<float> kv((size_t) T * KV);
  for (uint32_t t = 0; t < T; t++)
    for (size_t r = 0; r < KV; r++) {
      double s = 0;
      for (uint32_t i = 0; i < E; i++) s += (double) W[r * E + i] * e[(size_t) t * E + i];
      kv[(size_t) t * KV + r] = (float) s;
    }
  float* h = up(A("h"));
  float* dkv = up(kv);
  float *kvb = up(A("eg_kv_b")), *gate = up(A("eg_gate")), *hs = up(A("eg_hc_scale")), *pg = up(A("eg_pre_gamma"));
  CHECK(hy4_engram_update(h, dkv, kvb, gate, hs, pg, hc, D, T, 0));
  match("Engram: the streams after it", down(h, (size_t) T * hc * D), "h_engram");

  // the attention's iHC: its input (normed) and post; the attention's output back; the MLP's the same; the output
  float *fa = up(A("hc_attn_fn")), *sa = up(A("hc_attn_scale")), *ba = up(A("hc_attn_base")), *na = up(A("attn_norm"));
  float *ff = up(A("hc_ffn_fn")), *sf = up(A("hc_ffn_scale")), *bf = up(A("hc_ffn_base")), *nf = up(A("ffn_norm"));
  float* x = scratch((size_t) T * D);
  float* post = scratch((size_t) T * hc);
  CHECK(hy4_hc_pre(h, fa, sa, ba, 2 * hc, hc, D, eps, hc_eps, mag, na, T, x, post, 0));
  match("iHC: the attention's input", down(x, (size_t) T * D), "attn_in");
  match("iHC: the attention's post", down(post, (size_t) T * hc), "attn_post");
  float* ya = up(A("y_attn"));
  CHECK(hy4_hc_post(h, ya, post, hc, D, T, 0));
  CHECK(hy4_hc_pre(h, ff, sf, bf, 2 * hc, hc, D, eps, hc_eps, mag, nf, T, x, post, 0));
  match("iHC: the MLP's input", down(x, (size_t) T * D), "ffn_in");
  float* yf = up(A("y_ffn"));
  CHECK(hy4_hc_post(h, yf, post, hc, D, T, 0));
  match("iHC: the layer's output streams", down(h, (size_t) T * hc * D), "h_out");

  // the head's collapse: pre alone (hc rows), then output_norm - from the layer's input streams
  float* h0 = up(A("h"));
  float *fo = up(A("out_hc_fn")), *so = up(A("out_hc_scale")), *bo = up(A("out_hc_base")), *no = up(A("output_norm"));
  CHECK(hy4_hc_pre(h0, fo, so, bo, hc, hc, D, eps, hc_eps, mag, no, T, x, nullptr, 0));
  match("the head's collapse", down(x, (size_t) T * D), "head");

  // a token's bits its own: the head's collapse of token 5 alone, the same as among the 16
  std::vector<float> all = down(x, (size_t) T * D);
  CHECK(hy4_hc_pre(h0 + (size_t) 5 * hc * D, fo, so, bo, hc, hc, D, eps, hc_eps, mag, no, 1, x, nullptr, 0));
  const std::vector<float> one = down(x, D);
  CHECK(std::memcmp(one.data(), all.data() + (size_t) 5 * D, D * sizeof(float)) == 0);
  std::printf("  a token alone: its collapse the same bits as among %u\n", T);
  std::printf("[test_hy4_hc_gpu] ok\n");
  return 0;
}
