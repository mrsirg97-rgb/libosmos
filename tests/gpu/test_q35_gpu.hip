// the qwen35 forward against its reference (tools/convert/q35_ref.py: HF's Qwen3.5 on the NVFP4 file's values,
// BF16 activations): the reference's tokens in one step, every layer's residual (relative error, cosine) and the
// logits (top-1, KL from the reference). the model Q35_MODEL (the NVFP4 GGUF), the reference Q35_REF; skipped (77)
// without them. Q35_CHUNK > 0: the same tokens in steps of that many (the state carried), the same answers within
// the step's own rounding
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/q35/forward.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

static std::vector<char> slurp(const std::string& p) {
  std::vector<char> b;
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) return b;
  std::fseek(f, 0, SEEK_END);
  b.resize((size_t) std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  if (std::fread(b.data(), 1, b.size(), f) != b.size()) b.clear();
  std::fclose(f);
  return b;
}

struct Tap {
  const float* ref;  // layers x T x D
  uint32_t T, D, row0;
  std::vector<double> err, cos;
  std::vector<float> host;
};

static void tap(void* arg, int l, const float* dev, uint32_t n, uint32_t D) {
  Tap* t = (Tap*) arg;
  t->host.resize((size_t) n * D);
  CHECK(hipMemcpy(t->host.data(), dev, t->host.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t i = 0; i < n; i++) {
    const float* r = t->ref + ((size_t) l * t->T + t->row0 + i) * D;
    const float* o = t->host.data() + (size_t) i * D;
    double rr = 0, oo = 0, ro = 0, dd = 0;
    for (uint32_t c = 0; c < D; c++) rr += (double) r[c] * r[c], oo += (double) o[c] * o[c], ro += (double) r[c] * o[c],
                                      dd += ((double) r[c] - o[c]) * ((double) r[c] - o[c]);
    t->err[l] = std::fmax(t->err[l], std::sqrt(dd / rr));
    t->cos[l] = std::fmin(t->cos[l], ro / std::sqrt(rr * oo));
  }
}

int main() {
  std::printf("[test_q35_gpu] start\n");
  const char* mp = std::getenv("Q35_MODEL");
  const char* rp = std::getenv("Q35_REF");
  int devs = 0;
  if (!mp || !rp || hipGetDeviceCount(&devs) != hipSuccess || !devs) {
    std::printf("[test_q35_gpu] skipped: no Q35_MODEL / Q35_REF, or no GPU\n");
    return 77;
  }
  const std::string ref = rp;
  std::vector<char> ids = slurp(ref + "/ids.i32"), hb = slurp(ref + "/h.f32"), lb = slurp(ref + "/logits.f32");
  CHECK(!ids.empty() && !lb.empty());
  char err[Q35_ERR] = "";
  Gguf* g = gguf_create(mp, err);
  CHECK(g || (std::fprintf(stderr, "%s\n", err), false));
  Q35Weights* w = q35_weights_create(g, err);
  CHECK(w || (std::fprintf(stderr, "%s\n", err), false));
  const uint32_t T = (uint32_t) (ids.size() / 4), D = (uint32_t) w->geo.n_embd, V = (uint32_t) w->geo.n_vocab;
  const uint32_t L = (uint32_t) w->geo.n_layers;
  CHECK(lb.size() == (size_t) T * V * 4);
  const bool have_h = hb.size() == (size_t) L * T * D * 4;
  const char* cs = std::getenv("Q35_CHUNK");
  const uint32_t chunk = cs && std::atoi(cs) > 0 ? (uint32_t) std::atoi(cs) : T;
  Q35Forward* f = q35_forward_create(w, chunk, T < 4096 ? 4096 : T, err);
  CHECK(f || (std::fprintf(stderr, "%s\n", err), false));
  Q35Seq* s = q35_seq_create(f, err);
  CHECK(s || (std::fprintf(stderr, "%s\n", err), false));
  Tap t{(const float*) hb.data(), T, D, 0, std::vector<double>(L, 0.0), std::vector<double>(L, 1.0), {}};
  if (have_h) q35_forward_tap(f, tap, &t);
  const int32_t* tok = (const int32_t*) ids.data();
  const float* rl = (const float*) lb.data();
  double kl = 0, maxd = 0;
  uint32_t agree = 0;
  FILE* dump = std::getenv("Q35_DUMP") ? std::fopen(std::getenv("Q35_DUMP"), "wb") : nullptr;  // the rows' bits, to compare
  for (uint32_t t0 = 0; t0 < T; t0 += chunk) {
    const uint32_t n = T - t0 < chunk ? T - t0 : chunk;
    t.row0 = t0;
    CHECK(q35_step(f, s, tok + t0, n, n, err) || (std::fprintf(stderr, "%s\n", err), false));
    const float* lo = q35_logits(f);
    if (dump) CHECK(std::fwrite(lo, (size_t) V * 4, n, dump) == n);
    for (uint32_t i = 0; i < n; i++) {
      const float* a = rl + (size_t) (t0 + i) * V;
      const float* b = lo + (size_t) i * V;
      double ma = -1e30, mb = -1e30;
      uint32_t ia = 0, ib = 0;
      for (uint32_t v = 0; v < V; v++) {
        if (a[v] > ma) ma = a[v], ia = v;
        if (b[v] > mb) mb = b[v], ib = v;
        maxd = std::fmax(maxd, std::fabs((double) a[v] - b[v]));
      }
      double za = 0, zb = 0;
      for (uint32_t v = 0; v < V; v++) za += std::exp(a[v] - ma), zb += std::exp(b[v] - mb);
      double k = 0;
      for (uint32_t v = 0; v < V; v++) {
        const double pa = std::exp(a[v] - ma) / za;
        k += pa * ((a[v] - ma - std::log(za)) - (b[v] - mb - std::log(zb)));
      }
      kl += k, agree += ia == ib;
    }
  }
  if (dump) std::fclose(dump);
  if (have_h)
    for (uint32_t l = 0; l < L; l += (l < 4 || l + 4 >= L) ? 1 : 8)
      std::printf("  layer %2u (%s): residual's relative error %.4f, cosine %.6f\n", l,
                  w->geo.recurrent[l] ? "DeltaNet" : "attention", t.err[l], t.cos[l]);
  std::printf("  %u tokens (steps of %u): top-1 %u / %u, KL %.5f, largest logit gap %.3f\n", T, chunk, agree, T,
              kl / T, maxd);
  if (std::getenv("Q35_BENCH")) {  // the speeds: a fresh sequence, the reference's tokens as a prompt, then decode
    q35_forward_tap(f, nullptr, nullptr);  // the residuals' tap off: its sync and copy a layer are the check's, not the forward's
    Q35Seq* b = q35_seq_create(f, err);
    CHECK(b);
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0) == hipSuccess && hipEventCreate(&e1) == hipSuccess);
    float ms = 0;
    CHECK(hipEventRecord(e0, NULL) == hipSuccess);
    for (uint32_t t0 = 0; t0 < T; t0 += chunk) CHECK(q35_step(f, b, tok + t0, T - t0 < chunk ? T - t0 : chunk, 1, err));
    CHECK(hipEventRecord(e1, NULL) == hipSuccess && hipEventSynchronize(e1) == hipSuccess &&
          hipEventElapsedTime(&ms, e0, e1) == hipSuccess);
    std::printf("  prompt: %u tokens in %.1f ms, %.0f tokens/s\n", T, ms, T / (ms / 1e3));
    const int steps = std::getenv("Q35_STEPS") ? std::atoi(std::getenv("Q35_STEPS")) : 64;
    CHECK(hipEventRecord(e0, NULL) == hipSuccess);
    for (int i = 0; i < steps; i++) CHECK(q35_step(f, b, tok + (i % T), 1, 1, err));
    CHECK(hipEventRecord(e1, NULL) == hipSuccess && hipEventSynchronize(e1) == hipSuccess &&
          hipEventElapsedTime(&ms, e0, e1) == hipSuccess);
    std::printf("  decode at %u: %.2f ms a token, %.1f tokens/s\n", T, ms / steps, steps / (ms / 1e3));
    q35_seq_destroy(b);
  }
  q35_seq_destroy(s);
  q35_forward_destroy(f);
  q35_weights_destroy(w);
  gguf_destroy(g);
  CHECK(agree * 100 >= T * 95 && kl / T < 0.05);
  std::printf("[test_q35_gpu] ok\n");
  return 0;
}
