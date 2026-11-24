// what a long context's rotation does to a short one (osmos/lfm3/forward.h: past the native length the full layers'
// rotation is YaRN's at max_ctx / native, every position's): lfm3_yarn_check <GGUF> <fixture> <ctx>... - the
// fixture's ids (oracle/lfm3_forward.py's: its last 64 rows' logits the reference's) through a forward made for each
// context, a read of them all; the last rows' logits against the reference's and against the native context's
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"
}

#include "../../tests/gpu/oracle_fixture.h"

static double kl(const float* p, const float* q, uint32_t V) {
  double mp = -INFINITY, mq = -INFINITY, zp = 0, zq = 0, k = 0;
  for (uint32_t i = 0; i < V; i++) mp = std::fmax(mp, p[i]), mq = std::fmax(mq, q[i]);
  for (uint32_t i = 0; i < V; i++) zp += std::exp(p[i] - mp), zq += std::exp(q[i] - mq);
  for (uint32_t i = 0; i < V; i++) {
    const double lp = p[i] - mp - std::log(zp), lq = q[i] - mq - std::log(zq);
    k += std::exp(lp) * (lp - lq);
  }
  return k;
}

static uint32_t argmax(const float* x, uint32_t V) {
  uint32_t b = 0;
  for (uint32_t i = 1; i < V; i++) b = x[i] > x[b] ? i : b;
  return b;
}

int main(int argc, char** argv) {
  char err[OSMOS_ERR] = "", gerr[OSMOS_ERR] = "";
  if (argc < 4) return std::fprintf(stderr, "usage: lfm3_yarn_check <GGUF> <fixture> <ctx>...\n"), 2;
  Gguf* g = gguf_create(argv[1], gerr);
  CHECK(g && load(argv[2]));
  Lfm3Weights* w = lfm3_weights_create(g, err);
  CHECK(w);
  const std::vector<float>& fid = A("ids");
  const uint32_t T = (uint32_t) fid.size(), V = (uint32_t) w->geo.n_vocab, last = 64;
  std::vector<int32_t> ids(T);
  for (uint32_t t = 0; t < T; t++) ids[t] = (int32_t) fid[t];
  const std::vector<float>& ref = A("logits");
  const size_t ref_rows = fx["rows"].v.size();
  std::vector<float> native;
  for (int a = 3; a < argc; a++) {
    const uint32_t ctx = (uint32_t) strtoul(argv[a], nullptr, 10);
    const Lfm3Options o = {T, ctx, last, LFM3_HEAD_FILE, nullptr, 0, nullptr, 0, 0, 0};
    Lfm3Forward* f = lfm3_forward_create(w, &o, err);
    CHECK(f);
    Lfm3Seq* s = lfm3_seq_create(f, err);
    CHECK(s && lfm3_step(f, s, ids.data(), T, last, err));
    const float* got = lfm3_logits(f);
    if (native.empty()) native.assign(got, got + (size_t) last * V);
    double kr = 0, kn = 0, krm = 0;
    uint32_t sr = 0, sn = 0;
    for (uint32_t r = 0; r < last; r++) {
      const float *p = ref.data() + (ref_rows - last + r) * (size_t) V, *q = got + (size_t) r * V;
      const double k1 = kl(p, q, V), k2 = kl(native.data() + (size_t) r * V, q, V);
      kr += k1, krm = std::fmax(krm, k1), kn += k2;
      sr += argmax(p, V) == argmax(q, V), sn += argmax(native.data() + (size_t) r * V, V) == argmax(q, V);
    }
    std::printf("  ctx %8u (YaRN x%.2f): against the reference KL mean %.3g, max %.3g, top-1 %u/%u; against the "
                "native context's KL mean %.3g, top-1 %u/%u\n", ctx, ctx > w->geo.native_ctx ? (double) ctx / w->geo.native_ctx : 1.0,
                kr / last, krm, sr, last, kn / last, sn, last);
    lfm3_seq_destroy(s), lfm3_forward_destroy(f);
  }
  lfm3_weights_destroy(w), gguf_destroy(g);
  return 0;
}
