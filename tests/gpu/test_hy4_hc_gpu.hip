// Hy4-flash's iHC and Engram's ends (osmos/gpu/hy4_ops.h) against the reference's own code (oracle/hy4_pieces.py:
// modeling_hy_v4_flash.py's decoder layer with its attention and MLP stubbed, its head's collapse, its Engram
// layer - in F32 on the GGUF's own weights): one layer replayed step by step - Engram's rows to its residual, the
// attention's input and iHC's post, the MLP's input, the layer's output - and the head's collapse, each within a
// stated error of the reference's (the summation order alone differs: a few ulps of the largest value). Engram's
// projection (a matmul, not under test) on the host in double. ~/osmos-oracle/hy4_pieces.bin; skipped (77) without.
extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

#include "hy4_fixture.h"

// F32's summation order over a 12,288-long dot moves a result by a few ulps of the largest term; 4e-6 of an array's
// largest value is ~30 ulps of it - room for the order, none for a wrong term (one of 3,072 weighs ~3e-4)
constexpr double kTol = 4e-6;

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
  match("Engram: the streams after it", down(h, (size_t) T * hc * D), "h_engram", kTol);

  // the attention's iHC: its input (normed) and post; the attention's output back; the MLP's the same; the output
  float *fa = up(A("hc_attn_fn")), *sa = up(A("hc_attn_scale")), *ba = up(A("hc_attn_base")), *na = up(A("attn_norm"));
  float *ff = up(A("hc_ffn_fn")), *sf = up(A("hc_ffn_scale")), *bf = up(A("hc_ffn_base")), *nf = up(A("ffn_norm"));
  float* x = scratch((size_t) T * D);
  float* post = scratch((size_t) T * hc);
  CHECK(hy4_hc_pre(h, fa, sa, ba, 2 * hc, hc, D, eps, hc_eps, mag, na, T, x, post, 0));
  match("iHC: the attention's input", down(x, (size_t) T * D), "attn_in", kTol);
  match("iHC: the attention's post", down(post, (size_t) T * hc), "attn_post", kTol);
  float* ya = up(A("y_attn"));
  CHECK(hy4_hc_post(h, ya, post, hc, D, T, 0));
  CHECK(hy4_hc_pre(h, ff, sf, bf, 2 * hc, hc, D, eps, hc_eps, mag, nf, T, x, post, 0));
  match("iHC: the MLP's input", down(x, (size_t) T * D), "ffn_in", kTol);
  float* yf = up(A("y_ffn"));
  CHECK(hy4_hc_post(h, yf, post, hc, D, T, 0));
  match("iHC: the layer's output streams", down(h, (size_t) T * hc * D), "h_out", kTol);

  // the head's collapse: pre alone (hc rows), then output_norm - from the layer's input streams
  float* h0 = up(A("h"));
  float *fo = up(A("out_hc_fn")), *so = up(A("out_hc_scale")), *bo = up(A("out_hc_base")), *no = up(A("output_norm"));
  CHECK(hy4_hc_pre(h0, fo, so, bo, hc, hc, D, eps, hc_eps, mag, no, T, x, nullptr, 0));
  match("the head's collapse", down(x, (size_t) T * D), "head", kTol);

  // a token's bits its own: the head's collapse of token 5 alone, the same as among the 16
  std::vector<float> all = down(x, (size_t) T * D);
  CHECK(hy4_hc_pre(h0 + (size_t) 5 * hc * D, fo, so, bo, hc, hc, D, eps, hc_eps, mag, no, 1, x, nullptr, 0));
  const std::vector<float> one = down(x, D);
  CHECK(std::memcmp(one.data(), all.data() + (size_t) 5 * D, D * sizeof(float)) == 0);
  std::printf("  a token alone: its collapse the same bits as among %u\n", T);
  std::printf("[test_hy4_hc_gpu] ok\n");
  return 0;
}
