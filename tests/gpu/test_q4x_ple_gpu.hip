// Qwen's n-gram table (PLE, layer 1) against llama.cpp's own tensors (q4x_ref.h), the dump's own tokens: the hash's
// rows decoded (ple_embd: exact - a wrong row is a wrong embedding), the gate and gated value from the residual layer
// 0 leaves (l_last-0: through Q8_0 key and value matmuls, llama.cpp's rounding), the dilated conv from llama.cpp's
// own gated value (F32: exact), and the residual after it, normed by layer 1's mixer (hc_norm-1). the conv's
// history carried: two halves the bits of one run. needs a gfx1151, the UD-IQ3_XXS set and the dump; skipped (77)
// without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/q4x/ple.h"
#include "osmos/q4x/weights.h"
}

constexpr uint32_t kTok = 256, kHalf = kTok / 2, kDim = 2560, kHc = 4, kHd = kHc * kDim, kTaps = 4;
constexpr float kEps = 1e-6f;

static std::vector<int32_t> dump_ids() {
  FILE* f = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  std::vector<int32_t> v;
  for (int32_t x; f && std::fscanf(f, "%d,", &x) == 1;) v.push_back(x);
  if (f) std::fclose(f);
  return v;
}

int main() {
  std::printf("[test_q4x_ple_gpu] start\n");
  int ndev = 0;
  Ref probe;
  char err[GGUF_ERR] = "", werr[Q4X_ERR] = "";
  Gguf* g = gguf_create(home_path(kQwenFirst).c_str(), err);
  const std::vector<int32_t> ids = dump_ids();
  if (!g || !ref_read("ple_embd", &probe) || ids.size() != kTok || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_ple_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump or a GPU\n");
    return 77;
  }
  Q4xWeights* w = q4x_weights_create(g, werr);
  CHECK(w);
  const Q4xGeometry& geo = w->geo;
  const uint32_t heads = (uint32_t) geo.ple_heads, dil = (uint32_t) geo.ple_ngram;

  // the hash's rows, a token at a time with its predecessors (the first's none), decoded
  std::vector<int32_t> rows((size_t) kTok * heads);
  for (uint32_t t = 0; t < kTok; t++) {
    const int32_t prev[2] = {t >= 1 ? ids[t - 1] : -1, t >= 2 ? ids[t - 2] : -1};
    q4x_ple_rows(&geo, ids[t], prev, (int) (t < 2 ? t : 2), &rows[(size_t) t * heads]);
  }
  const GgufTensor* tab = w->ple_table;
  const void* table = dev_tensor(g, "per_layer_token_embd.weight");
  int32_t* drows = dev_alloc<int32_t>(rows.size());
  CHECK(hipMemcpy(drows, rows.data(), rows.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  float* emb = dev_alloc<float>((size_t) kTok * kDim);
  CHECK(q4x_rows(tab->type, table, tab->bytes / tab->dims[1], (uint32_t) tab->dims[0], drows, kTok * heads, emb,
                 nullptr));
  held("the rows decoded (ple_embd)", host_of(emb, (size_t) kTok * kDim), probe, 1.0, kExactRel);

  // the gate and gated value from layer 0's residual
  const GgufTensor *tk, *tv;
  const void *wk = dev_tensor(g, "blk.1.ple_key.weight", &tk), *wv = dev_tensor(g, "blk.1.ple_value.weight", &tv);
  const float* nk = (const float*) dev_tensor(g, "blk.1.ple_norm_key.weight");
  const float* nq = (const float*) dev_tensor(g, "blk.1.ple_norm_query.weight");
  const float* nc = (const float*) dev_tensor(g, "blk.1.ple_norm_conv.weight");
  const float* cw = (const float*) dev_tensor(g, "blk.1.ple_conv1d.weight");
  const Ref h0 = ref("l_last-0");
  float *h = dev_from(h0.v), *key = dev_alloc<float>((size_t) kTok * kHd), *val = dev_alloc<float>((size_t) kTok * kDim);
  float *kn = dev_alloc<float>((size_t) kTok * kHd), *qn = dev_alloc<float>((size_t) kTok * kHd);
  float *gate = dev_alloc<float>(kTok * kHc), *gated = dev_alloc<float>((size_t) kTok * kHd);
  CHECK(mv_strided(tk->type, wk, kHd, kDim, emb, kDim, kTok, key, kHd, nullptr));
  CHECK(mv_strided(tv->type, wv, kDim, kDim, emb, kDim, kTok, val, kDim, nullptr));
  CHECK(q4x_hc_norm(key, nk, kTok, kHc, kDim, kEps, kn, nullptr, nullptr));
  CHECK(q4x_hc_norm(h, nq, kTok, kHc, kDim, kEps, qn, nullptr, nullptr));
  CHECK(q4x_ple_gate(kn, qn, val, kTok, kHc, kDim, gate, gated, nullptr));
  held("the gate (ple_gate)", host_of(gate, kTok * kHc), ref("ple_gate-1"), 1.0, kRelL2);
  held("the gated value (ple_gated_value)", host_of(gated, (size_t) kTok * kHd), ref("ple_gated_value-1"), 1.0,
       kRelL2);

  // the dilated conv from llama.cpp's own gated value; then in two halves: the same bits
  const Ref rg = ref("ple_gated_value-1");
  const uint32_t nh = (kTaps - 1) * dil;
  float *rgd = dev_from(rg.v), *nrm = dev_alloc<float>((size_t) kTok * kHd), *hist = dev_alloc<float>((size_t) nh * kHd);
  float* conv = dev_alloc<float>((size_t) kTok * kHd);
  CHECK(q4x_hc_norm(rgd, nc, kTok, kHc, kDim, kEps, nrm, nullptr, nullptr));
  CHECK(hipMemset(hist, 0, (size_t) nh * kHd * 4) == hipSuccess);
  CHECK(q4x_conv(nrm, kTok, kHd, kTaps, dil, cw, hist, 1, 0, conv, nullptr));
  const std::vector<float> whole = host_of(conv, (size_t) kTok * kHd);
  held("the dilated conv (ple_conv_out)", whole, ref("ple_conv_out-1"), 1.0, kExactRel);
  CHECK(hipMemset(hist, 0, (size_t) nh * kHd * 4) == hipSuccess);
  CHECK(q4x_conv(nrm, kHalf, kHd, kTaps, dil, cw, hist, 1, 0, conv, nullptr));
  CHECK(q4x_conv(nrm + (size_t) kHalf * kHd, kHalf, kHd, kTaps, dil, cw, hist, 1, 0, conv + (size_t) kHalf * kHd, nullptr));
  CHECK(std::memcmp(host_of(conv, whole.size()).data(), whole.data(), whole.size() * 4) == 0);

  // the residual after it (llama.cpp's own gated value and conv), through layer 1's mixer norm
  float* hn = dev_alloc<float>((size_t) kTok * kHd);
  CHECK(q4x_add2(h, rgd, conv, (uint64_t) kTok * kHd, nullptr));
  CHECK(q4x_hc_norm(h, (const float*) dev_tensor(g, "blk.1.hc_attn_norm.weight"), kTok, kHc, kDim, kEps, hn, nullptr, nullptr));
  held("the residual after it (hc_norm-1)", host_of(hn, (size_t) kTok * kHd), ref("hc_norm-1"), 1.0, kExactRel);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_ple_gpu] ok\n");
  return 0;
}
