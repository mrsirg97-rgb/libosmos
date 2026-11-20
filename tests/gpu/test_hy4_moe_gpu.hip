// Hy4-flash's latent MoE and dense MLP, composed of osmos's ops as the forward runs them (mv_run: the router F32, the
// latent's projections and the shared expert Q8_0, each picked expert the file's type (Q3_K's layer 6: gate and up Q3_K, down Q4_K; the
// fixture's exp_types); ops_router_scored's sigmoid router;
// ops_swiglu, the routed experts clamped at the file's limit, their weight before the down projection; ops_rms's
// combine norm), against the reference's own (oracle/hy4_moe.py: modeling_hy_v4_flash.py's HYV4FlashMoE on a sparse
// layer, HYV4FlashMLP on the dense layer 0), in F32 on the GGUF's weights, 8 tokens: each token's 16 experts and
// their weights, the latent's input, the experts' sum, the shared expert, the block's output; the dense MLP's.
// the expert sum on the host, in the picks' order (the forward's op for it comes with step 2).
// ~/osmos-oracle/hy4_moe.bin; skipped (77) without.
extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
}

#include <algorithm>
#include <set>

#include "hy4_fixture.h"

// dots of 1,536 to 13,824 terms in another order than torch's, the weight applied before the down projection (the
// reference after it): a few ulps a term - 1e-5 of an array's largest value, ~80 ulps of it
constexpr double kTol = 1e-5;

static std::vector<float> mv(uint32_t type, const float* w, uint32_t rows, uint32_t cols, const float* x,
                             uint32_t n) {
  float* y = scratch((size_t) n * rows);
  CHECK(mv_run(type, w, rows, cols, x, n, y, 0));
  std::vector<float> out = down(y, (size_t) n * rows);
  CHECK(hipFree(y) == hipSuccess);
  return out;
}

static std::vector<float> swiglu(const std::vector<float>& g, const std::vector<float>& u, uint32_t n, float lim,
                                 const float* scale_dev, uint32_t n_tok) {
  float *dg = up(g), *du = up(u), *h = scratch(g.size());
  CHECK(ops_swiglu(dg, du, n, lim, scale_dev, n_tok, h, 0));
  std::vector<float> out = down(h, g.size());
  for (float* d : {dg, du, h}) CHECK(hipFree(d) == hipSuccess);
  return out;
}

int main() {
  char path[512];
  std::snprintf(path, sizeof path, "%s/osmos-oracle/hy4_moe.bin", getenv("HOME") ? getenv("HOME") : "");
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0 || !load(path)) {
    std::printf("[test_hy4_moe_gpu] skipped: no GPU or no %s (oracle/hy4_moe.py)\n", path);
    return 77;
  }
  const std::vector<float>& shape = A("shape");
  const uint32_t E = (uint32_t) shape[0], K = (uint32_t) shape[1], I = (uint32_t) shape[2], L = (uint32_t) shape[3];
  const uint32_t SH = (uint32_t) shape[4], FF = (uint32_t) shape[5];
  const float lim = shape[6], rscale = shape[7], eps = shape[8];
  const uint32_t T = (uint32_t) fx["x"].dims[0], D = (uint32_t) fx["x"].dims[1];
  std::printf("[test_hy4_moe_gpu] %u tokens; %u experts, %u a token, latent %u, clamp %g, scale %g\n", T, E, K, L,
              lim, rscale);

  // the router: its logits, each token's picks (as a set) and their weights
  float* x = up(A("x"));
  const std::vector<float> logits = mv(QTYPE_F32, up(A("router")), E, D, x, T);
  match("the router's logits", logits, "logits", kTol);
  float *dl = up(logits), *bias = up(A("router_bias")), *w = scratch((size_t) T * K);
  int32_t* idx = nullptr;
  CHECK(hipMalloc(&idx, (size_t) T * K * sizeof(int32_t)) == hipSuccess);
  CHECK(ops_router_scored(dl, bias, E, K, rscale, OPS_SCORE_SIGMOID, T, idx, w, 0));
  std::vector<int32_t> picks((size_t) T * K);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(picks.data(), idx, picks.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  const std::vector<float> wt = down(w, (size_t) T * K);
  const std::vector<float>&rp = A("picks"), &rw = A("weights");
  double werr = 0;
  for (uint32_t t = 0; t < T; t++)
    for (uint32_t s = 0; s < K; s++) {  // each pick the reference's, its weight the reference's for that expert
      const int32_t e = picks[(size_t) t * K + s];
      uint32_t r = 0;
      while (r < K && (int32_t) rp[(size_t) t * K + r] != e) r++;
      CHECK(r < K);
      werr = std::fmax(werr, std::fabs((double) wt[(size_t) t * K + s] - rw[(size_t) t * K + r]));
    }
  std::printf("  each token's %u experts the reference's; their weights within %.3g\n", K, werr);
  CHECK(werr <= kTol * rscale);

  // the latent's input; each picked expert (the file's types) on its token, weighted, summed in the picks' order
  const std::vector<float> rin = mv(QTYPE_Q8_0, up(A("routed_down_q8")), L, D, x, T);
  match("the latent's input", rin, "routed_in", kTol);
  const std::vector<float>& hit = A("hit");
  const std::vector<float>& ty = A("exp_types");
  CHECK(ty.size() == 3);
  const uint32_t tg = (uint32_t) ty[0], tu = (uint32_t) ty[1], td = (uint32_t) ty[2];
  const uint64_t eb = fx["exp_gate"].v.size() / hit.size() * 4, ebu = fx["exp_up"].v.size() / hit.size() * 4,
                 ebd = fx["exp_down"].v.size() / hit.size() * 4;  // an expert's bytes, each role's
  uint8_t *eg = (uint8_t*) up(A("exp_gate")), *eu = (uint8_t*) up(A("exp_up")), *ed = (uint8_t*) up(A("exp_down"));
  float* drin = up(rin);
  std::vector<float> sum((size_t) T * L, 0.0f);
  for (uint32_t t = 0; t < T; t++)
    for (uint32_t s = 0; s < K; s++) {
      const uint64_t at = std::find(hit.begin(), hit.end(), (float) picks[(size_t) t * K + s]) - hit.begin();
      CHECK(at < hit.size());
      const std::vector<float> g = mv(tg, (const float*) (eg + at * eb), I, L, drin + (size_t) t * L, 1);
      const std::vector<float> u = mv(tu, (const float*) (eu + at * ebu), I, L, drin + (size_t) t * L, 1);
      float* h = up(swiglu(g, u, I, lim, w + (size_t) t * K + s, 1));
      const std::vector<float> y = mv(td, (const float*) (ed + at * ebd), L, I, h, 1);
      for (uint32_t d = 0; d < L; d++) sum[(size_t) t * L + d] += y[d];
      CHECK(hipFree(h) == hipSuccess);
    }
  match("the experts' sum (latent)", sum, "experts", kTol);
  const std::vector<float> rout = mv(QTYPE_Q8_0, up(A("routed_up_q8")), D, L, up(sum), T);
  match("the experts' output", rout, "routed_out", kTol);

  // the shared expert (unclamped), the sum, the combine norm
  const std::vector<float> sg = mv(QTYPE_Q8_0, up(A("sh_gate_q8")), SH, D, x, T);
  const std::vector<float> su = mv(QTYPE_Q8_0, up(A("sh_up_q8")), SH, D, x, T);
  const std::vector<float> sh = mv(QTYPE_Q8_0, up(A("sh_down_q8")), D, SH, up(swiglu(sg, su, SH, INFINITY, nullptr, T)), T);
  match("the shared expert", sh, "shared", kTol);
  std::vector<float> both(sh.size());
  for (size_t i = 0; i < both.size(); i++) both[i] = rout[i] + sh[i];
  float *db = up(both), *out = scratch(both.size());
  CHECK(ops_rms(db, D, D, up(A("post_norm")), eps, T, out, D, 0));
  match("the MoE's output", down(out, both.size()), "out", kTol);

  // the dense MLP of layer 0 (unclamped), its post norm
  const std::vector<float> dg = mv(QTYPE_Q8_0, up(A("d_gate_q8")), FF, D, x, T);
  const std::vector<float> du = mv(QTYPE_Q8_0, up(A("d_up_q8")), FF, D, x, T);
  const std::vector<float> dd = mv(QTYPE_Q8_0, up(A("d_down_q8")), D, FF, up(swiglu(dg, du, FF, INFINITY, nullptr, T)), T);
  CHECK(ops_rms(up(dd), D, D, up(A("d_post_norm")), eps, T, out, D, 0));
  match("the dense MLP's output", down(out, both.size()), "dense_out", kTol);
  std::printf("[test_hy4_moe_gpu] ok\n");
  return 0;
}
