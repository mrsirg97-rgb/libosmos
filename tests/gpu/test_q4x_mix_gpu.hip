// Qwen's hyper-connections against llama.cpp's own tensors (q4x_ref.h): layer 0's attention mixer from the residual
// it starts with (hc_init) - its norm, gate, mixed input and inject - its MoE mixer from llama.cpp's own combine,
// the combine itself from llama.cpp's inputs, and the end mixer (the output norm) from the last layer's residual.
// needs a gfx1151, the UD-IQ3_XXS set and the dump; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/quant.h"
#include "osmos/gpu/q4x_ops.h"
}

constexpr uint32_t kHc = 4, kDim = 2560, kRank = 320, kTok = 256;
constexpr float kEps = 1e-6f;

struct Mixer {
  const float *norm;
  const void *down, *up;
  const float *inject; /* or null: the end mixer has none */
  uint32_t t_down, t_up;
};

static Mixer mixer(const Gguf* g, const char* pre) {
  const GgufTensor *d, *u;
  char n[96];
  Mixer m{};
  std::snprintf(n, sizeof n, "%s_norm.weight", pre), m.norm = (const float*) dev_tensor(g, n);
  std::snprintf(n, sizeof n, "%s_down.weight", pre), m.down = dev_tensor(g, n, &d);
  std::snprintf(n, sizeof n, "%s_up.weight", pre), m.up = dev_tensor(g, n, &u);
  std::snprintf(n, sizeof n, "%s_inject.weight", pre);
  m.inject = gguf_tensor(g, n) ? (const float*) dev_tensor(g, n) : nullptr;
  m.t_down = d->type, m.t_up = u->type;
  return m;
}

// the mixer on h: xn, the gate, the mixed input, the inject (scaled by 1 / hc, as llama.cpp's)
struct Out {
  std::vector<float> xn, gate, mixed, inject;
};

static Out mix(const Mixer& m, const float* h) {
  const size_t hd = (size_t) kHc * kDim;
  float *xn = dev_alloc<float>(kTok * hd), *lo = dev_alloc<float>(kTok * kRank), *gate = dev_alloc<float>(kTok * hd);
  float *mixed = dev_alloc<float>(kTok * kDim), *inj = dev_alloc<float>(kTok * kHc);
  CHECK(q4x_hc_norm(h, m.norm, kTok, kHc, kDim, kEps, xn, nullptr, nullptr));
  CHECK(mv_strided(m.t_down, m.down, kRank, hd, xn, hd, kTok, lo, kRank, nullptr));
  CHECK(q4x_scale_silu(lo, (uint64_t) kTok * kRank, 1.0f / kHc, nullptr, nullptr));
  CHECK(mv_strided(m.t_up, m.up, hd, kRank, lo, kRank, kTok, gate, hd, nullptr));
  CHECK(q4x_hc_gated_mean(xn, gate, kTok, kHc, kDim, mixed, nullptr, nullptr));
  if (m.inject) CHECK(mv_strided(0, m.inject, kHc, hd, xn, hd, kTok, inj, kHc, nullptr));
  Out o{host_of(xn, kTok * hd), host_of(gate, kTok * hd), host_of(mixed, kTok * kDim), {}};
  if (m.inject) o.inject = host_of(inj, kTok * kHc);
  for (float* p : {xn, lo, gate, mixed, inj}) CHECK(hipFree(p) == hipSuccess);
  return o;
}

// a matrix's exact values (quant_dequant: bit for bit gguf-py's), rows x cols on the host
static std::vector<float> exact_weights(uint32_t type, const void* w, uint32_t rows, uint32_t cols) {
  const size_t n = (size_t) rows * cols;
  float* d = dev_alloc<float>(n);
  if (type == 0)
    CHECK(hipMemcpy(d, w, n * 4, hipMemcpyDeviceToDevice) == hipSuccess);
  else
    CHECK(quant_dequant(type, w, n, d, nullptr));
  std::vector<float> v = host_of(d, n);
  CHECK(hipFree(d) == hipSuccess);
  return v;
}

// the mixer's gate and mixed input in double from osmos's own norm (q4x_hc_norm, held above to 6e-8 of llama.cpp's)
// and the exact weights: what a forward with no rounding gives. osmos is held to it; llama.cpp's distance from it is
// its own rounding (its matmuls take the activations at 8 bits)
static void exact(const Mixer& m, const Out& o, const char* ref_gate, const char* ref_mixed) {
  const size_t hd = (size_t) kHc * kDim;
  const std::vector<float> wd = exact_weights(m.t_down, m.down, kRank, hd), wu = exact_weights(m.t_up, m.up, hd, kRank);
  std::vector<double> gate(kTok * hd), mixed(kTok * kDim, 0), lo(kRank);
  for (uint32_t t = 0; t < kTok; t++) {
    const float* x = &o.xn[(size_t) t * hd];
    for (uint32_t r = 0; r < kRank; r++) {
      double s = 0;
      for (size_t c = 0; c < hd; c++) s += (double) wd[(size_t) r * hd + c] * x[c];
      const double z = s / kHc;
      lo[r] = z / (1 + std::exp(-z));
    }
    for (size_t c = 0; c < hd; c++) {
      double s = 0;
      for (uint32_t r = 0; r < kRank; r++) s += (double) wu[c * kRank + r] * lo[r];
      gate[(size_t) t * hd + c] = s;
    }
    for (uint32_t d = 0; d < kDim; d++) {
      double s = 0;
      for (uint32_t k = 0; k < kHc; k++) {
        const size_t at = (size_t) t * hd + (size_t) k * kDim + d;
        s += x[(size_t) k * kDim + d] / (1 + std::exp(-gate[at]));
      }
      mixed[(size_t) t * kDim + d] = s / kHc;
    }
  }
  Ref eg, em;
  eg.v.assign(gate.begin(), gate.end()), em.v.assign(mixed.begin(), mixed.end());
  const double og = held("  osmos against exact: gate", o.gate, eg, 1.0, kExactRel);
  const double om = held("  osmos against exact: mixed", o.mixed, em, 1.0, kExactRel);
  const double lg = held("  llama.cpp against exact: gate", ref(ref_gate).v, eg, 1.0, 1.0);
  const double lm = held("  llama.cpp against exact: mixed", ref(ref_mixed).v, em, 1.0, 1.0);
  CHECK(og < lg && om < lm);
}

int main() {
  std::printf("[test_q4x_mix_gpu] start\n");
  int ndev = 0;
  Ref probe;
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(home_path(kQwenFirst).c_str(), err);
  if (!g || !ref_read("hc_init", &probe) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_mix_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump or a GPU\n");
    return 77;
  }
  CHECK(!q4x_hc_norm(nullptr, nullptr, 0, kHc, kDim, kEps, nullptr, nullptr, nullptr) &&
        !q4x_hc_combine(nullptr, nullptr, nullptr, 1, 9, kDim, 1.0f, nullptr));

  // layer 0's attention mixer, from the residual llama.cpp starts with
  float* h = dev_from(probe.v);
  const Mixer att = mixer(g, "blk.0.hc_attn");
  const Out a = mix(att, h);
  held("layer 0 attention: hc_norm", a.xn, ref("hc_norm-0"));
  held("layer 0 attention: hc_gate", a.gate, ref("hc_gate-0"), 1.0, kChainRel);
  held("layer 0 attention: hc_mixed", a.mixed, ref("hc_mixed-0"), 1.0, kChainRel);
  held("layer 0 attention: hc_inject", a.inject, ref("hc_inject-0"), 1.0 / kHc);
  exact(att, a, "hc_gate-0", "hc_mixed-0");

  // the combine, on llama.cpp's own residual, block output and inject (already 1 / hc)
  const Ref out0 = ref("linear_attn_out-0"), inj0 = ref("hc_inject-0");
  float *o = dev_from(out0.v), *i = dev_from(inj0.v);
  CHECK(q4x_hc_combine(h, o, i, kTok, kHc, kDim, 1.0f, nullptr));
  held("layer 0: hc_combine", host_of(h, probe.v.size()), ref("hc_combine-0"));

  // layer 0's MoE mixer from llama.cpp's combine, and the end mixer from its last residual
  const Ref c0 = ref("hc_combine-0"), last = ref("l_last-47");
  float *hc0 = dev_from(c0.v), *hl = dev_from(last.v);
  const Out f = mix(mixer(g, "blk.0.hc_ffn"), hc0);
  held("layer 0 MoE: hc_mixed", f.mixed, ref("hc_mixed-0#1"), 1.0, kChainRel);
  held("layer 0 MoE: hc_inject", f.inject, ref("hc_inject-0#1"), 1.0 / kHc);
  const Out e = mix(mixer(g, "output_hc"), hl);
  held("the end mixer: result_norm", e.mixed, ref("result_norm"), 1.0, kChainRel);
  gguf_destroy(g);
  std::printf("[test_q4x_mix_gpu] ok\n");
  return 0;
}
