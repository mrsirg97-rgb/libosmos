// ink2_run - Inkling-2-Small's forward on the GPU, a layer at a time (0.8.0's first whole-model check): every token of
// one prompt through layer 0, then 1, ... - each layer's tensors uploaded, run, freed - so a model of any size runs in
// a layer's memory (the reference's 384 GB: ~8 GB a layer). the definition is transformers' modeling_inkling2.py:
// the normed embedding; each layer's attention residual over the block states (a block every res_block layers: its
// first layer's input), the attention (sliding or full), its conv, the residual or a new block, the MoE's residual,
// the MoE (sigmoid router, NVFP4 / MXFP8 experts with their scales), its conv; the final residual, norm, head, softcap.
// every op F32 on the GPU (include/osmos/gpu/ink2_ops.h, ops.h's rms and router, mv_run's matrices), one prompt
// read whole (no cache: its keys are its own tokens).
//
//   ink2_run <first shard> <ids> [out.f32]   ids: comma-separated token ids; out: the last token's log-probabilities
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                                  \
  do {                                                                            \
    if (!(x)) {                                                                   \
      std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x);        \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

constexpr uint32_t kF32 = 0, kBF16 = 30, kNVFP4 = 40;

static const Gguf* g_model;

static std::vector<uint8_t> bytes_of(const GgufTensor* t) {
  std::vector<uint8_t> b(t->bytes);
  FILE* f = std::fopen(g_model->paths[t->file], "rb");
  CHECK(f);
  for (uint64_t at = 0; at < t->bytes;) {
    const ssize_t n = pread(fileno(f), b.data() + at, t->bytes - at, (off_t) (t->offset + at));
    CHECK(n > 0);
    at += (uint64_t) n;
  }
  std::fclose(f);
  return b;
}

static float bf16(uint16_t v) {
  uint32_t u = (uint32_t) v << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// a small tensor's values as F32 (BF16 or F32 in the file)
static std::vector<float> f32_of(const GgufTensor* t) {
  auto b = bytes_of(t);
  std::vector<float> v;
  if (t->type == kF32) {
    v.resize(b.size() / 4), std::memcpy(v.data(), b.data(), b.size());
  } else {
    CHECK(t->type == kBF16);
    v.resize(b.size() / 2);
    for (size_t i = 0; i < v.size(); i++) v[i] = bf16((uint16_t) (b[2 * i] | b[2 * i + 1] << 8));
  }
  return v;
}

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc(&p, n * sizeof(T) + 16) == hipSuccess);
  return p;
}
static float* dev_f(const std::vector<float>& v) {
  float* p = dev<float>(v.size());
  CHECK(hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  return p;
}
static void* dev_raw(const GgufTensor* t) {  // a matrix's bytes as stored (mv_run reads its type)
  auto b = bytes_of(t);
  void* p = dev<uint8_t>(b.size());
  CHECK(hipMemcpy(p, b.data(), b.size(), hipMemcpyHostToDevice) == hipSuccess);
  return p;
}
static float* dev_tf(const GgufTensor* t) { return dev_f(f32_of(t)); }

// y [n][rows] = W x for n tokens of x [n][cols]
static void mat(const GgufTensor* t, const void* w, const float* x, uint32_t n, float* y) {
  CHECK(mv_run(t->type, w, (uint32_t) t->dims[1], (uint32_t) t->dims[0], x, n, y, nullptr));
}
static void matt(const GgufTensor* t, const float* x, uint32_t n, float* y) {  // uploaded, run, freed
  void* w = dev_raw(t);
  mat(t, w, x, n, y);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  hipFree(w);
}

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "ink2_run <first shard> <ids> [out.f32]\n"), 2;
  char err[GGUF_ERR > INK2_ERR ? GGUF_ERR : INK2_ERR];
  Gguf* model = gguf_create(argv[1], err);
  CHECK(model || (std::fprintf(stderr, "%s\n", err), false));
  g_model = model;
  Ink2Weights* W = ink2_weights_create(model, err);
  CHECK(W || (std::fprintf(stderr, "%s\n", err), false));
  const Ink2Geometry& G = W->geo;
  std::vector<int32_t> ids;
  for (char* s = std::strtok(argv[2], ","); s; s = std::strtok(nullptr, ",")) ids.push_back(std::atoi(s));
  const uint32_t T = (uint32_t) ids.size(), C = (uint32_t) G.n_embd, H = (uint32_t) G.n_heads,
                 KV = (uint32_t) G.n_kv_heads, D = (uint32_t) G.head_dim, R = (uint32_t) G.d_rel,
                 taps = (uint32_t) G.conv, E = (uint32_t) G.n_experts, K = (uint32_t) G.n_used,
                 F = (uint32_t) G.ff_exp, maxb = (uint32_t) ((G.n_layers + G.res_block - 1) / G.res_block);
  std::fprintf(stderr, "ink2_run: %u tokens, %d layers (a block every %d), %u experts (%u a token)\n", T, G.n_layers,
               G.res_block, E, K);

  // the normed embedding: the tokens' rows of the table, its type the file's (ops_embed)
  float* h = dev<float>((size_t) T * C);
  {
    void* tb = dev_raw(W->embed);
    int32_t* di = dev<int32_t>(T);
    CHECK(hipMemcpy(di, ids.data(), T * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(ops_embed(W->embed->type, tb, C, di, T, 1, h, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(tb), hipFree(di);
  }
  float *blocks = dev<float>((size_t) T * maxb * C), *a3 = dev<float>((size_t) 3 * T * C);
  float *xn = dev<float>((size_t) T * C), *kn = dev<float>((size_t) T * C), *vn = dev<float>((size_t) T * C);
  float *q = dev<float>((size_t) T * H * D), *k = dev<float>((size_t) T * KV * D), *v = dev<float>((size_t) T * KV * D);
  float *kc = dev<float>((size_t) T * KV * D), *vc = dev<float>((size_t) T * KV * D), *rel = dev<float>((size_t) T * H * R);
  float *gate = dev<float>((size_t) T * H * D), *sink = dev<float>((size_t) T * H), *att = dev<float>((size_t) T * H * D);
  float *dyn = dev<float>((size_t) T * KV * taps), *o = dev<float>((size_t) T * C), *r = dev<float>((size_t) T * C);
  float *moe = dev<float>((size_t) T * C), *logit = dev<float>((size_t) T * E), *eg = dev<float>(F), *eu = dev<float>(F);
  float *eh = dev<float>(F), *ed = dev<float>(C), *tmp = dev<float>((size_t) T * C);
  int32_t* pick = dev<int32_t>((size_t) T * K);
  float* pw = dev<float>((size_t) T * K);
  {
    auto w = f32_of(W->embed_norm);
    float* dw = dev_f(w);
    CHECK(ops_rms(h, C, C, dw, G.rms_eps, T, h, C, nullptr));
    hipFree(dw);
  }
  uint32_t nb = 0;
  const float res_scale = 1.0f / std::sqrt((float) C), conv_scale = 1.0f / std::sqrt((float) taps + 1);
  auto rms = [&](const GgufTensor* w, const float* x, float* y) {
    float* dw = dev_tf(w);
    CHECK(ops_rms(x, C, C, dw, G.rms_eps, T, y, C, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(dw);
  };
  auto conv = [&](const GgufTensor* w, const float* x, uint32_t ch, float* y) {
    float* dw = dev_tf(w);
    CHECK(ink2_conv(x, T, ch, taps, dw, nullptr, conv_scale, y, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(dw);
  };
  for (int l = 0; l < G.n_layers; l++) {
    const Ink2Layer& L = W->layer[l];
    const bool full = !G.sliding[l], boundary = l % G.res_block == 0;
    // the attention's inputs: the residual over the blocks (a full layer's three: q, k, v), or the layer's input
    const float *aq = h, *ak = h, *av = h;
    if (L.attn_res_query) {
      float* dq = dev_tf(L.attn_res_query);
      const uint32_t nq = full ? 3 : 1;
      CHECK(ink2_attn_res(dq, nq, blocks, nb, maxb, h, T, C, G.rms_eps, res_scale, a3, nullptr));
      CHECK(hipDeviceSynchronize() == hipSuccess);
      hipFree(dq);
      aq = a3, ak = full ? a3 + (size_t) T * C : a3, av = full ? a3 + (size_t) 2 * T * C : a3;
    }
    rms(full ? L.q_norm : L.attn_norm, aq, xn);
    if (full) rms(L.k_norm, ak, kn), rms(L.v_norm, av, vn);
    const float *xk = full ? kn : xn, *xv = full ? vn : xn;
    matt(L.q, xn, T, q), matt(L.k, xk, T, k), matt(L.v, xv, T, v);
    matt(L.rel, xn, T, rel), matt(L.gate, xn, T, gate), matt(L.sink, xn, T, sink);
    if (full) {
      for (int z = 0; z < 2; z++) {
        const GgufTensor *st = z ? L.v_conv_static : L.k_conv_static, *pr = z ? L.v_conv_proj : L.k_conv_proj;
        matt(pr, z ? xv : xk, T, dyn);
        std::vector<float> sw = f32_of(st);
        if (G.llama_names) {  // (llama.cpp's layout: [ch][taps] - the kernel's [taps][ch])
          std::vector<float> tw(sw.size());
          const size_t ch = KV * D;
          for (size_t c = 0; c < ch; c++)
            for (uint32_t j = 0; j < taps; j++) tw[(size_t) j * ch + c] = sw[c * taps + j];
          sw.swap(tw);
        }
        float* dw = dev_f(sw);
        CHECK(ink2_conv_dyn(z ? v : k, T, KV * D, taps, KV, dw, dyn, nullptr, 1.0f / std::sqrt(2.0f * taps + 1),
                            z ? vc : kc, nullptr));
        CHECK(hipDeviceSynchronize() == hipSuccess);
        hipFree(dw);
      }
    } else {
      conv(L.k_conv, k, KV * D, kc), conv(L.v_conv, v, KV * D, vc);
    }
    {  // the rows: q's and rel's weights and clips (then tau, a full layer's), k's clip, v's norm
      float *qw = dev_tf(L.q_clip), *rw = dev_tf(L.rel_clip), *rp = dev_tf(L.rel_proj);
      const Ink2Tau none = {0, 0, 0}, tau = {0, full ? G.log_alpha : 0.0f, (float) G.log_floor};
      CHECK(ink2_clip_rows(q, T, H, D, qw, G.q_clip, tau, nullptr) && ink2_clip_rows(rel, T, H, R, rw, G.rel_clip, tau, nullptr));
      CHECK(ink2_clip_rows(kc, T, KV, D, nullptr, G.k_clip, none, nullptr));
      CHECK(ops_rms(vc, D, D, nullptr, G.rms_eps, T * KV, vc, D, nullptr));
      const Ink2Attn at = {H, KV, D, R, full ? (uint32_t) G.rel_extent : (uint32_t) G.window,
                           full ? 0u : (uint32_t) G.window, 1.0f / std::sqrt((float) D)};
      CHECK(ink2_attn(&at, q, rel, sink, gate, T, kc, vc, T, rp, att, nullptr));
      CHECK(hipDeviceSynchronize() == hipSuccess);
      hipFree(qw), hipFree(rw), hipFree(rp);
    }
    matt(L.out, att, T, o);
    conv(L.out_conv, o, C, r);  // r: the attention's output after its conv
    if (boundary) {             // a new block: the layer's input joins the blocks, the attention's output its start
      CHECK(nb < maxb);
      for (uint32_t t = 0; t < T; t++)
        CHECK(hipMemcpy(blocks + ((size_t) t * maxb + nb) * C, h + (size_t) t * C, C * 4, hipMemcpyDeviceToDevice) == hipSuccess);
      nb++;
    } else {
      CHECK(ink2_axpy(r, h, (size_t) T * C, 1.0f, nullptr));
    }
    {  // the MoE's input: its residual over the blocks, normed
      float* dq = dev_tf(L.ffn_res_query);
      CHECK(ink2_attn_res(dq, 1, blocks, nb, maxb, r, T, C, G.rms_eps, res_scale, tmp, nullptr));
      CHECK(hipDeviceSynchronize() == hipSuccess);
      hipFree(dq);
    }
    rms(L.ffn_norm, tmp, xn);
    matt(L.router, xn, T, logit);
    {
      float* bias = dev_tf(L.router_bias);
      CHECK(ops_router_scored(logit, bias, E, K, G.route_scale, OPS_SCORE_SIGMOID, T, pick, pw, nullptr));
      CHECK(hipDeviceSynchronize() == hipSuccess);
      hipFree(bias);
    }
    std::vector<int32_t> hp((size_t) T * K);
    std::vector<float> hw((size_t) T * K);
    CHECK(hipMemcpy(hp.data(), pick, hp.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(hw.data(), pw, hw.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    // the experts: the layer's three tensors whole, each expert a slice; NVFP4's scales a value an expert
    uint8_t *wg = (uint8_t*) dev_raw(L.exp_gate), *wu = (uint8_t*) dev_raw(L.exp_up), *wd = (uint8_t*) dev_raw(L.exp_down);
    const uint64_t sg = L.exp_gate->bytes / E, su = L.exp_up->bytes / E, sd = L.exp_down->bytes / E;
    std::vector<float> scg(E, 1.0f), scu(E, 1.0f), scd(E, 1.0f);
    if (L.exp_gate->type == kNVFP4) scg = f32_of(L.exp_gate_scale), scu = f32_of(L.exp_up_scale), scd = f32_of(L.exp_down_scale);
    CHECK(hipMemset(moe, 0, (size_t) T * C * 4) == hipSuccess);
    for (uint32_t t = 0; t < T; t++)
      for (uint32_t s = 0; s < K; s++) {
        const uint32_t e = (uint32_t) hp[(size_t) t * K + s];
        const float* x = xn + (size_t) t * C;
        CHECK(mv_run(L.exp_gate->type, wg + e * sg, F, C, x, 1, eg, nullptr));
        CHECK(mv_run(L.exp_up->type, wu + e * su, F, C, x, 1, eu, nullptr));
        if (scg[e] != 1.0f) CHECK(ink2_scale(eg, F, scg[e], nullptr));
        if (scu[e] != 1.0f) CHECK(ink2_scale(eu, F, scu[e], nullptr));
        CHECK(ink2_swiglu(eg, eu, F, G.gate_softcap, G.up_softcap, 1, eh, nullptr));
        CHECK(mv_run(L.exp_down->type, wd + e * sd, C, F, eh, 1, ed, nullptr));
        CHECK(ink2_axpy(moe + (size_t) t * C, ed, C, hw[(size_t) t * K + s] * scd[e], nullptr));
      }
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(wg), hipFree(wu), hipFree(wd);
    conv(L.ffn_out_conv, moe, C, tmp);
    CHECK(ink2_axpy(r, tmp, (size_t) T * C, 1.0f, nullptr));  // the layer's output: its residual + the MoE's
    CHECK(hipMemcpy(h, r, (size_t) T * C * 4, hipMemcpyDeviceToDevice) == hipSuccess);
    std::fprintf(stderr, "  layer %2d %s%s\n", l, full ? "full" : "sliding", boundary ? ", a block starts" : "");
  }
  {  // the head: the last token's final residual, norm, logits, softcap, log-softmax
    float* dq = dev_tf(W->output_res_query);
    CHECK(ink2_attn_res(dq, 1, blocks, nb, maxb, h, T, C, G.rms_eps, res_scale, tmp, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(dq);
    rms(W->output_norm, tmp, xn);
    const uint32_t V = (uint32_t) W->output->dims[1];
    float* lg = dev<float>(V);
    matt(W->output, xn + (size_t) (T - 1) * C, 1, lg);
    CHECK(ink2_softcap(lg, V, G.logit_softcap, nullptr));
    std::vector<float> l(V);
    CHECK(hipMemcpy(l.data(), lg, V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    l.resize((size_t) G.n_vocab);  // (the real tokens: the rest padding)
    double m = -INFINITY, z = 0;
    for (float x : l) m = x > m ? x : m;
    for (float x : l) z += std::exp((double) x - m);
    const double lz = m + std::log(z);
    size_t best = 0;
    for (size_t i = 0; i < l.size(); i++) best = l[i] > l[best] ? i : best;
    for (float& x : l) x = (float) (x - lz);
    std::printf("ink2_run: the last token's top %zu (log-prob %.4f)\n", best, l[best]);
    if (argc > 3) {
      FILE* f = std::fopen(argv[3], "wb");
      CHECK(f && std::fwrite(l.data(), 4, l.size(), f) == l.size());
      std::fclose(f);
    }
  }
  return 0;
}
