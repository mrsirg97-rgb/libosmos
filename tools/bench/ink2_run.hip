// ink2_run - Inkling-2-Small's forward on the GPU, a layer at a time (0.8.0's first whole-model check): every token of
// one prompt through layer 0, then 1, ... - each layer's tensors uploaded, run, freed - so a model of any size runs in
// a layer's memory (the reference's 384 GB: ~8 GB a layer). the definition is transformers' modeling_inkling2.py:
// the normed embedding; each layer's attention residual over the block states (a block every res_block layers: its
// first layer's input), the attention (sliding or full), its conv, the residual or a new block, the MoE's residual,
// the MoE (sigmoid router, NVFP4 / MXFP8 experts with their scales), its conv; the final residual, norm, head, softcap.
// every op F32 on the GPU (include/osmos/gpu/ink2_ops.h, ops.h's rms and router, mv_run's matrices), one prompt
// read whole (no cache: its keys are its own tokens); MAT=mm and ATTN=fa the engine's WMMA in their place.
//
//   ink2_run <first shard> <ids> [out.f32]   ids: comma-separated token ids, or a file of them; out: the last
//                                             token's log-probabilities
//   REF_OUT=path  every scored row's log-probabilities written (F32, [rows][n_vocab]): the run is the reference
//   REF=path      each scored row's KL(reference || run) over the vocabulary (the reference's rows read in turn)
//   DUMP=path     a record a scored row: kl F32, nll F32, top I32, ref_top I32 (-1 without REF) - kld's, so
//                 tools/bench/kld_compare.py pairs runs
//   MASK=path     0 / 1 a token (as ids): row r scored only when token r + 1 is 1 (an agent trace's assistant tokens)
//   ATTN=fa       the attention on WMMA (ink2_fa: f16 in) - long prompts; default ink2_attn (F32: the reference's)
//   MAT=mm        the matrices on WMMA where mm has their type (mm.h: f16 in), the rest the matvec's - the engine's
//                 kernels (its bits with ATTN=fa); default every matrix the matvec's (F32 in)
//   HESS=dir      GPTQ's sums: each dense matrix's input's X^T X added into dir/<name>.f64 (float64, [dim][dim]:
//                 runs add up) - blk.L.attn_x (a sliding layer's q, k, v, rel, gate, sink), blk.L.attn_q_x /
//                 attn_k_x / attn_v_x (a full layer's: q, rel, gate, sink / k / v), blk.L.attn_out_x (the output's),
//                 output (the head's); dir/tokens the rows summed
//   TAPS=path     the drafter's taps, F32 [token][8][n_embd]: h at the entry of layers 5, 11, .., 41 (the attention
//                 residual's running partial) and the final normed hidden (the head's input) - mtp_config's
//                 target_layer_ids, TokenSpeed's capture
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mm.h"
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

static const bool g_mm = std::getenv("MAT") && !std::strcmp(std::getenv("MAT"), "mm");
static void* g_x16;         // MAT=mm's f16 activations (MM_MAX_TOKENS rows of the widest input)
static int32_t* g_overflow;  // an activation past f16's range (read at the end: a refusal)
static bool mm_takes(uint32_t type, uint32_t rows, uint32_t cols) {
  return g_mm && mm_known(type) && mm_fits(type, rows, cols);
}

// y [n][rows] = W x for n tokens of x [n][cols]: mm's (MAT=mm, its type's), else MV_MAX_TOKENS at a time (a token's
// bits its own: mv_run's)
static void mat(const GgufTensor* t, const void* w, const float* x, uint32_t n, float* y) {
  const uint32_t rows = (uint32_t) t->dims[1], cols = (uint32_t) t->dims[0];
  if (mm_takes(t->type, rows, cols)) {
    CHECK(mm_tokens(t->type, w, rows, y, rows, nullptr, 0, nullptr, 0, cols, x, cols, n, g_x16, g_overflow, false,
                    nullptr));
    return;
  }
  for (uint32_t at = 0; at < n; at += MV_MAX_TOKENS) {
    const uint32_t k = n - at < MV_MAX_TOKENS ? n - at : MV_MAX_TOKENS;
    CHECK(mv_run(t->type, w, rows, cols, x + (size_t) at * cols, k, y + (size_t) at * rows, nullptr));
  }
}
// x [n][dim]'s X^T X added into HESS's dir/<name>.f64 (made at zero)
static void hess(const char* name, const float* x, uint32_t n, uint32_t dim) {
  static const char* dir = std::getenv("HESS");
  if (!dir) return;
  const size_t nn = (size_t) dim * dim;
  float* dh = nullptr;
  CHECK(hipMalloc(&dh, nn * 4) == hipSuccess && hipMemset(dh, 0, nn * 4) == hipSuccess);
  CHECK(ink2_gram(x, n, dim, dh, nullptr) && hipDeviceSynchronize() == hipSuccess);
  std::vector<float> h(nn);
  CHECK(hipMemcpy(h.data(), dh, nn * 4, hipMemcpyDeviceToHost) == hipSuccess);
  hipFree(dh);
  const std::string path = std::string(dir) + "/" + name + ".f64";
  std::vector<double> sum(nn, 0.0);
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    CHECK(std::fread(sum.data(), 8, nn, f) == nn);
    std::fclose(f);
  }
  for (size_t i = 0; i < nn; i++) sum[i] += h[i];
  FILE* f = std::fopen((path + ".part").c_str(), "wb");
  CHECK(f && std::fwrite(sum.data(), 8, nn, f) == nn && std::fclose(f) == 0);
  CHECK(std::rename((path + ".part").c_str(), path.c_str()) == 0);
}

// an expert's y [rows] = W x for one token (W a slice of the layer's tensor of `type`): mm's at one token (MAT=mm),
// else mv_run's
static void xmat(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const float* x, float* y) {
  if (g_mm) {
    CHECK(mm_takes(type, rows, cols));
    CHECK(mm_x16(x, cols, 1, cols, g_x16, g_overflow, nullptr) && mm_run(type, w, rows, cols, g_x16, 1, y, rows, nullptr));
    return;
  }
  CHECK(mv_run(type, w, rows, cols, x, 1, y, nullptr));
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
  auto read_ints = [](const char* arg) {  // a list "1,2,3", or a file of one
    std::string text = arg;
    if (FILE* f = std::fopen(arg, "rb")) {
      text.clear();
      for (int c; (c = std::fgetc(f)) != EOF;) text.push_back(c == '\n' ? ',' : (char) c);
      std::fclose(f);
    }
    std::vector<int32_t> v;
    for (char* s = std::strtok(text.data(), ","); s; s = std::strtok(nullptr, ","))
      if (*s && *s != '\r') v.push_back(std::atoi(s));
    return v;
  };
  std::vector<int32_t> ids = read_ints(argv[2]);
  const uint32_t T = (uint32_t) ids.size(), C = (uint32_t) G.n_embd, H = (uint32_t) G.n_heads,
                 KV = (uint32_t) G.n_kv_heads, D = (uint32_t) G.head_dim, R = (uint32_t) G.d_rel,
                 taps = (uint32_t) G.conv, E = (uint32_t) G.n_experts, K = (uint32_t) G.n_used,
                 F = (uint32_t) G.ff_exp, maxb = (uint32_t) ((G.n_layers + G.res_block - 1) / G.res_block);
  std::fprintf(stderr, "ink2_run: %u tokens, %d layers (a block every %d), %u experts (%u a token)\n", T, G.n_layers,
               G.res_block, E, K);
  if (g_mm) {
    const size_t x16 = (size_t) MM_MAX_TOKENS * std::max(C, H * D) * 2;
    CHECK(hipMalloc(&g_x16, x16) == hipSuccess && hipMalloc(&g_overflow, 4) == hipSuccess &&
          hipMemset(g_overflow, 0, 4) == hipSuccess);
  }

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
    // (llama.cpp's weights: the residual and the scale folded in - res 0, scale 1)
    CHECK(ink2_conv(x, T, ch, taps, dw, nullptr, G.llama_names ? 0.0f : 1.0f, G.llama_names ? 1.0f : conv_scale, y,
                    nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(dw);
  };
  static const int kTapLayers[7] = {5, 11, 17, 23, 29, 35, 41};
  const char* taps_p = std::getenv("TAPS");
  std::vector<float> tapbuf(taps_p ? (size_t) T * 8 * C : 0);
  auto tap = [&](int slot, const float* x) {  // x [T][C] into slot of every token's 8
    std::vector<float> hx((size_t) T * C);
    CHECK(hipMemcpy(hx.data(), x, hx.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t t = 0; t < T; t++)
      std::memcpy(&tapbuf[((size_t) t * 8 + slot) * C], &hx[(size_t) t * C], C * 4);
  };
  for (int l = 0; l < G.n_layers; l++) {
    const Ink2Layer& L = W->layer[l];
    for (int k = 0; taps_p && k < 7; k++)
      if (kTapLayers[k] == l) tap(k, h);
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
    char hn[64];
    std::snprintf(hn, sizeof hn, full ? "blk.%d.attn_q_x" : "blk.%d.attn_x", l), hess(hn, xn, T, C);
    if (full) std::snprintf(hn, sizeof hn, "blk.%d.attn_k_x", l), hess(hn, kn, T, C);
    if (full) std::snprintf(hn, sizeof hn, "blk.%d.attn_v_x", l), hess(hn, vn, T, C);
    matt(L.q, xn, T, q), matt(L.k, xk, T, k), matt(L.v, xv, T, v);
    matt(L.rel, xn, T, rel), matt(L.gate, xn, T, gate), matt(L.sink, xn, T, sink);
    if (full) {
      for (int z = 0; z < 2; z++) {
        const GgufTensor *st = z ? L.v_conv_static : L.k_conv_static, *pr = z ? L.v_conv_proj : L.k_conv_proj;
        matt(pr, z ? xv : xk, T, dyn);
        std::vector<float> sw = f32_of(st);
        if (G.llama_names) {  // (llama.cpp's: [ch][taps], tap 0 last - the kernel's [taps][ch], tap 0 first)
          std::vector<float> tw(sw.size());
          const size_t ch = KV * D;
          for (size_t c = 0; c < ch; c++)
            for (uint32_t j = 0; j < taps; j++) tw[(size_t) j * ch + c] = sw[c * taps + (taps - 1 - j)];
          sw.swap(tw);
        }
        float* dw = dev_f(sw);
        CHECK(ink2_conv_dyn(z ? v : k, T, KV * D, taps, KV, dw, dyn, nullptr, G.llama_names ? 0.0f : 1.0f,
                            G.llama_names ? 1.0f : 1.0f / std::sqrt(2.0f * taps + 1), z ? vc : kc, nullptr));
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
      static const bool fa = std::getenv("ATTN") && !std::strcmp(std::getenv("ATTN"), "fa");
      CHECK(fa ? ink2_fa(&at, q, rel, sink, gate, T, kc, vc, T, 0, rp, att, nullptr, nullptr)
               : ink2_attn(&at, q, rel, sink, gate, T, kc, vc, T, rp, att, nullptr));
      CHECK(hipDeviceSynchronize() == hipSuccess);
      hipFree(qw), hipFree(rw), hipFree(rp);
    }
    std::snprintf(hn, sizeof hn, "blk.%d.attn_out_x", l), hess(hn, att, T, H * D);
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
    if (L.exp_gate_scale) scg = f32_of(L.exp_gate_scale);  // (NVFP4's own, or the checkpoint's beside a requantized one)
    if (L.exp_up_scale) scu = f32_of(L.exp_up_scale);
    if (L.exp_down_scale) scd = f32_of(L.exp_down_scale);
    CHECK(hipMemset(moe, 0, (size_t) T * C * 4) == hipSuccess);
    for (uint32_t t = 0; t < T; t++)
      for (uint32_t s = 0; s < K; s++) {
        const uint32_t e = (uint32_t) hp[(size_t) t * K + s];
        const float* x = xn + (size_t) t * C;
        xmat(L.exp_gate->type, wg + e * sg, F, C, x, eg);
        xmat(L.exp_up->type, wu + e * su, F, C, x, eu);
        if (scg[e] != 1.0f) CHECK(ink2_scale(eg, F, scg[e], nullptr));
        if (scu[e] != 1.0f) CHECK(ink2_scale(eu, F, scu[e], nullptr));
        CHECK(ink2_swiglu(eg, eu, F, G.gate_softcap, G.up_softcap, 1, eh, nullptr));
        xmat(L.exp_down->type, wd + e * sd, C, F, eh, ed);
        CHECK(ink2_axpy(moe + (size_t) t * C, ed, C, hw[(size_t) t * K + s] * scd[e], nullptr));
      }
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(wg), hipFree(wu), hipFree(wd);
    conv(L.ffn_out_conv, moe, C, tmp);
    CHECK(ink2_axpy(r, tmp, (size_t) T * C, 1.0f, nullptr));  // the layer's output: its residual + the MoE's
    CHECK(hipMemcpy(h, r, (size_t) T * C * 4, hipMemcpyDeviceToDevice) == hipSuccess);
    std::fprintf(stderr, "  layer %2d %s%s\n", l, full ? "full" : "sliding", boundary ? ", a block starts" : "");
  }
  {  // the head: every row's final residual, norm, logits, softcap, log-softmax in double; scored as kld scores
    float* dq = dev_tf(W->output_res_query);
    CHECK(ink2_attn_res(dq, 1, blocks, nb, maxb, h, T, C, G.rms_eps, res_scale, tmp, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    hipFree(dq);
    rms(W->output_norm, tmp, xn);
    hess("output", xn, T, C);
    if (taps_p) {
      tap(7, xn);
      FILE* f = std::fopen(taps_p, "wb");
      CHECK(f && std::fwrite(tapbuf.data(), 4, tapbuf.size(), f) == tapbuf.size() && std::fclose(f) == 0);
    }
    if (const char* dir = std::getenv("HESS")) {  // the rows summed, runs added
      const std::string path = std::string(dir) + "/tokens";
      long long had = 0;
      if (FILE* f = std::fopen(path.c_str(), "r")) {
        CHECK(std::fscanf(f, "%lld", &had) == 1);
        std::fclose(f);
      }
      FILE* f = std::fopen(path.c_str(), "w");
      CHECK(f && std::fprintf(f, "%lld\n", had + T) > 0 && std::fclose(f) == 0);
    }
    const uint32_t VR = (uint32_t) W->output->dims[1], V = (uint32_t) G.n_vocab, chunk = 64;
    void* head = dev_raw(W->output);
    float* lg = dev<float>((size_t) chunk * VR);
    const char *ro_p = std::getenv("REF_OUT"), *ri_p = std::getenv("REF"), *dp_p = std::getenv("DUMP"),
               *mk_p = std::getenv("MASK");
    std::vector<int32_t> mask = mk_p ? read_ints(mk_p) : std::vector<int32_t>(T, 1);
    CHECK(mask.size() == T);
    FILE *ro = ro_p ? std::fopen(ro_p, "wb") : nullptr, *ri = ri_p ? std::fopen(ri_p, "rb") : nullptr,
         *dp = dp_p ? std::fopen(dp_p, "wb") : nullptr;
    CHECK((!ro_p || ro) && (!ri_p || ri) && (!dp_p || dp));
    struct Rec {
      float kl, nll;
      int32_t top, ref_top;
    };
    std::vector<float> host((size_t) chunk * VR), rl(V), last(V);
    std::vector<double> lp(V), kls;
    double nll = 0;
    uint32_t scored = 0, agree = 0;
    for (uint32_t c0 = 0; c0 < T; c0 += chunk) {
      const uint32_t k = T - c0 < chunk ? T - c0 : chunk;
      mat(W->output, head, xn + (size_t) c0 * C, k, lg);
      CHECK(ink2_softcap(lg, (uint64_t) k * VR, G.logit_softcap, nullptr));
      CHECK(hipMemcpy(host.data(), lg, (size_t) k * VR * 4, hipMemcpyDeviceToHost) == hipSuccess);
      for (uint32_t r = 0; r < k; r++) {
        const uint32_t row = c0 + r;
        const float* x = &host[(size_t) r * VR];
        int32_t top = 0;
        for (uint32_t v = 1; v < V; v++) top = x[v] > x[top] ? (int32_t) v : top;
        double sum = 0;
        for (uint32_t v = 0; v < V; v++) sum += std::exp((double) x[v] - x[top]);
        const double lse = x[top] + std::log(sum);
        for (uint32_t v = 0; v < V; v++) lp[v] = (double) x[v] - lse;
        if (row + 1 == T)
          for (uint32_t v = 0; v < V; v++) last[v] = (float) lp[v];
        if (row + 1 >= T || !mask[row + 1]) continue;
        Rec rec = {0.0f, (float) -lp[ids[row + 1]], top, -1};
        nll += rec.nll;
        if (ro) {
          for (uint32_t v = 0; v < V; v++) rl[v] = (float) lp[v];
          CHECK(std::fwrite(rl.data(), 4, V, ro) == V);
        }
        if (ri) {
          CHECK(std::fread(rl.data(), 4, V, ri) == V);
          double kl = 0;
          int32_t rt = 0;
          for (uint32_t v = 0; v < V; v++) {
            rt = rl[v] > rl[rt] ? (int32_t) v : rt;
            const double p = std::exp((double) rl[v]);
            if (p > 0) kl += p * ((double) rl[v] - lp[v]);
          }
          rec.kl = (float) std::max(kl, 0.0), rec.ref_top = rt;
          agree += rt == top;
          kls.push_back(rec.kl);
        }
        if (dp) CHECK(std::fwrite(&rec, sizeof rec, 1, dp) == 1);
        scored++;
      }
    }
    hipFree(head), hipFree(lg);
    if (g_mm) {
      int32_t overflow = 0;
      CHECK(hipMemcpy(&overflow, g_overflow, 4, hipMemcpyDeviceToHost) == hipSuccess);
      CHECK(!overflow);  // an activation past f16's range: the run's rows not to be trusted
    }
    for (FILE* f : {ro, ri, dp})
      if (f) std::fclose(f);
    size_t best = 0;
    for (size_t i = 0; i < V; i++) best = last[i] > last[best] ? i : best;
    std::printf("ink2_run: the last token's top %zu (log-prob %.4f)\n", best, last[best]);
    if (scored) std::printf("ppl %.4f over %u scored rows\n", std::exp(nll / scored), scored);
    if (!kls.empty()) {
      double m = 0;
      for (double v : kls) m += v;
      std::printf("kld mean %.6f, top-1 agreement %.2f%%\n", m / (double) kls.size(), 100.0 * agree / (double) kls.size());
    }
    if (argc > 3) {
      FILE* f = std::fopen(argv[3], "wb");
      CHECK(f && std::fwrite(last.data(), 4, V, f) == V);
      std::fclose(f);
    }
  }
  return 0;
}
