// ds1017's layer 0 on the GPU, a token at a time (the decode: positions 0 .. T-1 of the 32 real ids), each step's
// output against the oracle's (oracle/layer_dump.py -> npz2bin.py: ref.py's prompt path at that position). the
// weights read in place from the GGUF (mmap + registration, as the pool does). prints each step's worst difference
// over the positions; fails past kMaxRel (F32's rounding, and the FP8 rounding of o: a value at a rounding edge
// flips by a step, 1/16 of itself) or on a route that differs. the stream out's fingerprint, printed, is the
// same every run (determinism). needs a gfx1151, ~/models/ds1017 and
// ~/substrate-oracle/layers3; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/model/gguf.h"
#include "substrate/model/rope.h"
#include "substrate/model/weights.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static std::string g_dir;
constexpr double kMaxRel = 1e-3;

struct Arr {
  std::vector<uint64_t> dims;
  std::vector<float> v;  // integers as floats (small, exact)
  size_t row(int t) const { return v.size() / dims[0] * t; }
};

static Arr load(const char* name) {
  Arr a;
  FILE* f = std::fopen((g_dir + "/" + name + ".bin").c_str(), "rb");
  CHECK(f);
  char m[4];
  uint32_t ty, nd;
  CHECK(std::fread(m, 1, 4, f) == 4 && std::memcmp(m, "SUBA", 4) == 0 && std::fread(&ty, 4, 1, f) == 1);
  CHECK(std::fread(&nd, 4, 1, f) == 1 && nd <= 4);
  a.dims.resize(nd);
  CHECK(std::fread(a.dims.data(), 8, nd, f) == nd);
  uint64_t n = 1;
  for (uint64_t d : a.dims) n *= d;
  a.v.resize(n);
  if (ty == 0) {
    CHECK(std::fread(a.v.data(), 4, n, f) == n);
  } else {
    std::vector<int32_t> iv(n);
    CHECK(std::fread(iv.data(), 4, n, f) == n);
    for (uint64_t i = 0; i < n; i++) a.v[i] = (float) iv[i];
  }
  std::fclose(f);
  return a;
}

// the files mapped whole, the tensors' pages registered (one registration a run of touching pages)
struct Mapped {
  std::vector<uint8_t*> base;
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> spans;
  const uint8_t* at(const GgufTensor* t) const { return base[t->file] + t->offset; }
};

static void map(const Gguf* g, Mapped* m, const std::vector<const GgufTensor*>& ts) {
  const uint64_t pg = (uint64_t) sysconf(_SC_PAGESIZE);
  m->base.resize(g->n_files);
  m->spans.resize(g->n_files);
  for (int f = 0; f < g->n_files; f++) {
    const int fd = open(g->paths[f], O_RDONLY);
    CHECK(fd >= 0);
    m->base[f] = (uint8_t*) mmap(nullptr, g->sizes[f], PROT_READ, MAP_SHARED, fd, 0);
    CHECK(m->base[f] != MAP_FAILED);
    close(fd);
  }
  for (const GgufTensor* t : ts)
    m->spans[t->file].push_back({t->offset / pg * pg, (t->offset + t->bytes + pg - 1) / pg * pg});
  for (int f = 0; f < g->n_files; f++) {
    auto& s = m->spans[f];
    std::sort(s.begin(), s.end());
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    for (auto& r : s)
      if (!merged.empty() && r.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, r.second);
      else merged.push_back(r);
    for (auto& r : merged) {
      void* p = m->base[f] + r.first;
      CHECK(hipHostRegister(p, r.second - r.first, hipHostRegisterMapped | hipHostRegisterReadOnly) == hipSuccess);
      void* dev = nullptr;
      CHECK(hipHostGetDevicePointer(&dev, p, 0) == hipSuccess && dev == p);
    }
    s = merged;
  }
}

static float* dalloc(size_t n) {
  float* p = nullptr;
  CHECK(hipMalloc(&p, n * 4) == hipSuccess);
  CHECK(hipMemset(p, 0, n * 4) == hipSuccess);
  return p;
}

static std::vector<float> host(const float* d, size_t n) {
  std::vector<float> v(n);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemcpy(v.data(), d, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

// the worst over positions of a step: max |ours - ref| / max |ref| (a position's), and the elements past 1e-3 of that
struct Diff {
  const char* name;
  double rel = 0, frac = 0;
  void add(const std::vector<float>& got, const float* ref, size_t n) {
    double mx = 0, md = 0;
    for (size_t i = 0; i < n; i++) mx = std::max(mx, (double) std::fabs(ref[i]));
    size_t far = 0;
    for (size_t i = 0; i < n; i++) {
      const double d = std::fabs((double) got[i] - ref[i]);
      md = std::max(md, d);
      far += d > 1e-3 * mx;
    }
    rel = std::max(rel, mx > 0 ? md / mx : md);
    frac = std::max(frac, (double) far / n);
  }
};

int main() {
  std::printf("[test_layer0_gpu] start\n");
  const char* home = std::getenv("HOME");
  g_dir = std::string(home ? home : "") + "/substrate-oracle/layers3";
  const std::string mdir = std::string(home ? home : "") + "/models/ds1017/";
  int ndev = 0;
  if (access((g_dir + "/0.h_out.bin").c_str(), R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_layer0_gpu] skipped: no oracle dump (oracle/layer_dump.py) or no GPU\n");
    return 77;
  }
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "";
  Gguf* g = gguf_create((mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").c_str(), err);
  Gguf* eg = gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err);
  CHECK(g && eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  const Geometry& G = w->geo;
  const LayerWeights& L = w->layer[0];
  CHECK(G.compress[0] == 0 && !G.engram[0]);
  const uint32_t hc = G.hc, D = G.n_embd, H = G.n_heads, HD = G.head_dim, RD = G.rope_dim, QL = G.q_lora;
  const uint32_t OG = G.o_groups, OL = G.o_lora, FF = G.expert_ff, E = G.n_experts, K = G.n_used, W = G.window;
  const uint32_t slots = W + MV_MAX_TOKENS, nmix = (2 + hc) * hc, oin = H * HD / OG;
  std::vector<const GgufTensor*> need = {w->embed, L.attn_norm, L.q_a, L.q_a_norm, L.q_b, L.kv, L.kv_norm, L.out_a,
                                         L.out_b, L.sinks, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base,
                                         L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, L.ffn_norm, L.router,
                                         L.router_bias, L.sh_gate, L.sh_up, L.sh_down, L.exp_gate, L.exp_up,
                                         L.exp_down};
  Mapped m;
  map(g, &m, need);
  auto F = [&](const GgufTensor* t) { return (const float*) m.at(t); };

  const Arr ids = load("ids"), r_attn_x = load("0.attn_x"), r_q = load("0.q"), r_kv = load("0.kv"), r_o = load("0.o"),
            r_attn = load("0.attn_out"), r_mid = load("0.h_mid"), r_ffn_x = load("0.ffn_x"),
            r_ridx = load("0.route_idx"), r_rw = load("0.route_w"), r_moe = load("0.moe_out"),
            r_out = load("0.h_out"), r_apre = load("0.a_pre"), r_fpre = load("0.f_pre"), r_hin = load("0.h_in");
  const int T = (int) ids.dims[0];

  float rf[ROPE_MAX_DIM / 2];
  const Rope window_rope = {(int) RD, G.rope_base, 0, G.rope_factor, G.yarn_beta_fast, G.yarn_beta_slow};
  CHECK(rope_freqs(&window_rope, rf));
  float* freqs = dalloc(RD / 2);
  CHECK(hipMemcpy(freqs, rf, RD / 2 * 4, hipMemcpyHostToDevice) == hipSuccess);
  int32_t *d_ids = nullptr, *d_pos = nullptr, *d_ridx = nullptr;
  CHECK(hipMalloc(&d_ids, 4) == hipSuccess && hipMalloc(&d_pos, 4) == hipSuccess);
  CHECK(hipMalloc(&d_ridx, K * 4) == hipSuccess);
  float *h = dalloc(hc * D), *mix = dalloc(nmix), *a_pre = dalloc(hc), *a_post = dalloc(hc), *a_comb = dalloc(hc * hc);
  float *f_pre = dalloc(hc), *f_post = dalloc(hc), *f_comb = dalloc(hc * hc), *pre0 = dalloc(hc);
  float *x = dalloc(D), *qa = dalloc(QL), *q = dalloc(H * HD), *kv = dalloc(HD), *ring = dalloc((size_t) slots * HD);
  float *o = dalloc(H * HD), *oa = dalloc(OG * OL), *y = dalloc(D), *mid = dalloc(hc * D), *xf = dalloc(D);
  float *logits = dalloc(E), *rw = dalloc(K), *ga = dalloc(FF), *up = dalloc(FF), *hh = dalloc(FF), *ye = dalloc(D);
  float *moe = dalloc(D), *out = dalloc(hc * D);
  const float one = 1.0f;
  CHECK(hipMemcpy(pre0, &one, 4, hipMemcpyHostToDevice) == hipSuccess);  // the first block's pre: stream 0

  const uint64_t gate_slice = L.exp_gate->bytes / E, up_slice = L.exp_up->bytes / E, down_slice = L.exp_down->bytes / E;
  const uint64_t oa_rows_bytes = L.out_a->bytes / OG;
  const float scale = 1.0f / std::sqrt((float) HD);
  Diff d_in{"h_in (embed)"}, d_x{"attn_x"}, d_q{"q"}, d_kv{"kv"}, d_o{"o"}, d_attn{"attn_out"}, d_mid{"h_mid"},
      d_apre{"a_pre"}, d_fpre{"f_pre"}, d_fx{"ffn_x"}, d_rw{"route_w"}, d_moe{"moe_out"}, d_out{"h_out"};
  int route_same = 0;
  uint64_t print = 1469598103934665603ull;  // FNV-1a over every position's stream out
  for (int p = 0; p < T; p++) {
    const int32_t id = (int32_t) ids.v[p], pos = p;
    CHECK(hipMemcpy(d_ids, &id, 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(d_pos, &pos, 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(ops_embed(w->embed->type, m.at(w->embed), D, d_ids, 1, hc, h, nullptr));
    d_in.add(host(h, hc * D), &r_hin.v[r_hin.row(p)], hc * D);
    // the attention's mixes, its input (the first block's pre: stream 0), q and kv
    CHECK(mv_run(0, F(L.hc_attn_fn), nmix, hc * D, h, 1, mix, nullptr));
    CHECK(ops_hc_mix(h, mix, F(L.hc_attn_scale), F(L.hc_attn_base), hc, D, G.rms_eps, G.hc_eps, G.sinkhorn_iters, 1,
                     a_pre, a_post, a_comb, nullptr));
    d_apre.add(host(a_pre, hc), &r_apre.v[r_apre.row(p)], hc);
    CHECK(ops_hc_pre_rms(h, pre0, hc, D, F(L.attn_norm), G.rms_eps, 1, x, nullptr));
    d_x.add(host(x, D), &r_attn_x.v[r_attn_x.row(p)], D);
    CHECK(mv_run(L.q_a->type, m.at(L.q_a), QL, D, x, 1, qa, nullptr));
    CHECK(ops_rms(qa, QL, QL, F(L.q_a_norm), G.rms_eps, 1, qa, QL, nullptr));
    CHECK(mv_run(L.q_b->type, m.at(L.q_b), H * HD, QL, qa, 1, q, nullptr));
    CHECK(ops_rope(q, H, HD, RD, H * HD, freqs, d_pos, false, 1, nullptr));
    d_q.add(host(q, H * HD), &r_q.v[r_q.row(p)], H * HD);
    CHECK(mv_run(L.kv->type, m.at(L.kv), HD, D, x, 1, kv, nullptr));
    CHECK(ops_rms(kv, HD, HD, F(L.kv_norm), G.rms_eps, 1, kv, HD, nullptr));
    CHECK(ops_rope(kv, 1, HD, RD, HD, freqs, d_pos, false, 1, nullptr));
    CHECK(ops_fp8_round(kv, HD, nullptr));
    d_kv.add(host(kv, HD), &r_kv.v[r_kv.row(p)], HD);
    CHECK(hipMemcpy(ring + (size_t) (p % slots) * HD, kv, HD * 4, hipMemcpyDeviceToDevice) == hipSuccess);
    // the window, the inverse rope, the FP8 rounding, wo_a (a group at a time) and wo_b
    CHECK(ops_window_attn(q, ring, slots, d_pos, W, F(L.sinks), scale, H, HD, 1, o, nullptr));
    d_o.add(host(o, H * HD), &r_o.v[r_o.row(p)], H * HD);
    CHECK(ops_rope(o, H, HD, RD, H * HD, freqs, d_pos, true, 1, nullptr));
    CHECK(ops_fp8_round(o, H * HD, nullptr));
    for (uint32_t k = 0; k < OG; k++)
      CHECK(mv_strided(L.out_a->type, m.at(L.out_a) + k * oa_rows_bytes, OL, oin, o + k * oin, H * HD, 1, oa + k * OL,
                       OG * OL, nullptr));
    CHECK(mv_run(L.out_b->type, m.at(L.out_b), D, OG * OL, oa, 1, y, nullptr));
    d_attn.add(host(y, D), &r_attn.v[r_attn.row(p)], D);
    CHECK(ops_hc_post(y, h, a_post, a_comb, hc, D, 1, mid, nullptr));
    d_mid.add(host(mid, hc * D), &r_mid.v[r_mid.row(p)], hc * D);
    // the FFN: its mixes, its input (the attention's pre - the lag), the router, the experts and the shared one
    CHECK(mv_run(0, F(L.hc_ffn_fn), nmix, hc * D, mid, 1, mix, nullptr));
    CHECK(ops_hc_mix(mid, mix, F(L.hc_ffn_scale), F(L.hc_ffn_base), hc, D, G.rms_eps, G.hc_eps, G.sinkhorn_iters, 1,
                     f_pre, f_post, f_comb, nullptr));
    d_fpre.add(host(f_pre, hc), &r_fpre.v[r_fpre.row(p)], hc);
    CHECK(ops_hc_pre_rms(mid, a_pre, hc, D, F(L.ffn_norm), G.rms_eps, 1, xf, nullptr));
    d_fx.add(host(xf, D), &r_ffn_x.v[r_ffn_x.row(p)], D);
    CHECK(mv_run(L.router->type, m.at(L.router), E, D, xf, 1, logits, nullptr));
    CHECK(ops_router(logits, F(L.router_bias), E, K, G.expert_scale, 1, d_ridx, rw, nullptr));
    std::vector<int32_t> ri(K);
    CHECK(hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(ri.data(), d_ridx, K * 4, hipMemcpyDeviceToHost) == hipSuccess);
    bool same = true;
    for (uint32_t k = 0; k < K; k++) same &= ri[k] == (int32_t) r_ridx.v[r_ridx.row(p) + k];
    route_same += same;
    d_rw.add(host(rw, K), &r_rw.v[r_rw.row(p)], K);
    std::vector<uint32_t> order(K);  // ref.py adds the experts in ascending id
    for (uint32_t k = 0; k < K; k++) order[k] = k;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return ri[a] < ri[b]; });
    CHECK(hipMemset(moe, 0, D * 4) == hipSuccess);
    for (uint32_t k : order) {
      const uint64_t e = (uint64_t) ri[k];
      CHECK(mv_run(L.exp_gate->type, m.at(L.exp_gate) + e * gate_slice, FF, D, xf, 1, ga, nullptr));
      CHECK(mv_run(L.exp_up->type, m.at(L.exp_up) + e * up_slice, FF, D, xf, 1, up, nullptr));
      CHECK(ops_swiglu(ga, up, FF, G.clamp_exp[0], rw + k, 1, hh, nullptr));
      CHECK(mv_run(L.exp_down->type, m.at(L.exp_down) + e * down_slice, D, FF, hh, 1, ye, nullptr));
      CHECK(ops_add(moe, ye, D, nullptr));
    }
    CHECK(mv_run(L.sh_gate->type, m.at(L.sh_gate), FF, D, xf, 1, ga, nullptr));
    CHECK(mv_run(L.sh_up->type, m.at(L.sh_up), FF, D, xf, 1, up, nullptr));
    CHECK(ops_swiglu(ga, up, FF, G.clamp_shexp[0], nullptr, 1, hh, nullptr));
    CHECK(mv_run(L.sh_down->type, m.at(L.sh_down), D, FF, hh, 1, ye, nullptr));
    CHECK(ops_add(moe, ye, D, nullptr));
    d_moe.add(host(moe, D), &r_moe.v[r_moe.row(p)], D);
    CHECK(ops_hc_post(moe, mid, f_post, f_comb, hc, D, 1, out, nullptr));
    const std::vector<float> ho = host(out, hc * D);
    d_out.add(ho, &r_out.v[r_out.row(p)], hc * D);
    for (size_t i = 0; i < ho.size() * 4; i++) print = (print ^ ((const uint8_t*) ho.data())[i]) * 1099511628211ull;
  }
  const Diff* steps[] = {&d_in, &d_apre, &d_x, &d_q, &d_kv, &d_o, &d_attn, &d_mid,
                         &d_fpre, &d_fx, &d_rw, &d_moe, &d_out};
  std::printf("  %d positions, a token at a time; the worst position of each step, against ref.py:\n", T);
  for (const Diff* d : steps)
    std::printf("    %-14s max |diff| / max |ref| %.2e   elements past 1e-3 of it %.4f%%\n", d->name, d->rel,
                100 * d->frac);
  std::printf("    routes         %d of %d positions the same 6 experts in the same order\n", route_same, T);
  std::printf("  the stream out, every position: fingerprint %016llx\n", (unsigned long long) print);
  for (const Diff* d : steps)
    CHECK(d->rel <= kMaxRel);
  CHECK(route_same == T);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_layer0_gpu] ok\n");
  return 0;
}
