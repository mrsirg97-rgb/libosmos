// Inkling-2-Small's own ops (include/osmos/gpu/ink2_ops.h, ops_router_scored's sigmoid) against transformers' own
// modules at the model's sizes (tools/convert/ink2_cases.py's files, ~/osmos-oracle/ink2-cases: each case's inputs
// and output; tools/convert/test_cases.py holds the files to the formulas in float64). F32 against F32: kExactRel
#include <algorithm>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/ops.h"
}

static const char* kCases = "%s/osmos-oracle/ink2-cases";

static Ref in(const char* c, const char* a) { return ref(std::string(c) + "/" + a, kCases); }

static std::vector<float> run_conv(const Ref& x, const Ref& w, uint32_t taps, float scale) {
  const uint32_t ch = (uint32_t) x.ne[0], n = (uint32_t) x.ne[1];
  float *dx = dev_from(x.v), *dw = dev_from(w.v), *dy = dev_alloc<float>(x.v.size());
  CHECK(ink2_conv(dx, n, ch, taps, dw, nullptr, scale, dy, nullptr));
  auto y = host_of(dy, x.v.size());
  hipFree(dx), hipFree(dw), hipFree(dy);
  return y;
}

int main() {
  std::printf("[test_ink2_ops_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_ink2_ops_gpu] skipped: no GPU\n"), 77;
  Ref probe;
  if (!ref_read("softcap/x", &probe, kCases))
    return std::printf("[test_ink2_ops_gpu] skipped: no cases (tools/convert/ink2_cases.py --out %s)\n", kCases), 77;

  {  // the logits' softcap, in place
    Ref x = in("softcap", "x");
    float* d = dev_from(x.v);
    CHECK(ink2_softcap(d, x.v.size(), 30.0f, nullptr));
    held("softcap (30 tanh(x / 30))", host_of(d, x.v.size()), in("softcap", "out"), 1.0, kExactRel);
    hipFree(d);
  }
  {  // the experts' softcapped SwiGLU
    Ref g = in("swiglu", "gate"), u = in("swiglu", "up");
    const uint32_t n = (uint32_t) g.ne[0], t = (uint32_t) g.ne[1];
    float *dg = dev_from(g.v), *du = dev_from(u.v), *dh = dev_alloc<float>(g.v.size());
    CHECK(ink2_swiglu(dg, du, n, 4.0f, 25.0f, t, dh, nullptr));
    held("swiglu (softcaps 4, 25)", host_of(dh, g.v.size()), in("swiglu", "out"), 1.0, kExactRel);
    hipFree(dg), hipFree(du), hipFree(dh);
  }
  // the static convs: the post-attention one, and a sliding layer's k and v
  held("conv (post-attention, 4 taps)", run_conv(in("conv", "x"), in("conv", "w"), 4, 1.0f / std::sqrt(5.0f)),
       in("conv", "out"), 1.0, kExactRel);
  held("conv (sliding layer 0's k)", run_conv(in("sliding", "k_raw"), in("sliding", "k_conv"), 4, 1.0f / std::sqrt(5.0f)),
       in("sliding", "k_conv_out"), 1.0, kExactRel);
  held("conv (sliding layer 0's v)", run_conv(in("sliding", "v_raw"), in("sliding", "v_conv"), 4, 1.0f / std::sqrt(5.0f)),
       in("sliding", "v_conv_out"), 1.0, kExactRel);
  for (const char* kv : {"k", "v"}) {  // a full layer's dynamic conv: static taps + a head's own a token
    const std::string raw = std::string(kv) + "_raw", st = std::string(kv) + "_conv_static",
                      dy = std::string(kv) + "_conv_dyn", out = std::string(kv) + "_conv_out";
    Ref x = in("full_p0", raw.c_str()), w = in("full_p0", st.c_str()), d = in("full_p0", dy.c_str());
    const uint32_t ch = (uint32_t) x.ne[0], n = (uint32_t) x.ne[1], heads = (uint32_t) d.ne[0] / 4;
    float *dx = dev_from(x.v), *dw = dev_from(w.v), *dd = dev_from(d.v), *dout = dev_alloc<float>(x.v.size());
    CHECK(ink2_conv_dyn(dx, n, ch, 4, heads, dw, dd, nullptr, 1.0f / 3.0f, dout, nullptr));
    held(("conv_dyn (full layer 5's " + std::string(kv) + ")").c_str(), host_of(dout, x.v.size()),
         in("full_p0", out.c_str()), 1.0, kExactRel);
    hipFree(dx), hipFree(dw), hipFree(dd), hipFree(dout);
  }
  for (const char* c : {"res1_nb1", "res1_nb8", "res3_nb1", "res3_nb8"}) {  // attn_res
    Ref q = in(c, "query"), b = in(c, "blocks"), p = in(c, "partial");
    const uint32_t dim = (uint32_t) q.ne[0], nq = (uint32_t) q.ne[1], nb = (uint32_t) b.ne[1], n = (uint32_t) p.ne[1];
    float *dq = dev_from(q.v), *db = dev_from(b.v), *dp = dev_from(p.v), *dout = dev_alloc<float>((size_t) nq * n * dim);
    CHECK(ink2_attn_res(dq, nq, db, nb, dp, n, dim, 1e-6f, 1.0f / std::sqrt((float) dim), dout, nullptr));
    auto o = host_of(dout, (size_t) nq * n * dim);
    for (uint32_t i = 0; i < nq; i++) {
      char what[64], name[16];
      std::snprintf(what, sizeof what, "attn_res %s, query %u", c, i);
      std::snprintf(name, sizeof name, "out%u", i);
      held(what, std::vector<float>(o.begin() + (size_t) i * n * dim, o.begin() + (size_t) (i + 1) * n * dim),
           in(c, name), 1.0, kExactRel);
    }
    hipFree(dq), hipFree(db), hipFree(dp), hipFree(dout);
  }
  {  // the router: sigmoid scores, the bias for the pick alone, normalized x 8 (the case's picks sorted by expert)
    Ref l = in("router", "logits"), bias = in("router", "bias"), picks = in("router", "picks"), wts = in("router", "weights");
    const uint32_t ne = (uint32_t) l.ne[0], n = (uint32_t) l.ne[1], k = (uint32_t) picks.ne[0];
    float *dl = dev_from(l.v), *db = dev_from(bias.v), *dw = dev_alloc<float>((size_t) n * k);
    int32_t* di = dev_alloc<int32_t>((size_t) n * k);
    CHECK(ops_router_scored(dl, db, ne, k, 8.0f, OPS_SCORE_SIGMOID, n, di, dw, nullptr));
    std::vector<int32_t> idx((size_t) n * k);
    auto w = host_of(dw, (size_t) n * k);
    CHECK(hipMemcpy(idx.data(), di, idx.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    std::vector<float> got((size_t) n * k);
    for (uint32_t t = 0; t < n; t++) {
      std::vector<std::pair<int32_t, float>> pw;
      for (uint32_t s = 0; s < k; s++) pw.push_back({idx[(size_t) t * k + s], w[(size_t) t * k + s]});
      std::sort(pw.begin(), pw.end());
      for (uint32_t s = 0; s < k; s++) {
        CHECK(pw[s].first == (int32_t) picks.v[(size_t) t * k + s]);
        got[(size_t) t * k + s] = pw[s].second;
      }
    }
    std::printf("  router (sigmoid): every token's 8 picks the reference's\n");
    held("router weights (normalized x 8)", got, wts, 1.0, kExactRel);
    hipFree(dl), hipFree(db), hipFree(dw), hipFree(di);
  }
  for (const char* c : {"sliding", "full_p0", "full_p300k"}) {  // the attention: its rows, then the core
    const bool full = std::string(c) != "sliding";
    const uint32_t pos0 = std::string(c) == "full_p300k" ? 300000 : 0;
    Ref qr = in(c, "q_raw"), qw = in(c, "q_clip"), rr = in(c, "rel_raw"), rw = in(c, "rel_clip");
    Ref kc = in(c, "k_conv_out"), vc = in(c, "v_conv_out"), sink = in(c, "sink_raw"), gate = in(c, "gate_raw");
    const uint32_t n = (uint32_t) qr.ne[1], H = (uint32_t) qw.ne[1], D = (uint32_t) qw.ne[0], R = (uint32_t) rw.ne[0];
    const uint32_t KV = (uint32_t) kc.ne[0] / D;
    const float clip = 22.627416997969522f, rclip = 8.0f;
    const Ink2Tau none = {pos0, 0, 0}, tau = {pos0, full ? 0.1f : 0.0f, 128000.0f};
    float *dq = dev_from(qr.v), *dqw = dev_from(qw.v), *dr = dev_from(rr.v), *drw = dev_from(rw.v);
    float *dk = dev_from(kc.v), *dv = dev_from(vc.v);
    CHECK(ink2_clip_rows(dq, n, H, D, dqw, clip, none, nullptr) && ink2_clip_rows(dr, n, H, R, drw, rclip, none, nullptr));
    CHECK(ink2_clip_rows(dk, n, KV, D, nullptr, clip, none, nullptr));
    CHECK(ops_rms(dv, D, D, nullptr, 1e-6f, n * KV, dv, D, nullptr));
    std::string w = std::string(c) + ": ";
    held((w + "q clipped").c_str(), host_of(dq, qr.v.size()), in(c, "q_clipped"), 1.0, kExactRel);
    held((w + "k clipped").c_str(), host_of(dk, kc.v.size()), in(c, "k"), 1.0, kExactRel);
    held((w + "v normed").c_str(), host_of(dv, vc.v.size()), in(c, "v"), 1.0, kExactRel);
    held((w + "rel clipped").c_str(), host_of(dr, rr.v.size()), in(c, "rel_clipped"), 1.0, kExactRel);
    if (tau.alpha > 0) {  // tau on q and the relative states (a full layer's), after their clips
      const Ink2Tau only = {pos0, 0.1f, 128000.0f};
      CHECK(ink2_clip_rows(dq, n, H, D, nullptr, 0, only, nullptr) && ink2_clip_rows(dr, n, H, R, nullptr, 0, only, nullptr));
    }
    Ref rp = in(c, "rel_proj");
    float *drp = dev_from(rp.v), *ds = dev_from(sink.v), *dg = dev_from(gate.v), *dout = dev_alloc<float>(qr.v.size());
    const Ink2Attn at = {H, KV, D, R, (uint32_t) rp.ne[0], full ? 0u : 1024u, 0.08838834764831845f};
    CHECK(ink2_attn(&at, dq, dr, ds, nullptr, n, dk, dv, n, drp, dout, nullptr));
    held((w + "attention, ungated").c_str(), host_of(dout, qr.v.size()), in(c, "ungated"), 1.0, kExactRel);
    CHECK(ink2_attn(&at, dq, dr, ds, dg, n, dk, dv, n, drp, dout, nullptr));
    held((w + "attention, gated").c_str(), host_of(dout, qr.v.size()), in(c, "out"), 1.0, kExactRel);
    for (float* d : {dq, dqw, dr, drw, dk, dv, drp, ds, dg, dout}) hipFree(d);
  }
  std::printf("[test_ink2_ops_gpu] ok\n");
  return 0;
}
