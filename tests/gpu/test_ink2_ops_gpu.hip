// Inkling-2-Small's own ops (include/osmos/gpu/ink2_ops.h, ops_router_scored's sigmoid) against transformers' own
// modules at the model's sizes (tools/convert/ink2_cases.py's files, ~/osmos-oracle/ink2-cases: each case's inputs
// and output; tools/convert/test_cases.py holds the files to the formulas in float64). F32 against F32: kExactRel
#include <algorithm>
#include <cstring>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/ops.h"
}

static const char* kCases = "%s/osmos-oracle/ink2-cases";

static Ref in(const char* c, const char* a) { return ref(std::string(c) + "/" + a, kCases); }

// res 1, scale the model's: the checkpoint's weights; res 0, scale 1: llama.cpp's (unsloth's GGUFs) - the residual and
// the scale folded into the weights, (w + 1 at the current tap) * scale
static std::vector<float> run_conv(const Ref& x, const Ref& w, uint32_t taps, float scale, bool folded = false) {
  const uint32_t ch = (uint32_t) x.ne[0], n = (uint32_t) x.ne[1];
  std::vector<float> wv = w.v;
  if (folded)
    for (uint32_t c = 0; c < ch; c++)
      for (uint32_t j = 0; j < taps; j++) wv[c * taps + j] = (wv[c * taps + j] + (j == taps - 1 ? 1.0f : 0.0f)) * scale;
  float *dx = dev_from(x.v), *dw = dev_from(wv), *dy = dev_alloc<float>(x.v.size());
  CHECK(ink2_conv(dx, n, ch, taps, dw, nullptr, folded ? 0.0f : 1.0f, folded ? 1.0f : scale, dy, nullptr));
  auto y = host_of(dy, x.v.size());
  hipFree(dx), hipFree(dw), hipFree(dy);
  return y;
}

// Q8_0 records (ink2_ops.h's INK2_KV8_RECORD): a row of INK2_MAX_HEAD values its int8s, then a block's f16 scale
// each - d = amax / 127, q = round(x / d) as x * (1 / d)
static std::vector<uint8_t> kv8_records(const std::vector<float>& x) {
  const size_t rows = x.size() / INK2_MAX_HEAD;
  std::vector<uint8_t> out(rows * INK2_KV8_RECORD);
  for (size_t r = 0; r < rows; r++) {
    uint8_t* rec = out.data() + r * INK2_KV8_RECORD;
    for (int b = 0; b < INK2_MAX_HEAD / INK2_KV8_BLOCK; b++) {
      const float* xb = x.data() + r * INK2_MAX_HEAD + b * INK2_KV8_BLOCK;
      float amax = 0;
      for (int i = 0; i < INK2_KV8_BLOCK; i++) amax = std::max(amax, std::fabs(xb[i]));
      const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
      for (int i = 0; i < INK2_KV8_BLOCK; i++) rec[b * INK2_KV8_BLOCK + i] = (uint8_t) (int8_t) std::roundf(xb[i] * id);
      const _Float16 h = (_Float16) d;
      std::memcpy(rec + INK2_MAX_HEAD + 2 * b, &h, 2);
    }
  }
  return out;
}

// the records' values as the kernel takes them: q times the block's f16 scale, rounded to f16
static std::vector<_Float16> kv8_values(const std::vector<uint8_t>& rec) {
  const size_t rows = rec.size() / INK2_KV8_RECORD;
  std::vector<_Float16> out(rows * INK2_MAX_HEAD);
  for (size_t r = 0; r < rows; r++)
    for (int i = 0; i < INK2_MAX_HEAD; i++) {
      _Float16 d;
      std::memcpy(&d, rec.data() + r * INK2_KV8_RECORD + INK2_MAX_HEAD + 2 * (i / INK2_KV8_BLOCK), 2);
      out[r * INK2_MAX_HEAD + i] = (_Float16) ((float) (int8_t) rec[r * INK2_KV8_RECORD + i] * (float) d);
    }
  return out;
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
  held("conv, folded (llama.cpp's weights)", run_conv(in("conv", "x"), in("conv", "w"), 4, 1.0f / std::sqrt(5.0f), true),
       in("conv", "out"), 1.0, kExactRel);
  for (const char* kv : {"k", "v"}) {  // a full layer's dynamic conv: static taps + a head's own a token
    const std::string raw = std::string(kv) + "_raw", st = std::string(kv) + "_conv_static",
                      dy = std::string(kv) + "_conv_dyn", out = std::string(kv) + "_conv_out";
    Ref x = in("full_p0", raw.c_str()), w = in("full_p0", st.c_str()), d = in("full_p0", dy.c_str());
    const uint32_t ch = (uint32_t) x.ne[0], n = (uint32_t) x.ne[1], heads = (uint32_t) d.ne[0] / 4;
    float *dx = dev_from(x.v), *dw = dev_from(w.v), *dd = dev_from(d.v), *dout = dev_alloc<float>(x.v.size());
    CHECK(ink2_conv_dyn(dx, n, ch, 4, heads, dw, dd, nullptr, 1.0f, 1.0f / 3.0f, dout, nullptr));
    held(("conv_dyn (full layer 5's " + std::string(kv) + ")").c_str(), host_of(dout, x.v.size()),
         in("full_p0", out.c_str()), 1.0, kExactRel);
    std::vector<float> wf = w.v, df = d.v;  // folded: (static + 1 at tap 0) / 3, dyn / 3, no residual
    for (uint32_t c = 0; c < ch; c++) wf[c] = (wf[c] + 1.0f) / 3.0f;
    for (size_t i = ch; i < wf.size(); i++) wf[i] /= 3.0f;
    for (float& x2 : df) x2 /= 3.0f;
    float *dwf = dev_from(wf), *ddf = dev_from(df);
    CHECK(ink2_conv_dyn(dx, n, ch, 4, heads, dwf, ddf, nullptr, 0.0f, 1.0f, dout, nullptr));
    held(("conv_dyn, folded (full layer 5's " + std::string(kv) + ")").c_str(), host_of(dout, x.v.size()),
         in("full_p0", out.c_str()), 1.0, kExactRel);
    hipFree(dwf), hipFree(ddf);
    hipFree(dx), hipFree(dw), hipFree(dd), hipFree(dout);
  }
  for (const char* c : {"res1_nb1", "res1_nb8", "res3_nb1", "res3_nb8"}) {  // attn_res
    Ref q = in(c, "query"), b = in(c, "blocks"), p = in(c, "partial");
    const uint32_t dim = (uint32_t) q.ne[0], nq = (uint32_t) q.ne[1], nb = (uint32_t) b.ne[1], n = (uint32_t) p.ne[1];
    float *dq = dev_from(q.v), *db = dev_from(b.v), *dp = dev_from(p.v), *dout = dev_alloc<float>((size_t) nq * n * dim);
    CHECK(ink2_attn_res(dq, nq, db, nb, nb, dp, n, dim, 1e-6f, 1.0f / std::sqrt((float) dim), dout, nullptr,
                        nullptr));
    auto o = host_of(dout, (size_t) nq * n * dim);
    {  // the split (a few tokens, its scratch): the first ones' bits the one-block kernel's
      const uint32_t m = n < INK2_RES_SPLIT_TOKENS ? n : INK2_RES_SPLIT_TOKENS;
      float *dsc = dev_alloc<float>(INK2_RES_SCRATCH), *dsplit = dev_alloc<float>((size_t) nq * m * dim);
      CHECK(ink2_attn_res(dq, nq, db, nb, nb, dp, m, dim, 1e-6f, 1.0f / std::sqrt((float) dim), dsplit, dsc,
                          nullptr));
      auto os = host_of(dsplit, (size_t) nq * m * dim);
      for (uint32_t i = 0; i < nq; i++)
        CHECK(!std::memcmp(os.data() + (size_t) i * m * dim, o.data() + (size_t) i * n * dim,
                           (size_t) m * dim * sizeof(float)));
      hipFree(dsc), hipFree(dsplit);
    }
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
    // the prompt's attention (ink2_fa: WMMA tiles, q / k / v / p in f16): f16's rounding of the same thing
    CHECK(ink2_fa(&at, dq, dr, ds, nullptr, n, dk, dv, n, 0, drp, dout, nullptr, nullptr));
    held((w + "flash attention, ungated").c_str(), host_of(dout, qr.v.size()), in(c, "ungated"), 1.0, kF16Rel);
    CHECK(ink2_fa(&at, dq, dr, ds, dg, n, dk, dv, n, 0, drp, dout, nullptr, nullptr));
    held((w + "flash attention, gated").c_str(), host_of(dout, qr.v.size()), in(c, "out"), 1.0, kF16Rel);
    {  // a step's rows the prompt's bits: queries s .. e - 1 over the keys to e, a window's from a tile's start - a
       // prompt's later chunk, one ending inside it, a decode's single token
      const std::vector<float> whole = host_of(dout, qr.v.size());
      float* dpart = dev_alloc<float>(qr.v.size());
      float* fa_scr = (float*) dev_alloc<uint8_t>(ink2_fa_scratch(H, n));
      const std::pair<uint32_t, uint32_t> steps[] = {{1, n},          {7, n},       {17, n},     {100, n},
                                                     {n / 2 + 3, n}, {n - 1, n},   {0, 1},      {0, 37},
                                                     {5, 6},         {100, 164},   {547, 548},  {n / 2, n / 2 + 1},
                                                     {1030, 1031},   {1029, 1100}, {0, n - 1}};
      int splits = 0;
      for (auto [s, e] : steps) {
        if (e > n || s >= e) continue;
        const uint32_t want = full ? s : (s < at.window - 1 ? s : at.window - 1);
        const uint32_t k0 = full ? 0 : s - (want + (s - want) % INK2_FA_TILE), m = e - s;
        const uint64_t kd = (uint64_t) KV * D;
        // a step of a decode's size also split (a block a unit, then the fold): the same bits
        for (int split = 0; split < (m <= INK2_FA_TILE ? 2 : 1); split++) {
        CHECK(ink2_fa(&at, dq + (uint64_t) s * H * D, dr + (uint64_t) s * H * R, ds + (uint64_t) s * H,
                      dg + (uint64_t) s * H * D, m, dk + k0 * kd, dv + k0 * kd, e - k0, k0, drp, dpart,
                      split ? fa_scr : nullptr, nullptr));
        const std::vector<float> part = host_of(dpart, (size_t) m * H * D);
        uint32_t bad = 0;
        for (size_t i = 0; i < part.size(); i++) bad += part[i] != whole[(size_t) s * H * D + i];
        CHECK(bad == 0 || (std::printf("  %s: the step %u .. %u%s: %u values not the prompt's bits\n", c, s, e,
                                       split ? " (split)" : "", bad), false));
        splits++;
        }
      }
      std::printf("  %s: flash attention's rows the same bits in %d steps (chunks, decodes, split decodes) as in the prompt\n",
                  c, splits);
      {  // the keys and values stored f16 (a cache's): ink2_fa's bits
        const size_t kvn = (size_t) n * KV * D;
        uint16_t *k16 = dev_alloc<uint16_t>(kvn), *v16 = dev_alloc<uint16_t>(kvn);
        CHECK(ink2_to_f16(dk, kvn, k16, nullptr) && ink2_to_f16(dv, kvn, v16, nullptr));
        CHECK(ink2_fa16(&at, dq, dr, ds, dg, n, k16, v16, n, 0, drp, dpart, nullptr, nullptr));
        CHECK(host_of(dpart, qr.v.size()) == whole || (std::printf("  %s: f16 keys not ink2_fa's bits\n", c), false));
        std::printf("  %s: keys and values stored f16, the same bits\n", c);
        hipFree(k16), hipFree(v16);
      }
      {  // the keys and values stored Q8 (a full layer's cache): the host's Q8_0 records; ink2_fa8 the bits of
         // ink2_fa16 on those records' values; a split decode's rows the prompt's
        const size_t rows = (size_t) n * KV, bytes = rows * INK2_KV8_RECORD;
        uint8_t *k8 = dev_alloc<uint8_t>(bytes), *v8 = dev_alloc<uint8_t>(bytes);
        CHECK(ink2_to_kv8(dk, rows, k8, nullptr) && ink2_to_kv8(dv, rows, v8, nullptr));
        std::vector<uint8_t> gk(bytes), gv(bytes);
        CHECK(hipMemcpy(gk.data(), k8, bytes, hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(gv.data(), v8, bytes, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(gk == kv8_records(host_of(dk, rows * D)) && gv == kv8_records(host_of(dv, rows * D)) ||
              (std::printf("  %s: Q8 records not the host's\n", c), false));
        const std::vector<_Float16> hk = kv8_values(gk), hv = kv8_values(gv);
        uint16_t *k16 = dev_alloc<uint16_t>(rows * D), *v16 = dev_alloc<uint16_t>(rows * D);
        CHECK(hipMemcpy(k16, hk.data(), rows * D * 2, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(v16, hv.data(), rows * D * 2, hipMemcpyHostToDevice) == hipSuccess);
        CHECK(ink2_fa16(&at, dq, dr, ds, dg, n, k16, v16, n, 0, drp, dpart, nullptr, nullptr));
        const std::vector<float> want16 = host_of(dpart, qr.v.size());
        CHECK(ink2_fa8(&at, dq, dr, ds, dg, n, k8, v8, n, 0, drp, dpart, nullptr, nullptr));
        const std::vector<float> got8 = host_of(dpart, qr.v.size());
        CHECK(got8 == want16 || (std::printf("  %s: Q8 keys not f16's bits on their values\n", c), false));
        if (full) {  // the last token a decode's, split: its row the prompt's
          const uint32_t s = n - 1;
          CHECK(ink2_fa8(&at, dq + (uint64_t) s * H * D, dr + (uint64_t) s * H * R, ds + (uint64_t) s * H,
                         dg + (uint64_t) s * H * D, 1, k8, v8, n, 0, drp, dpart, fa_scr, nullptr));
          const std::vector<float> row = host_of(dpart, (size_t) H * D);
          CHECK(std::equal(row.begin(), row.end(), got8.begin() + (size_t) s * H * D) ||
                (std::printf("  %s: a Q8 split decode not the prompt's bits\n", c), false));
        }
        double worst = 0, sq = 0;
        for (size_t i = 0; i < got8.size(); i++)
          worst = std::max(worst, (double) std::fabs(got8[i] - whole[i])), sq += (double) whole[i] * whole[i];
        std::printf("  %s: keys and values stored Q8: the host's records, f16's bits on their values (against f16 "
                    "keys: max |err| %.2e of the output's rms %.2e)\n", c, worst, std::sqrt(sq / got8.size()));
        hipFree(k8), hipFree(v8), hipFree(k16), hipFree(v16);
      }
      hipFree(fa_scr);
      hipFree(dpart);
    }
    for (float* d : {dq, dqw, dr, drw, dk, dv, drp, ds, dg, dout}) hipFree(d);
  }
  {  // ink2_gram: X^T X against float64 on the host, sizes past whole tiles, added onto what h held
    const uint32_t n = 37, dim = 45;
    std::vector<float> x((size_t) n * dim), h0((size_t) dim * dim);
    uint32_t r = 12345;
    for (float& v : x) r = r * 1664525u + 1013904223u, v = (float) (r >> 8) / (1 << 24) - 0.5f;
    for (float& v : h0) r = r * 1664525u + 1013904223u, v = (float) (r >> 8) / (1 << 24);
    float *dx = dev_from(x), *dh = dev_from(h0);
    CHECK(ink2_gram(dx, n, dim, dh, nullptr));
    const std::vector<float> h = host_of(dh, h0.size());
    double worst = 0;
    for (uint32_t i = 0; i < dim; i++)
      for (uint32_t j = 0; j < dim; j++) {
        double want = h0[(size_t) i * dim + j];
        for (uint32_t t = 0; t < n; t++) want += (double) x[(size_t) t * dim + i] * x[(size_t) t * dim + j];
        worst = std::max(worst, std::fabs(want - h[(size_t) i * dim + j]));
      }
    CHECK(worst < 1e-5);
    std::printf("  gram: X^T X of 37 x 45 added onto h, max |err| %.2e\n", worst);
    hipFree(dx), hipFree(dh);
  }
  std::printf("[test_ink2_ops_gpu] ok\n");
  return 0;
}
