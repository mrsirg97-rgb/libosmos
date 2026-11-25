// Inkling-2-Small's own ops (include/osmos/gpu/ink2_ops.h, ops_router_scored's sigmoid) against transformers' own
// modules at the model's sizes (tools/convert/ink2_cases.py's files, ~/osmos-oracle/ink2-cases: each case's inputs
// and output; tools/convert/test_cases.py holds the files to the formulas in float64). F32 against F32: kExactRel
#include <algorithm>
#include <cstring>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/ops.h"
}

static const char* kCases = "%s/osmos-oracle/ink2-cases";

static Ref in(const char* c, const char* a) { return ref(std::string(c) + "/" + a, kCases); }

template <class T>
static bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}

static std::vector<uint8_t> bytes_of(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  CHECK(hipMemcpy(b.data(), p, n, hipMemcpyDeviceToHost) == hipSuccess);
  return b;
}

// values a layer's might be: a normal's around mean, of spread sd (a fixed seed's LCG, Box-Muller)
static std::vector<float> normals(size_t n, uint32_t seed, float sd, float mean = 0.0f) {
  std::vector<float> v(n);
  uint32_t r = seed;
  auto uni = [&] { return r = r * 1664525u + 1013904223u, ((float) (r >> 8) + 0.5f) / (1 << 24); };
  for (float& x : v) x = mean + sd * std::sqrt(-2.0f * std::log(uni())) * std::cos(6.2831853f * uni());
  return v;
}

// ink2_attn_prep against the ops it stands for, run one after another on copies: every output's bytes theirs -
// a window's (the static convs, f16 stores) and a full layer's (the dynamic convs, Q8 / Q5 records, tau), a decode's
// token and a short step's, a tail's history or none
static void attn_prep_case(bool full, uint32_t n, bool hist) {
  const uint32_t H = 48, KV = 8, D = INK2_MAX_HEAD, R = 16, taps = 4, ch = KV * D;
  const float q_clip = 5.0f, rel_clip = 2.0f, k_clip = 6.0f, eps = 1e-6f;
  const float res = 1.0f, scale = full ? 1.0f / std::sqrt(9.0f) : 1.0f / std::sqrt(5.0f);
  const Ink2Tau tau = {1000, full ? 0.25f : 0.0f, 128.0f}, none = {0, 0, 0};
  const uint32_t seed = (full ? 100 : 200) + n * 10 + hist;
  const auto q = normals((size_t) n * H * D, seed + 1, 0.5f), rel = normals((size_t) n * H * R, seed + 2, 0.6f);
  const auto qw = normals((size_t) H * D, seed + 3, 0.2f, 1.0f), rw = normals((size_t) H * R, seed + 4, 0.2f, 1.0f);
  const auto k = normals((size_t) n * ch, seed + 5, 1.0f), v = normals((size_t) n * ch, seed + 6, 1.0f);
  const auto wk = normals((size_t) ch * taps, seed + 7, 0.4f), wv = normals((size_t) ch * taps, seed + 8, 0.4f);
  const auto dk = normals((size_t) n * KV * taps, seed + 9, 0.3f), dv = normals((size_t) n * KV * taps, seed + 10, 0.3f);
  const auto hk = normals((size_t) (taps - 1) * ch, seed + 11, 1.0f), hv = normals((size_t) (taps - 1) * ch, seed + 12, 1.0f);
  const size_t kb = full ? (size_t) n * KV * INK2_KV8_RECORD : (size_t) n * ch * 2;
  const size_t vb = full ? (size_t) n * KV * INK2_KV5_RECORD : (size_t) n * ch * 2;
  // the ops one after another
  float *q0 = dev_from(q), *r0 = dev_from(rel), *dqw = dev_from(qw), *drw = dev_from(rw), *dkx = dev_from(k);
  float *dvx = dev_from(v), *dwk = dev_from(wk), *dwv = dev_from(wv), *ddk = dev_from(dk), *ddv = dev_from(dv);
  float *dhk = hist ? dev_from(hk) : nullptr, *dhv = hist ? dev_from(hv) : nullptr;
  float *kc0 = dev_alloc<float>((size_t) n * ch), *vc0 = dev_alloc<float>((size_t) n * ch);
  float *kk0 = dev_alloc<float>((size_t) n * ch), *kv0 = dev_alloc<float>((size_t) n * ch);
  uint8_t *sk0 = dev_alloc<uint8_t>(kb), *sv0 = dev_alloc<uint8_t>(vb);
  CHECK(ink2_clip_rows(q0, n, H, D, dqw, q_clip, tau, nullptr) && ink2_clip_rows(r0, n, H, R, drw, rel_clip, tau, nullptr));
  if (full)
    CHECK(ink2_conv_dyn_keep(dkx, n, ch, taps, KV, dwk, ddk, dhk, res, scale, kc0, kk0, nullptr) &&
          ink2_conv_dyn_keep(dvx, n, ch, taps, KV, dwv, ddv, dhv, res, scale, vc0, kv0, nullptr));
  else
    CHECK(ink2_conv_keep(dkx, n, ch, taps, dwk, dhk, res, scale, nullptr, kc0, kk0, nullptr) &&
          ink2_conv_keep(dvx, n, ch, taps, dwv, dhv, res, scale, nullptr, vc0, kv0, nullptr));
  CHECK(ink2_clip_rows(kc0, n, KV, D, nullptr, k_clip, none, nullptr) &&
        ops_rms(vc0, D, D, nullptr, eps, n * KV, vc0, D, nullptr));
  if (full)
    CHECK(ink2_to_kv8(kc0, (uint64_t) n * KV, sk0, nullptr) && ink2_to_kv5(vc0, (uint64_t) n * KV, sv0, nullptr));
  else
    CHECK(ink2_to_f16(kc0, (uint64_t) n * ch, (uint16_t*) sk0, nullptr) &&
          ink2_to_f16(vc0, (uint64_t) n * ch, (uint16_t*) sv0, nullptr));
  // the one launch
  float *q1 = dev_from(q), *r1 = dev_from(rel), *kc1 = dev_alloc<float>((size_t) n * ch);
  float *vc1 = dev_alloc<float>((size_t) n * ch), *kk1 = dev_alloc<float>((size_t) n * ch);
  float *kv1 = dev_alloc<float>((size_t) n * ch);
  uint8_t *sk1 = dev_alloc<uint8_t>(kb), *sv1 = dev_alloc<uint8_t>(vb);
  const Ink2Prep p = {n, H, KV, D, R, taps, q1, r1, dqw, drw, q_clip, rel_clip, k_clip, eps, res, scale, tau, full, false,
                      {dkx, dwk, full ? ddk : nullptr, dhk, kc1, kk1, sk1},
                      {dvx, dwv, full ? ddv : nullptr, dhv, vc1, kv1, sv1}};
  CHECK(ink2_attn_prep(&p, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const size_t nq = (size_t) n * H * D, nr = (size_t) n * H * R, nk = (size_t) n * ch;
  CHECK(same_bits(host_of(q1, nq), host_of(q0, nq)) && same_bits(host_of(r1, nr), host_of(r0, nr)));
  CHECK(same_bits(host_of(kc1, nk), host_of(kc0, nk)) && same_bits(host_of(vc1, nk), host_of(vc0, nk)));
  CHECK(same_bits(host_of(kk1, nk), k) && same_bits(host_of(kv1, nk), v));
  CHECK(same_bits(bytes_of(sk1, kb), bytes_of(sk0, kb)) && same_bits(bytes_of(sv1, vb), bytes_of(sv0, vb)));
  std::printf("  attn_prep (%s, %u token%s, %s): the ops' bits\n", full ? "full: Q8 / Q5 records" : "window: f16", n,
              n > 1 ? "s" : "", hist ? "a history" : "none");
  for (void* d : {(void*) q0, (void*) r0, (void*) dqw, (void*) drw, (void*) dkx, (void*) dvx, (void*) dwk, (void*) dwv,
                  (void*) ddk, (void*) ddv, (void*) dhk, (void*) dhv, (void*) kc0, (void*) vc0, (void*) kk0,
                  (void*) kv0, (void*) sk0, (void*) sv0, (void*) q1, (void*) r1, (void*) kc1, (void*) vc1,
                  (void*) kk1, (void*) kv1, (void*) sk1, (void*) sv1})
    if (d) hipFree(d);
}

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

// Q8_0 as ggml lays it: a 32's f16 d, then its int8s - d = amax / 127, q = round(x / d) as x * (1 / d)
static std::vector<uint8_t> q8_0_blocks(const std::vector<float>& x) {
  std::vector<uint8_t> out(x.size() / 32 * 34);
  for (size_t b = 0; b < x.size() / 32; b++) {
    const float* xb = x.data() + b * 32;
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(xb[i]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const _Float16 h = (_Float16) d;
    std::memcpy(out.data() + b * 34, &h, 2);
    for (int i = 0; i < 32; i++) out[b * 34 + 2 + i] = (uint8_t) (int8_t) std::roundf(xb[i] * id);
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

// Q5 records (ink2_ops.h's INK2_KV5_RECORD): a row of INK2_MAX_HEAD values its low nibbles (value i at byte i / 2,
// the high nibble the odd one), its fifth bits (value i at byte i / 8, bit i % 8), then a block's f16 scale each -
// d = amax / 15, q = round(x / d) as x * (1 / d), stored q + 16
static std::vector<uint8_t> kv5_records(const std::vector<float>& x) {
  constexpr int H = INK2_MAX_HEAD;
  const size_t rows = x.size() / H;
  std::vector<uint8_t> out(rows * INK2_KV5_RECORD, 0);
  for (size_t r = 0; r < rows; r++) {
    uint8_t* rec = out.data() + r * INK2_KV5_RECORD;
    for (int b = 0; b < H / INK2_KV8_BLOCK; b++) {
      const float* xb = x.data() + r * H + b * INK2_KV8_BLOCK;
      float amax = 0;
      for (int i = 0; i < INK2_KV8_BLOCK; i++) amax = std::max(amax, std::fabs(xb[i]));
      const float d = amax / 15.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
      for (int i = 0; i < INK2_KV8_BLOCK; i++) {
        const int v = b * INK2_KV8_BLOCK + i, q = (int) std::roundf(xb[i] * id) + 16;
        rec[v / 2] |= (uint8_t) ((q & 15) << (v % 2 * 4));
        rec[H / 2 + v / 8] |= (uint8_t) ((q >> 4 & 1) << (v % 8));
      }
      const _Float16 h = (_Float16) d;
      std::memcpy(rec + H / 2 + H / 8 + 2 * b, &h, 2);
    }
  }
  return out;
}

static std::vector<_Float16> kv5_values(const std::vector<uint8_t>& rec) {
  constexpr int H = INK2_MAX_HEAD;
  const size_t rows = rec.size() / INK2_KV5_RECORD;
  std::vector<_Float16> out(rows * H);
  for (size_t r = 0; r < rows; r++) {
    const uint8_t* p = rec.data() + r * INK2_KV5_RECORD;
    for (int v = 0; v < H; v++) {
      const int q = (p[v / 2] >> (v % 2 * 4) & 15) | (p[H / 2 + v / 8] >> (v % 8) & 1) << 4;
      _Float16 d;
      std::memcpy(&d, p + H / 2 + H / 8 + 2 * (v / INK2_KV8_BLOCK), 2);
      out[r * H + v] = (_Float16) ((float) (q - 16) * (float) d);
    }
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
  {  // the swiglu straight to the down's f16: ink2_swiglu_rows then mm_x16_rows' bytes (padded slots, a value past f16)
    const uint32_t n = 256, rows = 21, slots = 48;
    auto g = normals((size_t) n * rows, 31, 3.0f), u = normals((size_t) n * rows, 32, 20.0f);
    const auto sg = normals(rows, 33, 0.5f, 1.0f), su = normals(rows, 34, 0.5f, 1.0f);
    std::vector<int32_t> idx(slots, -1);
    for (uint32_t s = 0; s < rows; s++) idx[(s * 7) % slots] = (int32_t) s;
    float *dg = dev_from(g), *du = dev_from(u), *dsg = dev_from(sg), *dsu = dev_from(su);
    float* dh = dev_alloc<float>((size_t) n * rows);
    int32_t *di = dev_alloc<int32_t>(slots), *dov = dev_alloc<int32_t>(2);
    void *x0 = dev_alloc<uint16_t>((size_t) slots * n), *x1 = dev_alloc<uint16_t>((size_t) slots * n);
    CHECK(hipMemcpy(di, idx.data(), slots * 4, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemset(dov, 0, 8) == hipSuccess);
    CHECK(ink2_swiglu_rows(dg, du, dsg, dsu, n, 4.0f, 25.0f, rows, dh, nullptr) &&
          mm_x16_rows(dh, n, di, slots, n, x0, dov, nullptr));
    CHECK(ink2_swiglu_x16(dg, du, dsg, dsu, n, 4.0f, 25.0f, di, slots, x1, dov + 1, nullptr));
    CHECK(same_bits(bytes_of(x1, (size_t) slots * n * 2), bytes_of(x0, (size_t) slots * n * 2)));
    int32_t ov[2];
    CHECK(hipMemcpy(ov, dov, 8, hipMemcpyDeviceToHost) == hipSuccess && ov[0] == 0 && ov[1] == 0);
    g[5] = 1e30f, u[5] = 1e30f;  // a value past f16: both flag it
    CHECK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(du, u.data(), u.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    hipFree(dsu), dsu = dev_from(std::vector<float>(rows, 1e30f));
    CHECK(ink2_swiglu_rows(dg, du, dsg, dsu, n, 4.0f, 1e30f, rows, dh, nullptr) &&
          mm_x16_rows(dh, n, di, slots, n, x0, dov, nullptr) &&
          ink2_swiglu_x16(dg, du, dsg, dsu, n, 4.0f, 1e30f, di, slots, x1, dov + 1, nullptr));
    CHECK(same_bits(bytes_of(x1, (size_t) slots * n * 2), bytes_of(x0, (size_t) slots * n * 2)));
    CHECK(hipMemcpy(ov, dov, 8, hipMemcpyDeviceToHost) == hipSuccess && ov[0] == 1 && ov[1] == 1);
    std::printf("  swiglu to f16: ink2_swiglu_rows then mm_x16_rows' bytes, and its flag\n");
    for (void* d : {(void*) dg, (void*) du, (void*) dsg, (void*) dsu, (void*) dh, (void*) di, (void*) dov, x0, x1})
      hipFree(d);
  }
  for (uint32_t n : {1u, 5u})  // the experts' sums in the conv: ink2_moe_sum then ink2_conv_keep's bytes
    for (bool hist : {false, true}) {
      const uint32_t ch = 512, k = 8, taps = 4, P = n * k;
      const auto x = normals((size_t) P * ch, 41 + n, 1.0f), a = normals(P, 42 + n, 0.3f);
      const auto w = normals((size_t) ch * taps, 43, 0.4f), hv = normals((size_t) (taps - 1) * ch, 44, 1.0f);
      const auto add = normals((size_t) n * ch, 45, 1.0f);
      std::vector<int32_t> row(P);
      for (uint32_t m = 0; m < P; m++) row[m] = (int32_t) ((m * 13) % P);
      float *dx = dev_from(x), *da = dev_from(a), *dw = dev_from(w), *dh = hist ? dev_from(hv) : nullptr;
      float *dadd = dev_from(add), *dm = dev_alloc<float>((size_t) n * ch), *y0 = dev_alloc<float>((size_t) n * ch);
      float *y1 = dev_alloc<float>((size_t) n * ch), *k0 = dev_alloc<float>((size_t) n * ch);
      float* k1 = dev_alloc<float>((size_t) n * ch);
      int32_t* dr = dev_alloc<int32_t>(P);
      CHECK(hipMemcpy(dr, row.data(), P * 4, hipMemcpyHostToDevice) == hipSuccess);
      const float scale = 1.0f / std::sqrt(5.0f);
      CHECK(ink2_moe_sum(dm, dx, da, dr, k, n, ch, nullptr) &&
            ink2_conv_keep(dm, n, ch, taps, dw, dh, 1.0f, scale, dadd, y0, k0, nullptr));
      CHECK(ink2_moe_conv(dx, da, dr, k, n, ch, taps, dw, dh, 1.0f, scale, dadd, y1, k1, nullptr));
      const size_t m = (size_t) n * ch;
      CHECK(same_bits(host_of(y1, m), host_of(y0, m)) && same_bits(host_of(k1, m), host_of(k0, m)));
      std::printf("  moe_conv (%u token%s, %s): ink2_moe_sum then ink2_conv_keep's bits\n", n, n > 1 ? "s" : "",
                  hist ? "a history" : "none");
      for (void* d : {(void*) dx, (void*) da, (void*) dw, (void*) dh, (void*) dadd, (void*) dm, (void*) y0, (void*) y1,
                      (void*) k0, (void*) k1, (void*) dr})
        if (d) hipFree(d);
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
  {  // a residual's sum in the conv's launch: ink2_conv then ink2_axpy's bits, out of place and in place
    Ref x = in("conv", "x"), w = in("conv", "w");
    const uint32_t ch = (uint32_t) x.ne[0], n = (uint32_t) x.ne[1];
    const float scale = 1.0f / std::sqrt(5.0f);
    std::vector<float> res(x.v.size());
    for (size_t i = 0; i < res.size(); i++) res[i] = x.v[(i * 7919) % x.v.size()] * 3.0f - 0.25f;
    float *dx = dev_from(x.v), *dw = dev_from(w.v), *dr = dev_from(res), *dy = dev_alloc<float>(x.v.size());
    float *dk = dev_alloc<float>(x.v.size()), *dz = dev_alloc<float>(x.v.size());
    CHECK(ink2_conv(dx, n, ch, 4, dw, nullptr, 1.0f, scale, dy, nullptr) &&
          ink2_axpy(dy, dr, x.v.size(), 1.0f, nullptr));
    const auto want = host_of(dy, x.v.size());
    CHECK(ink2_conv_keep(dx, n, ch, 4, dw, nullptr, 1.0f, scale, dr, dz, dk, nullptr));
    CHECK(same_bits(host_of(dz, x.v.size()), want) && same_bits(host_of(dk, x.v.size()), x.v));
    CHECK(ink2_conv_keep(dx, n, ch, 4, dw, nullptr, 1.0f, scale, dr, dr, nullptr, nullptr));
    CHECK(same_bits(host_of(dr, x.v.size()), want));
    std::printf("  conv + a residual: ink2_conv then ink2_axpy's bits (in place too)\n");
    hipFree(dx), hipFree(dw), hipFree(dr), hipFree(dy), hipFree(dk), hipFree(dz);
  }
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
    // the prompt's attention on the core (attn_core: WMMA tiles, q / k / v / p in f16): f16's rounding of the same
    // thing. fa: a core of the layer's desc over the rows given (k, v from key k0, nk of them), queries s .. s + m - 1
    auto fa = [&](AttnRows kr, AttnRows vr, uint32_t s, uint32_t m, const void* k, const void* v, uint32_t nk,
                  uint32_t k0, const float* g, float* o, void* x16, int32_t* ov, uint32_t d_rel = 0) -> bool {
      AttnCoreDesc d{};
      d.heads = H, d.kv_heads = KV, d.dk = d.dv = D, d.k_rows = kr, d.v_rows = vr, d.window = full ? 0u : 1024u;
      d.scaling = 0.08838834764831845f, d.term = ATTN_TERM_REL, d.d_rel = d_rel ? d_rel : R, d.extent = (uint32_t) rp.ne[0];
      d.sink = true, d.gate = ATTN_GATE_VALUE, d.block = ATTN_BLOCK_TOKENS, d.max_tokens = n, d.max_keys = n;
      const uint64_t sb = attn_core_scratch(&d);
      void* sc = dev_alloc<uint8_t>(sb);
      char e[OSMOS_ERR] = "";
      AttnCore* core = attn_core_create(&d, sc, sb, e);
      bool ok = false;
      if (core) {
        const AttnStep st = {dq + (uint64_t) s * H * D, m, k0 + nk - m, {k, v, k0, nk, nullptr, nullptr, nullptr, nullptr, 0},
                             dr + (uint64_t) s * H * R, drp, ds + (uint64_t) s * H, g ? g + (uint64_t) s * H * D : nullptr,
                             o, x16, ov, 0, false};
        ok = attn_core_run(core, &st, nullptr) && hipDeviceSynchronize() == hipSuccess;
      }
      attn_core_destroy(core), hipFree(sc);
      return ok;
    };
    CHECK(fa(ATTN_ROWS_F32, ATTN_ROWS_F32, 0, n, dk, dv, n, 0, nullptr, dout, nullptr, nullptr));
    held((w + "flash attention, ungated").c_str(), host_of(dout, qr.v.size()), in(c, "ungated"), 1.0, kF16Rel);
    CHECK(fa(ATTN_ROWS_F32, ATTN_ROWS_F32, 0, n, dk, dv, n, 0, dg, dout, nullptr, nullptr));
    held((w + "flash attention, gated").c_str(), host_of(dout, qr.v.size()), in(c, "out"), 1.0, kF16Rel);
    {  // its relative bias ATTN_REL states a head, unrolled: another count refused, not summed short
      CHECK(R == ATTN_REL);
      CHECK(!fa(ATTN_ROWS_F32, ATTN_ROWS_F32, 0, n, dk, dv, n, 0, dg, dout, nullptr, nullptr, R / 2));
      std::printf("  %s: flash attention's relative states %u a head (ATTN_REL), %u refused\n", c, R, R / 2);
    }
    {  // a step's rows the prompt's bits: queries s .. e - 1 over the keys to e, a window's from a tile's start - a
       // prompt's later chunk, one ending inside it, a decode's single token
      const std::vector<float> whole = host_of(dout, qr.v.size());
      float* dpart = dev_alloc<float>(qr.v.size());
      const std::pair<uint32_t, uint32_t> steps[] = {{1, n},          {7, n},       {17, n},     {100, n},
                                                     {n / 2 + 3, n}, {n - 1, n},   {0, 1},      {0, 37},
                                                     {5, 6},         {100, 164},   {547, 548},  {n / 2, n / 2 + 1},
                                                     {1030, 1031},   {1029, 1100}, {0, n - 1}};
      int splits = 0;
      for (auto [s, e] : steps) {
        if (e > n || s >= e) continue;
        const uint32_t want = full ? s : (s < at.window - 1 ? s : at.window - 1);
        const uint32_t k0 = full ? 0 : s - (want + (s - want) % ATTN_TILE), m = e - s;
        const uint64_t kd = (uint64_t) KV * D;
        // a step of a decode's size split (a block a unit, then the fold): the same bits
        const bool split = m <= ATTN_TILE;
        CHECK(fa(ATTN_ROWS_F32, ATTN_ROWS_F32, s, m, dk + k0 * kd, dv + k0 * kd, e - k0, k0, dg, dpart, nullptr, nullptr));
        const std::vector<float> part = host_of(dpart, (size_t) m * H * D);
        uint32_t bad = 0;
        for (size_t i = 0; i < part.size(); i++) bad += part[i] != whole[(size_t) s * H * D + i];
        CHECK(bad == 0 || (std::printf("  %s: the step %u .. %u%s: %u values not the prompt's bits\n", c, s, e,
                                       split ? " (split)" : "", bad), false));
        if (split) {  // the fold's f16 of its rows too: mm_x16's bytes, the rows its own
          const size_t xb = (size_t) (m + 15) / 16 * 16 * H * D * 2;
          void *xa = dev_alloc<uint8_t>(xb), *xw = dev_alloc<uint8_t>(xb);
          int32_t* ov = dev_alloc<int32_t>(1);
          CHECK(hipMemset(ov, 0, 4) == hipSuccess && hipMemset(xa, 0x5A, xb) == hipSuccess);
          CHECK(fa(ATTN_ROWS_F32, ATTN_ROWS_F32, s, m, dk + k0 * kd, dv + k0 * kd, e - k0, k0, dg, dpart, xa, ov));
          CHECK(same_bits(host_of(dpart, (size_t) m * H * D), part));
          CHECK(mm_x16(dpart, H * D, m, H * D, xw, ov, nullptr));
          CHECK(same_bits(bytes_of(xa, xb), bytes_of(xw, xb)));
          hipFree(xa), hipFree(xw), hipFree(ov);
        }
        splits++;
      }
      std::printf("  %s: flash attention's rows the same bits in %d steps (chunks, decodes, split decodes) as in the prompt; "
                  "a split decode's f16 mm_x16's\n",
                  c, splits);
      {  // the keys and values stored f16 (a cache's): the F32 rows' bits
        const size_t kvn = (size_t) n * KV * D;
        uint16_t *k16 = dev_alloc<uint16_t>(kvn), *v16 = dev_alloc<uint16_t>(kvn);
        CHECK(ink2_to_f16(dk, kvn, k16, nullptr) && ink2_to_f16(dv, kvn, v16, nullptr));
        CHECK(fa(ATTN_ROWS_F16, ATTN_ROWS_F16, 0, n, k16, v16, n, 0, dg, dpart, nullptr, nullptr));
        CHECK(host_of(dpart, qr.v.size()) == whole || (std::printf("  %s: f16 keys not the prompt's bits\n", c), false));
        std::printf("  %s: keys and values stored f16, the same bits\n", c);
        hipFree(k16), hipFree(v16);
      }
      {  // the keys and values stored Q8 (a full layer's cache): the host's Q8_0 records; the core the bits of
         // f16 rows of those records' values; a split decode's rows the prompt's
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
        CHECK(fa(ATTN_ROWS_F16, ATTN_ROWS_F16, 0, n, k16, v16, n, 0, dg, dpart, nullptr, nullptr));
        const std::vector<float> want16 = host_of(dpart, qr.v.size());
        CHECK(fa(ATTN_ROWS_KV8, ATTN_ROWS_KV8, 0, n, k8, v8, n, 0, dg, dpart, nullptr, nullptr));
        const std::vector<float> got8 = host_of(dpart, qr.v.size());
        CHECK(got8 == want16 || (std::printf("  %s: Q8 keys not f16's bits on their values\n", c), false));
        if (full) {  // the last token a decode's, split: its row the prompt's
          const uint32_t s = n - 1;
          CHECK(fa(ATTN_ROWS_KV8, ATTN_ROWS_KV8, s, 1, k8, v8, n, 0, dg, dpart, nullptr, nullptr));
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
      {  // keys Q8, values Q5: the host's Q5 records; f16's bits on both records' values; a split decode's
         // row the prompt's
        const size_t rows = (size_t) n * KV;
        uint8_t *k8 = dev_alloc<uint8_t>(rows * INK2_KV8_RECORD), *v5 = dev_alloc<uint8_t>(rows * INK2_KV5_RECORD);
        CHECK(ink2_to_kv8(dk, rows, k8, nullptr) && ink2_to_kv5(dv, rows, v5, nullptr));
        std::vector<uint8_t> gk(rows * INK2_KV8_RECORD), gv(rows * INK2_KV5_RECORD);
        CHECK(hipMemcpy(gk.data(), k8, gk.size(), hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(gv.data(), v5, gv.size(), hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(gv == kv5_records(host_of(dv, rows * D)) || (std::printf("  %s: Q5 records not the host's\n", c), false));
        const std::vector<_Float16> hk = kv8_values(gk), hv = kv5_values(gv);
        uint16_t *k16 = dev_alloc<uint16_t>(rows * D), *v16 = dev_alloc<uint16_t>(rows * D);
        CHECK(hipMemcpy(k16, hk.data(), rows * D * 2, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(v16, hv.data(), rows * D * 2, hipMemcpyHostToDevice) == hipSuccess);
        CHECK(fa(ATTN_ROWS_F16, ATTN_ROWS_F16, 0, n, k16, v16, n, 0, dg, dpart, nullptr, nullptr));
        const std::vector<float> want16 = host_of(dpart, qr.v.size());
        CHECK(fa(ATTN_ROWS_KV8, ATTN_ROWS_KV5, 0, n, k8, v5, n, 0, dg, dpart, nullptr, nullptr));
        const std::vector<float> got = host_of(dpart, qr.v.size());
        CHECK(got == want16 || (std::printf("  %s: Q8 / Q5 not f16's bits on their values\n", c), false));
        if (full) {
          const uint32_t s = n - 1;
          CHECK(fa(ATTN_ROWS_KV8, ATTN_ROWS_KV5, s, 1, k8, v5, n, 0, dg, dpart, nullptr, nullptr));
          const std::vector<float> row = host_of(dpart, (size_t) H * D);
          CHECK(std::equal(row.begin(), row.end(), got.begin() + (size_t) s * H * D) ||
                (std::printf("  %s: a Q8 / Q5 split decode not the prompt's bits\n", c), false));
        }
        double worst = 0, sq = 0;
        for (size_t i = 0; i < got.size(); i++)
          worst = std::max(worst, (double) std::fabs(got[i] - whole[i])), sq += (double) whole[i] * whole[i];
        std::printf("  %s: keys Q8, values Q5: the host's records, f16's bits on their values (against f16: max |err| "
                    "%.2e of the output's rms %.2e)\n", c, worst, std::sqrt(sq / got.size()));
        hipFree(k8), hipFree(v5), hipFree(k16), hipFree(v16);
      }
      hipFree(dpart);
    }
    for (float* d : {dq, dqw, dr, drw, dk, dv, drp, ds, dg, dout}) hipFree(d);
  }
  for (bool full : {false, true})
    for (uint32_t n : {1u, 5u})
      for (bool hist : {false, true}) attn_prep_case(full, n, hist);
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
  {  // ink2_transpose_q8_0: the fold's layout, the host's Q8_0 of the transpose byte for byte (a zero block's d 0)
    const uint32_t rows = 96, cols = 37;
    std::vector<float> x((size_t) rows * cols);
    uint32_t r = 777;
    for (float& v : x) r = r * 1664525u + 1013904223u, v = ((float) (r >> 8) / (1 << 24) - 0.5f) * 9.0f;
    for (uint32_t k = 0; k < 32; k++) x[(size_t) (32 + k) * cols + 5] = 0.0f;
    std::vector<float> t((size_t) cols * rows);
    for (uint32_t i = 0; i < rows; i++)
      for (uint32_t c = 0; c < cols; c++) t[(size_t) c * rows + i] = x[(size_t) i * cols + c];
    const std::vector<uint8_t> want = q8_0_blocks(t);
    float* dx = dev_from(x);
    uint8_t* dq = dev_alloc<uint8_t>(want.size());
    CHECK(ink2_transpose_q8_0(dx, rows, cols, dq, nullptr));
    CHECK(!ink2_transpose_q8_0(dx, rows + 1, cols, dq, nullptr));
    std::vector<uint8_t> got(want.size());
    CHECK(hipMemcpy(got.data(), dq, got.size(), hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(got == want || (std::printf("  transpose_q8_0: not the host's Q8_0 blocks\n"), false));
    std::printf("  transpose_q8_0: [96][37] as [37][96] Q8_0, the host's bytes\n");
    hipFree(dx), hipFree(dq);
  }
  {  // ink2_mtp_markov_pick: a Markov step's greatest and its probability against float64 on the host - past the
     // blocks' slices (79: fewer values than blocks), the greatest in the last slice, a tie (the lowest id)
    for (const uint32_t n : {201024u, 1000u, 79u}) {
      std::vector<float> b(n), c(n), fb(n);
      uint32_t r = 4242 + n;
      for (uint32_t i = 0; i < n; i++) {
        r = r * 1664525u + 1013904223u, b[i] = ((float) (r >> 8) / (1 << 24) - 0.5f) * 40.0f;
        r = r * 1664525u + 1013904223u, c[i] = ((float) (r >> 8) / (1 << 24) - 0.5f) * 4.0f;
        fb[i] = 0.25f;
      }
      for (int pass = 0; pass < 3; pass++) {
        if (pass == 1) b[n - 1] = 400.0f, c[n - 1] = 0.0f;  // the last id's
        if (pass == 2) b[n / 3] = b[n - 1], c[n / 3] = 0.0f;  // a tie: the lower id
        float *db = dev_from(b), *dc = dev_from(c), *dfb = dev_from(fb), *dscr = dev_alloc<float>(ink2_mtp_pick_scratch() / 4);
        int32_t* did = dev_alloc<int32_t>(1);
        float* dp = dev_alloc<float>(1);
        CHECK(ink2_mtp_markov_pick(db, dc, dfb, 30.0f, n, dscr, did, dp, nullptr));
        int32_t id = -1;
        float p = 0, p2 = 0;
        CHECK(hipMemcpy(&id, did, 4, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(hipMemcpy(&p, dp, 4, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(ink2_mtp_markov_pick(db, dc, dfb, 30.0f, n, dscr, did, dp, nullptr));
        CHECK(hipMemcpy(&p2, dp, 4, hipMemcpyDeviceToHost) == hipSuccess);
        std::vector<double> x(n);
        uint32_t best = 0;
        for (uint32_t i = 0; i < n; i++) {
          x[i] = 30.0 * std::tanh(((double) b[i] + c[i] + fb[i]) / 30.0);
          best = x[i] > x[best] ? i : best;
        }
        double sum = 0;
        for (uint32_t i = 0; i < n; i++) sum += std::exp(x[i] - x[best]);
        const double rel = std::fabs(p - 1.0 / sum) * sum;
        CHECK((uint32_t) id == best || (std::printf("  pick: %u values: id %d, not %u\n", n, id, best), false));
        CHECK(rel < 1e-5 || (std::printf("  pick: %u values: p %.9g, not %.9g\n", n, p, 1.0 / sum), false));
        CHECK(std::memcmp(&p, &p2, 4) == 0);
        if (pass == 2) CHECK(id == (int32_t) (n / 3));
        hipFree(db), hipFree(dc), hipFree(dfb), hipFree(dscr), hipFree(did), hipFree(dp);
      }
      std::printf("  markov_pick: %u values: the greatest's id (ties the lowest), p against float64 within 1e-5, the "
                  "same bits again\n", n);
    }
  }
  {  // ink2_mtp_attn: the drafter's block against float64 on the host - Inkling-2's 48 heads on 8 KV heads, 8 queries:
     // every context position (one span, and past a span's), a local layer's window over a ring that wraps
    const uint32_t nq = 8, H = 48, KH = 8, D = 128, row = KH * D;
    struct Case {
      int32_t pos0, from;
      uint32_t ring, window;
    } cases[] = {{700, 0, 1024, 0}, {20000, 37, 20480, 0}, {3000, 1960, 1040, 1031}};
    for (const Case& c : cases) {
      std::vector<float> q((size_t) nq * H * D), kb((size_t) nq * row), vb((size_t) nq * row);
      std::vector<_Float16> kr((size_t) c.ring * row), vr((size_t) c.ring * row);
      uint32_t r = 99 + (uint32_t) c.pos0;
      auto rnd = [&r](float s) { return r = r * 1664525u + 1013904223u, ((float) (r >> 8) / (1 << 24) - 0.5f) * s; };
      for (float& v : q) v = rnd(4.0f);
      for (float& v : kb) v = rnd(2.0f);
      for (float& v : vb) v = rnd(2.0f);
      for (auto& v : kr) v = (_Float16) rnd(2.0f);
      for (auto& v : vr) v = (_Float16) rnd(2.0f);
      std::vector<float> want((size_t) nq * H * D);
      for (uint32_t i = 0; i < nq; i++)
        for (uint32_t h = 0; h < H; h++) {
          const uint32_t g = h / (H / KH);
          const int32_t qp = c.pos0 + (int32_t) i;
          const int32_t lo = c.window && qp - (int32_t) c.window + 1 > c.from ? qp - (int32_t) c.window + 1 : c.from;
          std::vector<double> s, o(D, 0.0);
          std::vector<const _Float16*> kc, vc;
          for (int32_t p = lo; p < c.pos0; p++) {
            const _Float16* k = &kr[(size_t) ((uint32_t) p % c.ring) * row + g * D];
            double a = 0;
            for (uint32_t d = 0; d < D; d++) a += (double) q[((size_t) i * H + h) * D + d] * (double) (float) k[d];
            s.push_back(a / std::sqrt((double) D)), vc.push_back(&vr[(size_t) ((uint32_t) p % c.ring) * row + g * D]);
          }
          const size_t nctx = s.size();
          for (uint32_t b = 0; b < nq; b++) {
            double a = 0;
            for (uint32_t d = 0; d < D; d++) a += (double) q[((size_t) i * H + h) * D + d] * kb[(size_t) b * row + g * D + d];
            s.push_back(a / std::sqrt((double) D));
          }
          const double mx = *std::max_element(s.begin(), s.end());
          double l = 0;
          for (size_t j = 0; j < s.size(); j++) {
            const double w = std::exp(s[j] - mx);
            l += w;
            for (uint32_t d = 0; d < D; d++)
              o[d] += w * (j < nctx ? (double) (float) vc[j][d] : (double) vb[(j - nctx) * row + g * D + d]);
          }
          for (uint32_t d = 0; d < D; d++) want[((size_t) i * H + h) * D + d] = (float) (o[d] / l);
        }
      float *dq = dev_from(q), *dkb = dev_from(kb), *dvb = dev_from(vb), *dout = dev_alloc<float>(want.size()),
            *dscr = dev_alloc<float>(ink2_mtp_attn_scratch(nq, H, KH) / 4);
      uint16_t *dkr = dev_alloc<uint16_t>(kr.size()), *dvr = dev_alloc<uint16_t>(vr.size());
      CHECK(hipMemcpy(dkr, kr.data(), kr.size() * 2, hipMemcpyHostToDevice) == hipSuccess);
      CHECK(hipMemcpy(dvr, vr.data(), vr.size() * 2, hipMemcpyHostToDevice) == hipSuccess);
      CHECK(ink2_mtp_attn(dq, nq, c.pos0, dkb, dvb, dkr, dvr, c.ring, c.from, H, KH, c.window, dscr, dout, nullptr));
      const std::vector<float> got = host_of(dout, want.size());
      CHECK(ink2_mtp_attn(dq, nq, c.pos0, dkb, dvb, dkr, dvr, c.ring, c.from, H, KH, c.window, dscr, dout, nullptr));
      const std::vector<float> again = host_of(dout, want.size());
      CHECK(std::memcmp(got.data(), again.data(), got.size() * 4) == 0);
      CHECK(!ink2_mtp_attn(dq, nq + 1, c.pos0, dkb, dvb, dkr, dvr, c.ring, c.from, H, KH, c.window, dscr, dout, nullptr));
      double worst = 0;
      for (size_t i = 0; i < got.size(); i++) worst = std::max(worst, (double) std::fabs(got[i] - want[i]));
      CHECK(worst < 1e-5 || (std::printf("  mtp_attn: %d keys back: max |err| %.2e\n", c.pos0 - c.from, worst), false));
      std::printf("  mtp_attn: from %d to %d (ring %u, window %u): max |err| %.2e against float64, the same bits again\n",
                  c.from, c.pos0, c.ring, c.window, worst);
      for (void* d : {(void*) dq, (void*) dkb, (void*) dvb, (void*) dout, (void*) dscr, (void*) dkr, (void*) dvr}) hipFree(d);
    }
  }
  {  // the loop's controls in one launch (a decode's q, k and v): each part ink2_loop_ctrl's bits; out may be x
    const uint64_t n = 3 * 2304 + 5;  // a token's rows and a few past a block
    std::vector<float*> dx, dm, dg, df, d1, dn;
    Ink2LoopCtrlPart parts[INK2_LOOP_CTRL_PARTS];
    for (uint32_t c = 0; c < 3; c++) {
      dx.push_back(dev_from(normals(n, 11 + c, 1.0f))), dm.push_back(dev_from(normals(n, 21 + c, 1.0f)));
      dg.push_back(dev_from(normals(n, 31 + c, 2.0f))), df.push_back(dev_from(normals(n, 41 + c, 2.0f)));
      d1.push_back(dev_alloc<float>(n)), dn.push_back(dev_alloc<float>(n));
      CHECK(ink2_loop_ctrl(dx[c], dm[c], dg[c], df[c], n, d1[c], nullptr));
      parts[c] = Ink2LoopCtrlPart{dx[c], dm[c], dg[c], df[c], dn[c]};
    }
    CHECK(ink2_loop_ctrl_many(parts, 3, n, nullptr));
    for (uint32_t c = 0; c < 3; c++) CHECK(same_bits(host_of(d1[c], n), host_of(dn[c], n)));
    parts[0].out = dx[0];  // in place
    CHECK(ink2_loop_ctrl_many(parts, 1, n, nullptr) && same_bits(host_of(dx[0], n), host_of(d1[0], n)));
    CHECK(!ink2_loop_ctrl_many(parts, 0, n, nullptr) && !ink2_loop_ctrl_many(parts, INK2_LOOP_CTRL_PARTS + 1, n, nullptr) &&
          !ink2_loop_ctrl_many(parts, 3, 0, nullptr) && !ink2_loop_ctrl_many(nullptr, 3, n, nullptr));
    for (uint32_t c = 0; c < 3; c++)
      for (float* d : {dx[c], dm[c], dg[c], df[c], d1[c], dn[c]}) hipFree(d);
    std::printf("  loop_ctrl_many: 3 parts in one launch, each ink2_loop_ctrl's bits (in place too); refusals\n");
  }
  {  // a loop's FFN control then the router's norm in one launch: ink2_loop_ctrl then ops_rms_x16's bits (its out, the
     // norm's y and x16), out in place of x too; 1, 3 and 16 rows
    const uint32_t C = 2304;
    const std::vector<float> nw = normals(C, 7, 0.3f, 1.0f);
    float* dw = dev_from(nw);
    int32_t* ovf = dev_alloc<int32_t>(1);
    for (uint32_t rows : {1u, 3u, 16u}) {
      const uint64_t n = (uint64_t) rows * C;
      float *dx = dev_from(normals(n, 51, 1.0f)), *dm = dev_from(normals(n, 52, 1.0f)), *dg = dev_from(normals(n, 53, 2.0f)),
            *df = dev_from(normals(n, 54, 2.0f));
      float *o1 = dev_alloc<float>(n), *y1 = dev_alloc<float>(n), *y2 = dev_alloc<float>(n), *x2 = dev_alloc<float>(n);
      void *h1 = dev_alloc<uint16_t>((size_t) 16 * C), *h2 = dev_alloc<uint16_t>((size_t) 16 * C);
      CHECK(ink2_loop_ctrl(dx, dm, dg, df, n, o1, nullptr) &&
            ops_rms_x16(o1, C, C, dw, 1e-6f, rows, y1, C, h1, ovf, nullptr));
      CHECK(hipMemcpy(x2, dx, n * 4, hipMemcpyDeviceToDevice) == hipSuccess);
      const OpsCtrl ctl = {x2, dm, dg, df, x2};  // in place
      CHECK(ops_rms_x16_ctrl(&ctl, C, dw, 1e-6f, rows, y2, C, h2, ovf, nullptr));
      CHECK(same_bits(host_of(o1, n), host_of(x2, n)) && same_bits(host_of(y1, n), host_of(y2, n)));
      CHECK(bytes_of(h1, (size_t) 16 * C * 2) == bytes_of(h2, (size_t) 16 * C * 2));
      for (void* d : {(void*) dx, (void*) dm, (void*) dg, (void*) df, (void*) o1, (void*) y1, (void*) y2, (void*) x2, h1, h2})
        hipFree(d);
    }
    const OpsCtrl none = {nullptr, nullptr, nullptr, nullptr, nullptr};
    CHECK(!ops_rms_x16_ctrl(&none, C, dw, 1e-6f, 1, dw, C, dw, ovf, nullptr) &&
          !ops_rms_x16_ctrl(nullptr, C, dw, 1e-6f, 1, dw, C, dw, ovf, nullptr));
    hipFree(dw), hipFree(ovf);
    std::printf("  rms_x16_ctrl: the loop's control then the norm in one launch, their bits (1 / 3 / 16 rows, in place)\n");
  }
  std::printf("[test_ink2_ops_gpu] ok\n");
  return 0;
}
