// the attention core on Qwen3.8-Flash-Next's attention (docs/attention-core.md, C1): its bits change once, so the core
// is held to exact math - an F64 attention on the host over the same Q8_0 cache values - and its error may be no
// larger than q4x_attn's (today's) on any case; then its bits a token's own (all picked == dense, a prompt in one
// step == three, a decode == its prompt row), and what it refuses
#include "attn_core_test.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/q4x_ops.h"
}

// Qwen's: 24 heads over 2 KV heads (12 a KV head), heads of 256, QSA's blocks of 4
constexpr uint32_t H = 24, KV = 2, D = 256, kRatio = 4;
constexpr float kScale = 0.0625f;  // 256^-0.5

static float h2f(uint16_t h) { return (float) *(const _Float16*) &h; }

// a sequence: K cache rows (positions 0 ..), its last T tokens the step's (positions K - T ..), picks of at most
// `stride` blocks a token (all: every block)
struct Seq {
  uint32_t T, K, stride;
  bool all;
  std::vector<float> q, gate;
  std::vector<uint8_t> kc, vc;  // Q8_0 rows (q4x_kv_put's), host copies
  std::vector<int32_t> sel, npick;
  float *dq, *dg;
  void *dkc, *dvc;
  int32_t *dsel, *dnpick;
};

static Seq make(uint32_t T, uint32_t K, uint32_t stride, bool all) {
  Seq s{T, K, stride, all};
  s.q = rnd((size_t) T * H * D, 0.3f), s.gate = rnd((size_t) T * H * D, 1.0f);
  const std::vector<float> k = rnd((size_t) K * KV * D, 1.0f), v = rnd((size_t) K * KV * D, 1.0f);
  float *dk = up(k), *dv = up(v);
  const uint64_t row = q4x_kv_row_bytes(KV * D);
  s.dkc = dev<uint8_t>(K * row), s.dvc = dev<uint8_t>(K * row);
  HIP_OK(q4x_kv_put(dk, dv, K, KV * D, s.dkc, s.dvc, nullptr) ? hipSuccess : hipErrorUnknown);
  s.kc = down(s.dkc, K * row), s.vc = down(s.dvc, K * row);
  (void) hipFree(dk), (void) hipFree(dv);
  s.sel.assign((size_t) T * stride, 0), s.npick.assign(T, 0);
  for (uint32_t t = 0; t < T; t++) {  // a token's picks: every visible block, or `stride` of them (spread, ascending)
    const uint32_t p = K - T + t, nb = (p + 1) / kRatio;
    if (all || nb <= stride) {
      s.npick[t] = (int32_t) nb;
      continue;
    }
    std::vector<int32_t> pick;
    for (uint32_t b = 0; b < nb && pick.size() < stride; b++)
      if ((uint32_t) (uni() * nb) < stride) pick.push_back((int32_t) b);
    s.npick[t] = (int32_t) pick.size();
    std::copy(pick.begin(), pick.end(), s.sel.begin() + (size_t) t * stride);
  }
  s.dq = up(s.q), s.dg = up(s.gate);
  s.dsel = dev<int32_t>(s.sel.size()), s.dnpick = dev<int32_t>(T);
  HIP_OK(hipMemcpy(s.dsel, s.sel.data(), s.sel.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(s.dnpick, s.npick.data(), T * 4, hipMemcpyHostToDevice));
  return s;
}

static void free_seq(Seq& s) {
  for (void* p : {(void*) s.dq, (void*) s.dg, s.dkc, s.dvc, (void*) s.dsel, (void*) s.dnpick}) (void) hipFree(p);
}

// a Q8_0 row's value (ggml's blocks: an f16 d, 32 int8s), exact in double
static double q8(const std::vector<uint8_t>& c, uint32_t pos, uint32_t i) {
  const uint8_t* b = c.data() + (size_t) pos * q4x_kv_row_bytes(KV * D) + i / 32 * 34;
  return (double) h2f(*(const uint16_t*) b) * (int8_t) b[2 + i % 32];
}

// a token's keys: its picked blocks' positions, then its tail - every position to its own when all are picked
static std::vector<uint32_t> keys_of(const Seq& s, uint32_t t) {
  const uint32_t p = s.K - s.T + t, nb = (p + 1) / kRatio;
  std::vector<uint32_t> ks;
  if ((uint32_t) s.npick[t] == nb)
    for (uint32_t x = 0; x <= p; x++) ks.push_back(x);
  else {
    for (int32_t i = 0; i < s.npick[t]; i++)
      for (uint32_t x = 0; x < kRatio; x++) ks.push_back((uint32_t) s.sel[(size_t) t * s.stride + i] * kRatio + x);
    for (uint32_t x = nb * kRatio; x <= p; x++) ks.push_back(x);
  }
  return ks;
}

// exact math: out[t][h][d] = sigmoid(gate) sum softmax(q . k scale) v over the token's keys, in double
static std::vector<double> exact(const Seq& s) {
  std::vector<double> out((size_t) s.T * H * D);
  for (uint32_t t = 0; t < s.T; t++) {
    const std::vector<uint32_t> ks = keys_of(s, t);
    for (uint32_t h = 0; h < H; h++) {
      const uint32_t g = h / (H / KV);
      const float* q = s.q.data() + ((size_t) t * H + h) * D;
      std::vector<double> sc(ks.size());
      double mx = -INFINITY, z = 0;
      for (size_t i = 0; i < ks.size(); i++) {
        double d = 0;
        for (uint32_t e = 0; e < D; e++) d += (double) q[e] * q8(s.kc, ks[i], g * D + e);
        mx = std::max(mx, sc[i] = d * kScale);
      }
      for (double& x : sc) z += x = std::exp(x - mx);
      for (uint32_t e = 0; e < D; e++) {
        double o = 0;
        for (size_t i = 0; i < ks.size(); i++) o += sc[i] * q8(s.vc, ks[i], g * D + e);
        const size_t at = ((size_t) t * H + h) * D + e;
        out[at] = o / z / (1.0 + std::exp(-(double) s.gate[at]));
      }
    }
  }
  return out;
}

// today's kernel on the sequence (q4x_attn: its picks, no shape - every key, a gate a value)
static std::vector<float> today(const Seq& s) {
  float *out = dev<float>((size_t) s.T * H * D), *scratch = dev<float>(q4x_attn_scratch(H, s.K) + 1);
  const Q4xPick pick = {s.dsel, s.dnpick, kRatio, s.stride};
  HIP_OK(q4x_attn(s.dq, s.dkc, s.dvc, s.T, H, KV, D, s.K - s.T, s.dg, &pick, nullptr, scratch, out, nullptr, nullptr)
             ? hipSuccess : hipErrorUnknown);
  std::vector<float> o((size_t) s.T * H * D);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(o.data(), out, o.size() * 4, hipMemcpyDeviceToHost));
  (void) hipFree(out), (void) hipFree(scratch);
  return o;
}

static AttnCoreDesc qwen_desc(uint32_t max_tokens, uint32_t max_keys) {
  AttnCoreDesc d{};
  d.heads = H, d.kv_heads = KV, d.dk = d.dv = D, d.k_rows = d.v_rows = ATTN_ROWS_Q8_0, d.window = 0;
  d.scaling = kScale, d.term = ATTN_TERM_NONE, d.sink = false, d.gate = ATTN_GATE_VALUE;
  d.block = ATTN_BLOCK_HEADS, d.ratio = kRatio, d.max_tokens = max_tokens, d.max_keys = max_keys, d.unit = 128;
  return d;
}

// the core on tokens t0 .. t0 + n - 1 of the sequence (their picks, keys to the last of them); empty when refused
static std::vector<float> core(const Seq& s, uint32_t t0, uint32_t n, std::string* why, bool picks = true) {
  const AttnCoreDesc d = qwen_desc(s.T, s.K);
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, sb, err);
  std::vector<float> o;
  if (!c) {
    *why = err;
  } else {
    float* out = dev<float>((size_t) n * H * D);
    AttnStep st{};
    st.q = s.dq + (size_t) t0 * H * D, st.n_tok = n, st.pos0 = s.K - s.T + t0;
    st.keys.k = s.dkc, st.keys.v = s.dvc, st.keys.key0 = 0, st.keys.n_keys = s.K - s.T + t0 + n;
    if (picks) st.keys.sel = s.dsel + (size_t) t0 * s.stride, st.keys.npick = s.dnpick + t0, st.keys.stride = s.stride;
    st.gate = s.dg + (size_t) t0 * H * D, st.out = out;
    if (attn_core_run(c, &st, nullptr)) {
      o.resize((size_t) n * H * D);
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipMemcpy(o.data(), out, o.size() * 4, hipMemcpyDeviceToHost));
    } else
      *why = "attn_core_run refused the step";
    (void) hipFree(out);
  }
  attn_core_destroy(c);
  (void) hipFree(scratch);
  return o;
}

static double rms_err(const std::vector<float>& got, const std::vector<double>& ref) {
  double e = 0;
  for (size_t i = 0; i < ref.size(); i++) e += (got[i] - ref[i]) * (got[i] - ref[i]);
  return std::sqrt(e / ref.size());
}
static std::vector<uint8_t> bytes(const std::vector<float>& v) {
  return std::vector<uint8_t>((const uint8_t*) v.data(), (const uint8_t*) (v.data() + v.size()));
}

// against exact math: the core's error no larger than q4x_attn's, case by case
static void accuracy() {
  struct Case { uint32_t T, K, stride; bool all; };
  const Case cases[] = {{37, 37, 64, true},   {600, 600, 1024, true},   {1, 1500, 64, false}, {3, 1503, 64, false},
                        {16, 1520, 64, false}, {200, 3000, 128, false}, {1, 3000, 1024, true}};
  for (const Case& c : cases) {
    Seq s = make(c.T, c.K, c.stride, c.all);
    const std::vector<double> ref = exact(s);
    const std::vector<float> was = today(s);
    std::string why;
    const std::vector<float> got = core(s, 0, c.T, &why);
    // no larger than today's past the rms estimate's own noise: two of its relative standard errors, 1 / sqrt(2 N)
    // over N values - an equally exact kernel's sums in another order land on either side of it
    const double ew = rms_err(was, ref), eg = got.empty() ? INFINITY : rms_err(got, ref);
    const double noise = 2.0 / std::sqrt(2.0 * (double) ref.size());
    char name[240];
    std::snprintf(name, sizeof name, "qwen accuracy: %u tokens of %u keys%s - rms error core %.6g, q4x_attn %.6g (%+.3f%%, "
                  "noise %.3f%%)%s", c.T, c.K, c.all ? " (dense)" : " (picks)", eg, ew, 100 * (eg / ew - 1), 100 * noise,
                  got.empty() ? (" (" + why + ")").c_str() : "");
    check(!got.empty() && eg <= ew * (1 + noise), name);
    std::printf("  %s\n", name);
    free_seq(s);
  }
}

// a token's bits its own: every block picked == dense keys; a prompt in one step == three; a decode == its prompt row
static void own_bits() {
  Seq s = make(600, 1200, 64, false);
  std::string why;
  const std::vector<float> whole = core(s, 0, 600, &why);
  std::vector<float> steps, dec;
  for (uint32_t t0 = 0; t0 < 600 && !whole.empty(); t0 += 200) {
    const std::vector<float> p = core(s, t0, 200, &why);
    steps.insert(steps.end(), p.begin(), p.end());
  }
  for (uint32_t t = 596; t < 600 && !whole.empty(); t++) {
    const std::vector<float> p = core(s, t, 1, &why);
    dec.insert(dec.end(), p.begin(), p.end());
  }
  const std::vector<float> tail = whole.empty() ? whole : std::vector<float>(whole.end() - 4 * H * D, whole.end());
  check(!whole.empty() && bytes(whole) == bytes(steps), "qwen own bits: one step == three (" + why + ")");
  check(!whole.empty() && bytes(tail) == bytes(dec), "qwen own bits: a decode == its prompt row (" + why + ")");
  free_seq(s);
  Seq a = make(300, 300, 1024, true);  // every block picked (npick = visible), against no picks at all
  const std::vector<float> picked = core(a, 0, 300, &why), dense = core(a, 0, 300, &why, false);
  check(!picked.empty() && bytes(picked) == bytes(dense), "qwen own bits: every block picked == no picks (" + why + ")");
  free_seq(a);
}

static void refusals() {
  struct Bad { const char* field; void (*bend)(AttnCoreDesc*); };
  const Bad bad[] = {
      {"ratio", [](AttnCoreDesc* d) { d->ratio = 3; }},                      // not a tile's divisor
      {"dk", [](AttnCoreDesc* d) { d->dk = d->dv = 192; }},
      {"group", [](AttnCoreDesc* d) { d->heads = 34, d->kv_heads = 2; }},   // 17 a KV head: past a tile's 16 rows
      {"ratio", [](AttnCoreDesc* d) { d->block = ATTN_BLOCK_TOKENS; }},      // picks on token rows
  };
  const AttnCoreDesc good = qwen_desc(64, 4096);
  const uint64_t sb = attn_core_scratch(&good);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&good, scratch, sb, err);
  check(c != nullptr, std::string("qwen refusals: Qwen's own desc made (") + err + ")");
  attn_core_destroy(c);
  for (const Bad& b : bad) {
    AttnCoreDesc d = good;
    b.bend(&d);
    err[0] = 0;
    AttnCore* x = attn_core_create(&d, scratch, sb, err);
    check(!x && std::strstr(err, b.field), std::string("qwen refusals: ") + b.field + " refused by name (\"" + err + "\")");
    attn_core_destroy(x);
  }
  (void) hipFree(scratch);
}

int main() {
  accuracy();
  own_bits();
  refusals();
  std::printf("attn_core qwen: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
