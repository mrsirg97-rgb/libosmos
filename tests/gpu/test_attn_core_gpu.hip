// the attention core (osmos/gpu/attn_core.h, docs/attention-core.md): Inkling's attention byte for byte (ink2_fa,
// ink2_fa16, ink2_fa8, ink2_fa85 - prompts, a step at depth, split decodes and verifies, its f16 out), its bits a
// token's own (a prompt in one step and in three, a decode and its prompt row), and what it refuses, named. seeded
// inputs; every comparison's own mutation checked first
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/ink2_ops.h"
}

#define HIP_OK(x)                                                                \
  do {                                                                           \
    if ((x) != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const std::string& what) {
  if (ok)
    g_pass++;
  else
    g_fail++, std::printf("  FAIL %s\n", what.c_str());
}

template <class T>
static T* dev(size_t n) {
  T* p;
  HIP_OK(hipMalloc((void**) &p, n * sizeof(T) + 256));
  HIP_OK(hipMemset(p, 0, n * sizeof(T) + 256));
  return p;
}
static float* up(const std::vector<float>& v) {
  float* d = dev<float>(v.size());
  HIP_OK(hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice));
  return d;
}
static std::vector<uint8_t> down(const void* p, size_t bytes) {
  std::vector<uint8_t> b(bytes);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(b.data(), p, bytes, hipMemcpyDeviceToHost));
  return b;
}
static bool same(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size());
}

// Inkling-2-Mini's attention: 24 heads over 8 KV heads (GQA 3), dim 128, 16 relative states over 1,024 positions
constexpr uint32_t H = 24, KV = 8, D = 128, R = 16, E = 1024;
constexpr float kScale = 0.08838834764831845f;  // 128^-0.5

static std::mt19937 g_rng(1017);
static std::vector<float> rnd(size_t n, float s) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = s * nd(g_rng);
  return v;
}

enum Kind { KIND_F32, KIND_F16, KIND_KV8, KIND_KV85 };
static const char* kKind[] = {"F32", "f16", "Q8", "Q8 / Q5"};

// a sequence's inputs: K keys (positions key0 ..), T queries (the last T keys)
struct Inputs {
  uint32_t T, K, key0;
  float *q, *rel, *sink, *gate, *rp, *k, *v;
  const void *kc, *vc;  // the cache as the kind stores it
};

static Inputs make(Kind kind, uint32_t T, uint32_t K, uint32_t key0) {
  Inputs in{T, K, key0};
  in.q = up(rnd((size_t) T * H * D, 0.3f)), in.rel = up(rnd((size_t) T * H * R, 0.5f));
  in.sink = up(rnd((size_t) T * H, 1.0f)), in.gate = up(rnd((size_t) T * H * D, 1.0f)), in.rp = up(rnd((size_t) R * E, 0.5f));
  const size_t kv = (size_t) K * KV * D;
  in.k = up(rnd(kv, 1.0f)), in.v = up(rnd(kv, 1.0f));
  if (kind == KIND_F32)
    in.kc = in.k, in.vc = in.v;
  else if (kind == KIND_F16) {
    uint16_t *k16 = dev<uint16_t>(kv), *v16 = dev<uint16_t>(kv);
    HIP_OK(ink2_to_f16(in.k, kv, k16, nullptr) ? hipSuccess : hipErrorUnknown);
    HIP_OK(ink2_to_f16(in.v, kv, v16, nullptr) ? hipSuccess : hipErrorUnknown);
    in.kc = k16, in.vc = v16;
  } else {
    uint8_t *k8 = dev<uint8_t>((size_t) K * KV * INK2_KV8_RECORD), *v8 = dev<uint8_t>((size_t) K * KV * INK2_KV8_RECORD);
    HIP_OK(ink2_to_kv8(in.k, (uint64_t) K * KV, k8, nullptr) ? hipSuccess : hipErrorUnknown);
    const bool ok = kind == KIND_KV8 ? ink2_to_kv8(in.v, (uint64_t) K * KV, v8, nullptr)
                                     : ink2_to_kv5(in.v, (uint64_t) K * KV, v8, nullptr);
    HIP_OK(ok ? hipSuccess : hipErrorUnknown);
    in.kc = k8, in.vc = v8;
  }
  return in;
}

static AttnCoreDesc desc(Kind kind, uint32_t window, uint32_t max_tokens, uint32_t max_keys) {
  AttnCoreDesc d{};
  d.heads = H, d.kv_heads = KV, d.dk = D, d.dv = D;
  d.k_rows = kind == KIND_F32 ? ATTN_ROWS_F32 : kind == KIND_F16 ? ATTN_ROWS_F16 : ATTN_ROWS_KV8;
  d.v_rows = kind == KIND_F32 ? ATTN_ROWS_F32 : kind == KIND_F16 ? ATTN_ROWS_F16 : kind == KIND_KV8 ? ATTN_ROWS_KV8 : ATTN_ROWS_KV5;
  d.window = window, d.scaling = kScale;
  d.term = ATTN_TERM_REL, d.d_rel = R, d.extent = E;
  d.sink = true, d.gate = ATTN_GATE_VALUE;
  d.max_tokens = max_tokens, d.max_keys = max_keys;
  return d;
}

// Inkling's kernel on the inputs (the oracle): its out, and its x16 when asked (a split decode's)
static std::vector<uint8_t> ink2(Kind kind, uint32_t window, const Inputs& in, uint32_t t0, uint32_t n, bool x16,
                                 std::vector<uint8_t>* x16_out) {
  float* out = dev<float>((size_t) n * H * D);
  float* scratch = dev<float>(ink2_fa_scratch(H, in.K) / 4 + 1);
  void* xb = x16 ? (void*) dev<uint16_t>((size_t) 16 * H * D + 4096) : nullptr;
  int32_t* of = dev<int32_t>(1);
  const Ink2Attn a = {H, KV, D, R, E, window, kScale, xb, x16 ? of : nullptr};
  const uint32_t keys = in.K - in.T + t0 + n;  // the step's last token's key and those before it
  const uint64_t q = (uint64_t) t0 * H * D, r = (uint64_t) t0 * H * R, s = (uint64_t) t0 * H;
  bool ok;
  if (kind == KIND_F32)
    ok = ink2_fa(&a, in.q + q, in.rel + r, in.sink + s, in.gate + q, n, in.k, in.v, keys, in.key0, in.rp, out, scratch, nullptr);
  else if (kind == KIND_F16)
    ok = ink2_fa16(&a, in.q + q, in.rel + r, in.sink + s, in.gate + q, n, (const uint16_t*) in.kc, (const uint16_t*) in.vc,
                   keys, in.key0, in.rp, out, scratch, nullptr);
  else if (kind == KIND_KV8)
    ok = ink2_fa8(&a, in.q + q, in.rel + r, in.sink + s, in.gate + q, n, (const uint8_t*) in.kc, (const uint8_t*) in.vc,
                  keys, in.key0, in.rp, out, scratch, nullptr);
  else
    ok = ink2_fa85(&a, in.q + q, in.rel + r, in.sink + s, in.gate + q, n, (const uint8_t*) in.kc, (const uint8_t*) in.vc,
                   keys, in.key0, in.rp, out, scratch, nullptr);
  HIP_OK(ok ? hipSuccess : hipErrorUnknown);
  std::vector<uint8_t> o = down(out, (size_t) n * H * D * 4);
  if (x16_out) *x16_out = down(xb, (size_t) 16 * H * D * 2);
  (void) hipFree(out), (void) hipFree(scratch), (void) hipFree(xb), (void) hipFree(of);
  return o;
}

// the core on the same: tokens t0 .. t0 + n - 1 of the sequence, its keys up to the last of them; nothing (an empty
// vector) when the core is not made or refuses the step
static std::vector<uint8_t> core(Kind kind, uint32_t window, const Inputs& in, uint32_t t0, uint32_t n, bool x16,
                                 std::vector<uint8_t>* x16_out, std::string* why) {
  char err[OSMOS_ERR] = "";
  const AttnCoreDesc d = desc(kind, window, in.T, in.K);
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  AttnCore* c = attn_core_create(&d, scratch, sb, err);
  if (!c) return (void) hipFree(scratch), *why = err, std::vector<uint8_t>();
  float* out = dev<float>((size_t) n * H * D);
  void* xb = x16 ? (void*) dev<uint16_t>((size_t) 16 * H * D + 4096) : nullptr;
  int32_t* of = dev<int32_t>(1);
  const uint64_t q = (uint64_t) t0 * H * D, r = (uint64_t) t0 * H * R, s = (uint64_t) t0 * H;
  AttnStep st{};
  st.q = in.q + q, st.n_tok = n, st.pos0 = in.key0 + in.K - in.T + t0;
  st.keys.k = in.kc, st.keys.v = in.vc, st.keys.key0 = in.key0, st.keys.n_keys = in.K - in.T + t0 + n;
  st.rel = in.rel + r, st.rel_proj = in.rp, st.sink = in.sink + s, st.gate = in.gate + q;
  st.out = out, st.x16 = xb, st.overflow = x16 ? of : nullptr;
  const bool ok = attn_core_run(c, &st, nullptr);
  std::vector<uint8_t> o;
  if (ok) o = down(out, (size_t) n * H * D * 4);
  else *why = "attn_core_run refused the step";
  if (ok && x16_out) *x16_out = down(xb, (size_t) 16 * H * D * 2);
  attn_core_destroy(c);
  (void) hipFree(out), (void) hipFree(xb), (void) hipFree(of), (void) hipFree(scratch);
  return o;
}

static void free_inputs(Inputs& in) {
  for (void* p : {(void*) in.q, (void*) in.rel, (void*) in.sink, (void*) in.gate, (void*) in.rp, (void*) in.k, (void*) in.v})
    (void) hipFree(p);
  if (in.kc != in.k) (void) hipFree((void*) in.kc), (void) hipFree((void*) in.vc);
}

// the comparison itself: a bit flipped in a copy must be caught (else every "same" below proves nothing)
static void mutation() {
  std::vector<uint8_t> a(4096, 0x3c), b = a;
  check(same(a, b), "mutation: equal buffers compare the same");
  b[2049] ^= 0x01;
  check(!same(a, b), "mutation: one flipped bit caught");
}

// Inkling's every format, window, length: the core's out (and a split decode's x16) byte for byte its kernel's
static void inkling() {
  struct Case { uint32_t T, K, key0; };
  const Case cases[] = {{37, 37, 0}, {1000, 1000, 0}, {2968, 2968, 0}, {512, 3008, 0}, {512, 1536, 1024},
                        {1, 1504, 0}, {3, 1507, 0}, {4, 1504, 0}, {16, 1520, 0}};
  for (int kind = 0; kind < 4; kind++)
    for (uint32_t window : {0u, 1024u})
      for (const Case& c : cases) {
        if ((c.K - c.T) % ATTN_TILE) continue;  // (a step from a whole tile: the engine's)
        Inputs in = make((Kind) kind, c.T, c.K, c.key0);
        const bool split = c.T <= 16;
        std::vector<uint8_t> xr, xc;
        std::string why;
        const std::vector<uint8_t> want = ink2((Kind) kind, window, in, 0, c.T, split, split ? &xr : nullptr);
        const std::vector<uint8_t> got = core((Kind) kind, window, in, 0, c.T, split, split ? &xc : nullptr, &why);
        char name[160];
        std::snprintf(name, sizeof name, "inkling %s window %u: %u tokens of %u keys from %u%s", kKind[kind], window, c.T,
                      c.K, c.key0, got.empty() ? (" (" + why + ")").c_str() : "");
        check(!got.empty() && same(want, got), name);
        if (split) check(!got.empty() && same(xr, xc), std::string(name) + ": its x16");
        free_inputs(in);
      }
}

// a token's bits its own: a 600-token prompt in one step and in three (200 each); its last 4 tokens decoded one at a
// time - each row the one-step prompt's
static void own_bits() {
  for (int kind = 0; kind < 4; kind++) {
    Inputs in = make((Kind) kind, 600, 600, 0);
    std::string why;
    const std::vector<uint8_t> whole = core((Kind) kind, 0, in, 0, 600, false, nullptr, &why);
    std::vector<uint8_t> steps, decodes;
    for (uint32_t t0 = 0; t0 < 600 && !whole.empty(); t0 += 200) {
      const std::vector<uint8_t> s = core((Kind) kind, 0, in, t0, 200, false, nullptr, &why);
      steps.insert(steps.end(), s.begin(), s.end());
    }
    for (uint32_t t = 596; t < 600 && !whole.empty(); t++) {
      const std::vector<uint8_t> s = core((Kind) kind, 0, in, t, 1, false, nullptr, &why);
      decodes.insert(decodes.end(), s.begin(), s.end());
    }
    const size_t row = (size_t) H * D * 4;
    const std::vector<uint8_t> tail = whole.empty() ? whole : std::vector<uint8_t>(whole.end() - 4 * row, whole.end());
    check(!whole.empty() && same(whole, steps), std::string("own bits ") + kKind[kind] + ": one step == three (" + why + ")");
    check(!whole.empty() && same(tail, decodes), std::string("own bits ") + kKind[kind] + ": decode == its prompt row (" + why + ")");
    free_inputs(in);
  }
}

// what it does not take, refused at create with the field named; a step past its desc refused at run
static void refusals() {
  struct Bad { const char* field; void (*bend)(AttnCoreDesc*); };
  const Bad bad[] = {
      {"dk", [](AttnCoreDesc* d) { d->dk = 96, d->dv = 96; }},
      {"kv_heads", [](AttnCoreDesc* d) { d->kv_heads = 5; }},             // 24 heads over 5: not a group
      {"group", [](AttnCoreDesc* d) { d->heads = 136, d->kv_heads = 8; }}, // 17 a KV head: past the tile's 16
      {"k_rows", [](AttnCoreDesc* d) { d->k_rows = ATTN_ROWS_KV5; }},       // Q5 keys: none made
      {"d_rel", [](AttnCoreDesc* d) { d->d_rel = 8; }},
      {"window", [](AttnCoreDesc* d) { d->window = 1000; }},               // not a whole tile
      {"max_tokens", [](AttnCoreDesc* d) { d->max_tokens = 0; }},
      {"max_keys", [](AttnCoreDesc* d) { d->max_keys = 1u << 30; }},     // past the scratch it was given
  };
  const AttnCoreDesc good = desc(KIND_KV85, 0, 64, 4096);
  char err[OSMOS_ERR] = "";
  const uint64_t sb = attn_core_scratch(&good);
  void* scratch = dev<uint8_t>(sb);
  AttnCore* c = attn_core_create(&good, scratch, sb, err);
  check(c != nullptr, std::string("refusals: Inkling's own desc made (") + err + ")");
  for (const Bad& b : bad) {
    AttnCoreDesc d = good;
    b.bend(&d);
    err[0] = 0;
    AttnCore* x = attn_core_create(&d, scratch, sb, err);
    check(!x && std::strstr(err, b.field), std::string("refusals: ") + b.field + " refused by name (\"" + err + "\")");
    attn_core_destroy(x);
  }
  if (c) {
    Inputs in = make(KIND_KV85, 64, 64, 0);
    AttnStep st{};
    st.q = in.q, st.n_tok = 65, st.pos0 = 0;  // past max_tokens
    st.keys.k = in.kc, st.keys.v = in.vc, st.keys.key0 = 0, st.keys.n_keys = 64;
    st.rel = in.rel, st.rel_proj = in.rp, st.sink = in.sink, st.gate = in.gate, st.out = in.k;
    check(!attn_core_run(c, &st, nullptr), "refusals: a step past max_tokens");
    st.n_tok = 16, st.keys.key0 = 8;  // key0 off a tile
    check(!attn_core_run(c, &st, nullptr), "refusals: key0 off a tile");
    free_inputs(in);
  }
  attn_core_destroy(c);
  (void) hipFree(scratch);
}

int main() {
  mutation();
  inkling();
  own_bits();
  refusals();
  std::printf("attn_core: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
