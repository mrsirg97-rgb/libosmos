// the attention core on the reads-apart families (osmos1.7b, q35: attn_core.h's apart; docs/attention-core.md C2):
// each of their paths byte for byte their kernels' as their callers called them - a dense prompt (LFM3's spans, at 0
// and at depth), a sliding window over a ring, decodes and verifies, LSA's prompt lists, LSA's decode read - at
// osmos1.7b's shape (16 query heads over 4, heads of 128, a gate a head) and a q35-like one (heads of 256, a gate a
// value); and what it refuses
#include "attn_core_test.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/q4x_ops.h"
}

struct Shape {
  const char* name;
  uint32_t H, KV, D;
  bool head_gate;
};
static const Shape kLfm3 = {"osmos1.7b", 16, 4, 128, true}, kQ35 = {"q35", 16, 4, 256, false};

struct Cache {
  uint32_t K;
  void *kc, *vc;
};
static Cache cache(const Shape& sh, uint32_t K) {
  const std::vector<float> k = rnd((size_t) K * sh.KV * sh.D, 1.0f), v = rnd((size_t) K * sh.KV * sh.D, 1.0f);
  float *dk = up(k), *dv = up(v);
  const uint64_t row = q4x_kv_row_bytes(sh.KV * sh.D);
  Cache c{K, dev<uint8_t>(K * row), dev<uint8_t>(K * row)};
  HIP_OK(q4x_kv_put(dk, dv, K, sh.KV * sh.D, c.kc, c.vc, nullptr) ? hipSuccess : hipErrorUnknown);
  (void) hipFree(dk), (void) hipFree(dv);
  return c;
}

static AttnCoreDesc desc(const Shape& sh, uint32_t window, uint32_t slots, uint32_t max_tokens, uint32_t max_keys) {
  AttnCoreDesc d{};
  d.heads = sh.H, d.kv_heads = sh.KV, d.dk = d.dv = sh.D, d.k_rows = d.v_rows = ATTN_ROWS_Q8_0;
  d.window = window, d.slots = slots, d.scaling = 1.0f / std::sqrt((float) sh.D), d.term = ATTN_TERM_NONE;
  d.gate = sh.head_gate ? ATTN_GATE_HEAD : ATTN_GATE_VALUE, d.block = ATTN_BLOCK_HEADS, d.unit = 128, d.apart = true;
  d.max_tokens = max_tokens, d.max_keys = max_keys;
  return d;
}

// a step's inputs: q, the gate, an out
struct Step {
  uint32_t n;
  float *q, *gate, *out;
};
static Step step(const Shape& sh, uint32_t n) {
  return {n, up(rnd((size_t) n * sh.H * sh.D, 0.3f)), up(rnd((size_t) n * sh.H * (sh.head_gate ? 1 : sh.D), 1.0f)),
          dev<float>((size_t) n * sh.H * sh.D)};
}
static void free_step(Step& s) { (void) hipFree(s.q), (void) hipFree(s.gate), (void) hipFree(s.out); }

// the core's run of a step (keys as given), its out's bytes; empty when not made or refused
static std::vector<uint8_t> core(const Shape& sh, const AttnCoreDesc& d, const Step& s, AttnKeys keys, uint32_t pos0,
                                 bool read, std::string* why) {
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, sb, err);
  std::vector<uint8_t> o;
  if (!c)
    *why = err;
  else {
    AttnStep st{};
    st.q = s.q, st.n_tok = s.n, st.pos0 = pos0, st.keys = keys, st.gate = s.gate, st.out = s.out, st.read = read;
    HIP_OK(hipMemset(s.out, 0x7f, (size_t) s.n * sh.H * sh.D * 4));
    if (attn_core_run(c, &st, nullptr))
      o = down(s.out, (size_t) s.n * sh.H * sh.D * 4);
    else
      *why = "attn_core_run refused the step";
  }
  attn_core_destroy(c);
  (void) hipFree(scratch);
  return o;
}

static std::vector<uint8_t> out_of(const Shape& sh, const Step& s) { return down(s.out, (size_t) s.n * sh.H * sh.D * 4); }

static void compare(const char* what, const Shape& sh, const std::vector<uint8_t>& was, const std::vector<uint8_t>& got,
                    const std::string& why) {
  check(!got.empty() && same(was, got), std::string(sh.name) + " " + what + (got.empty() ? " (" + why + ")" : ""));
}

// a dense prompt (a read): q4x_attn's, no shape's window (LFM3's spans at 128) - at 0 and at depth
static void dense_prompt(const Shape& sh) {
  for (auto [n, pos0] : {std::pair<uint32_t, uint32_t>{600, 0}, {500, 1000}}) {
    Cache c = cache(sh, pos0 + n);
    Step s = step(sh, n);
    const AttnShape ash = {sh.head_gate ? sh.D : 1u, 0, 0};
    float* scr = dev<float>(q4x_attn_scratch_dim(sh.H, sh.KV, sh.D, c.K) + 1);
    HIP_OK(q4x_attn(s.q, c.kc, c.vc, n, sh.H, sh.KV, sh.D, pos0, s.gate, nullptr, &ash, scr, s.out, nullptr, nullptr)
               ? hipSuccess : hipErrorUnknown);
    const std::vector<uint8_t> was = out_of(sh, s);
    std::string why;
    AttnKeys k{};
    k.k = c.kc, k.v = c.vc, k.n_keys = c.K;
    const std::vector<uint8_t> got = core(sh, desc(sh, 0, 0, n, c.K), s, k, pos0, true, &why);
    char name[96];
    std::snprintf(name, sizeof name, "dense prompt: %u tokens from %u", n, pos0);
    compare(name, sh, was, got, why);
    (void) hipFree(scr), (void) hipFree(c.kc), (void) hipFree(c.vc), free_step(s);
  }
}

// a sliding layer's prompt and decode: a window of 512 over a ring of 2,048 rows (a position's row position % 2048)
static void sliding(const Shape& sh) {
  const uint32_t W = 512, slots = 2048, pos0 = 1200, n = 300;
  Cache c = cache(sh, slots);
  const AttnShape ash = {sh.head_gate ? sh.D : 1u, W, slots};
  AttnKeys k{};
  k.k = c.kc, k.v = c.vc, k.n_keys = slots;
  std::string why;
  {  // the prompt (a read): q4x_attn under the shape
    Step s = step(sh, n);
    float* scr = dev<float>(q4x_attn_scratch_dim(sh.H, sh.KV, sh.D, slots) + 1);
    HIP_OK(q4x_attn(s.q, c.kc, c.vc, n, sh.H, sh.KV, sh.D, pos0, s.gate, nullptr, &ash, scr, s.out, nullptr, nullptr)
               ? hipSuccess : hipErrorUnknown);
    const std::vector<uint8_t> was = out_of(sh, s);
    compare("sliding prompt: window 512, a ring of 2,048", sh, was, core(sh, desc(sh, W, slots, n, slots), s, k, pos0, true, &why), why);
    (void) hipFree(scr), free_step(s);
  }
  for (uint32_t m : {1u, 3u}) {  // a decode and a verify (not a read): q35_attn_decode under the shape
    Step s = step(sh, m);
    float* scr = dev<float>(q35_attn_decode_scratch(sh.H, Q35_ATTN_DECODE_MAX) + 1);
    HIP_OK(q35_attn_decode(s.q, c.kc, c.vc, m, sh.H, sh.KV, sh.D, &ash, pos0 + n, s.gate, scr, s.out, nullptr)
               ? hipSuccess : hipErrorUnknown);
    const std::vector<uint8_t> was = out_of(sh, s);
    compare(m == 1 ? "sliding decode" : "sliding verify of 3", sh, was,
            core(sh, desc(sh, W, slots, 16, slots), s, k, pos0 + n, false, &why), why);
    (void) hipFree(scr), free_step(s);
  }
  (void) hipFree(c.kc), (void) hipFree(c.vc);
}

// a full layer's decodes and verifies (not reads): q35_attn_decode, every key to a token's own, in steps of its most
static void decodes(const Shape& sh) {
  for (auto [m, pos0] : {std::pair<uint32_t, uint32_t>{1, 2999}, {4, 1500}, {7, 900}}) {
    Cache c = cache(sh, pos0 + m);
    Step s = step(sh, m);
    const AttnShape ash = {sh.head_gate ? sh.D : 1u, 0, 0};
    float* scr = dev<float>(q35_attn_decode_scratch(sh.H, Q35_ATTN_DECODE_MAX) + 1);
    for (uint32_t t0 = 0; t0 < m; t0 += Q35_ATTN_DECODE_MAX) {  // (its callers' steps)
      const uint32_t k = m - t0 < Q35_ATTN_DECODE_MAX ? m - t0 : Q35_ATTN_DECODE_MAX, g = sh.head_gate ? sh.H : sh.H * sh.D;
      HIP_OK(q35_attn_decode(s.q + (size_t) t0 * sh.H * sh.D, c.kc, c.vc, k, sh.H, sh.KV, sh.D, &ash, pos0 + t0,
                             s.gate + (size_t) t0 * g, scr, s.out + (size_t) t0 * sh.H * sh.D, nullptr)
                 ? hipSuccess : hipErrorUnknown);
    }
    const std::vector<uint8_t> was = out_of(sh, s);
    std::string why;
    AttnKeys k{};
    k.k = c.kc, k.v = c.vc, k.n_keys = c.K;
    char name[96];
    std::snprintf(name, sizeof name, "decode: %u tokens from %u", m, pos0);
    compare(name, sh, was, core(sh, desc(sh, 0, 0, 16, c.K), s, k, pos0, false, &why), why);
    (void) hipFree(scr), (void) hipFree(c.kc), (void) hipFree(c.vc), free_step(s);
  }
}

// LSA's lists: a prompt's (a read: q4x_attn_lists, heads of 128) and a decode's read (q35_attn_read, a token at a time)
static void lists(const Shape& sh) {
  const uint32_t K = 3000, n = 40, stride = 700;
  Cache c = cache(sh, K);
  std::vector<uint32_t> lst((size_t) n * stride);
  std::vector<int32_t> cnt(n);
  for (uint32_t t = 0; t < n; t++) {  // a token's list: its positions in any order, some of those to its own
    const uint32_t p = K - n + t, len = 100 + (uint32_t) (uni() * 600);
    cnt[t] = (int32_t) len;
    for (uint32_t i = 0; i < len; i++) lst[(size_t) t * stride + i] = (uint32_t) (uni() * (p + 1)) % (p + 1);
  }
  uint32_t* dl = dev<uint32_t>(lst.size());
  int32_t* dc = dev<int32_t>(n);
  HIP_OK(hipMemcpy(dl, lst.data(), lst.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dc, cnt.data(), n * 4, hipMemcpyHostToDevice));
  const AttnShape ash = {sh.head_gate ? sh.D : 1u, 0, 0};
  AttnKeys k{};
  k.k = c.kc, k.v = c.vc, k.n_keys = K, k.lists = dl, k.counts = dc, k.stride = stride;
  std::string why;
  if (sh.D == Q4X_QK_HEAD_NARROW) {  // the prompt's (q4x_attn_lists takes heads of 128)
    Step s = step(sh, n);
    HIP_OK(q4x_attn_lists(s.q, c.kc, c.vc, n, sh.H, sh.KV, sh.D, dl, dc, stride, s.gate, &ash, s.out, nullptr)
               ? hipSuccess : hipErrorUnknown);
    const std::vector<uint8_t> was = out_of(sh, s);
    compare("LSA prompt lists: 40 tokens", sh, was, core(sh, desc(sh, 0, 0, n, K), s, k, K - n, true, &why), why);
    free_step(s);
  }
  {  // the decode's read: its last token alone
    Step s = step(sh, 1);
    float* scr = dev<float>(q35_attn_decode_scratch(sh.H, Q35_ATTN_DECODE_MAX) + 1);
    HIP_OK(q35_attn_read(s.q, c.kc, c.vc, sh.H, sh.KV, sh.D, &ash, dl + (size_t) (n - 1) * stride,
                         (const uint32_t*) dc + (n - 1), stride, s.gate, scr, s.out, nullptr)
               ? hipSuccess : hipErrorUnknown);
    const std::vector<uint8_t> was = out_of(sh, s);
    AttnKeys k1 = k;
    k1.lists = dl + (size_t) (n - 1) * stride, k1.counts = dc + (n - 1);
    compare("LSA decode read", sh, was, core(sh, desc(sh, 0, 0, 16, K), s, k1, K - 1, false, &why), why);
    (void) hipFree(scr), free_step(s);
  }
  (void) hipFree(dl), (void) hipFree(dc), (void) hipFree(c.kc), (void) hipFree(c.vc);
}

static void refusals() {
  AttnCoreDesc d = desc(kLfm3, 0, 0, 16, 4096);
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, sb, err);
  check(c != nullptr, std::string("apart refusals: osmos1.7b's desc made (") + err + ")");
  attn_core_destroy(c);
  struct Bad { const char* field; void (*bend)(AttnCoreDesc*); };
  const Bad bad[] = {
      {"group", [](AttnCoreDesc* x) { x->heads = 12, x->kv_heads = 4; }},  // 3 a KV head: no decode made
      {"slots", [](AttnCoreDesc* x) { x->window = 512, x->slots = 100; }},  // a ring short of its window
      {"gate", [](AttnCoreDesc* x) { x->gate = ATTN_GATE_NONE; }},
      {"dk", [](AttnCoreDesc* x) { x->dk = x->dv = 192; }},
  };
  for (const Bad& b : bad) {
    AttnCoreDesc x = d;
    b.bend(&x);
    err[0] = 0;
    AttnCore* y = attn_core_create(&x, scratch, sb, err);
    check(!y && std::strstr(err, b.field), std::string("apart refusals: ") + b.field + " refused by name (\"" + err + "\")");
    attn_core_destroy(y);
  }
  (void) hipFree(scratch);
}

int main() {
  for (const Shape* sh : {&kLfm3, &kQ35}) {
    dense_prompt(*sh);
    sliding(*sh);
    decodes(*sh);
    lists(*sh);
  }
  refusals();
  std::printf("attn_core apart: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
