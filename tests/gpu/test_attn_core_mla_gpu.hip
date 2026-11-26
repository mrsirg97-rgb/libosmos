// the attention core on Hy4's MLA (attn_core.h's MLA rows; docs/attention-core.md C3): each of its paths byte for
// byte hy4_mla_attn's as its callers called it - a dense prompt (mla_mm's read), dense decodes and verifies (the
// splits), a sliding window over a ring, the indexer's selected groups (a prompt's and a decode's) - in F32 and Q8
// latent rows, at Hy4-flash's shape (64 heads over one latent of 512 and a rope of 64) and at 32, 16 and 8 heads (the
// read's block sizes and its fallback); and what it refuses
#include "attn_core_test.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/hy4_ops.h"
}

static const uint32_t kLat = 512, kRope = 64, kRow = kLat + kRope;

// a cache of K positions' rows (hy4_mla_kv's from random inputs), at a row a position or in a ring of slots
struct Cache {
  void* rows;
  uint32_t slots;
  bool q8;
};
static int32_t* dpos_of(const std::vector<int32_t>& p) {
  int32_t* d = dev<int32_t>(p.size());
  HIP_OK(hipMemcpy(d, p.data(), p.size() * 4, hipMemcpyHostToDevice));
  return d;
}
static std::vector<float> freqs() {
  std::vector<float> f(kRope / 2);
  for (uint32_t i = 0; i < kRope / 2; i++) f[i] = std::pow(10000.0f, -2.0f * (float) i / (float) kRope);
  return f;
}
static Cache cache(uint32_t K, uint32_t slots, bool q8) {
  const uint32_t rows = slots ? slots : K;
  Cache c{dev<uint8_t>((size_t) rows * hy4_mla_row_bytes(kLat, kRope, q8)), slots, q8};
  float *kva = up(rnd((size_t) K * kRow, 1.0f)), *norm = up(rnd(kLat, 1.0f)), *fr = up(freqs());
  std::vector<int32_t> p(K);
  for (uint32_t i = 0; i < K; i++) p[i] = (int32_t) i;
  int32_t* dp = dpos_of(p);
  // a ring: the last slots positions' rows (an earlier one's overwritten as the engine's)
  const uint32_t from = slots && K > slots ? K - slots : 0;
  HIP_OK(hy4_mla_kv(kva + (size_t) from * kRow, norm, kLat, kRope, 1e-5f, fr, dp + from, K - from, c.rows, slots, q8, nullptr)
             ? hipSuccess : hipErrorUnknown);
  (void) hipFree(kva), (void) hipFree(norm), (void) hipFree(fr), (void) hipFree(dp);
  return c;
}

static AttnCoreDesc desc(uint32_t H, bool q8, uint32_t window, uint32_t slots, uint32_t pool) {
  AttnCoreDesc d{};
  d.heads = H, d.kv_heads = 1, d.dk = kRow, d.dv = kLat, d.k_rows = d.v_rows = q8 ? ATTN_ROWS_MLA_Q8 : ATTN_ROWS_MLA_F32;
  d.window = window, d.slots = slots, d.scaling = 0.0625f, d.term = ATTN_TERM_NONE, d.sink = ATTN_SINK_HEAD;
  d.gate = ATTN_GATE_NONE, d.block = ATTN_BLOCK_HEADS, d.ratio = pool, d.unit = 32, d.apart = true;
  d.max_tokens = 512, d.max_keys = HY4_MLA_MAX_KEYS;
  return d;
}

// a step's queries at positions pos, the groups each reads (sel / n_sel, cap a token's: none for positions)
struct Step {
  uint32_t n;
  float *q, *sink, *out;
  int32_t *pos, *sel, *nsel;
  uint32_t cap;
};
static Step step(uint32_t H, uint32_t p0, uint32_t n, uint32_t pool, uint32_t cap) {
  std::vector<int32_t> p(n);
  for (uint32_t t = 0; t < n; t++) p[t] = (int32_t) (p0 + t);
  Step s{n, up(rnd((size_t) n * H * kRow, 0.3f)), up(rnd(H, 1.0f)), dev<float>((size_t) n * H * kLat), dpos_of(p), nullptr,
         nullptr, cap};
  if (pool) {  // a token's groups: some of the earlier ones, ascending, then its own (its last, to itself)
    std::vector<int32_t> sel((size_t) n * cap, 0), ns(n);
    for (uint32_t t = 0; t < n; t++) {
      const uint32_t own = (p0 + t) / pool;
      uint32_t c = 0;
      for (uint32_t g = 0; g < own && c + 1 < cap; g++)
        if (uni() < 0.4f) sel[(size_t) t * cap + c++] = (int32_t) g;
      sel[(size_t) t * cap + c++] = (int32_t) own;
      ns[t] = (int32_t) c;
    }
    s.sel = dpos_of(sel), s.nsel = dpos_of(ns);
  }
  return s;
}
static void free_step(Step& s) {
  (void) hipFree(s.q), (void) hipFree(s.sink), (void) hipFree(s.out), (void) hipFree(s.pos);
  (void) hipFree(s.sel), (void) hipFree(s.nsel);
}

// the step's out's bytes, through the core; empty when refused
static std::vector<uint8_t> run(const AttnCoreDesc& d, const Cache& c, const Step& s, bool read, std::string* why) {
  const size_t ob = (size_t) s.n * d.heads * kLat * 4;
  HIP_OK(hipMemset(s.out, 0x7f, ob));
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* core = attn_core_create(&d, scratch, sb, err);
  std::vector<uint8_t> o;
  if (!core)
    *why = err;
  else {
    AttnStep st{};
    st.q = s.q, st.n_tok = s.n, st.pos = s.pos, st.sink = s.sink, st.out = s.out, st.read = read;
    st.keys.k = st.keys.v = c.rows, st.keys.sel = s.sel, st.keys.npick = s.nsel, st.keys.stride = s.cap;
    if (attn_core_run(core, &st, nullptr))
      o = down(s.out, ob);
    else
      *why = "attn_core_run refused the step";
  }
  attn_core_destroy(core);
  (void) hipFree(scratch);
  return o;
}

// each path's output as hy4_mla_attn gave it (before the core took its code: FNV-1a of the bytes, the cases' order
// below, 3 runs the same)
static const uint64_t kWas[] = {
    0x580b26e7a8654ed0ull, // 64 heads F32: dense prompt of 300 from 0
    0x453319ba5a6b70f3ull, // 64 heads F32: dense prompt of 200 from 2,400
    0xdd3fcafb623881caull, // 64 heads F32: dense decode at 2,999
    0x4f879227c387c634ull, // 64 heads F32: dense verify of 20 from 1,500
    0x1eb94e92c39d47edull, // 64 heads F32: sliding prompt of 200, a ring of 1,024
    0xb063f301cfbd85a2ull, // 64 heads F32: sliding decode of 3
    0xdba06cdd4b88bb91ull, // 64 heads F32: selected groups: a prompt of 40
    0xd41201a3a02f2cc2ull, // 64 heads F32: selected groups: a decode
    0x55567cf249c7e8ebull, // 64 heads Q8: dense prompt of 300 from 0
    0xfd990ef55ebe07d0ull, // 64 heads Q8: dense prompt of 200 from 2,400
    0x82797806f1d6dbf4ull, // 64 heads Q8: dense decode at 2,999
    0x98e1d697bc765d17ull, // 64 heads Q8: dense verify of 20 from 1,500
    0x14fe3c6b945cabb9ull, // 64 heads Q8: sliding prompt of 200, a ring of 1,024
    0x06a9c6882723e555ull, // 64 heads Q8: sliding decode of 3
    0xb5d819d1b4989130ull, // 64 heads Q8: selected groups: a prompt of 40
    0xcf914e7f0e9f2de8ull, // 64 heads Q8: selected groups: a decode
    0x39182fba0c6fc741ull, // 32 heads Q8: dense prompt of 300 from 0
    0xc2ebaa08e9d9e63dull, // 16 heads Q8: dense prompt of 300 from 0
    0x525c670cc9476047ull, // 8 heads Q8: dense prompt of 300 from 0
};
static size_t g_case = 0;
static void compare(const std::string& what, const std::vector<uint8_t>& got, const std::string& why) {
  const size_t i = g_case++;
  check(!got.empty() && i < sizeof kWas / sizeof *kWas && fnv(got) == kWas[i],
        what + (got.empty() ? " (" + why + ")" : ""));
}

static std::string name(uint32_t H, bool q8, const char* what) {
  return std::to_string(H) + " heads " + (q8 ? "Q8" : "F32") + ": " + what;
}

// a dense layer's: a prompt (a read, mla_mm's) at 0 and at depth, a decode and a verify of 20 (the splits, past a
// launch's 16 tokens)
static void dense(uint32_t H, bool q8) {
  const uint32_t K = 3000;
  Cache c = cache(K, 0, q8);
  std::string why;
  struct Case { uint32_t p0, n; bool read; const char* what; };
  for (const Case& k : {Case{0, 300, true, "dense prompt of 300 from 0"}, Case{2400, 200, true, "dense prompt of 200 from 2,400"},
                        Case{K - 1, 1, false, "dense decode at 2,999"}, Case{1500, 20, false, "dense verify of 20 from 1,500"}}) {
    Step s = step(H, k.p0, k.n, 0, 0);
    compare(name(H, q8, k.what), run(desc(H, q8, 0, 0, 0), c, s, k.read, &why), why);
    free_step(s);
  }
  (void) hipFree(c.rows);
}

// a sliding layer's: a window of 512 over a ring of 1,024 rows, a prompt's step (a read) and a decode's of 3
static void sliding(uint32_t H, bool q8) {
  const uint32_t W = 512, slots = 1024, p0 = 1400;
  std::string why;
  for (auto [n, read] : {std::pair<uint32_t, bool>{200, true}, {3, false}}) {
    Cache c = cache(p0 + n, slots, q8);
    Step s = step(H, p0, n, 0, 0);
    compare(name(H, q8, read ? "sliding prompt of 200, a ring of 1,024" : "sliding decode of 3"),
            run(desc(H, q8, W, slots, 0), c, s, read, &why), why);
    free_step(s), (void) hipFree(c.rows);
  }
}

// a sparse layer's: the indexer's groups of 4 (at most 512 a token), a prompt's 40 (a read) and a decode
static void selected(uint32_t H, bool q8) {
  const uint32_t K = 3000, pool = 4, cap = 512;
  Cache c = cache(K, 0, q8);
  std::string why;
  for (auto [n, read] : {std::pair<uint32_t, bool>{40, true}, {1, false}}) {
    Step s = step(H, K - n, n, pool, cap);
    compare(name(H, q8, read ? "selected groups: a prompt of 40" : "selected groups: a decode"),
            run(desc(H, q8, 0, 0, pool), c, s, read, &why), why);
    free_step(s);
  }
  (void) hipFree(c.rows);
}

static void refusals() {
  AttnCoreDesc d = desc(64, true, 512, 1024, 0);
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, sb, err);
  check(c != nullptr, std::string("MLA refusals: Hy4's desc made (") + err + ")");
  attn_core_destroy(c);
  struct Bad { const char* field; void (*bend)(AttnCoreDesc*); };
  const Bad bad[] = {
      {"kv_heads", [](AttnCoreDesc* x) { x->kv_heads = 2; }},  // MLA's one latent for every head
      {"dv", [](AttnCoreDesc* x) { x->dv = 520; }},           // the latent past its most
      {"dk", [](AttnCoreDesc* x) { x->dk = x->dv + 256; }},   // a rope past its most
      {"slots", [](AttnCoreDesc* x) { x->slots = 100; }},     // a ring short of its window
      {"sink", [](AttnCoreDesc* x) { x->sink = ATTN_SINK_TOKEN; }},
      {"gate", [](AttnCoreDesc* x) { x->gate = ATTN_GATE_VALUE; }},
      {"unit", [](AttnCoreDesc* x) { x->unit = 64; }},
      {"ratio", [](AttnCoreDesc* x) { x->ratio = 4; }},       // groups under a window
      {"max_keys", [](AttnCoreDesc* x) { x->max_keys = HY4_MLA_MAX_KEYS + 1; }},
  };
  for (const Bad& b : bad) {
    AttnCoreDesc x = d;
    b.bend(&x);
    err[0] = 0;
    AttnCore* y = attn_core_create(&x, scratch, sb, err);
    check(!y && std::strstr(err, b.field), std::string("MLA refusals: ") + b.field + " refused by name (\"" + err + "\")");
    attn_core_destroy(y);
  }
  (void) hipFree(scratch);
}

int main() {
  for (bool q8 : {false, true}) {
    dense(64, q8);
    sliding(64, q8);
    selected(64, q8);
  }
  for (uint32_t H : {32u, 16u, 8u}) {  // the read's blocks of 2 and 1 head groups, and its fallback (the splits)
    Cache c = cache(600, 0, true);
    Step s = step(H, 0, 300, 0, 0);
    std::string why;
    compare(name(H, true, "dense prompt of 300 from 0"), run(desc(H, true, 0, 0, 0), c, s, true, &why), why);
    free_step(s), (void) hipFree(c.rows);
  }
  refusals();
  std::printf("attn_core MLA: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
