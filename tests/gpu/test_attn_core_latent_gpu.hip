// the attention core on DeepSeek's latent (attn_core.h's FP8 rows; docs/attention-core.md C3): one 512-wide row a key,
// its value the same row, for every head - each path byte for byte ops_sparse_attn's as its callers called it: the
// window's FP8 ring and the selected FP4 compressed rows, both unpacked to F32 (a prompt's step), the window alone, a
// floor (the decoder's first position), large and small tiles, 128 heads, and the drafter's block (its own rows F32,
// every one to each row); and what it refuses
#include "attn_core_test.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/ops.h"
}

static const uint32_t kHd = 512, kWindow = 128, kSlots = 256, kRows = 2048, kTop = 512;

// the window's ring (FP8, and unpacked) and the compressed rows (FP4 with an E4M3 scale, and unpacked) as a sequence
// keeps them
struct Keys {
  uint8_t *ring, *rows;
  float *ring_f, *rows_f;
};
static Keys keys() {
  Keys k{dev<uint8_t>(kSlots * kv_row_bytes(KV_FP8, kHd)), dev<uint8_t>(kRows * kv_row_bytes(KV_FP4_E4M3, kHd)),
         up(rnd((size_t) kSlots * kHd, 1.0f)), up(rnd((size_t) kRows * kHd, 1.0f))};
  HIP_OK(ops_fp8_round(k.ring_f, (uint64_t) kSlots * kHd, k.ring, kHd, nullptr) &&
                 ops_fp4_round(k.rows_f, (uint64_t) kRows * kHd, KV_FP4_E4M3_BLOCK, true, k.rows, kHd, nullptr)
             ? hipSuccess : hipErrorUnknown);
  return k;
}

// a step: n tokens at positions p0 .., each its window and up to kTop of the rows its position covers (any of them,
// ascending), a head's sink
struct Step {
  uint32_t n, H;
  float *q, *sink, *out;
  int32_t *pos, *sel, *nsel;
};
static int32_t* up_i(const std::vector<int32_t>& v) {
  int32_t* d = dev<int32_t>(v.size());
  HIP_OK(hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice));
  return d;
}
static Step step(uint32_t H, uint32_t p0, uint32_t n, uint32_t top) {
  std::vector<int32_t> p(n), sel((size_t) n * OPS_MAX_SELECT, 0), ns(n, 0);
  for (uint32_t t = 0; t < n; t++) {
    p[t] = (int32_t) (p0 + t);
    uint32_t c = 0;
    for (uint32_t r = 0; r < kRows && c < top; r++)
      if (uni() < 0.3f) sel[(size_t) t * OPS_MAX_SELECT + c++] = (int32_t) r;
    ns[t] = (int32_t) c;
  }
  return {n, H, up(rnd((size_t) n * H * kHd, 0.05f)), up(rnd(H, 1.0f)), dev<float>((size_t) n * H * kHd), up_i(p), up_i(sel),
          up_i(ns)};
}
static void free_step(Step& s) {
  (void) hipFree(s.q), (void) hipFree(s.sink), (void) hipFree(s.out), (void) hipFree(s.pos), (void) hipFree(s.sel);
  (void) hipFree(s.nsel);
}

// a call's keys: the window's ring (rfmt: KV_FP8 or its unpacked F32), the selected rows (fmt: FP4, F32, or none)
struct Call {
  const void *ring, *rows;
  uint32_t rfmt, fmt, slots, window, top;
  int32_t floor;
};

static AttnCoreDesc desc(uint32_t H, const Call& c, uint32_t max_tokens) {
  AttnCoreDesc d{};
  d.heads = H, d.kv_heads = 1, d.dk = d.dv = kHd, d.k_rows = d.v_rows = ATTN_ROWS_FP8;
  d.s_rows = c.rows ? (c.fmt == KV_F32 && c.slots == c.window ? ATTN_ROWS_F32 : ATTN_ROWS_FP4) : ATTN_ROWS_F32;
  d.max_sel = c.rows ? c.top : 0, d.window = c.window, d.slots = c.slots, d.scaling = 1.0f / std::sqrt((float) kHd);
  d.term = ATTN_TERM_NONE, d.sink = ATTN_SINK_HEAD, d.gate = ATTN_GATE_NONE, d.block = ATTN_BLOCK_HEADS, d.unit = 0;
  d.max_tokens = max_tokens, d.max_keys = c.window + d.max_sel;
  return d;
}

// the step's out's bytes, through the core; empty when refused
static std::vector<uint8_t> run(const Step& s, const Call& c, std::string* why, bool drafter = false) {
  const size_t ob = (size_t) s.n * s.H * kHd * 4;
  HIP_OK(hipMemset(s.out, 0x7f, ob));
  AttnCoreDesc d = desc(s.H, c, s.n);
  if (drafter) d.s_rows = ATTN_ROWS_F32;
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* core = attn_core_create(&d, scratch, sb, err);
  std::vector<uint8_t> o;
  if (!core)
    *why = err;
  else {
    AttnStep st{};
    st.q = s.q, st.n_tok = s.n, st.pos = s.pos, st.sink = s.sink, st.out = s.out;
    st.keys.k = st.keys.v = c.ring, st.keys.k_wide = c.rfmt == KV_F32, st.keys.floor = c.floor;
    if (c.rows)
      st.keys.rows = c.rows, st.keys.rows_wide = c.fmt == KV_F32 && !drafter, st.keys.sel = s.sel, st.keys.npick = s.nsel,
      st.keys.stride = OPS_MAX_SELECT;
    if (attn_core_run(core, &st, nullptr))
      o = down(s.out, ob);
    else
      *why = "attn_core_run refused the step";
  }
  attn_core_destroy(core);
  (void) hipFree(scratch);
  return o;
}

// each path's output as ops_sparse_attn gave it (before the core took its code: FNV-1a of the bytes, the cases'
// order below, 3 runs the same)
static const uint64_t kWas[] = {
    0x87a5b575ae284ca8ull, // 24 tokens: the FP8 ring, the FP4 rows
    0xdd887171fbffdb26ull, // 3 tokens (small tiles): FP8, FP4
    0x17a48eeb166eab47ull, // a decode: FP8, FP4
    0xc7713df660aa44ccull, // 24 tokens: the ring unpacked, FP4
    0xf51f34c63f8a2abaull, // 24 tokens: FP8, the rows unpacked
    0xb8d6a5900f1c8343ull, // 24 tokens: both unpacked (F32 alone)
    0x880f0684492a48c2ull, // 24 tokens: the window alone
    0x6fab403fd84d1deaull, // a decode: the window alone
    0x9481e05c30ed6583ull, // 4 tokens: a floor 60 below
    0x1698711077abe690ull, // 20 tokens: 128 heads
    0xc46757bcee2f0f0cull, // the drafter's block of 8: the ring's window, its own rows
};
static size_t g_case = 0;
static void compare(const std::string& what, const std::vector<uint8_t>& got, const std::string& why) {
  const size_t i = g_case++;
  check(!got.empty() && i < sizeof kWas / sizeof *kWas && fnv(got) == kWas[i], what + (got.empty() ? " (" + why + ")" : ""));
}

static void refusals() {
  const Call c{(const void*) 1, (const void*) 1, KV_FP8, KV_FP4_E4M3, kSlots, kWindow, kTop, 0};
  AttnCoreDesc d = desc(64, c, 16);
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* core = attn_core_create(&d, scratch, sb, err);
  check(core != nullptr, std::string("latent refusals: DeepSeek's desc made (") + err + ")");
  attn_core_destroy(core);
  struct Bad { const char* field; void (*bend)(AttnCoreDesc*); };
  const Bad bad[] = {
      {"heads", [](AttnCoreDesc* x) { x->heads = 96; }},           // not whole blocks of 64
      {"kv_heads", [](AttnCoreDesc* x) { x->kv_heads = 2; }},
      {"dv", [](AttnCoreDesc* x) { x->dv = 256; }},                // the value the key's row
      {"dk", [](AttnCoreDesc* x) { x->dk = x->dv = 320; }},        // not whole blocks of 256
      {"window", [](AttnCoreDesc* x) { x->window = 0; }},
      {"slots", [](AttnCoreDesc* x) { x->slots = 100; }},
      {"max_sel", [](AttnCoreDesc* x) { x->max_sel = OPS_MAX_SELECT + 1; }},
      {"s_rows", [](AttnCoreDesc* x) { x->s_rows = ATTN_ROWS_Q8_0; }},
      {"sink", [](AttnCoreDesc* x) { x->sink = ATTN_SINK_NONE; }},
      {"unit", [](AttnCoreDesc* x) { x->unit = 256; }},
      {"apart", [](AttnCoreDesc* x) { x->apart = true; }},
  };
  for (const Bad& b : bad) {
    AttnCoreDesc x = d;
    b.bend(&x);
    err[0] = 0;
    AttnCore* y = attn_core_create(&x, scratch, sb, err);
    check(!y && std::strstr(err, b.field), std::string("latent refusals: ") + b.field + " refused by name (\"" + err + "\")");
    attn_core_destroy(y);
  }
  (void) hipFree(scratch);
}

int main() {
  Keys k = keys();
  std::string why;
  const uint32_t p0 = 6000;  // every window full
  struct Case { const char* what; uint32_t H, n; uint32_t rfmt, fmt; bool rows; int32_t floor; };
  const Case cases[] = {
      {"24 tokens: the FP8 ring, the FP4 rows", 64, 24, KV_FP8, KV_FP4_E4M3, true, 0},
      {"3 tokens (small tiles): FP8, FP4", 64, 3, KV_FP8, KV_FP4_E4M3, true, 0},
      {"a decode: FP8, FP4", 64, 1, KV_FP8, KV_FP4_E4M3, true, 0},
      {"24 tokens: the ring unpacked, FP4", 64, 24, KV_F32, KV_FP4_E4M3, true, 0},
      {"24 tokens: FP8, the rows unpacked", 64, 24, KV_FP8, KV_F32, true, 0},
      {"24 tokens: both unpacked (F32 alone)", 64, 24, KV_F32, KV_F32, true, 0},
      {"24 tokens: the window alone", 64, 24, KV_FP8, KV_F32, false, 0},
      {"a decode: the window alone", 64, 1, KV_FP8, KV_F32, false, 0},
      {"4 tokens: a floor 60 below", 64, 4, KV_FP8, KV_FP4_E4M3, true, (int32_t) p0 - 60},
      {"20 tokens: 128 heads", 128, 20, KV_FP8, KV_FP4_E4M3, true, 0},
  };
  for (const Case& c : cases) {
    Step s = step(c.H, p0, c.n, kTop);
    const Call call{c.rfmt == KV_F32 ? (const void*) k.ring_f : (const void*) k.ring,
                    !c.rows ? nullptr : c.fmt == KV_F32 ? (const void*) k.rows_f : (const void*) k.rows, c.rfmt, c.fmt, kSlots,
                    kWindow, kTop, c.floor};
    compare(c.what, run(s, call, &why), why);
    free_step(s);
  }
  {  // the drafter's block: T rows at p0 .., the ring's window before p0 (its slots the window), then the block's own
     // T rows (F32, not kept) every one to each row
    const uint32_t T = 8;
    Step s = step(64, p0, T, 0);
    std::vector<int32_t> pa(T, (int32_t) p0 - 1), sel((size_t) T * OPS_MAX_SELECT, 0), ns(T, (int32_t) T);
    for (uint32_t t = 0; t < T; t++)
      for (uint32_t j = 0; j < T; j++) sel[(size_t) t * OPS_MAX_SELECT + j] = (int32_t) j;
    (void) hipFree(s.pos), (void) hipFree(s.sel), (void) hipFree(s.nsel);
    s.pos = up_i(pa), s.sel = up_i(sel), s.nsel = up_i(ns);
    float* own = up(rnd((size_t) T * kHd, 1.0f));
    const Call call{k.ring, own, KV_FP8, KV_F32, kWindow, kWindow, T, 0};
    compare("the drafter's block of 8: the ring's window, its own rows", run(s, call, &why, true), why);
    (void) hipFree(own), free_step(s);
  }
  refusals();
  std::printf("attn_core latent: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
