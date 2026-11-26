// the attention core on the drafters' blocks (attn_core.h's ATTN_BLOCK_DRAFT; docs/attention-core.md C3): a block's
// queries against the ring's context and the block's own keys, not causal - each byte for byte its family's kernel as
// its caller called it: Inkling's MTP (an f16 ring: ink2_mtp_attn - 48 heads over 8, a block of 8, one span and many,
// a window over a ring that wraps) and Qwen's DFlash (an F32 ring: q4x_df_attn - 32 heads over 8, a block of 16, its
// window either way); and what it refuses
#include "attn_core_test.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/q4x_ops.h"
}

static const uint32_t kD = 128;

struct Shape {
  const char* name;
  uint32_t H, KH, nq;
  bool f16;  // the ring's rows: f16 (Inkling's MTP) or F32 (Qwen's DFlash)
};
static const Shape kMtp = {"MTP", 48, 8, 8, true}, kDf = {"DFlash", 32, 8, 16, false};

struct Case {
  int32_t pos0, from;
  uint32_t ring, window;
};

static AttnCoreDesc desc(const Shape& sh, const Case& c) {
  AttnCoreDesc d{};
  d.heads = sh.H, d.kv_heads = sh.KH, d.dk = d.dv = kD, d.k_rows = d.v_rows = sh.f16 ? ATTN_ROWS_F16 : ATTN_ROWS_F32;
  d.window = c.window, d.slots = c.ring, d.scaling = 1.0f / std::sqrt((float) kD), d.term = ATTN_TERM_NONE;
  d.sink = ATTN_SINK_NONE, d.gate = ATTN_GATE_NONE, d.block = ATTN_BLOCK_DRAFT, d.max_tokens = sh.nq;
  d.max_keys = c.ring + sh.nq;
  return d;
}

// the case's out's bytes, through the core; empty when refused
static std::vector<uint8_t> run(const Shape& sh, const Case& c, std::string* why) {
  const uint32_t row = sh.KH * kD;
  float *q = up(rnd((size_t) sh.nq * sh.H * kD, 1.0f)), *kb = up(rnd((size_t) sh.nq * row, 1.0f)),
        *vb = up(rnd((size_t) sh.nq * row, 1.0f)), *out = dev<float>((size_t) sh.nq * sh.H * kD);
  const std::vector<float> kr = rnd((size_t) c.ring * row, 1.0f), vr = rnd((size_t) c.ring * row, 1.0f);
  void *dkr, *dvr;
  if (sh.f16) {
    std::vector<_Float16> k16(kr.size()), v16(vr.size());
    for (size_t i = 0; i < kr.size(); i++) k16[i] = (_Float16) kr[i], v16[i] = (_Float16) vr[i];
    dkr = dev<uint16_t>(k16.size()), dvr = dev<uint16_t>(v16.size());
    HIP_OK(hipMemcpy(dkr, k16.data(), k16.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dvr, v16.data(), v16.size() * 2, hipMemcpyHostToDevice));
  } else {
    dkr = up(kr), dvr = up(vr);
  }
  const size_t ob = (size_t) sh.nq * sh.H * kD * 4;
  HIP_OK(hipMemset(out, 0x7f, ob));
  std::vector<uint8_t> o;
  const AttnCoreDesc d = desc(sh, c);
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = sb ? dev<uint8_t>(sb) : nullptr;
  char err[OSMOS_ERR] = "";
  AttnCore* core = attn_core_create(&d, scratch, sb, err);
  if (!core)
    *why = err;
  else {
    AttnStep st{};
    st.q = q, st.n_tok = sh.nq, st.pos0 = (uint32_t) c.pos0, st.out = out;
    st.keys.k = dkr, st.keys.v = dvr, st.keys.key0 = (uint32_t) c.from, st.keys.own_k = kb, st.keys.own_v = vb;
    if (attn_core_run(core, &st, nullptr)) o = down(out, ob); else *why = "attn_core_run refused the step";
  }
  attn_core_destroy(core);
  (void) hipFree(scratch);
  (void) hipFree(q), (void) hipFree(kb), (void) hipFree(vb), (void) hipFree(out), (void) hipFree(dkr), (void) hipFree(dvr);
  return o;
}

// each case's output as its family's kernel gave it (before the core took its code: FNV-1a of the bytes, the cases'
// order below, 3 runs the same)
static const uint64_t kWas[] = {
    0x432d279701b95d99ull, // MTP: 700 keys back, window 0
    0x8ef5a68b4213847aull, // MTP: 19963 keys back, window 0
    0x0789f47e7068970aull, // MTP: 1040 keys back, window 1031
    0x12cb5ae2c1b8c655ull, // DFlash: 300 keys back, window 4096
    0x17cde246af1b382dull, // DFlash: 2048 keys back, window 4096
    0xadac2199e2e79a70ull, // DFlash: 2000 keys back, window 700
};
static size_t g_case = 0;
static void compare(const std::string& what, const std::vector<uint8_t>& got, const std::string& why) {
  const size_t i = g_case++;
  check(!got.empty() && i < sizeof kWas / sizeof *kWas && fnv(got) == kWas[i], what + (got.empty() ? " (" + why + ")" : ""));
}

static void refusals() {
  const AttnCoreDesc d = desc(kMtp, {700, 0, 1024, 0});
  const uint64_t sb = attn_core_scratch(&d);
  void* scratch = dev<uint8_t>(sb);
  char err[OSMOS_ERR] = "";
  AttnCore* c = attn_core_create(&d, scratch, sb, err);
  check(c != nullptr, std::string("draft refusals: the MTP's desc made (") + err + ")");
  attn_core_destroy(c);
  struct Bad { const char* field; void (*bend)(AttnCoreDesc*); };
  const Bad bad[] = {
      {"dk", [](AttnCoreDesc* x) { x->dk = x->dv = 256; }},
      {"kv_heads", [](AttnCoreDesc* x) { x->heads = 50; }},
      {"max_tokens", [](AttnCoreDesc* x) { x->max_tokens = 9; }},  // 9 x 6 rows past the MTP's 48
      {"k_rows", [](AttnCoreDesc* x) { x->k_rows = x->v_rows = ATTN_ROWS_Q8_0; }},
      {"sink", [](AttnCoreDesc* x) { x->sink = ATTN_SINK_HEAD; }},
      {"slots", [](AttnCoreDesc* x) { x->slots = 0; }},
      {"window", [](AttnCoreDesc* x) { x->k_rows = x->v_rows = ATTN_ROWS_F32, x->window = 0; }},  // DFlash's: at least the block
  };
  for (const Bad& b : bad) {
    AttnCoreDesc x = d;
    b.bend(&x);
    err[0] = 0;
    AttnCore* y = attn_core_create(&x, scratch, sb, err);
    check(!y && std::strstr(err, b.field), std::string("draft refusals: ") + b.field + " refused by name (\"" + err + "\")");
    attn_core_destroy(y);
  }
  (void) hipFree(scratch);
}

int main() {
  // the MTP's (its test's): every context position in one span and past many, a local layer's window over a wrapping ring
  for (const Case& c : {Case{700, 0, 1024, 0}, Case{20000, 37, 20480, 0}, Case{3000, 1960, 1040, 1031}}) {
    std::string why;
    compare(std::string("MTP: ") + std::to_string(c.pos0 - c.from) + " keys back, window " + std::to_string(c.window),
            run(kMtp, c, &why), why);
  }
  // DFlash's: a short context, a ring that wraps (the context its whole), a window shorter than the context
  for (const Case& c : {Case{300, 0, 2048, 4096}, Case{9000, 9000 - 2048, 2048, 4096}, Case{5000, 3000, 2048, 700}}) {
    std::string why;
    compare(std::string("DFlash: ") + std::to_string(c.pos0 - c.from) + " keys back, window " + std::to_string(c.window),
            run(kDf, c, &why), why);
  }
  refusals();
  std::printf("attn_core draft: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
