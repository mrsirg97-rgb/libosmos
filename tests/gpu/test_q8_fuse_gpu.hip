// q8_gate_up against what it stands for - q8_group_tok on the gate, on the up, then ops_swiglu_lane - bit for bit, on
// ds1017's or UD-IQ1_M's experts (three layers of their types; experts 7 and 300 a launch), 1 to 8 tokens each at a
// launch of that width and of 8, at a clamp that bites and one that does not; the rows past a matrix's count left
// as they were; the rows read by token (xin, win) the gathered rows' bits; refusals
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/gpu/q8.h"
#include "substrate/model/gguf.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kTok = MV_MAX_TOKENS, kExperts = 2, kBlock = 32;
constexpr uint32_t kGuard = 0xFFFFFFFFu;  // a NaN's bits: a row the launch must not write

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

// expert e's slice of a [cols][rows][experts] tensor
static std::vector<uint8_t> slice(const Gguf* g, const GgufTensor* t, int e) {
  const uint64_t bytes = t->bytes / t->dims[2];
  std::vector<uint8_t> w(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, w.data(), bytes, (off_t) (t->offset + (uint64_t) e * bytes)) == (ssize_t) bytes);
  close(fd);
  return w;
}

// the lane's rows by token (xin, win): kTokens tokens' rows rounded once, the 16 rows reading theirs through xin and
// their weights through win - the bits of the same rows gathered a copy a pair, each copy rounded, the weights gathered
// (the lane's way before). the table entries past each count an index far past any buffer: read, the launch faults
static void by_token(uint32_t type, const void* const* gates, const void* const* ups, int32_t* dcnt, const int32_t* dxrow,
                     const int32_t* xrow, uint32_t rows, uint32_t cols, uint32_t nrow, const std::vector<float>& sc,
                     float* dx, int8_t* dq, float* dd, float* dga, float* dup, float* dh, float* dref) {
  constexpr uint32_t kTokens = 5;
  constexpr int32_t kPoison = 0x40000000;
  const float lim = 0.75f;  // (a clamp that bites)
  std::vector<float> tx((size_t) kTokens * cols), gx((size_t) nrow * cols), gs(nrow);
  for (float& v : tx) v = uniform() * 3.0f;
  std::vector<int32_t> xin(nrow), win(nrow);
  for (uint32_t r = 0; r < nrow; r++) {
    xin[r] = (int32_t) ((r * 3 + 1) % kTokens), win[r] = (int32_t) ((r * 5 + 3) % nrow);
    std::memcpy(&gx[(size_t) r * cols], &tx[(size_t) xin[r] * cols], cols * 4);
    gs[r] = sc[win[r]];
  }
  float *dtx, *dtd, *dgs, *dws;
  int8_t* dtq;
  int32_t *dxin, *dwin;
  CHECK(hipMalloc(&dtx, tx.size() * 4) == hipSuccess && hipMalloc(&dtq, tx.size()) == hipSuccess &&
        hipMalloc(&dtd, (size_t) kTokens * (cols / kBlock) * 4) == hipSuccess && hipMalloc(&dgs, nrow * 4) == hipSuccess &&
        hipMalloc(&dws, nrow * 4) == hipSuccess && hipMalloc(&dxin, nrow * 4) == hipSuccess &&
        hipMalloc(&dwin, nrow * 4) == hipSuccess);
  CHECK(hipMemcpy(dtx, tx.data(), tx.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dx, gx.data(), gx.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dgs, gs.data(), nrow * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dws, sc.data(), nrow * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q8_round(dtx, kTokens, cols, cols, dtq, dtd, nullptr) && q8_round(dx, nrow, cols, cols, dq, dd, nullptr));
  const size_t out = (size_t) nrow * rows;
  std::vector<float> ref(out), got(out), refg(out), gotg(out);
  for (uint32_t n = 1; n <= kTok; n++) {
    const int32_t cnt[kExperts] = {(int32_t) n, (int32_t) (kTok + 1 - n)};
    std::vector<int32_t> xp = xin, wp = win;
    for (uint32_t e = 0; e < kExperts; e++)
      for (uint32_t t = (uint32_t) cnt[e]; t < kTok; t++) xp[xrow[e] + t] = wp[xrow[e] + t] = kPoison;
    CHECK(hipMemcpy(dcnt, cnt, sizeof cnt, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(dxin, xp.data(), nrow * 4, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(dwin, wp.data(), nrow * 4, hipMemcpyHostToDevice) == hipSuccess);
    for (float* p : {dref, dh, dga, dup}) CHECK(hipMemset(p, 0xFF, out * 4) == hipSuccess);
    CHECK(q8_gate_up(type, gates, ups, dcnt, dxrow, nullptr, nullptr, kExperts, kTok, rows, cols, dq, dd, dgs, lim,
                     dref, rows, nullptr) &&
          q8_group_tok(type, gates, dcnt, dxrow, nullptr, kExperts, kTok, rows, cols, dq, dd, dga, rows, nullptr));
    CHECK(q8_gate_up(type, gates, ups, dcnt, dxrow, dxin, dwin, kExperts, kTok, rows, cols, dtq, dtd, dws, lim, dh,
                     rows, nullptr) &&
          q8_group_tok(type, gates, dcnt, dxrow, dxin, kExperts, kTok, rows, cols, dtq, dtd, dup, rows, nullptr) &&
          hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(ref.data(), dref, out * 4, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(got.data(), dh, out * 4, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(refg.data(), dga, out * 4, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(gotg.data(), dup, out * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(got.data(), ref.data(), out * 4) == 0 && std::memcmp(gotg.data(), refg.data(), out * 4) == 0);
  }
  for (void* p : {(void*) dtx, (void*) dtq, (void*) dtd, (void*) dgs, (void*) dws, (void*) dxin, (void*) dwin})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("  %-26s by token: %u tokens rounded once, 16 rows through xin / win - the gathered rows' bits\n", "",
              kTokens);
}

static void layer(const Gguf* g, int l) {
  char gn[64], un[64];
  std::snprintf(gn, sizeof gn, "blk.%d.ffn_gate_exps.weight", l);
  std::snprintf(un, sizeof un, "blk.%d.ffn_up_exps.weight", l);
  const GgufTensor *tg = gguf_tensor(g, gn), *tu = gguf_tensor(g, un);
  CHECK(tg && tu && tg->type == tu->type && tg->dims[0] == tu->dims[0] && tg->dims[1] == tu->dims[1]);
  const uint32_t cols = (uint32_t) tg->dims[0], rows = (uint32_t) tg->dims[1], nrow = kExperts * kTok;
  const uint64_t bytes = tg->bytes / tg->dims[2];
  uint8_t *dw, **dp;
  float *dx, *dd, *ds, *dga, *dup, *dh, *dref;
  int8_t* dq;
  int32_t *dcnt, *dxrow;
  const size_t out = (size_t) nrow * rows * 4;
  CHECK(hipMalloc(&dw, 2 * kExperts * bytes) == hipSuccess &&
        hipMalloc(&dp, 2 * kExperts * sizeof(void*)) == hipSuccess);
  CHECK(hipMalloc(&dx, (size_t) nrow * cols * 4) == hipSuccess && hipMalloc(&dq, (size_t) nrow * cols) == hipSuccess &&
        hipMalloc(&dd, (size_t) nrow * (cols / kBlock) * 4) == hipSuccess && hipMalloc(&ds, nrow * 4) == hipSuccess);
  CHECK(hipMalloc(&dga, out) == hipSuccess && hipMalloc(&dup, out) == hipSuccess && hipMalloc(&dh, out) == hipSuccess &&
        hipMalloc(&dref, out) == hipSuccess && hipMalloc(&dcnt, kExperts * 4) == hipSuccess &&
        hipMalloc(&dxrow, kExperts * 4) == hipSuccess);
  const uint8_t* hp[2 * kExperts];  // the gates, then the ups
  for (uint32_t e = 0; e < kExperts; e++) {
    const std::vector<uint8_t> wg = slice(g, tg, e ? 300 : 7), wu = slice(g, tu, e ? 300 : 7);
    CHECK(hipMemcpy(dw + e * bytes, wg.data(), bytes, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(dw + (kExperts + e) * bytes, wu.data(), bytes, hipMemcpyHostToDevice) == hipSuccess);
    hp[e] = dw + e * bytes, hp[kExperts + e] = dw + (kExperts + e) * bytes;
  }
  std::vector<float> x((size_t) nrow * cols), sc(nrow);
  for (float& v : x) v = uniform() * 3.0f;
  for (float& v : sc) v = 0.25f + uniform() * uniform();  // a routed expert's weight
  const int32_t xrow[kExperts] = {0, (int32_t) kTok};
  CHECK(hipMemcpy(dp, hp, sizeof hp, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(ds, sc.data(), sc.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dxrow, xrow, sizeof xrow, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q8_round(dx, nrow, cols, cols, dq, dd, nullptr));
  const void* const* gates = (const void* const*) dp;
  const void* const* ups = (const void* const*) dp + kExperts;
  std::vector<float> ref((size_t) nrow * rows), got(ref.size());
  for (float lim : {0.75f, 1e30f}) {  // (the activations' |gate| and |up| reach a few: the first clamps)
    uint32_t clamped = 0;
    for (uint32_t n = 1; n <= kTok; n++) {
      const int32_t cnt[kExperts] = {(int32_t) n, (int32_t) (kTok + 1 - n)};
      CHECK(hipMemcpy(dcnt, cnt, sizeof cnt, hipMemcpyHostToDevice) == hipSuccess);
      for (uint32_t width : {std::max(n, kTok + 1 - n), kTok}) {
        CHECK(hipMemset(dref, 0xFF, ref.size() * 4) == hipSuccess && hipMemset(dh, 0xFF, got.size() * 4) == hipSuccess);
        CHECK(q8_group_tok(tg->type, gates, dcnt, dxrow, nullptr, kExperts, width, rows, cols, dq, dd, dga, rows, nullptr) &&
              q8_group_tok(tu->type, ups, dcnt, dxrow, nullptr, kExperts, width, rows, cols, dq, dd, dup, rows, nullptr) &&
              ops_swiglu_lane(dga, dup, rows, lim, ds, nrow, dref, nullptr));
        CHECK(q8_gate_up(tg->type, gates, ups, dcnt, dxrow, nullptr, nullptr, kExperts, width, rows, cols, dq, dd, ds, lim, dh, rows,
                         nullptr) &&
              hipDeviceSynchronize() == hipSuccess);
        CHECK(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(got.data(), dh, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
        for (uint32_t e = 0; e < kExperts; e++) {
          const size_t at = (size_t) xrow[e] * rows, live = (size_t) cnt[e] * rows;
          CHECK(std::memcmp(&got[at], &ref[at], live * 4) == 0);
          for (size_t i = at + live; i < at + (size_t) kTok * rows; i++) {  // past the count: left as it was
            uint32_t b;
            std::memcpy(&b, &got[i], 4);
            CHECK(b == kGuard);
          }
        }
      }
    }
    std::vector<float> ga((size_t) kTok * rows);  // how many of the gates the clamp took (the first expert, 8 rows)
    CHECK(hipMemcpy(ga.data(), dga, ga.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (float v : ga) clamped += v > lim;
    std::printf("  %-26s type %2u %4u x %4u, clamp %g (%u gates past it): the unfused bits at 1 .. %u tokens\n", gn,
                tg->type, rows, cols, lim, clamped, kTok);
  }
  by_token(tg->type, gates, ups, dcnt, dxrow, xrow, rows, cols, nrow, sc, dx, dq, dd, dga, dup, dh, dref);
  // refusals: no scale, h's rows shorter than a row, a type not the contract's
  CHECK(!q8_gate_up(tg->type, gates, ups, dcnt, dxrow, nullptr, nullptr, kExperts, kTok, rows, cols, dq, dd, nullptr, 1.0f, dh, rows,
                    nullptr) &&
        !q8_gate_up(tg->type, gates, ups, dcnt, dxrow, nullptr, nullptr, kExperts, kTok, rows, cols, dq, dd, ds, 1.0f, dh, rows - 1,
                    nullptr) &&
        !q8_gate_up(13, gates, ups, dcnt, dxrow, nullptr, nullptr, kExperts, kTok, rows, cols, dq, dd, ds, 1.0f, dh, rows, nullptr));
  for (void* p : {(void*) dw, (void*) dp, (void*) dx, (void*) dq, (void*) dd, (void*) ds, (void*) dga, (void*) dup,
                  (void*) dh, (void*) dref, (void*) dcnt, (void*) dxrow})
    CHECK(hipFree(p) == hipSuccess);
}

int main() {
  std::printf("[test_q8_fuse_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_q8_fuse_gpu] skipped: no ds1017 or UD-IQ1_M, or no GPU\n");
    return 77;
  }
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  // ds1017's IQ1_M, IQ2_XXS, IQ2_XS; UD-IQ1_M's IQ1_S, IQ2_XXS, IQ2_S (test_model.h's gates)
  for (int l : which == TestModel::kUd ? std::vector<int>{0, 20, 38} : std::vector<int>{0, 10, 17}) layer(g, l);
  gguf_destroy(g);
  std::printf("[test_q8_fuse_gpu] ok\n");
  return 0;
}
