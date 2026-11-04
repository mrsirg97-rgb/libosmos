// the indexer's selections: ops_topk (a token's k highest rows, ties to the lower row) plain and under the candidate
// blocks' mask, and ops_cand_blocks (ref.py's `candidates`: blocks of rows by their highest score, the token's last
// block forced in, the topk_blocks highest kept) - each against the host's replay, bit for bit. scores drawn from a
// few values (ties everywhere) and negatives; tokens with no rows, rows a partial block, fewer blocks than kept (all
// kept) and many more. a small candidate_topk_blocks (16) so the mask bites at test sizes. needs a gfx1151; skipped
// (77) without one.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kTok = 40, kRatio = 4, kStride = 3200, kBlock = 8, kKeepBlocks = 16, kK = 64;
constexpr uint32_t kWords = (kStride / kBlock + 31) / 32;

static uint32_t rng = 1017;
static uint32_t next() { return rng = rng * 1664525u + 1013904223u; }

static uint32_t visible(int32_t pos) { return (uint32_t) (pos + 1) / kRatio; }

// ref.py's candidates for one token's n rows: the kept blocks' bits
static std::vector<uint32_t> host_cand(const float* s, uint32_t n) {
  std::vector<uint32_t> bits(kWords, 0);
  const uint32_t nb = (n + kBlock - 1) / kBlock;
  std::vector<std::pair<float, uint32_t>> key(nb);
  for (uint32_t j = 0; j < nb; j++) {
    float m = -INFINITY;
    for (uint32_t c = j * kBlock; c < n && c < (j + 1) * kBlock; c++) m = std::max(m, s[c]);
    key[j] = {j == nb - 1 ? INFINITY : m, j};
  }
  std::stable_sort(key.begin(), key.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (uint32_t i = 0; i < nb && i < kKeepBlocks; i++) bits[key[i].second / 32] |= 1u << (key[i].second % 32);
  return bits;
}

// the k highest of n rows (a masked row -inf), ties to the lower row, ascending
static std::vector<int32_t> host_topk(const float* s, uint32_t n, const uint32_t* bits) {
  std::vector<int32_t> out;
  if (n <= kK) {
    for (uint32_t c = 0; c < n; c++) out.push_back((int32_t) c);
    return out;
  }
  std::vector<std::pair<float, uint32_t>> key(n);
  for (uint32_t c = 0; c < n; c++) {
    const uint32_t j = c / kBlock;
    key[c] = {bits && !((bits[j / 32] >> (j % 32)) & 1u) ? -INFINITY : s[c], c};
  }
  std::stable_sort(key.begin(), key.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (uint32_t i = 0; i < kK; i++) out.push_back((int32_t) key[i].second);
  std::sort(out.begin(), out.end());
  return out;
}

int main() {
  std::printf("[test_select_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_select_gpu] skipped: no GPU\n");
    return 77;
  }
  // positions: no rows, a few (fewer than k, fewer blocks than kept), a partial block, and many blocks
  std::vector<int32_t> pos(kTok);
  const int32_t fixed[] = {0, 2, 3, 40, 200, 257, 515, 520, 1023, 4 * kStride - 1};
  for (uint32_t t = 0; t < kTok; t++)
    pos[t] = t < sizeof fixed / sizeof *fixed ? fixed[t] : (int32_t) (next() % (4 * kStride));
  std::vector<float> score((size_t) kTok * kStride);
  for (auto& v : score) v = (float) ((int) (next() % 23) - 7) * 0.5f;  // 23 values, negatives: ties everywhere
  float* ds;
  int32_t *dpos, *dsel, *dn;
  uint32_t* dkeep;
  CHECK(hipMalloc(&ds, score.size() * 4) == hipSuccess && hipMalloc(&dpos, kTok * 4) == hipSuccess &&
        hipMalloc(&dsel, (size_t) kTok * OPS_MAX_SELECT * 4) == hipSuccess && hipMalloc(&dn, kTok * 4) == hipSuccess &&
        hipMalloc(&dkeep, (size_t) kTok * kWords * 4) == hipSuccess);
  CHECK(hipMemcpy(ds, score.data(), score.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dpos, pos.data(), kTok * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemset(dkeep, 0xA5, (size_t) kTok * kWords * 4) == hipSuccess);

  // the candidate blocks
  CHECK(ops_cand_blocks(ds, kStride, dpos, kRatio, kBlock, kKeepBlocks, kTok, dkeep, kWords, nullptr));
  std::vector<uint32_t> keep((size_t) kTok * kWords);
  CHECK(hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(keep.data(), dkeep, keep.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  uint32_t masked = 0;
  for (uint32_t t = 0; t < kTok; t++) {
    const uint32_t n = visible(pos[t]);
    const std::vector<uint32_t> want = host_cand(&score[(size_t) t * kStride], n);
    if (std::memcmp(want.data(), &keep[(size_t) t * kWords], kWords * 4)) {
      std::printf("  token %u (%u rows): the kept blocks differ\n", t, n);
      CHECK(false);
    }
    masked += (n + kBlock - 1) / kBlock > kKeepBlocks;
  }
  std::printf("  candidate blocks: %u tokens (%u past %u blocks, the mask biting), the kept bits the host's\n", kTok,
              masked, kKeepBlocks);

  // the top k, plain and masked
  for (int with = 0; with < 2; with++) {
    CHECK(hipMemset(dsel, 0xFF, (size_t) kTok * OPS_MAX_SELECT * 4) == hipSuccess);
    CHECK(ops_topk(ds, kStride, dpos, kRatio, kK, with ? dkeep : nullptr, kBlock, kWords, kTok, dsel, OPS_MAX_SELECT,
                   dn, nullptr));
    std::vector<int32_t> sel((size_t) kTok * OPS_MAX_SELECT), n_sel(kTok);
    CHECK(hipDeviceSynchronize() == hipSuccess &&
          hipMemcpy(sel.data(), dsel, sel.size() * 4, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(n_sel.data(), dn, kTok * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t t = 0; t < kTok; t++) {
      const std::vector<int32_t> want =
          host_topk(&score[(size_t) t * kStride], visible(pos[t]), with ? &keep[(size_t) t * kWords] : nullptr);
      CHECK(n_sel[t] == (int32_t) want.size());
      if (std::memcmp(want.data(), &sel[(size_t) t * OPS_MAX_SELECT], want.size() * 4)) {
        std::printf("  token %u (%u rows, %s): the selected rows differ\n", t, visible(pos[t]), with ? "masked" : "plain");
        CHECK(false);
      }
    }
    std::printf("  top %u %s: every token's rows the host's (ties to the lower row)\n", kK,
                with ? "under the candidate blocks' mask" : "plain");
  }
  // refusals
  CHECK(!ops_cand_blocks(ds, kStride, dpos, kRatio, 0, kKeepBlocks, kTok, dkeep, kWords, nullptr) &&
        !ops_cand_blocks(ds, kStride, dpos, kRatio, kBlock, kKeepBlocks, kTok, dkeep, kWords - 1, nullptr) &&
        !ops_topk(ds, kStride, dpos, kRatio, kK, dkeep, kBlock, kWords - 1, kTok, dsel, OPS_MAX_SELECT, dn, nullptr));
  std::printf("  refusals: a block of 0, bits shorter than the stride's blocks\n");
  for (void* p : {(void*) ds, (void*) dpos, (void*) dsel, (void*) dn, (void*) dkeep}) CHECK(hipFree(p) == hipSuccess);
  std::printf("[test_select_gpu] ok\n");
  return 0;
}
