// the type table against the kernels: for each type id, every GPU path - the decode, the matvec and its experts'
// form, the prompt matmul, the lane's int8 experts, the embedding - takes a block of it (zeros) exactly when the
// table's row names that path, and each call it takes runs. needs a gfx1151, not the model; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "substrate/gpu/mm.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/gpu/q8.h"
#include "substrate/gpu/quant.h"
#include "substrate/model/qtype.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kSweep = 256;  // past every ggml type id, as test_qtype's
constexpr uint32_t kRows = 16;    // mm_run's rows: a multiple of 16
constexpr uint32_t kCols = 512;   // mm_run's cols a multiple of 128; whole blocks of every type; two lanes' blocks
constexpr uint32_t kHc = 4;       // the model's hyper-connection streams (ops_embed's h)
constexpr uint32_t kWidest = 8;   // the widest type's bytes a value and more: F32's 4, so no read runs off a buffer
constexpr float kLim = 10.0f;     // a swiglu clamp: any finite one

static int failures;

static void agree(bool took, uint32_t id, uint32_t path, const char* what) {
  const bool named = qtype_paths(id) & path;
  if (took && hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "  type %u: %s took it and the launch failed\n", id, what);
    failures++;
    return;
  }
  if (took != named) {
    const QTypeInfo* t = qtype_info(id);
    std::fprintf(stderr, "  type %u (%s): %s %s it, the table %s\n", id, t ? t->name : "not in the table", what,
                 took ? "takes" : "refuses", named ? "names it" : "does not");
    failures++;
  }
}

template <typename T> static T* zeros(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc(&p, n * sizeof(T)) == hipSuccess);
  CHECK(hipMemset(p, 0, n * sizeof(T)) == hipSuccess);
  return p;
}

int main() {
  std::printf("[test_qtype_gpu] start\n");
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no GPU, skipped\n");
    return 77;
  }
  uint8_t* w = zeros<uint8_t>((size_t) kRows * kCols * kWidest);
  const void** wtab = zeros<const void*>(1);
  CHECK(hipMemcpy(wtab, &w, sizeof w, hipMemcpyHostToDevice) == hipSuccess);
  const int32_t one = 1;
  int32_t* cnt = zeros<int32_t>(1);
  int32_t* xrow = zeros<int32_t>(1);
  int32_t* ids = zeros<int32_t>(1);
  CHECK(hipMemcpy(cnt, &one, sizeof one, hipMemcpyHostToDevice) == hipSuccess);
  float* x = zeros<float>((size_t) MV_MAX_TOKENS * kCols);
  float* y = zeros<float>((size_t) MV_MAX_TOKENS * kRows);
  float* dec = zeros<float>(kCols);
  uint16_t* dec16 = zeros<uint16_t>(kCols);
  uint16_t* x16 = zeros<uint16_t>((size_t) MM_TILE * kCols);
  int32_t* overflow = zeros<int32_t>(1);
  int8_t* q = zeros<int8_t>((size_t) MV_MAX_TOKENS * kCols);
  float* d = zeros<float>((size_t) MV_MAX_TOKENS * kCols / 32);
  float* scale = zeros<float>(MV_MAX_TOKENS);
  float* h = zeros<float>((size_t) MV_MAX_TOKENS * kHc * kCols);
  CHECK(mm_x16(x, kCols, 1, kCols, x16, overflow, nullptr));
  CHECK(q8_round(x, 1, kCols, kCols, q, d, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);

  for (uint32_t id = 0; id < kSweep; id++) {
    agree(quant_known(id), id, QTYPE_DECODE, "quant_known");
    agree(quant_dequant(id, w, kCols, dec, nullptr), id, QTYPE_DECODE, "quant_dequant");
    agree(quant_dequant_f16(id, w, kCols, dec16, nullptr), id, QTYPE_DECODE, "quant_dequant_f16");
    agree(mv_group(id, wtab, 1, kRows, kCols, x, 0, 1, y, nullptr), id, QTYPE_MV, "mv_group");
    agree(mv_group_tok(id, wtab, cnt, xrow, 1, 1, kRows, kCols, x, kCols, y, kRows, nullptr), id, QTYPE_MV_TOK,
          "mv_group_tok");
    agree(mm_known(id), id, QTYPE_MM, "mm_known");
    agree(mm_col_tiles(id, 1) > 0, id, QTYPE_MM, "mm_col_tiles");
    agree(mm_run(id, w, kRows, kCols, x16, 1, y, kRows, nullptr), id, QTYPE_MM, "mm_run");
    agree(q8_group_tok(id, wtab, cnt, xrow, 1, 1, kRows, kCols, q, d, y, kRows, nullptr), id, QTYPE_Q8,
          "q8_group_tok");
    agree(q8_gate_up(id, wtab, wtab, cnt, xrow, 1, 1, kRows, kCols, q, d, scale, kLim, h, kRows, nullptr), id,
          QTYPE_Q8, "q8_gate_up");
    agree(ops_embed(id, w, kCols, ids, 1, kHc, h, nullptr), id, QTYPE_EMBED, "ops_embed");
  }
  if (failures) {
    std::fprintf(stderr, "[test_qtype_gpu] %d failures\n", failures);
    return 1;
  }
  std::printf("  %u ids: each kernel path takes exactly the table's types, and runs them\n", kSweep);
  std::printf("[test_qtype_gpu] ok\n");
  return 0;
}
