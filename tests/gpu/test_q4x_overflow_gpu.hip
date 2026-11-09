// a step whose activations pass f16's range (mm's inputs are f16: the overflow flag) gives no token: its results are
// refused where they are read (q4x_greedy), never a pick from them. the overflow made: an overlay of the first layer's
// shared expert's down (Q8_0) with every block's scale f16's largest - its output added into the residual past f16.
// then a clean forward of the same model the same tokens as before (the flag not left set). needs a gfx1151 and
// ~/Models/qwen38/UD-IQ3_XXS; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kCtx = 512, kMaxTokens = 64, kPrompt = 32;
constexpr const char* kTensor = "blk.0.ffn_down_shexp.weight";
constexpr uint32_t kQ8Type = 8, kQ8Block = 34;  // ggml's Q8_0: an f16 scale, then 32 int8
constexpr uint16_t kF16Max = 0x7BFF;            // 65504

// a GGUF v3 of the one tensor, every Q8_0 block's scale f16's largest
static void write_overlay(const Gguf* g, const std::string& path) {
  const GgufTensor* t = gguf_tensor(g, kTensor);
  CHECK(t && t->type == kQ8Type && t->bytes % kQ8Block == 0);
  std::vector<char> b(t->bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, b.data(), t->bytes, (off_t) t->offset) == (ssize_t) t->bytes);
  close(fd);
  for (uint64_t o = 0; o < t->bytes; o += kQ8Block) std::memcpy(&b[o], &kF16Max, 2);
  std::string s("GGUF", 4);
  auto u32 = [&](uint32_t v) { s.append((const char*) &v, 4); };
  auto u64 = [&](uint64_t v) { s.append((const char*) &v, 8); };
  auto str = [&](const std::string& x) { u64(x.size()), s += x; };
  u32(3), u64(1), u64(1);
  str("general.architecture"), u32(8), str("qwen4exp");
  str(kTensor), u32(t->n_dims);
  for (uint32_t d = 0; d < t->n_dims; d++) u64(t->dims[d]);
  u32(t->type), u64(0);
  s.append((32 - s.size() % 32) % 32, '\0');
  FILE* f = std::fopen(path.c_str(), "wb");
  CHECK(f && std::fwrite(s.data(), 1, s.size(), f) == s.size() && std::fwrite(b.data(), 1, b.size(), f) == b.size());
  std::fclose(f);
}

// the prompt, then its greedy pick: ok false with why when either refuses
struct Out {
  bool ok;
  int32_t t;
  std::string err;
};

static Out run(const Gguf* g, const Gguf* ov, const std::vector<int32_t>& p) {
  char err[Q4X_FWD_ERR] = "", werr[Q4X_ERR] = "";
  Q4xWeights* w = q4x_weights_create(g, ov, werr);
  CHECK(w);
  Q4xForward* f = q4x_forward_create(w, nullptr, kCtx, kMaxTokens, err);
  CHECK(f);
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);
  Out o{false, -1, ""};
  o.ok = q4x_batch(f, s, p.data(), kPrompt, 0, err) && q4x_greedy(f, kPrompt - 1, 1, &o.t, err);
  o.err = err;
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  q4x_weights_destroy(w);
  return o;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_overflow_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string model = home + "/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf";
  int ndev = 0;
  char gerr[GGUF_ERR] = "";
  Gguf* g = access(model.c_str(), R_OK) == 0 ? gguf_create(model.c_str(), gerr) : nullptr;
  if (!g || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_overflow_gpu] skipped: no model or GPU\n");
    return 77;
  }
  const uint32_t vocab = (uint32_t) gguf_tensor(g, "token_embd.weight")->dims[1];
  std::vector<int32_t> p(kPrompt);
  uint32_t r = 1017;
  for (auto& x : p) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (vocab - 1000)) + 100;

  const Out clean = run(g, nullptr, p);
  CHECK(clean.ok);
  char dir[] = "/tmp/test_q4x_overflow_XXXXXX";
  CHECK(mkdtemp(dir));
  const std::string path = std::string(dir) + "/overflow.gguf";
  write_overlay(g, path);
  Gguf* og = gguf_create(path.c_str(), gerr);
  CHECK(og);
  const Out over = run(g, og, p);
  std::printf("  the overflowing step: %s (%s)\n", over.ok ? "a token picked" : "refused", over.err.c_str());
  CHECK(!over.ok && std::strstr(over.err.c_str(), "f16"));
  const Out again = run(g, nullptr, p);
  std::printf("  a clean forward after it: token %d (the first clean one's %d)\n", again.t, clean.t);
  CHECK(again.ok && again.t == clean.t);
  gguf_destroy(og);
  unlink(path.c_str()), rmdir(dir);
  gguf_destroy(g);
  std::printf("[test_q4x_overflow_gpu] ok\n");
  return 0;
}
