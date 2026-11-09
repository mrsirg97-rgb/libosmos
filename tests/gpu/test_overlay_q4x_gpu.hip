// a Qwen overlay (q4x_weights_create's: tensors standing in for the model's of their names). an overlay of the model's
// own bytes (three tensors copied out: a DeltaNet's ssm_out, an attention's out, a shared expert's down) - the logits
// of a prompt and a few greedy steps the same bits as with none: the stand-ins read from their file, the shadowed
// tensors' place taken. huihui's abliteration (tools/install/overlay_q4x.py's, its 96 tensors) - the logits not the
// same, the same bits twice; the state identity another (a conversation kept with one is not loaded with the other).
// needs a gfx1151, ~/Models/qwen38/UD-IQ3_XXS and the converted overlay; skipped (77) without them.
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

constexpr uint32_t kCtx = 1024, kMaxTokens = 64, kPrompt = 48;
constexpr int kSteps = 6;

// a GGUF v3 of `names`' tensors copied from the model (their dims, types and bytes), general.architecture qwen4exp
static void write_overlay(const Gguf* g, const std::vector<std::string>& names, const std::string& path) {
  std::string head("GGUF", 4), info;
  auto u32 = [](std::string& s, uint32_t v) { s.append((const char*) &v, 4); };
  auto u64 = [](std::string& s, uint64_t v) { s.append((const char*) &v, 8); };
  auto str = [&](std::string& s, const std::string& t) { u64(s, t.size()), s += t; };
  u32(head, 3), u64(head, names.size()), u64(head, 1);
  str(head, "general.architecture"), u32(head, 8), str(head, "qwen4exp");
  std::vector<const GgufTensor*> ts;
  uint64_t at = 0;
  for (const auto& n : names) {
    const GgufTensor* t = gguf_tensor(g, n.c_str());
    CHECK(t);
    ts.push_back(t);
    at += (32 - at % 32) % 32;
    str(info, n), u32(info, t->n_dims);
    for (uint32_t d = 0; d < t->n_dims; d++) u64(info, t->dims[d]);
    u32(info, t->type), u64(info, at);
    at += t->bytes;
  }
  std::string body = head + info;
  body.append((32 - body.size() % 32) % 32, '\0');
  FILE* f = std::fopen(path.c_str(), "wb");
  CHECK(f && std::fwrite(body.data(), 1, body.size(), f) == body.size());
  uint64_t written = 0;
  for (const GgufTensor* t : ts) {
    const uint64_t pad = (32 - written % 32) % 32;
    std::vector<char> b(pad + t->bytes, 0);
    const int fd = open(g->paths[t->file], O_RDONLY);
    CHECK(fd >= 0 && pread(fd, b.data() + pad, t->bytes, (off_t) t->offset) == (ssize_t) t->bytes);
    close(fd);
    CHECK(std::fwrite(b.data(), 1, b.size(), f) == b.size());
    written += b.size();
  }
  std::fclose(f);
}

struct Run {
  std::vector<float> logits;  // the prompt's last row, then each step's
  uint64_t identity;
};

static Run run(const Gguf* g, const Gguf* ov, uint32_t vocab) {
  char err[Q4X_FWD_ERR] = "", werr[Q4X_ERR] = "";
  Q4xWeights* w = q4x_weights_create(g, ov, werr);
  if (!w) std::fprintf(stderr, "  weights: %s\n", werr);
  CHECK(w);
  Q4xForward* f = q4x_forward_create(w, nullptr, kCtx, kMaxTokens, err);
  if (!f) std::fprintf(stderr, "  forward: %s\n", err);
  CHECK(f);
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);
  std::vector<int32_t> p(kPrompt);
  uint32_t r = 1017;
  for (auto& x : p) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (vocab - 1000)) + 100;
  Run out{{}, q4x_identity(f)};
  std::vector<float> row(vocab);
  CHECK(q4x_batch(f, s, p.data(), kPrompt, 0, err));
  int32_t t;
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &t, err) && hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(row.data(), q4x_logits(f) + (size_t) (kPrompt - 1) * vocab, vocab * 4, hipMemcpyDeviceToHost) == hipSuccess);
  out.logits.insert(out.logits.end(), row.begin(), row.end());
  for (int i = 0; i < kSteps; i++) {
    CHECK(q4x_batch(f, s, &t, 1, 0, err));
    CHECK(q4x_greedy(f, 0, 1, &t, err) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(row.data(), q4x_logits(f), vocab * 4, hipMemcpyDeviceToHost) == hipSuccess);
    out.logits.insert(out.logits.end(), row.begin(), row.end());
  }
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  q4x_weights_destroy(w);
  return out;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_overlay_q4x_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string model = home + "/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf";
  const std::string abl = home + "/Models/qwen38/huihui-patch/qwen38-ud-iq3_xxs-abliterated-overlay.gguf";
  int ndev = 0;
  char gerr[GGUF_ERR] = "";
  Gguf* g = access(model.c_str(), R_OK) == 0 ? gguf_create(model.c_str(), gerr) : nullptr;
  if (!g || access(abl.c_str(), R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_overlay_q4x_gpu] skipped: no model, overlay or GPU\n");
    return 77;
  }
  const uint32_t vocab = (uint32_t) gguf_tensor(g, "token_embd.weight")->dims[1];
  const Run base = run(g, nullptr, vocab);

  char dir[] = "/tmp/test_overlay_q4x_XXXXXX";
  CHECK(mkdtemp(dir));
  const std::string same = std::string(dir) + "/same.gguf";
  write_overlay(g, {"blk.0.ssm_out.weight", "blk.3.attn_output.weight", "blk.0.ffn_down_shexp.weight"}, same);
  Gguf* sg = gguf_create(same.c_str(), gerr);
  CHECK(sg);
  const Run copy = run(g, sg, vocab);
  std::printf("  an overlay of the model's own bytes (3 tensors): logits %s\n",
              copy.logits == base.logits ? "the same bits" : "NOT the same");
  CHECK(copy.logits == base.logits);
  CHECK(copy.identity != base.identity);  // another file in the model: another model's states, though the same bits
  gguf_destroy(sg);
  unlink(same.c_str()), rmdir(dir);

  Gguf* ag = gguf_create(abl.c_str(), gerr);
  CHECK(ag);
  const Run a1 = run(g, ag, vocab), a2 = run(g, ag, vocab);
  size_t moved = 0;
  for (size_t i = 0; i < base.logits.size(); i++) moved += a1.logits[i] != base.logits[i];
  std::printf("  the abliteration (%zu tensors): %zu of %zu logits moved; twice the same bits: %s\n", ag->n_tensors, moved,
              base.logits.size(), a1.logits == a2.logits ? "yes" : "NO");
  CHECK(moved > 0 && a1.logits == a2.logits && a1.identity != base.identity);
  gguf_destroy(ag);
  gguf_destroy(g);
  std::printf("[test_overlay_q4x_gpu] ok\n");
  return 0;
}
