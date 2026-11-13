// llama_ref <first shard> <ids.txt> <n> <out.ref> [step]: a reference's log-probabilities from llama.cpp, in kld's
// osmosref file (tools/bench/kld.hip, q4x_kld.hip read it as REF) - for a model too large for osmos's own reference
// path: Qwen3.8-Flash-Next's BF16 (360 GB) read from its mmapped files a ubatch at a time (DESIGN.md, "0.6.9"). one
// sequence of the first n ids, `step` (2,048) tokens a decode with every row's logits, each row's log-softmax in
// double on the host - kld's protocol. the same run on a quant osmos also runs (UD-IQ3_XXS) is the engines'
// calibration: q4x_kld's KL against it is what the two implementations add between them.
//   MASK=mask.txt  0 / 1 a token: row r written only when token r + 1 is 1 (as kld's)
//   YARN=f         static YaRN, factor f over the native 262,144 (llama.cpp's: osmos's q4x_rope is its port)
//   NGL=n          layers in GPU memory (0 by default: the weights stay in the mapped files)
//   KV=q8_0        the cache in Q8_0 (osmos's); f16 by default
// build: g++ -std=c++17 -O2 tools/bench/llama_ref.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include
//        -L<llama.cpp>/build-hip/bin -lllama -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-hip/bin -o bin/llama_ref
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "llama.h"

struct RefHead {  // kld.hip's
  char magic[8];
  uint32_t vocab, n, step, rows;
};

static std::vector<int32_t> read_list(const char* path) {
  std::vector<int32_t> v;
  FILE* f = std::fopen(path, "r");
  if (!f) return v;
  for (long x; std::fscanf(f, "%ld,", &x) == 1;) v.push_back((int32_t) x);
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 5) return std::fprintf(stderr, "usage: llama_ref <first shard> <ids.txt> <n> <out.ref> [step]\n"), 2;
  const uint32_t n = (uint32_t) std::atoi(argv[3]), step = argc > 5 ? (uint32_t) std::atoi(argv[5]) : 2048;
  const std::vector<int32_t> ids = read_list(argv[2]);
  if (ids.size() < n || n < 2 || !step) return std::fprintf(stderr, "llama_ref: %zu ids, %u asked\n", ids.size(), n), 2;
  std::vector<int32_t> mask(n, 1);
  if (const char* mp = std::getenv("MASK")) {
    mask = read_list(mp);
    if (mask.size() < n) return std::fprintf(stderr, "llama_ref: a mask of %zu, %u asked\n", mask.size(), n), 2;
  }
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = std::getenv("NGL") ? std::atoi(std::getenv("NGL")) : 0;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) return std::fprintf(stderr, "llama_ref: the model did not load\n"), 1;
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const uint32_t V = (uint32_t) llama_vocab_n_tokens(vocab);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = n + 64, cp.n_batch = cp.n_ubatch = step;
  if (const char* kv = std::getenv("KV"); kv && !std::strcmp(kv, "q8_0"))
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0, cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  if (const char* f = std::getenv("YARN"))
    cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN, cp.rope_freq_scale = 1.0f / (float) std::atof(f),
    cp.yarn_orig_ctx = 262144;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) return std::fprintf(stderr, "llama_ref: no context\n"), 1;

  uint32_t want = 0;
  for (uint32_t r = 0; r + 1 < n; r++) want += mask[r + 1] != 0;
  FILE* o = std::fopen(argv[4], "wb");
  if (!o) return std::fprintf(stderr, "llama_ref: %s not writable\n", argv[4]), 1;
  const RefHead h = {{'o', 's', 'm', 'o', 's', 'r', 'e', 'f'}, V, n, step, want};
  std::fwrite(&h, sizeof h, 1, o);
  std::vector<float> row(V);
  llama_batch b = llama_batch_init((int32_t) step, 0, 1);
  double nll = 0;
  uint32_t written = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t at = 0; at < n; at += step) {
    const uint32_t k = n - at < step ? n - at : step;
    for (uint32_t i = 0; i < k; i++)
      b.token[i] = ids[at + i], b.pos[i] = (llama_pos) (at + i), b.n_seq_id[i] = 1, b.seq_id[i][0] = 0, b.logits[i] = true;
    b.n_tokens = (int32_t) k;
    if (llama_decode(ctx, b) != 0) return std::fprintf(stderr, "llama_ref: decode at %u failed\n", at), 1;
    for (uint32_t r = 0; r < k && at + r + 1 < n; r++) {
      if (!mask[at + r + 1]) continue;
      const float* x = llama_get_logits_ith(ctx, (int32_t) r);
      uint32_t top = 0;
      for (uint32_t v = 1; v < V; v++)
        if (x[v] > x[top]) top = v;
      double sum = 0;
      for (uint32_t v = 0; v < V; v++) sum += std::exp((double) x[v] - x[top]);
      const double lse = x[top] + std::log(sum);
      for (uint32_t v = 0; v < V; v++) row[v] = (float) ((double) x[v] - lse);
      nll -= (double) x[ids[at + r + 1]] - lse;
      if (std::fwrite(row.data(), 4, V, o) != V) return std::fprintf(stderr, "llama_ref: a write failed\n"), 1;
      written++;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %u of %u tokens, %u rows, %.0f s\n", at + k, n, written, s);
    std::fflush(stdout);
  }
  if (std::fclose(o) != 0 || written != want) return std::fprintf(stderr, "llama_ref: %u rows of %u\n", written, want), 1;
  std::printf("llama_ref: ppl %.4f over %u rows (mean log-prob %.5f), steps of %u\n", std::exp(nll / written), written,
              -nll / written, step);
  llama_batch_free(b);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
