// qwen_dump: llama.cpp's own tensors on a real text, the reference Qwen3.8-Flash-Next's forward is held to (DESIGN.md,
// "Qwen's forward, op by op"). the text tokenized by the model's own tokenizer, its first N tokens one batch, every
// tensor whose cb name (before its "-layer") is asked written as F32: <out>/<name>.bin (a name's second and later in
// the graph's order <name>#k.bin: a layer's two mixers share their names) - "QDMP", u32 dims, 4 x i64
// ne, then the values (ggml's order: ne0 fastest) - and <out>/ids.txt the tokens.
//
//   qwen_dump <first shard> <text file> <tokens> <out dir> <name,name,...>
//   qwen_dump <first shard> <text file> <tokens> <out dir> -ppl    (no tensors: the perplexity of the tokens, one
//   QWEN_DUMP_KV=q8_0: the cache in Q8_0 (keys and values; flash attention), as osmos keeps it
//   QWEN_DUMP_YARN=f: YaRN, factor f over the native context (the model card's static YaRN, original 262,144)
//   QWEN_DUMP_POS0=p: the tokens at positions p, p + 1, ... (a long context's rotation without its cache)
//   QWEN_DUMP_LAYER=l: layer l's tensors alone (and those of no layer); I32 ones (indexer_top_k) as F32, exact
//        sequence, each next token scored by log-softmax in double - tools/bench/q4x_ppl's protocol)
//
// build: g++ -std=c++17 -O2 oracle/qwen_dump.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include
//        -L<llama.cpp>/build-hip/bin -lllama -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-hip/bin -o bin/qwen_dump
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

struct Want {
  std::set<std::string> names;
  std::map<std::string, int> seen; /* a name's dumps so far: a layer's two mixers share one */
  std::string out;
  int written = 0, skipped = 0;
};

// the cb name a tensor was given: "<name>" or "<name>-<layer>"; a node ggml derived from one ("... (reshaped)") none
static std::string base_name(const char* name) {
  const char* dash = std::strrchr(name, '-');
  if (std::strchr(name, ' ')) return "";
  if (!dash || !dash[1]) return std::string(name);
  for (const char* p = dash + 1; *p; p++)
    if (*p < '0' || *p > '9') return std::string(name);
  return std::string(name, dash - name);
}

// the layer a cb name carries ("<name>-<layer>"), -1 for none
static int layer_of(const char* name) {
  const char* dash = std::strrchr(name, '-');
  return dash && dash[1] && base_name(name) != name ? std::atoi(dash + 1) : -1;
}

static bool dump(ggml_tensor* t, bool ask, void* arg) {
  Want* w = (Want*) arg;
  static const int only = std::getenv("QWEN_DUMP_LAYER") ? std::atoi(std::getenv("QWEN_DUMP_LAYER")) : -1;
  const int l = layer_of(t->name);
  const bool wanted = w->names.count(base_name(t->name)) > 0 && (only < 0 || l < 0 || l == only);
  if (ask) return wanted;
  if (!wanted) return true;
  if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32)) {
    std::fprintf(stderr, "qwen_dump: %s not dumped (%s, %s)\n", t->name, ggml_type_name(t->type),
                 ggml_is_contiguous(t) ? "contiguous" : "a view");
    w->skipped++;
    return true;
  }
  const int64_t n = ggml_nelements(t);
  std::vector<uint8_t> raw(ggml_nbytes(t));
  ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
  std::vector<float> f(n);
  if (t->type == GGML_TYPE_F32)
    std::memcpy(f.data(), raw.data(), n * 4);
  else if (t->type == GGML_TYPE_I32)
    for (int64_t i = 0; i < n; i++) f[i] = (float) ((const int32_t*) raw.data())[i];
  else
    for (int64_t i = 0; i < n; i++) f[i] = ggml_fp16_to_fp32(((const ggml_fp16_t*) raw.data())[i]);
  const int k = w->seen[t->name]++;
  const std::string path = w->out + "/" + t->name + (k ? "#" + std::to_string(k) : "") + ".bin";
  FILE* o = std::fopen(path.c_str(), "wb");
  if (!o) return std::fprintf(stderr, "qwen_dump: %s: cannot write\n", path.c_str()), false;
  const uint32_t dims = (uint32_t) ggml_n_dims(t);
  std::fwrite("QDMP", 1, 4, o);
  std::fwrite(&dims, 4, 1, o);
  std::fwrite(t->ne, 8, 4, o);
  std::fwrite(f.data(), 4, n, o);
  std::fclose(o);
  w->written++;
  return true;
}

int main(int argc, char** argv) {
  if (argc != 6) return std::fprintf(stderr, "usage: qwen_dump <first shard> <text> <tokens> <out dir> <names>\n"), 2;
  const int n_tok = std::atoi(argv[3]);
  Want want;
  want.out = argv[4];
  std::stringstream names(argv[5]);
  for (std::string s; std::getline(names, s, ',');) want.names.insert(s);
  std::ifstream in(argv[2]);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (n_tok < 1 || text.empty()) return std::fprintf(stderr, "qwen_dump: no tokens asked, or an empty text\n"), 2;

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) return std::fprintf(stderr, "qwen_dump: the model did not load\n"), 1;
  const llama_vocab* vocab = llama_model_get_vocab(model);
  std::vector<llama_token> ids(text.size() + 16);
  const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false, false);
  if (n < n_tok) return std::fprintf(stderr, "qwen_dump: the text is %d tokens, %d asked\n", n, n_tok), 1;
  ids.resize(n_tok);

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t) n_tok + 64;
  cp.n_batch = cp.n_ubatch = (uint32_t) n_tok;
  const bool ppl = want.names.count("-ppl") > 0;
  if (ppl) cp.n_ubatch = 512, cp.n_ctx = (uint32_t) n_tok + 64; /* every row's logits, read a ubatch at a time */
  else cp.cb_eval = dump, cp.cb_eval_user_data = &want;
  if (const char* kv = std::getenv("QWEN_DUMP_KV"); kv && !std::strcmp(kv, "q8_0")) /* osmos's cache: llama.cpp's on it */
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0, cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  if (const char* f = std::getenv("QWEN_DUMP_YARN")) /* static YaRN: llama.cpp's own ext / attn factors for it */
    cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN, cp.rope_freq_scale = 1.0f / (float) std::atof(f),
    cp.yarn_orig_ctx = 262144;
  const int pos0 = std::getenv("QWEN_DUMP_POS0") ? std::atoi(std::getenv("QWEN_DUMP_POS0")) : 0;
  if (pos0) cp.n_ctx = (uint32_t) (pos0 + n_tok + 64);
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) return std::fprintf(stderr, "qwen_dump: no context\n"), 1;
  llama_batch b = llama_batch_init(n_tok, 0, 1);
  for (int i = 0; i < n_tok; i++) {
    b.token[i] = ids[i], b.pos[i] = pos0 + i, b.n_seq_id[i] = 1, b.seq_id[i][0] = 0, b.logits[i] = true;
  }
  b.n_tokens = n_tok;
  if (llama_decode(ctx, b) != 0) return std::fprintf(stderr, "qwen_dump: decode failed\n"), 1;
  if (ppl) {
    const int V = llama_vocab_n_tokens(vocab);
    double nll = 0;
    for (int i = 0; i + 1 < n_tok; i++) {
      const float* l = llama_get_logits_ith(ctx, i);
      double m = l[0], s = 0;
      for (int k = 1; k < V; k++) m = l[k] > m ? l[k] : m;
      for (int k = 0; k < V; k++) s += std::exp(l[k] - m);
      nll -= l[ids[i + 1]] - m - std::log(s);
    }
    std::printf("qwen_dump: perplexity %.4f over %d tokens (mean log-probability %.6f)\n",
                std::exp(nll / (n_tok - 1)), n_tok - 1, -nll / (n_tok - 1));
  }
  FILE* o = std::fopen((want.out + "/ids.txt").c_str(), "w");
  if (!o) return std::fprintf(stderr, "qwen_dump: %s/ids.txt not writable (no such directory?)\n", want.out.c_str()), 1;
  for (int i = 0; i < n_tok; i++) std::fprintf(o, "%d%s", ids[i], i + 1 < n_tok ? "," : "\n");
  std::fclose(o);
  std::printf("qwen_dump: %d tokens, %d tensors written, %d not (views or other types)\n", n_tok, want.written,
              want.skipped);
  llama_batch_free(b);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
