// Hy4-flash's gated MLA, absorbed (osmos/gpu/hy4_ops.h's hy4_mla_*) against the reference's own attention
// (oracle/hy4_mla.py: modeling_hy_v4_flash.py's HYV4FlashAttention, which expands the latent to keys and values -
// the same maths in another order), in F32 on the GGUF's weights: 600 tokens from position 0 through a sliding layer
// (64 heads, past its 512-key window) and a sparse one (32 heads, every key: dense below 2,048) - from the
// projections' outputs (q_b, kv_a, the gate) to o_proj's input, k_b and v_b read as the file's Q8_0. then a token's
// bits its own: the last token's output alone, after the cache its earlier tokens wrote, the same as in the batch.
// ~/osmos-oracle/hy4_mla.bin; skipped (77) without.
extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

#include "hy4_fixture.h"

// absorbing changes the order of 192 + 512 products per score and the softmax's sums over up to 600 keys: errors of a
// few ulps of each term, the output a weighted mean of the latent - 1e-5 of the largest output is ~80 ulps of it
constexpr double kTol = 1e-5;

static std::vector<float> freqs_of(float base, uint32_t rope_dim) {
  std::vector<float> f(rope_dim / 2);  // the reference's: 1 / base^(2i / rope_dim), in F32
  for (uint32_t i = 0; i < rope_dim / 2; i++) f[i] = 1.0f / powf(base, (float) (2 * i) / (float) rope_dim);
  return f;
}

static void layer(int L) {
  char p[32];
  auto key = [&](const char* name) {
    std::snprintf(p, sizeof p, "L%d.%s", L, name);
    return std::string(p);
  };
  const std::vector<float>& shape = A(key("shape").c_str());
  const uint32_t H = (uint32_t) shape[0], nope = (uint32_t) shape[1], vd = (uint32_t) shape[2];
  const uint32_t lat = (uint32_t) shape[3], rd = (uint32_t) shape[4], window = (uint32_t) shape[5];
  const uint32_t T = (uint32_t) fx[key("x")].dims[0], row = lat + rd;
  std::printf("  layer %d: %u heads, %s\n", L, H, window ? "sliding (512 keys)" : "sparse (every key)");
  std::vector<int32_t> pos(T);
  for (uint32_t t = 0; t < T; t++) pos[t] = (int32_t) t;
  int32_t* dpos = nullptr;
  CHECK(hipMalloc(&dpos, T * sizeof(int32_t)) == hipSuccess);
  CHECK(hipMemcpy(dpos, pos.data(), T * sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess);
  float* freqs = up(freqs_of(A(key("rope_base").c_str())[0], rd));
  float *kva = up(A(key("kva").c_str())), *q = up(A(key("q").c_str())), *gate = up(A(key("gate").c_str()));
  float *kb = up(A(key("k_b_q8").c_str())), *vb = up(A(key("v_b_q8").c_str()));
  float *norm = up(A(key("kv_a_norm").c_str())), *sinks = up(A(key("sinks").c_str()));
  float *cache = scratch((size_t) T * row), *qa = scratch((size_t) T * H * row);
  float *o_lat = scratch((size_t) T * H * lat), *out = scratch((size_t) T * H * vd);
  const float scale = 1.0f / sqrtf((float) (nope + rd));
  CHECK(hy4_mla_kv(kva, norm, lat, rd, 1e-5f, freqs, dpos, T, cache, 0));
  CHECK(hy4_mla_q(q, kb, H, nope, lat, rd, freqs, dpos, T, qa, 0));
  CHECK(hy4_mla_attn(qa, cache, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale, dpos, T, o_lat, 0));
  CHECK(hy4_mla_out(o_lat, vb, gate, H, lat, vd, T, out, 0));
  const std::vector<float> all = down(out, (size_t) T * H * vd);
  match("o_proj's input (gated attention)", all, key("o_in").c_str(), kTol);

  // the last token alone, its queries' the batch's rows: the cache the batch's (every earlier row written)
  const uint32_t last = T - 1;
  CHECK(hy4_mla_attn(qa + (size_t) last * H * row, cache, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale,
                     dpos + last, 1, o_lat, 0));
  CHECK(hy4_mla_out(o_lat, vb, gate + (size_t) last * H * vd, H, lat, vd, 1, out, 0));
  const std::vector<float> one = down(out, (size_t) H * vd);
  CHECK(std::memcmp(one.data(), all.data() + (size_t) last * H * vd, (size_t) H * vd * sizeof(float)) == 0);
  std::printf("  the last token alone: the same bits as in the batch\n");
  for (float* d : {freqs, kva, q, gate, kb, vb, norm, sinks, cache, qa, o_lat, out}) CHECK(hipFree(d) == hipSuccess);
  CHECK(hipFree(dpos) == hipSuccess);
}

int main() {
  char path[512];
  std::snprintf(path, sizeof path, "%s/osmos-oracle/hy4_mla.bin", getenv("HOME") ? getenv("HOME") : "");
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0 || !load(path)) {
    std::printf("[test_hy4_mla_gpu] skipped: no GPU or no %s (oracle/hy4_mla.py)\n", path);
    return 77;
  }
  std::printf("[test_hy4_mla_gpu] start\n");
  CHECK(!hy4_mla_attn(nullptr, nullptr, nullptr, 64, HY4_MLA_MAX_LATENT + 32, 64, 0, nullptr, nullptr, 0, 0, 1.0f,
                      nullptr, 1, nullptr, 0));
  layer(2);
  layer(5);
  std::printf("[test_hy4_mla_gpu] ok\n");
  return 0;
}
