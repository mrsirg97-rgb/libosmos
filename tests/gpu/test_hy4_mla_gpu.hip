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

// the Q8_0 cache's gated attention output against the F32 reference: a latent value's rounding (1/254 of its
// block's largest) through the softmax and v_b - its error relative to the largest output under this
constexpr double kQ8Rel = 0.05;

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
  void* table = scratch(hy4_mla_table_bytes(T, H) / sizeof(float) + 1);
  float* part = scratch(hy4_mla_attn_scratch(H));
  float *o_lat = scratch((size_t) T * H * lat), *out = scratch((size_t) T * H * vd);
  const float scale = 1.0f / sqrtf((float) (nope + rd));
  CHECK(hy4_mla_kv(kva, norm, lat, rd, 1e-5f, freqs, dpos, T, cache, 0, false, 0));
  CHECK(hy4_mla_q(q, kb, H, nope, lat, rd, freqs, dpos, T, qa, table, 0));
  CHECK(hy4_mla_attn(qa, cache, 0, false, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale, dpos, T, o_lat, part, 0));
  CHECK(hy4_mla_out(o_lat, vb, gate, H, lat, vd, T, out, table, 0));
  const std::vector<float> all = down(out, (size_t) T * H * vd);
  match("o_proj's input (gated attention)", all, key("o_in").c_str(), kTol);

  // the last token alone, its queries' the batch's rows: the cache the batch's (every earlier row written)
  const uint32_t last = T - 1;
  CHECK(hy4_mla_attn(qa + (size_t) last * H * row, cache, 0, false, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale,
                     dpos + last, 1, o_lat, part, 0));
  CHECK(hy4_mla_out(o_lat, vb, gate + (size_t) last * H * vd, H, lat, vd, 1, out, table, 0));
  const std::vector<float> one = down(out, (size_t) H * vd);
  CHECK(std::memcmp(one.data(), all.data() + (size_t) last * H * vd, (size_t) H * vd * sizeof(float)) == 0);
  std::printf("  the last token alone: the same bits as in the batch\n");

  // a sliding layer's ring (the window and a step's tokens): steps of 8, each its keys then its attention - the
  // full cache's bits
  if (window) {
    const uint32_t step = 8, slots = window + step;
    float* ring = scratch((size_t) slots * row);
    for (uint32_t t0 = 0; t0 < T; t0 += step) {
      const uint32_t m = T - t0 < step ? T - t0 : step;
      CHECK(hy4_mla_kv(kva + (size_t) t0 * row, norm, lat, rd, 1e-5f, freqs, dpos + t0, m, ring, slots, false, 0));
      CHECK(hy4_mla_attn(qa + (size_t) t0 * H * row, ring, slots, false, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0,
                         scale, dpos + t0, m, o_lat + (size_t) t0 * H * lat, part, 0));
    }
    CHECK(hy4_mla_out(o_lat, vb, gate, H, lat, vd, T, out, table, 0));
    const std::vector<float> ringed = down(out, (size_t) T * H * vd);
    CHECK(std::memcmp(ringed.data(), all.data(), all.size() * sizeof(float)) == 0);
    CHECK(!hy4_mla_attn(qa, ring, window - 1, false, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale, dpos, 1, o_lat, part,
                        0));
    std::printf("  a ring of %u rows in steps of %u: the full cache's bits (a ring under the window refused)\n", slots,
                step);
    CHECK(hipFree(ring) == hipSuccess);
  }
  // the Q8_0 cache (0.11.0's default): its rows hy4_mla_kv's q8; the attention over them the F32 path's over the same
  // rows decoded on the host (d times a code: exact in F32) bit for bit; its distance from the reference stated
  {
    const uint64_t rb = hy4_mla_row_bytes(lat, rd, true);
    CHECK(rb == (uint64_t) lat / 32 * 34 + (uint64_t) rd * 4 && rb % 16 == 0);
    float* c8 = scratch((size_t) (T * rb / sizeof(float)) + 4);
    CHECK(hy4_mla_kv(kva, norm, lat, rd, 1e-5f, freqs, dpos, T, c8, 0, true, 0));
    std::vector<uint8_t> raw((size_t) T * rb);
    CHECK(hipMemcpy(raw.data(), c8, raw.size(), hipMemcpyDeviceToHost) == hipSuccess);
    std::vector<float> dec((size_t) T * row);
    for (uint32_t t = 0; t < T; t++) {
      const uint8_t* r = raw.data() + (size_t) t * rb;
      for (uint32_t b = 0; b < lat / 32; b++) {
        _Float16 d16;
        std::memcpy(&d16, r + b * 34, 2);
        for (int i = 0; i < 32; i++) dec[(size_t) t * row + b * 32 + i] = (float) d16 * (float) (int8_t) r[b * 34 + 2 + i];
      }
      std::memcpy(&dec[(size_t) t * row + lat], r + lat / 32 * 34, rd * sizeof(float));
    }
    float* cd = up(dec);
    CHECK(hy4_mla_attn(qa, c8, 0, true, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale, dpos, T, o_lat, part, 0));
    const std::vector<float> o8 = down(o_lat, (size_t) T * H * lat);
    CHECK(hy4_mla_attn(qa, cd, 0, false, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale, dpos, T, o_lat, part, 0));
    const std::vector<float> od = down(o_lat, (size_t) T * H * lat);
    CHECK(std::memcmp(o8.data(), od.data(), o8.size() * sizeof(float)) == 0);
    CHECK(hy4_mla_attn(qa, c8, 0, true, sinks, H, lat, rd, window, nullptr, nullptr, 0, 0, scale, dpos, T, o_lat, part, 0));
    CHECK(hy4_mla_out(o_lat, vb, gate, H, lat, vd, T, out, table, 0));
    const std::vector<float> q8out = down(out, (size_t) T * H * vd);
    double worst = 0, top = 0;
    const std::vector<float>& ref = A(key("o_in").c_str());
    for (size_t i = 0; i < ref.size(); i++)
      worst = std::fmax(worst, std::fabs((double) q8out[i] - ref[i])), top = std::fmax(top, std::fabs((double) ref[i]));
    std::printf("  the Q8_0 cache (%llu bytes a row, F32's %u): the F32 path's bits on its rows decoded; against the "
                "reference max error %.3g of %.4g (%.2g)\n", (unsigned long long) rb, row * 4, worst, top, worst / top);
    CHECK(worst / top < kQ8Rel);
    CHECK(hipFree(c8) == hipSuccess && hipFree(cd) == hipSuccess);
  }
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
  CHECK(!hy4_mla_attn(nullptr, nullptr, 0, false, nullptr, 64, HY4_MLA_MAX_LATENT + 32, 64, 0, nullptr, nullptr, 0, 0, 1.0f,
                      nullptr, 1, nullptr, nullptr, 0));
  layer(2);
  layer(5);
  std::printf("[test_hy4_mla_gpu] ok\n");
  return 0;
}
