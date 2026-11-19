// Hy4-flash's compressed indexer (osmos/gpu/hy4_ops.h's hy4_ix_*, hy4_mla_attn over selected groups) against the
// reference's own sparse attention (oracle/hy4_indexer.py: modeling_hy_v4_flash.py's HYV4FlashAttention on a
// sparse layer past 2,048 keys), in F32 on the GGUF's weights, 2,600 tokens from position 0: the token keys, the
// pooled group keys (pool_groups'), each token's groups the reference's recall exactly (its 511 best complete groups
// before its own and its own; fewer complete groups: all of them), then the gated attention over them entering
// o_proj. every group selected, the dense path's bits; run twice, the same bits. ~/osmos-oracle/hy4_indexer.bin; skipped (77) without.
extern "C" {
#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/ops.h"
}

#include <algorithm>
#include <set>

#include "hy4_fixture.h"

// the pooled keys: a 128-long dot, a softmax over 4 slots and a norm (a few ulps); the attention as
// test_hy4_mla_gpu's
constexpr double kTolKeys = 4e-6, kTolAttn = 1e-5;
constexpr float kEps = 1e-5f;  // config.json's rms_norm_eps: the indexer's norms', MLA's latent's

static std::vector<float> freqs_of(float base, uint32_t rope_dim) {
  std::vector<float> f(rope_dim / 2);
  for (uint32_t i = 0; i < rope_dim / 2; i++) f[i] = 1.0f / powf(base, (float) (2 * i) / (float) rope_dim);
  return f;
}

template <typename T>
static T* dev(const std::vector<T>& v) {
  T* d = nullptr;
  CHECK(hipMalloc(&d, v.size() * sizeof(T)) == hipSuccess);
  CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

int main() {
  char path[512];
  std::snprintf(path, sizeof path, "%s/osmos-oracle/hy4_indexer.bin", getenv("HOME") ? getenv("HOME") : "");
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0 || !load(path)) {
    std::printf("[test_hy4_indexer_gpu] skipped: no GPU or no %s (oracle/hy4_indexer.py)\n", path);
    return 77;
  }
  const std::vector<float>& shape = A("shape");
  const uint32_t H = (uint32_t) shape[0], nope = (uint32_t) shape[1], vd = (uint32_t) shape[2];
  const uint32_t lat = (uint32_t) shape[3], rd = (uint32_t) shape[4], IH = (uint32_t) shape[5];
  const uint32_t ID = (uint32_t) shape[6], pool = (uint32_t) shape[7], top = (uint32_t) shape[8];
  const uint32_t T = (uint32_t) fx["q"].dims[0], G = T / pool, row = lat + rd;
  std::printf("[test_hy4_indexer_gpu] %u tokens, %u groups of %u; indexer %u heads of %u, %u groups a query\n", T, G,
              pool, IH, ID, top);
  std::vector<int32_t> pos(T);
  for (uint32_t t = 0; t < T; t++) pos[t] = (int32_t) t;
  int32_t* dpos = dev(pos);
  float* freqs = up(freqs_of(A("rope_base")[0], rd));

  // the token keys, then the groups pooled (unroped: pool_groups' own), then roped (the scores' keys)
  float *ixk = up(A("ixk")), *knorm = up(A("ix_k_norm")), *keys = scratch((size_t) T * ID);
  CHECK(ops_rms(ixk, ID, ID, knorm, kEps, T, keys, ID, 0));
  float *pv = up(A("pool_value_q8")), *pg = up(A("pool_gate_q8")), *ape = up(A("pool_ape"));
  float *pn = up(A("pool_norm")), *groups = scratch((size_t) G * ID);
  CHECK(hy4_ix_pool(keys, pv, pg, ape, pn, ID, pool, rd, kEps, nullptr, 0, G, groups, 0));
  match("the pooled group keys", down(groups, (size_t) G * ID), "groups", kTolKeys);
  CHECK(hy4_ix_pool(keys, pv, pg, ape, pn, ID, pool, rd, kEps, freqs, 0, G, groups, 0));

  // the queries and the heads' weights, then each token's groups: the reference's recall, exactly
  float *ixq = up(A("ixq")), *ixw = up(A("ixw")), *scores = scratch((size_t) T * G);
  std::vector<int32_t> zeros((size_t) T * top, -1), nz(T, 0);
  int32_t *sel = dev(zeros), *n_sel = dev(nz);
  CHECK(hy4_ix_query(ixq, ixw, IH, ID, rd, freqs, dpos, T, 0));
  CHECK(hy4_ix_select(ixq, ixw, groups, IH, ID, pool, top, dpos, T, G, scores, sel, n_sel, 0));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<int32_t> s((size_t) T * top), n(T);
  CHECK(hipMemcpy(s.data(), sel, s.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(n.data(), n_sel, n.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  const std::vector<float>& recall = A("recall");
  uint32_t differ = 0, chosen = 0;
  for (uint32_t t = 0; t < T; t++) {
    const int32_t own = (int32_t) (t / pool);
    std::set<int32_t> want = {own};  // the reference's picks before its own group, and its own
    for (uint32_t i = 0; i < top; i++)
      if ((int32_t) recall[(size_t) t * top + i] < own) want.insert((int32_t) recall[(size_t) t * top + i]);
    std::vector<int32_t> got(s.begin() + (size_t) t * top, s.begin() + (size_t) t * top + n[t]);
    CHECK(std::is_sorted(got.begin(), got.end()) && !got.empty() && got.back() == own);
    const std::set<int32_t> gs(got.begin(), got.end());
    if (gs != want && differ++ < 6) {
      std::vector<int32_t> only_ours, only_ref;
      std::set_difference(gs.begin(), gs.end(), want.begin(), want.end(), std::back_inserter(only_ours));
      std::set_difference(want.begin(), want.end(), gs.begin(), gs.end(), std::back_inserter(only_ref));
      std::printf("  token %u: %zu groups ours alone, %zu the reference's alone (first %d / %d)\n", t,
                  only_ours.size(), only_ref.size(), only_ours.empty() ? -1 : only_ours[0],
                  only_ref.empty() ? -1 : only_ref[0]);
    }
    chosen += (uint32_t) own > top - 1;
  }
  std::printf("  each token's groups: %u of %u the reference's exactly (%u chosen among more than %u)\n", T - differ,
              T, chosen, top - 1);
  CHECK(!differ && chosen > 0);

  // run again: the pooled keys and every token's groups the same bits (no order a race picks)
  const std::vector<float> g1 = down(groups, (size_t) G * ID);
  CHECK(hy4_ix_pool(keys, pv, pg, ape, pn, ID, pool, rd, kEps, freqs, 0, G, groups, 0));
  CHECK(hy4_ix_select(ixq, ixw, groups, IH, ID, pool, top, dpos, T, G, scores, sel, n_sel, 0));
  const std::vector<float> g2 = down(groups, (size_t) G * ID);
  std::vector<int32_t> s2((size_t) T * top);
  CHECK(hipMemcpy(s2.data(), sel, s2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(g1.data(), g2.data(), g1.size() * 4) == 0 && s2 == s);
  std::printf("  again: the pooled keys and the groups the same bits\n");

  // the keys and groups a step at a time on the GPU's positions (a decode's: 1, 1, 3, then the rest) - the same bits
  {
    float *keys2 = scratch((size_t) T * ID), *groups2 = scratch((size_t) G * ID), *kn = scratch((size_t) T * ID);
    CHECK(ops_rms(ixk, ID, ID, knorm, kEps, T, kn, ID, 0));
    const uint32_t cuts[] = {0, 1, 2, 5, T};
    for (int c = 0; c + 1 < 5; c++) {
      const uint32_t a = cuts[c], m = cuts[c + 1] - a;
      CHECK(hy4_rows_at(keys2, kn + (size_t) a * ID, dpos + a, m, ID, 0, 0));
      CHECK(hy4_ix_pool_at(keys2, 0, pv, pg, ape, pn, ID, pool, rd, kEps, freqs, dpos + a, m, groups2, 0));
    }
    CHECK(std::memcmp(down(keys2, (size_t) T * ID).data(), down(keys, (size_t) T * ID).data(), (size_t) T * ID * 4) == 0);
    CHECK(std::memcmp(down(groups2, (size_t) G * ID).data(), g2.data(), g2.size() * 4) == 0);
    CHECK(!hy4_ix_pool_at(keys2, 0, pv, pg, ape, pn, ID, pool, rd, kEps, freqs, dpos, 0, groups2, 0));
    CHECK(!hy4_rows_at(keys2, kn, dpos, 0, ID, 0, 0));
    std::printf("  in steps of 1, 1, 3 and %u on the GPU's positions: the keys and the groups the same bits\n", T - 5);
    // the keys a ring (a sequence's: a step's tokens and a pool's - 0.11.0): steps of 8, a ring of 16 - the groups
    // the full rows' bits
    float* ring = scratch((size_t) 16 * ID);
    float* groups3 = scratch((size_t) G * ID);
    for (uint32_t a = 0; a < T; a += 8) {
      const uint32_t m = T - a < 8 ? T - a : 8;
      CHECK(hy4_rows_at(ring, kn + (size_t) a * ID, dpos + a, m, ID, 16, 0));
      CHECK(hy4_ix_pool_at(ring, 16, pv, pg, ape, pn, ID, pool, rd, kEps, freqs, dpos + a, m, groups3, 0));
    }
    CHECK(std::memcmp(down(groups3, (size_t) G * ID).data(), g2.data(), g2.size() * 4) == 0);
    CHECK(!hy4_ix_pool_at(ring, 6, pv, pg, ape, pn, ID, pool, rd, kEps, freqs, dpos, 1, groups3, 0));  // (not a pool's multiple)
    std::printf("  the keys a ring of 16 in steps of 8: the groups the full rows' bits\n");
    CHECK(hipFree(ring) == hipSuccess && hipFree(groups3) == hipSuccess);
  }

  // the attention over them (MLA as test_hy4_mla_gpu's), against the reference's
  float *kva = up(A("kva")), *q = up(A("q")), *gate = up(A("gate")), *kb = up(A("k_b_q8")), *vb = up(A("v_b_q8"));
  float *norm = up(A("kv_a_norm")), *sinks = up(A("sinks"));
  float *cache = scratch((size_t) T * row), *qa = scratch((size_t) T * H * row);
  void* table = scratch(hy4_mla_table_bytes(T, H) / sizeof(float) + 1);
  float* part = scratch(hy4_mla_attn_scratch(H));
  float *o_lat = scratch((size_t) T * H * lat), *out = scratch((size_t) T * H * vd);
  const float scale = 1.0f / sqrtf((float) (nope + rd));
  CHECK(hy4_mla_kv(kva, norm, lat, rd, kEps, freqs, dpos, T, cache, 0, false, 0));
  CHECK(hy4_mla_q(q, kb, H, nope, lat, rd, freqs, dpos, T, qa, table, 0));
  CHECK(hy4_mla_attn(qa, cache, 0, false, sinks, H, lat, rd, 0, sel, n_sel, top, pool, scale, dpos, T, o_lat, part, 0));
  CHECK(hy4_mla_out(o_lat, vb, gate, H, lat, vd, T, out, table, 0));
  const std::vector<float> sparse = down(out, (size_t) T * H * vd);
  match("o_proj's input (sparse attention)", sparse, "o_in", kTolAttn);

  // every group selected (the first 2,048 tokens' own selection is all of theirs): the dense path's bits
  const uint32_t dense_n = (top - 1) * pool;
  CHECK(hy4_mla_attn(qa, cache, 0, false, sinks, H, lat, rd, 0, nullptr, nullptr, 0, 0, scale, dpos, dense_n, o_lat, part, 0));
  CHECK(hy4_mla_out(o_lat, vb, gate, H, lat, vd, dense_n, out, table, 0));
  const std::vector<float> dense = down(out, (size_t) dense_n * H * vd);
  CHECK(std::memcmp(dense.data(), sparse.data(), dense.size() * sizeof(float)) == 0);
  std::printf("  the first %u tokens (every group theirs): the dense path's bits\n", dense_n);
  std::printf("[test_hy4_indexer_gpu] ok\n");
  return 0;
}
