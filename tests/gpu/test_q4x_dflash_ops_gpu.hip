// DFlash2's drafter kernels (q4x_dflash.hip) against host references on random inputs: the rope (rotate-half), the
// grouped dynamic convolution (its sums' order: the bits), the ring's rows, the windowed attention over the ring and
// the block (GQA, not causal), the top-k (ties to the lower id, NaNs never), the selector's greedy path (its scores'
// sums in order: the same picks). needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off) /* the references' sums as written: their bits the kernels' */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t HD = Q4X_DF_HEAD;
static uint32_t seed = 1017;
static float rnd() { return (seed = seed * 1664525u + 1013904223u, (float) ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
static std::vector<float> rand_vec(size_t n) {
  std::vector<float> v(n);
  for (auto& x : v) x = rnd();
  return v;
}
template <typename T>
static T* dev_of(const std::vector<T>& h) {
  T* d;
  CHECK(hipMalloc(&d, h.size() * sizeof(T) + 4) == hipSuccess && hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
template <typename T>
static std::vector<T> host_of(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}
static double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) m = std::fmax(m, std::fabs((double) a[i] - b[i]) / (std::fabs((double) b[i]) + 1e-3));
  return m;
}

static void rope() {
  const uint32_t n = 8, heads = 4, stride = heads * HD + 64;  // a token's row wider than its heads: the stride kept
  std::vector<float> x = rand_vec(n * stride), want = x;
  std::vector<int32_t> pos(n);
  for (uint32_t t = 0; t < n; t++) pos[t] = 1000 + 37 * (int32_t) t;
  for (uint32_t t = 0; t < n; t++)
    for (uint32_t h = 0; h < heads; h++)
      for (uint32_t i = 0; i < HD / 2; i++) {
        float* v = &want[t * stride + h * HD];
        const float a = (float) pos[t] * (float) (1.0 / std::pow(10000.0, 2.0 * i / HD));
        const float x1 = v[i], x2 = v[i + HD / 2];
        v[i] = x1 * std::cos(a) - x2 * std::sin(a), v[i + HD / 2] = x2 * std::cos(a) + x1 * std::sin(a);
      }
  float* dx = dev_of(x);
  int32_t* dp = dev_of(pos);
  CHECK(q4x_df_rope(dx, n, heads, stride, dp, 10000.0f, nullptr));
  const std::vector<float> got = host_of(dx, x.size());
  const double e = max_rel(got, want);
  std::printf("  rope: max rel error %.2e (rotate-half, %u heads of %u, the stride's tail untouched)\n", e, heads, HD);
  CHECK(e < 1e-3);
  for (uint32_t t = 0; t < n; t++)
    for (uint32_t i = heads * HD; i < stride; i++) CHECK(got[t * stride + i] == x[t * stride + i]);
  hipFree(dx), hipFree(dp);
}

static void conv() {
  const uint32_t n = 8, dim = 256, group = 16, groups = dim / group, stride = 2 * 2 * groups;
  const std::vector<float> x = rand_vec(n * dim), base = rand_vec(2 * dim), dyn = rand_vec(n * stride);
  for (int which = 0; which < 2; which++) {  // prepare's half of the projection, then finish's
    std::vector<float> want(n * dim);
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t c = 0; c < dim; c++) {
        float z = 0;
        for (uint32_t off = 0; off < 2; off++) {
          const float v = t >= off ? x[(t - off) * dim + c] : 0.0f;
          z = z + base[off * dim + c] * v;
          z = z + dyn[t * stride + which * 2 * groups + off * groups + c / group] * v;
        }
        want[t * dim + c] = z;
      }
    float *dx = dev_of(x), *db = dev_of(base), *dd = dev_of(dyn), *dy;
    CHECK(hipMalloc(&dy, n * dim * 4) == hipSuccess);
    CHECK(q4x_df_conv(dx, n, dim, group, db, dd + which * 2 * groups, stride, dy, nullptr));
    CHECK(std::memcmp(host_of(dy, n * dim).data(), want.data(), n * dim * 4) == 0);
    hipFree(dx), hipFree(db), hipFree(dd), hipFree(dy);
  }
  std::printf("  conv: prepare's and finish's halves the reference's bits (its sums' order)\n");
}

static void ring_attn() {
  const uint32_t heads = 8, kvh = 2, ring = 64, window = 40, nq = 8, row = kvh * HD;
  const int32_t pos0 = 150, from = pos0 - (int32_t) ring;  // the ring full: positions 86 .. 149 in it
  // the context's keys and values, written into the ring at their positions (and some overwritten: wrapped)
  std::vector<float> kr(ring * row, NAN), vr(ring * row, NAN);
  float *dkr = dev_of(kr), *dvr = dev_of(vr);
  std::vector<float> kc = rand_vec((size_t) (pos0 - 50) * row), vc = rand_vec((size_t) (pos0 - 50) * row);
  float *dkc = dev_of(kc), *dvc = dev_of(vc);
  CHECK(q4x_df_ring(dkr, ring, dkc, pos0 - 50, row, 50, nullptr) && q4x_df_ring(dvr, ring, dvc, pos0 - 50, row, 50, nullptr));
  const std::vector<float> hk = host_of(dkr, ring * row), hv = host_of(dvr, ring * row);
  for (int32_t p = from; p < pos0; p++)
    CHECK(std::memcmp(&hk[(p % ring) * row], &kc[(size_t) (p - 50) * row], row * 4) == 0 &&
          std::memcmp(&hv[(p % ring) * row], &vc[(size_t) (p - 50) * row], row * 4) == 0);
  const std::vector<float> q = rand_vec(nq * heads * HD), kb = rand_vec(nq * row), vb = rand_vec(nq * row);
  std::vector<float> want(nq * heads * HD);
  for (uint32_t i = 0; i < nq; i++)
    for (uint32_t h = 0; h < heads; h++) {
      const uint32_t g = h / (heads / kvh);
      const int32_t qp = pos0 + (int32_t) i;
      std::vector<std::pair<float, const float*>> keys;  // score, value row
      auto add = [&](const float* k, const float* v) {
        float s = 0;
        for (uint32_t d = 0; d < HD; d++) s += q[(i * heads + h) * HD + d] * k[d];
        keys.push_back({s / std::sqrt((float) HD), v});
      };
      for (int32_t p = from; p < pos0; p++)
        if (std::abs(qp - p) < (int32_t) window) add(&hk[(p % ring) * row + g * HD], &hv[(p % ring) * row + g * HD]);
      for (uint32_t j = 0; j < nq; j++) add(&kb[j * row + g * HD], &vb[j * row + g * HD]);
      float mx = -INFINITY, sum = 0;
      for (auto& k : keys) mx = std::fmax(mx, k.first);
      for (auto& k : keys) sum += std::exp(k.first - mx);
      for (uint32_t d = 0; d < HD; d++) {
        float o = 0;
        for (auto& k : keys) o += std::exp(k.first - mx) / sum * k.second[d];
        want[(i * heads + h) * HD + d] = o;
      }
    }
  float *dq = dev_of(q), *dkb = dev_of(kb), *dvb = dev_of(vb), *dout;
  CHECK(hipMalloc(&dout, want.size() * 4) == hipSuccess);
  CHECK(q4x_df_attn(dq, nq, pos0, dkb, dvb, dkr, dvr, ring, from, heads, kvh, window, dout, nullptr));
  const double e = max_rel(host_of(dout, want.size()), want);
  std::printf("  ring and attention: the ring's rows at p %% %u (wrapped); attention max rel error %.2e (%u of %u heads,"
              " window %u, not causal)\n", ring, e, heads, kvh, window);
  CHECK(e < 1e-4);
  CHECK(!q4x_df_attn(dq, nq, pos0, dkb, dvb, dkr, dvr, ring, from - 1, heads, kvh, window, dout, nullptr));  // past the ring
  hipFree(dkr), hipFree(dvr), hipFree(dkc), hipFree(dvc), hipFree(dq), hipFree(dkb), hipFree(dvb), hipFree(dout);
}

static void topk() {
  const uint32_t rows = 5, vocab = 248320, k = 16;
  std::vector<float> lg = rand_vec(rows * vocab);
  for (uint32_t i = 0; i < 40; i++) lg[3 * vocab + 1000 + i * 3] = 7.0f;  // row 3: 40 ties at the top
  lg[2 * vocab + 5] = NAN, lg[2 * vocab + 6] = 9.0f;                      // row 2: a NaN beside the best
  float *dl = dev_of(lg), *dv;
  int32_t* di;
  CHECK(hipMalloc(&dv, rows * k * 4) == hipSuccess && hipMalloc(&di, rows * k * 4) == hipSuccess);
  CHECK(q4x_df_topk(dl, rows, vocab, k, dv, di, nullptr));
  const std::vector<float> v = host_of(dv, rows * k);
  const std::vector<int32_t> id = host_of(di, rows * k);
  for (uint32_t r = 0; r < rows; r++) {
    std::vector<int32_t> order;
    for (uint32_t i = 0; i < vocab; i++)
      if (!std::isnan(lg[r * vocab + i])) order.push_back((int32_t) i);
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
      const float x = lg[r * vocab + a], y = lg[r * vocab + b];
      return x > y || (x == y && a < b);
    });
    for (uint32_t j = 0; j < k; j++) CHECK(id[r * k + j] == order[j] && v[r * k + j] == lg[r * vocab + order[j]]);
  }
  CHECK(id[3 * k] == 1000 && id[3 * k + 15] == 1045 && id[2 * k] == 6);
  CHECK(!q4x_df_topk(dl, rows, vocab, Q4X_DF_TOPK + 1, dv, di, nullptr));
  std::printf("  top-k: %u rows of %u, the reference's ids and values (40 ties: the lower ids; a NaN never)\n", rows, vocab);
  hipFree(dl), hipFree(dv), hipFree(di);
}

// a Q8_0 row of `rank` values: rank / 32 blocks of an f16 scale and 32 int8
static void q8_row(uint8_t* out, uint32_t rank, std::vector<float>& deq) {
  for (uint32_t b = 0; b < rank / 32; b++) {
    const float d = 0.01f + 0.02f * (rnd() + 1.0f);
    const _Float16 h = (_Float16) d;
    std::memcpy(out + b * 34, &h, 2);
    for (uint32_t i = 0; i < 32; i++) {
      const int8_t q = (int8_t) (rnd() * 127.0f);
      out[b * 34 + 2 + i] = (uint8_t) q;
      deq.push_back((float) h * (float) q);
    }
  }
}

static void select() {
  const uint32_t rows = 7, k = 16, rank = 256, vocab = 4096, rb = rank / 32 * 34;
  std::vector<uint8_t> pred(vocab * rb), succ(vocab * rb);
  std::vector<float> dp, ds;
  for (uint32_t t = 0; t < vocab; t++) q8_row(&pred[t * rb], rank, dp), q8_row(&succ[t * rb], rank, ds);
  std::vector<float> val(rows * k), hp = rand_vec(rows * rank);
  std::vector<int32_t> idx(rows * k);
  for (uint32_t j = 0; j < rows; j++)
    for (uint32_t c = 0; c < k; c++) val[j * k + c] = 3.0f - 0.01f * (float) c, idx[j * k + c] = (int32_t) ((j * 131 + c * 17) % vocab);
  const int32_t anchor = 77;
  std::vector<int32_t> want(rows);
  int32_t prev = anchor;
  for (uint32_t j = 0; j < rows; j++) {
    float best = -INFINITY;
    int32_t pick = -1;
    for (uint32_t c = 0; c < k; c++) {
      float s = 0;
      for (uint32_t r = 0; r < rank; r++) s += dp[(size_t) prev * rank + r] * hp[j * rank + r] * ds[(size_t) idx[j * k + c] * rank + r];
      s = val[j * k + c] + s;
      if (s > best) best = s, pick = idx[j * k + c];
    }
    want[j] = prev = pick;
  }
  uint8_t *dpd = dev_of(pred), *dsc = dev_of(succ);
  float *dv = dev_of(val), *dh = dev_of(hp);
  int32_t *di = dev_of(idx), *dpath;
  CHECK(hipMalloc(&dpath, rows * 4) == hipSuccess);
  CHECK(q4x_df_select(dv, di, rows, k, dh, rank, dpd, dsc, anchor, dpath, nullptr));
  const std::vector<int32_t> got = host_of(dpath, rows);
  CHECK(std::memcmp(got.data(), want.data(), rows * 4) == 0);
  // the score's val counted: candidate 5 lifted past every rank sum, its token each row's pick
  std::vector<float> lift = val;
  for (uint32_t j = 0; j < rows; j++) lift[j * k + 5] += 1000.0f;
  CHECK(hipMemcpy(dv, lift.data(), lift.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q4x_df_select(dv, di, rows, k, dh, rank, dpd, dsc, anchor, dpath, nullptr));
  const std::vector<int32_t> lifted = host_of(dpath, rows);
  for (uint32_t j = 0; j < rows; j++) CHECK(lifted[j] == idx[j * k + 5]);
  // a tie: candidates 3 and 9 of row 0 the same val and the same successor row (so the same score), both lifted
  // past the rest - the lower candidate, 3, picked
  std::vector<uint8_t> tied = succ;
  std::memcpy(&tied[(size_t) idx[9] * rb], &tied[(size_t) idx[3] * rb], rb);
  lift = val, lift[3] = lift[9] = 1000.0f;
  CHECK(hipMemcpy(dv, lift.data(), lift.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dsc, tied.data(), tied.size(), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q4x_df_select(dv, di, 1, k, dh, rank, dpd, dsc, anchor, dpath, nullptr));
  CHECK(host_of(dpath, 1)[0] == idx[3]);
  CHECK(!q4x_df_select(dv, di, rows, Q4X_DF_TOPK + 1, dh, rank, dpd, dsc, anchor, dpath, nullptr));
  std::printf("  select: the greedy path through %u rows of %u candidates the reference's (%d %d %d ..); its val"
              " counted; a tie the lower candidate\n", rows, k, got[0], got[1], got[2]);
  hipFree(dpd), hipFree(dsc), hipFree(dv), hipFree(dh), hipFree(di), hipFree(dpath);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_dflash_ops_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_q4x_dflash_ops_gpu] skipped: no GPU\n"), 77;
  rope();
  conv();
  ring_attn();
  topk();
  select();
  std::printf("[test_q4x_dflash_ops_gpu] ok\n");
  return 0;
}
