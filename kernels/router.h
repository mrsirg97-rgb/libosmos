#pragma once
// a token's routed picks (ops_router_scored, and ink2's moe_prep): its scores over the block's threads, then the k
// greatest of score + bias by the block's first wave alone (block.h's wave_pick: the larger, ties to the lower
// expert - the serial scan's picks), their weights - shared by every caller, so a token's picks the same bits however
// it is launched. every thread of the block calls it (its barrier); sc and keys the block's LDS (n_exp floats each)
#include <cstdint>

#include "block.h"
extern "C" {
#include "osmos/gpu/ops.h"
}

namespace router {
constexpr float kRouteEps = 1e-20f;
constexpr uint32_t kLane = OPS_MAX_EXPERTS / 32;  // a lane's experts at most
template <int Score>
__device__ inline void token(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float scale,
                             uint32_t t, uint32_t threads, float* sc, float* keys, int32_t* idx, float* w,
                             int32_t* idx_lds) {
  for (uint32_t e = threadIdx.x; e < n_exp; e += threads) {
    const float l = logits[(uint64_t) t * n_exp + e];
    sc[e] = Score == OPS_SCORE_SIGMOID ? sigmoid(l) : sqrtf(softplus(l));
    keys[e] = pick_key(sc[e] + bias[e]);
  }
  __syncthreads();
  if (threadIdx.x < 32u) {
    const uint32_t lane = threadIdx.x;
    float key[kLane];
    for (uint32_t r = 0; r < kLane; r++) key[r] = lane + 32 * r < n_exp ? keys[lane + 32 * r] : 0.0f;
    uint32_t taken = 0;
    int32_t pick[OPS_MAX_TOPK];
    for (uint32_t s = 0; s < k; s++) {
      float best;
      pick[s] = wave_pick<kLane>(key, n_exp, &taken, &best);
    }
    if (!lane) {
      float wt[OPS_MAX_TOPK], sum = 0;
      for (uint32_t s = 0; s < k; s++) sum += (wt[s] = (uint32_t) pick[s] < n_exp ? sc[pick[s]] : 0.0f);
      for (uint32_t s = 0; s < k; s++) {
        if (idx) idx[t * k + s] = pick[s], w[t * k + s] = wt[s] / (sum + kRouteEps) * scale;
        if (idx_lds) idx_lds[t * k + s] = pick[s];
      }
    }
  }
}

// the same picks by one wave alone, its token's: a lane's experts' scores and keys in its registers (lane + 32 r -
// the layout wave_pick reads), a pick's score fetched from its expert's lane - every value token()'s expression, so
// the same bits; a block's waves each a token at once (no barrier: the wave's registers and shuffles alone)
template <int Score, int R>
__device__ inline void token_wave(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float scale,
                                  uint32_t t, int32_t* idx, float* w, int32_t* idx_lds) {
  const uint32_t lane = threadIdx.x % 32;
  float sc[R], key[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const uint32_t e = lane + 32u * (uint32_t) r;
    const float l = e < n_exp ? logits[(uint64_t) t * n_exp + e] : 0.0f;
    sc[r] = Score == OPS_SCORE_SIGMOID ? sigmoid(l) : sqrtf(softplus(l));
    key[r] = e < n_exp ? pick_key(sc[r] + bias[e]) : 0.0f;
  }
  uint32_t taken = 0;
  int32_t pick[OPS_MAX_TOPK];
  float wt[OPS_MAX_TOPK];
  for (uint32_t s = 0; s < k; s++) {
    float best;
    pick[s] = wave_pick<R>(key, n_exp, &taken, &best);
    const uint32_t p = (uint32_t) pick[s] < n_exp ? (uint32_t) pick[s] : 0u;
    float mine = 0.0f;
#pragma unroll
    for (int r = 0; r < R; r++)
      if ((uint32_t) r == p / 32) mine = sc[r];
    const float v = __shfl(mine, (int) (p % 32), 32);
    wt[s] = (uint32_t) pick[s] < n_exp ? v : 0.0f;
  }
  if (lane) return;
  float sum = 0;
  for (uint32_t s = 0; s < k; s++) sum += wt[s];
  for (uint32_t s = 0; s < k; s++) {
    if (idx) idx[t * k + s] = pick[s], w[t * k + s] = wt[s] / (sum + kRouteEps) * scale;
    if (idx_lds) idx_lds[t * k + s] = pick[s];
  }
}
}  // namespace router
