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
}  // namespace router
