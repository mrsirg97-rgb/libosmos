// tools/bench/bench_common.h - what the bench tools share, once
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

// the q-quantile of sorted values as numpy's np.quantile (its default, "linear") gives it - kld_compare.py's - so a
// run's median and p99 are the same digits whichever tool printed them
static inline double bench_quantile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return NAN;
  const double at = q * (double) (sorted.size() - 1);
  const size_t lo = (size_t) std::floor(at), hi = lo + 1 < sorted.size() ? lo + 1 : lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - (double) lo);
}
