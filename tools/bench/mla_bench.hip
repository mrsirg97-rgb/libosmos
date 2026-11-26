// hy4_mla_attn alone at a decode's two shapes (Hy4-flash: a sliding layer's 64 heads over a 512 window in a ring; a
// sparse layer's 32 over `keys` dense keys), timed over many launches: mla_bench [keys]
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attn_core_call.h"
extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

static float* fill(size_t n) {
  std::vector<float> h(n);
  for (size_t i = 0; i < n; i++) h[i] = (float) ((i * 2654435761u) % 2001) / 1000.0f - 1.0f;
  float* d = nullptr;
  if (hipMalloc(&d, n * 4) != hipSuccess || hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice) != hipSuccess) std::exit(1);
  return d;
}

int main(int argc, char** argv) {
  const uint32_t keys = argc > 1 ? (uint32_t) atoi(argv[1]) : 1024, lat = 512, rope = 64, row = lat + rope, iters = 200;
  struct Shape {
    const char* name;
    uint32_t heads, window, slots;
  } shapes[] = {{"sliding (64 heads, window 512, ring)", 64, 512, 512 + 8}, {"sparse (32 heads, dense keys)", 32, 0, 0}};
  for (const Shape& s : shapes) {
    const uint32_t ctx = s.slots ? s.slots : keys;
    float *qa = fill((size_t) s.heads * row), *cache = fill((size_t) ctx * row), *sinks = fill(s.heads);
    float *o = fill((size_t) s.heads * lat), *scratch = fill(1);
    int32_t* pos = nullptr;
    const int32_t p = (int32_t) (s.slots ? 5000 : keys - 1);
    if (hipMalloc(&pos, 4) != hipSuccess || hipMemcpy(pos, &p, 4, hipMemcpyHostToDevice) != hipSuccess) return 1;
    hipEvent_t a, b;
    if (hipEventCreate(&a) != hipSuccess || hipEventCreate(&b) != hipSuccess) return 1;
    bool ok = true;
    for (uint32_t i = 0; i < 10 && ok; i++)
      ok = core_hy4_mla_attn(qa, cache, s.slots, false, sinks, s.heads, lat, rope, s.window, nullptr, nullptr, 0, 0, 0.05f, pos, 1, o,
                        scratch, false, nullptr);
    if (hipEventRecord(a, nullptr) != hipSuccess) return 1;
    for (uint32_t i = 0; i < iters && ok; i++)
      ok = core_hy4_mla_attn(qa, cache, s.slots, false, sinks, s.heads, lat, rope, s.window, nullptr, nullptr, 0, 0, 0.05f, pos, 1, o,
                        scratch, false, nullptr);
    float ms = 0;
    if (!ok || hipEventRecord(b, nullptr) != hipSuccess || hipEventSynchronize(b) != hipSuccess ||
        hipEventElapsedTime(&ms, a, b) != hipSuccess)
      return std::printf("refused\n"), 1;
    std::printf("%-38s %4u keys: %7.1f us a call\n", s.name, s.window ? s.window : keys, ms * 1000 / iters);
  }
  return 0;
}
