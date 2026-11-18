// Hy4-flash's small steps (include/osmos/gpu/hy4_ops.h): step 1's schema, refused until built
#include "port.h"

extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

extern "C" bool hy4_hc_pre(const float*, const float*, const float*, const float*, uint32_t, uint32_t, uint32_t, float,
                           float, float, const float*, uint32_t, float*, float*, void*) {
  return false;
}
extern "C" bool hy4_hc_post(float*, const float*, const float*, uint32_t, uint32_t, uint32_t, void*) { return false; }
extern "C" bool hy4_engram_rows(const float*, const float*, uint32_t, uint32_t, uint32_t, float*, void*) {
  return false;
}
extern "C" bool hy4_engram_update(float*, const float*, const float*, const float*, const float*, const float*,
                                  uint32_t, uint32_t, uint32_t, void*) {
  return false;
}
