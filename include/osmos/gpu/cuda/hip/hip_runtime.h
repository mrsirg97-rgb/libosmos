/* the CUDA build's <hip/hip_runtime.h>: the runtime's names (hip_runtime_api.h) and, in device code, HIP's warp
   functions by CUDA's (all 32 lanes: libosmos's waves are whole) */
#ifndef OSMOS_CUDA_HIP_RUNTIME_H
#define OSMOS_CUDA_HIP_RUNTIME_H

#include "hip_runtime_api.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda_fp16.h> /* (before the macros: its own __shfl overloads are declarations, not calls) */
#define OSMOS_FULL_WARP 0xFFFFFFFFu
#define __shfl_xor(v, m, ...) __shfl_xor_sync(OSMOS_FULL_WARP, (v), (m), ##__VA_ARGS__)
#define __shfl(v, s, ...) __shfl_sync(OSMOS_FULL_WARP, (v), (s), ##__VA_ARGS__)
#define __ballot(p) ((unsigned long long) __ballot_sync(OSMOS_FULL_WARP, (p)))
#define __all(p) __all_sync(OSMOS_FULL_WARP, (p))
#define __any(p) __any_sync(OSMOS_FULL_WARP, (p))
#endif

#endif
