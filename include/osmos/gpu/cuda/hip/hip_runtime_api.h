/* the CUDA build's <hip/hip_runtime_api.h> (DESIGN.md, "a second GPU: CUDA"): the HIP runtime's names libosmos calls,
   each CUDA's own one for one - on this include path only when GPU=cuda, so the host's hip_*.c are one source */
#ifndef OSMOS_CUDA_HIP_RUNTIME_API_H
#define OSMOS_CUDA_HIP_RUNTIME_API_H

#include <cuda_runtime_api.h>

typedef cudaError_t hipError_t;
typedef cudaStream_t hipStream_t;
typedef cudaEvent_t hipEvent_t;
#define hipSuccess cudaSuccess
#define hipErrorNotReady cudaErrorNotReady
#define hipGetErrorString cudaGetErrorString
#define hipGetLastError cudaGetLastError

#define hipMalloc cudaMalloc
#define hipGetDevice cudaGetDevice
#define hipFree cudaFree
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemGetInfo cudaMemGetInfo
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetDeviceCount cudaGetDeviceCount
#define hipMemcpyFromSymbol cudaMemcpyFromSymbol
#define hipMemcpyToSymbol cudaMemcpyToSymbol
#define HIP_SYMBOL(s) s

/* pinned host memory: CUDA's is coherent already; mapped the same flag */
#define hipHostMalloc(p, n, flags) cudaHostAlloc((p), (n), (flags))
#define hipHostMallocDefault cudaHostAllocDefault
#define hipHostMallocMapped cudaHostAllocMapped
#define hipHostMallocCoherent 0u
#define hipHostFree cudaFreeHost
#define hipHostRegister cudaHostRegister
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipHostUnregister cudaHostUnregister
#define hipHostGetDevicePointer cudaHostGetDevicePointer

#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamWaitEvent cudaStreamWaitEvent
#define hipEventCreate cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventQuery cudaEventQuery
#define hipEventElapsedTime cudaEventElapsedTime
#define hipLaunchKernel cudaLaunchKernel

#define hipSetDeviceFlags cudaSetDeviceFlags
#define hipDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync
#define hipInit(flags) cudaFree(0) /* (the runtime's context made: CUDA starts it on first use) */

/* the device clock's rate: port.h's osmos_wall_clock is %globaltimer, nanoseconds - 1,000,000 kHz */
typedef enum
{
  hipDeviceAttributeWallClockRate,
  hipDeviceAttributeComputeCapabilityMajor,
  hipDeviceAttributeComputeCapabilityMinor,
  hipDeviceAttributeMultiprocessorCount
} hipDeviceAttribute_t;
static inline hipError_t hipDeviceGetAttribute(int *v, hipDeviceAttribute_t a, int dev)
{ /* a compute capability CUDA's own */
  if (a == hipDeviceAttributeComputeCapabilityMajor)
    return cudaDeviceGetAttribute(v, cudaDevAttrComputeCapabilityMajor, dev);
  if (a == hipDeviceAttributeComputeCapabilityMinor)
    return cudaDeviceGetAttribute(v, cudaDevAttrComputeCapabilityMinor, dev);
  if (a == hipDeviceAttributeMultiprocessorCount)
    return cudaDeviceGetAttribute(v, cudaDevAttrMultiProcessorCount, dev);
  if (a != hipDeviceAttributeWallClockRate) return cudaErrorInvalidValue;
  (void)dev;
  *v = 1000000;
  return cudaSuccess;
}

#endif
