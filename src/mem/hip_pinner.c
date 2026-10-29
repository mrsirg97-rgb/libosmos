#include <hip/hip_runtime_api.h>

#include "substrate/mem/hip_pinner.h"

static bool pin(void *host, size_t bytes, void *ctx)
{
  (void)ctx;
  if (hipHostRegister(host, bytes, hipHostRegisterMapped | hipHostRegisterReadOnly) != hipSuccess)
  {
    (void)hipGetLastError();
    return false;
  }
  void *dev = NULL;
  if (hipHostGetDevicePointer(&dev, host, 0) != hipSuccess || dev != host)
  {
    (void)hipGetLastError();
    (void)hipHostUnregister(host);
    return false;
  }
  return true;
}

static void unpin(void *host, void *ctx)
{
  (void)ctx;
  (void)hipHostUnregister(host);
}

Pinner hip_pinner(void) { return (Pinner){pin, unpin, NULL}; }
