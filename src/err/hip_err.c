/* a GPU call's result in the error shape (osmos/err_hip.h) */
#include "osmos/err_hip.h"

__attribute__((weak)) bool osmos_hip_fault(const char *what)
{
  (void)what;
  return false;
}

bool osmos_hip(hipError_t e, char err[OSMOS_ERR], const char *what)
{
  if (e == hipSuccess && !osmos_hip_fault(what))
    return true;
  return osmos_fail(err, "%s: %s", what, e == hipSuccess ? "failed (the fault seam's)" : hipGetErrorString(e));
}
