#ifndef OSMOS_ERR_HIP_H
#define OSMOS_ERR_HIP_H
/* a GPU call's result in the error shape (osmos/err.h): every HIP call whose failure leaves state the code then relies
   on - a memset (the overflow flag's reset), a copy, a sync before memory is released or a result read - checked
   through osmos_hip, so a failure is false with its cause; a destroy's frees are exempt (nothing left to refuse) */
#include <hip/hip_runtime_api.h>

#include "osmos/err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* true when e is hipSuccess (and the fault seam names no failure here), else osmos_fail("<what>: <hip's reason>") */
bool osmos_hip(hipError_t e, char err[OSMOS_ERR], const char *what);

/* the fault seam: whether the call named `what` is to fail - false in the library (a weak definition); a test links
   its own to fail one call, so a failed GPU call's path is tested, not assumed (nothing installed at run time). it
   fails the call's report, after the call ran: the path a failure takes, not the device state a real one leaves */
bool osmos_hip_fault(const char *what);

#ifdef __cplusplus
}
#endif

#endif
