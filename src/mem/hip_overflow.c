/* a step's overflow flag (osmos/gpu/overflow.h) */
#include "osmos/gpu/overflow.h"

#include <hip/hip_runtime_api.h>

#include "osmos/err_hip.h"

bool overflow_reset(OverflowFlag *o, char err[OSMOS_ERR])
{
  o->owed = !osmos_hip(hipMemset(o->dev, 0, o->bytes), err, OVERFLOW_RESET);
  return !o->owed;
}

bool overflow_settle(OverflowFlag *o, char err[OSMOS_ERR]) { return !o->owed || overflow_reset(o, err); }
