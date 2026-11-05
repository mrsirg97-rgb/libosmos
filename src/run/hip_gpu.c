#include <hip/hip_runtime_api.h>

#include "osmos/run/gpu.h"

bool gpu_sleep_on_waits(void) { return hipSetDeviceFlags(hipDeviceScheduleBlockingSync) == hipSuccess; }
