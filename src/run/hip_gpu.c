#include <hip/hip_runtime_api.h>
#include <stdlib.h>

#include "osmos/run/gpu.h"

bool gpu_sleep_on_waits(void) { return hipSetDeviceFlags(hipDeviceScheduleBlockingSync) == hipSuccess; }

bool gpu_alive(void) { return hipDeviceSynchronize() == hipSuccess; }

struct GpuWait
{
  hipEvent_t mark, end;
  bool marked; /* a mark queued since the last settle */
};

GpuWait *gpu_wait_create(void)
{
  GpuWait *w = calloc(1, sizeof *w);
  if (w && (hipEventCreateWithFlags(&w->mark, hipEventDisableTiming) != hipSuccess ||
            hipEventCreateWithFlags(&w->end, hipEventDisableTiming) != hipSuccess))
    gpu_wait_destroy(w), w = NULL;
  return w;
}

void gpu_wait_destroy(GpuWait *w)
{
  if (!w)
    return;
  if (w->mark)
    (void)hipEventDestroy(w->mark);
  if (w->end)
    (void)hipEventDestroy(w->end);
  free(w);
}

/* the schema's stubs (0.6.2's tests fail on them): no mark, the stream's sleep - the wait as it was */
bool gpu_wait_mark(GpuWait *w) { return w != NULL; }

bool gpu_wait_settle(GpuWait *w) { return w && hipStreamSynchronize(NULL) == hipSuccess; }
