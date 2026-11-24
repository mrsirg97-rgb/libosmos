#ifndef GPU_H
#define GPU_H

#include <stdbool.h>

bool gpu_sleep_on_waits(void);
bool gpu_alive(void);

#endif

/*
gpu_sleep_on_waits
  the process's waits for the GPU sleep: the device scheduled blocking-sync, so hipEventSynchronize,
  hipStreamSynchronize and hipDeviceSynchronize put the waiting thread to sleep until the GPU passes them instead of
  spinning a core the GPU shares its package's power with (an event's own blocking-sync flag is not honoured - the
  device's is). call it before the process's first kernel: it takes after an allocation, not after a kernel has run.
  false: the runtime refused.

gpu_alive
  the GPU still usable after a failed call: every stream's work done, the device reporting no error - false after a
  sticky one (a kernel's fault: every later call fails the same way, so the process can only exit). not
  hipGetLastError's, whose last error may be a refused launch the caller has already reported.
*/
