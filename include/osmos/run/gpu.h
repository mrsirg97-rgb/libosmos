#ifndef GPU_H
#define GPU_H

#include <stdbool.h>

bool gpu_sleep_on_waits(void);
bool gpu_alive(void);

typedef struct GpuWait GpuWait;

GpuWait *gpu_wait_create(void);
void gpu_wait_destroy(GpuWait *w);
bool gpu_wait_mark(GpuWait *w);
bool gpu_wait_settle(GpuWait *w);

#endif

/*
gpu_sleep_on_waits
  the process's waits for the GPU sleep (DESIGN.md, "0.3.0: energy a token", target 2): the device scheduled
  blocking-sync, so hipEventSynchronize, hipStreamSynchronize and hipDeviceSynchronize put the waiting thread to sleep
  until the GPU passes them, the driver waking it, instead of spinning a core the GPU shares its package's power with
  (measured: a 300 ms wait 300 ms of a core spinning, 7 ms asleep; an event's own blocking-sync flag is not honoured on
  ROCm 7.15 - the device's is). a process calls it first, before its first kernel: it takes after an allocation, not
  after a kernel has run (measured) - the forward's creation runs kernels, so osmos-serve and dbench call it at the
  top of main. false: the runtime refused.

gpu_alive
  the GPU still usable after a failed call: every stream's work done, the device reporting no error - false after a
  sticky one (a kernel's fault: every later call fails the same way, so the process can only exit). not
  hipGetLastError's, whose last error may be a refused launch the caller has already reported.

GpuWait (DESIGN.md, "0.6.2: Qwen's waits")
  a thread's wait for the null stream's queued work, its wake hidden under the work's own tail: a sleeping wait's wake
  comes ~0.2-0.3 ms after the GPU is done (measured), the GPU idle meanwhile. gpu_wait_mark: a mark queued now - what
  is queued after it is the tail. gpu_wait_settle: returns once everything queued on the null stream is done - asleep
  (the device's blocking sync) until the mark is passed, then polling to the end; with no mark since the last settle,
  asleep to the end. a tail shorter than the wake: the idle what is left of the wake, never more than a sleep's.
  false: a GPU call failed (the work's own error among them). create NULL where the runtime refuses its events.
*/
