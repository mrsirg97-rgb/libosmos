#ifndef GPU_H
#define GPU_H

#include <stdbool.h>

bool gpu_sleep_on_waits(void);

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
*/
