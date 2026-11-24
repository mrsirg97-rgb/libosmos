#include "ds4_impl.h"

/* the host's wait for the GPU, one place for all of the forward's (DESIGN.md). a decode-sized step - its routing
   pairs fewer than a layer's experts, the lane's small step - spins on the event: its waits are short and many, a
   sleep's wake longer than one; a prompt's step sleeps - its waits are long, the core is freed. the sleep is the
   runtime's blocking sync (gpu_sleep_on_waits: the process's, set before its first kernel); without it the runtime
   spins there too */
static bool spun(Forward *f) { return (uint64_t)f->step_n * (uint64_t)f->g->n_used < (uint64_t)f->g->n_experts; }

bool fwd_wait(Forward *f, hipEvent_t ev)
{
  if (!spun(f))
    return hipEventSynchronize(ev) == hipSuccess;
  hipError_t e;
  while ((e = hipEventQuery(ev)) == hipErrorNotReady)
    ;
  return e == hipSuccess;
}

bool fwd_settle(Forward *f) { return hipEventRecord(f->passed, NULL) == hipSuccess && fwd_wait(f, f->passed); }
