#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "substrate/mem/hip_slots.h"

/* the GPU's coarse-grained global pool, and the CPU agent its memory is opened to */
typedef struct Where
{
  hsa_amd_memory_pool_t pool;
  hsa_agent_t cpu;
  bool has_pool, has_cpu;
} Where;

static Where where;

static hsa_status_t coarse(hsa_amd_memory_pool_t p, void *arg)
{
  Where *w = arg;
  hsa_amd_segment_t seg;
  uint32_t flags = 0;
  if (!w->has_pool &&
      hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) == HSA_STATUS_SUCCESS &&
      seg == HSA_AMD_SEGMENT_GLOBAL &&
      hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) == HSA_STATUS_SUCCESS &&
      flags == HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
    w->pool = p, w->has_pool = true;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t agent(hsa_agent_t a, void *arg)
{
  Where *w = arg;
  hsa_device_type_t type;
  if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (type == HSA_DEVICE_TYPE_GPU && !w->has_pool) /* the first GPU's: the forward runs on device 0 */
    (void)hsa_amd_agent_iterate_memory_pools(a, coarse, w);
  if (type == HSA_DEVICE_TYPE_CPU && !w->has_cpu)
    w->cpu = a, w->has_cpu = true;
  return HSA_STATUS_SUCCESS;
}

static void *alloc(uint64_t bytes, void *ctx)
{
  const Where *w = ctx;
  void *dev = NULL;
  if (!w->has_pool || !w->has_cpu || hsa_amd_memory_pool_allocate(w->pool, bytes, 0, &dev) != HSA_STATUS_SUCCESS)
    return NULL;
  if (hsa_amd_agents_allow_access(1, &w->cpu, NULL, dev) != HSA_STATUS_SUCCESS)
  {
    (void)hsa_amd_memory_pool_free(dev);
    return NULL;
  }
  return dev;
}

static void release(void *dev, void *ctx)
{
  (void)ctx;
  (void)hsa_amd_memory_pool_free(dev);
}

Slots hip_slots(void)
{
  /* the runtime up (HSA under it), then the pool and the CPU once: without them every alloc is refused */
  if (!where.has_pool && hipInit(0) == hipSuccess)
    (void)hsa_iterate_agents(agent, &where);
  return (Slots){alloc, release, &where};
}
