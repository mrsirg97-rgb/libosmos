#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "substrate/mem/hip_pinner.h"

/* where a lock goes: the CPU's coarse-grained system pool (the GPU caches it as hipHostRegister's pages; a plain
   hsa_amd_memory_lock's are fine-grained, a decode step 4% slower, measured), mapped for the GPUs alone */
typedef struct Where
{
  hsa_agent_t gpu[8];
  int n_gpu;
  hsa_amd_memory_pool_t pool;
  bool has_pool;
} Where;

static Where where;

static hsa_status_t coarse(hsa_amd_memory_pool_t p, void *arg)
{
  Where *w = arg;
  hsa_amd_segment_t seg;
  uint32_t flags = 0;
  if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) == HSA_STATUS_SUCCESS &&
      seg == HSA_AMD_SEGMENT_GLOBAL &&
      hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) == HSA_STATUS_SUCCESS &&
      flags == HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED && !w->has_pool)
    w->pool = p, w->has_pool = true;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t agent(hsa_agent_t a, void *arg)
{
  Where *w = arg;
  hsa_device_type_t type;
  if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (type == HSA_DEVICE_TYPE_GPU && w->n_gpu < (int)(sizeof w->gpu / sizeof *w->gpu))
    w->gpu[w->n_gpu++] = a;
  if (type == HSA_DEVICE_TYPE_CPU)
    (void)hsa_amd_agent_iterate_memory_pools(a, coarse, w);
  return HSA_STATUS_SUCCESS;
}

/* HSA's lock and unlock, what hipHostRegister and hipHostUnregister are made of, without the unregister's wait for
   every stream: the pool unpins only a unit no hold is on, and its holder releases it once the GPU is past it */
static bool pin(void *host, size_t bytes, void *ctx)
{
  const Where *w = ctx;
  void *dev = NULL;
  if (!w->has_pool || !w->n_gpu ||
      hsa_amd_memory_lock_to_pool(host, bytes, (hsa_agent_t *)w->gpu, w->n_gpu, w->pool, 0, &dev) != HSA_STATUS_SUCCESS)
    return false;
  if (dev != host)
  {
    (void)hsa_amd_memory_unlock(host);
    return false;
  }
  return true;
}

static void unpin(void *host, void *ctx)
{
  (void)ctx;
  (void)hsa_amd_memory_unlock(host);
}

Pinner hip_pinner(void)
{
  /* the runtime up (HSA under it), then the agents and the pool once: a pinner without them refuses every pin */
  if (!where.n_gpu && hipInit(0) == hipSuccess)
    (void)hsa_iterate_agents(agent, &where);
  return (Pinner){pin, unpin, &where};
}
