/* an MoE's experts as the membrane's units (osmos/model/expert_units.h) */
#include "osmos/model/expert_units.h"

size_t expert_units(const ExpertTensors *layers, uint32_t n_layers, uint32_t experts, const int *fds, PoolUnit *units,
                    size_t cap)
{
  if (!layers || !n_layers || !experts)
    return 0;
  for (uint32_t l = 0; l < n_layers; l++) /* every layer's tensors first: nothing written for a model refused */
  {
    const ExpertTensors *x = &layers[l];
    const GgufTensor *t[POOL_PARTS] = {x->gate, x->fused ? x->gate : x->up, x->down};
    for (int k = 0; k < POOL_PARTS; k++)
      if (!t[k] || t[k]->bytes % experts || (x->fused && k < 2 && t[k]->bytes / experts % 2))
        return 0;
  }
  const size_t n = (size_t)n_layers * experts;
  for (size_t i = 0; units && fds && i < n && i < cap; i++)
  {
    const ExpertTensors *x = &layers[i / experts];
    const GgufTensor *t[POOL_PARTS] = {x->gate, x->fused ? x->gate : x->up, x->down};
    units[i] = (PoolUnit){.n_parts = POOL_PARTS};
    for (int k = 0; k < POOL_PARTS; k++)
    {
      const uint64_t slice = t[k]->bytes / experts, half = x->fused && k < 2 ? slice / 2 : 0;
      units[i].part[k] = (PoolRange){fds[t[k]->file], t[k]->offset + (i % experts) * slice + (k == 1 ? half : 0),
                                     half ? half : slice};
    }
  }
  return n;
}
