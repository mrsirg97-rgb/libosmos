/* an MoE's experts as the membrane's units: every family's one slicing of its expert tensors */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/mem/pool.h"
#include "osmos/model/gguf.h"

/* a layer's expert tensors, each n_experts equal slices one after another; fused: gate holds each expert's gate then
   its up (Inkling's ffn_gate_up_exps: half a slice each), up unused */
typedef struct ExpertTensors
{
  const GgufTensor *gate, *up, *down;
  bool fused;
} ExpertTensors;

/*
expert_units
  every layer's experts as units (unit = layer * experts + expert): parts gate, up, down - each its slice of its
  tensor (fds[the tensor's file], its offset + expert * slice). units NULL: only checked. its units' number, or 0 -
  nothing written - for a layer's tensor missing, not experts equal slices, or a fused slice not halves.
*/
size_t expert_units(const ExpertTensors *layers, uint32_t n_layers, uint32_t experts, const int *fds, PoolUnit *units,
                    size_t cap);
