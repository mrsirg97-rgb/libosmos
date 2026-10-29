#ifndef HIP_PINNER_H
#define HIP_PINNER_H

#include "substrate/mem/pool.h"

Pinner hip_pinner(void);

#endif

/*
hip_pinner
  the pool's Pinner on this APU: hipHostRegister (mapped, read-only) and hipHostUnregister. a pin is refused - false,
  nothing left registered - unless the device's pointer for the range is the host's own address: the pool hands the
  GPU the mapping's addresses, which on gfx1151 is what registration gives (measured, DESIGN.md "the memory"); a
  device where it is not fails closed instead of reading the wrong bytes.
*/
