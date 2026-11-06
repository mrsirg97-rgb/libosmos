#ifndef OSMOS_MODEL_FNV_H
#define OSMOS_MODEL_FNV_H

#include <stdint.h>

/* 64-bit FNV-1a: a saved state's checksum, an identity's key (the families' seq states) */
#define FNV_SEED 1469598103934665603ull
#define FNV_PRIME 1099511628211ull

static inline uint64_t fnv(const void *p, uint64_t n, uint64_t h)
{
  const uint8_t *b = (const uint8_t *)p;
  for (uint64_t i = 0; i < n; i++)
    h = (h ^ b[i]) * FNV_PRIME;
  return h;
}

static inline uint64_t fnv_u64(uint64_t v, uint64_t h) { return fnv(&v, sizeof v, h); }

#endif
