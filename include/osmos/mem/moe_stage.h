#ifndef OSMOS_MEM_MOE_STAGE_H
#define OSMOS_MEM_MOE_STAGE_H
/* compact host->GPU routing table staging (one shape across DS4, Hy4, Q4x, Ink2): carves 16-byte-aligned slices out of
   a paired host buffer and device buffer so a layer's routing tables upload in one async H2D copy of only the step's
   live bytes */
#include <stddef.h>
#include <stdint.h>

#define MOE_STAGE_ALIGN 16ull

typedef struct MoeStage
{
  uint8_t *host; /* host staging base (pinned or cached build buffer) */
  uint8_t *dev;  /* matching device staging base */
  uint64_t used, cap;
} MoeStage;

static inline uint64_t moe_stage_align(uint64_t n)
{
  return (n + MOE_STAGE_ALIGN - 1u) & ~(MOE_STAGE_ALIGN - 1u);
}

static inline void *moe_stage_carve(MoeStage *s, uint64_t bytes, void **dev_out)
{
  const uint64_t at = moe_stage_align(s->used);
  if (dev_out)
    *dev_out = NULL;
  if (bytes > s->cap || at > s->cap - bytes)
    return NULL;
  if (dev_out)
    *dev_out = s->dev ? s->dev + at : NULL;
  s->used = at + bytes;
  return s->host + at;
}

#endif
