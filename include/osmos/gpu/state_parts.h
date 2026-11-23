#ifndef OSMOS_GPU_STATE_PARTS_H
#define OSMOS_GPU_STATE_PARTS_H

/* a sequence's saved state as a family lays it (osmos/model/state_io.h's pieces): its header, then its parts - device
   ranges, in the family's order - then a 64-bit FNV-1a of every byte before it. a family keeps its header and its
   parts' list; the moving and the sum are here (q4x's, q35's) */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/state_io.h"

typedef struct StatePart
{
  void *dev;
  uint64_t bytes;
} StatePart;

/* the string's bytes: the header, the parts, the sum */
uint64_t state_parts_bytes(uint64_t head_bytes, const StatePart *p, int n);
/* the header, the parts (each through a STATE_CHUNK host bounce), the sum, to the sink: NULL, or which step failed.
   the device's writes to the parts done first (hipDeviceSynchronize) */
const char *state_parts_save(const void *head, uint64_t head_bytes, const StatePart *p, int n, StateSink sink,
                             void *ctx);
/* the parts and the sum from the source, after the caller read and checked its header (head: those bytes, the sum's
   start): NULL, or which step failed - a source or copy, or the sum not the bytes' (the parts then part written: the
   caller empties the sequence) */
const char *state_parts_load(const void *head, uint64_t head_bytes, const StatePart *p, int n, StateSource src,
                             void *ctx);
/* a mark: the parts copied whole into (out) or from (in) a host buffer, in order (a part of no bytes skipped) - NULL,
   or the failed copy. its header and any sum the caller's */
const char *state_parts_to_host(const StatePart *p, int n, uint8_t *out);
const char *state_parts_from_host(const StatePart *p, int n, const uint8_t *in);

/* a loaded header's checks, every family's - each a rule and its words one place (0.16.0: seven copies; Qwen loaded
   any token in its history, Hy4 refused one; the 27B sized its parts from a header before checking it). each NULL, or
   why: the format and model (magic, version, identity); a position within the context; a history's tokens within the
   vocabulary (the first n of t). run before anything is sized from the header */
const char *state_head_check(uint64_t magic, uint64_t want_magic, uint32_t version, uint32_t want_version,
                             uint64_t identity, uint64_t want_identity);
const char *state_pos_check(int64_t pos, uint32_t max_ctx);
const char *state_tokens_check(const int32_t *t, int64_t n, uint32_t vocab);

#endif
