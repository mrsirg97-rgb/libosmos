#ifndef OSMOS_STATE_IO_H
#define OSMOS_STATE_IO_H

#include <stdbool.h>
#include <stdint.h>

/* a sequence's saved state moved in pieces, in order: a save gives them to a sink, a load takes them from a source
   (either false stops it). a piece is at most STATE_CHUNK bytes - a state of any size through that much host memory:
   the smallest piece the SSD takes at its full rate */
#define STATE_CHUNK (4ull << 20)

typedef bool (*StateSink)(void *ctx, const void *bytes, uint64_t n);
typedef bool (*StateSource)(void *ctx, void *bytes, uint64_t n);

#endif
