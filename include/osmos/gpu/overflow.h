#ifndef OSMOS_GPU_OVERFLOW_H
#define OSMOS_GPU_OVERFLOW_H
/* a step's overflow flag (an activation past f16's range, set by a kernel), one shape every family's step shares:
   reset after a step that set it; a reset that fails owed - its cause said - and tried again at the next step's start,
   which refuses with that cause while it fails (an unchecked reset left the flag set: every later step refused as
   past f16's range, its cause misstated) */
#include <stdbool.h>
#include <stddef.h>

#include "osmos/err.h"

#define OVERFLOW_RESET "the overflow flag's reset" /* its GPU call's name (osmos_hip: the fault seam's too) */

typedef struct OverflowFlag
{
  void *dev;    /* the flag's device bytes (the family's own: a forward's, a drafter's) */
  size_t bytes; /* its size: 4, or 8 with a second flag beside it (Hy4's router fault) */
  bool owed;    /* a reset failed: the next step's start tries it again */
} OverflowFlag;

/* the flag zeroed; false with the reset's cause (then owed) */
bool overflow_reset(OverflowFlag *o, char err[OSMOS_ERR]);
/* a step's start: an owed reset tried again (none owed: true, no GPU call); false with its cause while it fails */
bool overflow_settle(OverflowFlag *o, char err[OSMOS_ERR]);

#endif
