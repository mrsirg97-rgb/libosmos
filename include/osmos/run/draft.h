#ifndef OSMOS_RUN_DRAFT_H
#define OSMOS_RUN_DRAFT_H
/* shared speculative-drafting helpers (one shape across every drafter family: DS4, Hy4, Ink2, Q4x/LFM3 DFlash,
   Q4x MTP) */
#include <stdbool.h>

/* confidence gate for the next draft token: updates *cum and returns true iff p >= conf_min and (*cum * p) >= cum_min.
   negated >= fails closed on NaN (either p, *cum, conf_min or cum_min) rather than silently accepting NaN drafts */
static inline bool draft_accept(float p, float conf_min, float cum_min, float *cum)
{
  const float next = *cum * p;
  if (!(p >= conf_min && next >= cum_min))
    return false;
  *cum = next;
  return true;
}

#endif
