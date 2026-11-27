#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "osmos/run/draft.h"

static void check(bool ok, const char *what)
{
  if (!ok)
  {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
  }
}

int main(void)
{
  printf("[test_draft] start\n");
  float cum = 1.0f;
  check(draft_accept(0.8f, 0.5f, 0.5f, &cum) && fabsf(cum - 0.8f) < 1e-6f, "accept first token");
  check(draft_accept(0.7f, 0.5f, 0.5f, &cum) && fabsf(cum - 0.56f) < 1e-6f, "accept second token");
  check(!draft_accept(0.8f, 0.5f, 0.5f, &cum) && fabsf(cum - 0.56f) < 1e-6f, "reject when cumulative drops below floor");

  cum = 1.0f;
  check(!draft_accept(0.49f, 0.5f, 0.1f, &cum) && cum == 1.0f, "reject when per-token conf below floor");
  check(draft_accept(0.5f, 0.5f, 0.5f, &cum) && cum == 0.5f, "accept exact boundary");

  /* NaN and -Inf must fail closed */
  cum = 1.0f;
  check(!draft_accept(NAN, 0.5f, 0.5f, &cum) && cum == 1.0f, "reject NaN p");
  check(!draft_accept(0.9f, NAN, 0.5f, &cum) && cum == 1.0f, "reject NaN conf_min");
  check(!draft_accept(0.9f, 0.5f, NAN, &cum) && cum == 1.0f, "reject NaN cum_min");
  check(!draft_accept(-INFINITY, 0.5f, 0.5f, &cum) && cum == 1.0f, "reject -Inf p");
  cum = NAN;
  check(!draft_accept(0.9f, 0.5f, 0.5f, &cum), "reject NaN cum");

  printf("[test_draft] ok\n");
  return 0;
}
