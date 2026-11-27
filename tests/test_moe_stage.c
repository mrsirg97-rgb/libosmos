#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "osmos/mem/moe_stage.h"

static void check(int ok, const char *what)
{
  if (!ok)
  {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
  }
}

int main(void)
{
  printf("[test_moe_stage] start\n");
  uint8_t hbuf[128], dbuf[128];
  MoeStage s = {hbuf, dbuf, 0, sizeof hbuf};
  void *d0 = NULL, *d1 = NULL, *d2 = NULL;
  void *h0 = moe_stage_carve(&s, 5, &d0);
  check(h0 == hbuf && d0 == dbuf && s.used == 5, "first carve at 0");
  void *h1 = moe_stage_carve(&s, 20, &d1);
  check(h1 == hbuf + 16 && d1 == dbuf + 16 && s.used == 36, "second carve 16-byte aligned");
  void *h2 = moe_stage_carve(&s, 80, &d2);
  check(h2 == hbuf + 48 && d2 == dbuf + 48 && s.used == 128, "exact capacity boundary");
  void *d3 = (void *)1;
  check(moe_stage_carve(&s, 1, &d3) == NULL && s.used == 128 && d3 == NULL, "refuses overflow and zeroes dev_out");
  check(moe_stage_carve(&s, UINT64_MAX, &d3) == NULL && s.used == 128, "refuses wraparound");
  printf("[test_moe_stage] ok\n");
  return 0;
}
