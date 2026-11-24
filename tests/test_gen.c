/* osmos-serve's GEN line, parsed whole (osmos/serve/gen.h; 0.16.0: each number read as its prefix - "12x" as 12, an
   id of -5 a huge index, 99999999999 wrapped): every number its field's whole - digits to its end, no sign where none
   belongs, within its range - a float finite; anything else refused, saying which */
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "osmos/serve/gen.h"

static bool parses(const char *text, Gen *g, char *why)
{
  char line[512];
  snprintf(line, sizeof line, "%s", text);
  gen_free(g);
  return gen_parse(line, g, why, 256);
}

int main(void)
{
  printf("[test_gen] start\n");
  Gen g = {0};
  char why[256] = "";
  assert(parses("GEN 16 rid=7 spec=1 mark=3 temp=0.7 top_p=0.9 top_k=40 seed=42 1,2,3", &g, why));
  assert(g.max_new == 16 && g.rid == 7 && g.has_rid && g.spec == 1 && g.mark == 3 && g.n == 3 && g.ids[2] == 3);
  assert(g.samp.top_k == 40 && g.samp.seed == 42 && g.samp.temperature > 0.69f && g.samp.top_p < 0.91f);
  assert(parses("GEN 4 0", &g, why) && g.n == 1 && g.ids[0] == 0 && g.spec == -1);
  const char *bad[] = {
      "GEN 12x 1",          "GEN -1 1",          "GEN 4 1,2x,3",     "GEN 4 1,-5",        "GEN 4 1,,2",
      "GEN 4 1,2,",         "GEN 4 99999999999", "GEN 4 2147483648", "GEN 4 spec=2 1",    "GEN 4 spec=1x 1",
      "GEN 4 rid=7z 1",     "GEN 4 mark=-1 1",   "GEN 4 temp=nan 1", "GEN 4 temp=1e40 1", "GEN 4 temp=0.5x 1",
      "GEN 4 top_p=0 1",    "GEN 4 top_k=4294967296 1",                "GEN 4 seed=18446744073709551616 1",
      "GEN 4 seed=-3 1",    "GEN 4",             "GEN",              "GEN 4 foo=1 1",     "GEN 4 1 2",
  };
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
  {
    why[0] = 0;
    const bool ok = parses(bad[i], &g, why);
    if (ok)
      fprintf(stderr, "  taken: %s\n", bad[i]);
    assert(!ok && why[0]);
  }
  long long rid = 0;
  assert(gen_rid("17", &rid) && rid == 17 && !gen_rid("17x", &rid) && !gen_rid("", &rid) && !gen_rid("-1", &rid));
  gen_free(&g);
  printf("  a GEN's every number whole: %zu malformed lines refused, each saying why\n", sizeof bad / sizeof *bad);
  printf("[test_gen] ok\n");
  return 0;
}
