#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/json.h"

/* the JSON module against Python's own json (tests/fixtures/json.txt: tools/gen_json_fixtures.py) - each input read
   and written back as json.dumps(ensure_ascii=False) writes it, byte for byte; each refused input refused. */

static size_t unhex(const char *h, char *out)
{
  size_t n = 0;
  for (; h[0] && h[0] != ' ' && h[0] != '\n'; h += 2)
  {
    const int hi = h[0] <= '9' ? h[0] - '0' : h[0] - 'a' + 10, lo = h[1] <= '9' ? h[1] - '0' : h[1] - 'a' + 10;
    out[n++] = (char)(hi << 4 | lo);
  }
  out[n] = 0;
  return n;
}

int main(void)
{
  printf("[test_json] start\n");
  FILE *f = fopen("tests/fixtures/json.txt", "r");
  assert(f);
  static char line[8192], in[4096], want[4096];
  char err[JSON_ERR];
  int good = 0, bad = 0;
  while (fgets(line, sizeof line, f))
  {
    const size_t n = unhex(line + 2, in);
    Json *j = json_parse(in, n, err);
    if (line[0] == 'X')
    {
      if (j)
        printf("  read, not refused: \"%s\"\n", in);
      assert(!j && err[0]);
      bad++;
      continue;
    }
    if (!j)
      printf("  refused \"%s\": %s\n", in, err);
    assert(j);
    const size_t wn = unhex(strchr(line + 2, ' ') + 1, want);
    JsonBuf b = {0};
    json_dump(&b, j);
    if (b.oom || b.n != wn || memcmp(b.p, want, wn) != 0)
    {
      printf("  \"%s\": wrote \"%.*s\", python \"%s\"\n", in, (int)b.n, b.p, want);
      exit(1);
    }
    free(b.p);
    json_free(j);
    good++;
  }
  fclose(f);
  Json *o = json_parse("{\"a\":1,\"s\":\"x\",\"a\":2}", 21, err);
  assert(o && json_get(o, "a") && strcmp(json_get(o, "a")->s, "2") == 0 && strcmp(json_str(o, "s"), "x") == 0);
  assert(!json_get(o, "zz") && !json_str(o, "a") && o->n == 2);
  json_free(o);
  char deep[2 * JSON_MAX_DEPTH + 8];
  memset(deep, '[', JSON_MAX_DEPTH + 1), memset(deep + JSON_MAX_DEPTH + 1, ']', JSON_MAX_DEPTH + 1);
  assert(!json_parse(deep, 2 * JSON_MAX_DEPTH + 2, err) && strstr(err, "deep"));
  assert(!json_parse("\"\xff\"", 3, err));   /* invalid UTF-8 */
  assert(!json_parse("NaN", 3, err));           /* Python reads it; JSON has none */
  assert(!json_parse("\"a\0b\"", 5, err));   /* a raw control character in a string */
  printf("  %d inputs written back as python writes them, %d refused as it refuses them\n", good, bad);
  printf("[test_json] ok\n");
  return 0;
}
