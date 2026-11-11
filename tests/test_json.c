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
  /* a model's bytes as a JSON string: each invalid sequence's maximal subpart U+FFFD, as Python's
     bytes.decode("utf-8", "replace") (its verdicts), valid UTF-8 as it is */
  const char *const bytes[][2] = {
      {"ok \xf0\x9f\x98", "ok \xef\xbf\xbd"},                     /* a character cut by max_tokens */
      {"\xed\xa0\x80", "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd"},     /* a surrogate */
      {"\xc0\xaf", "\xef\xbf\xbd\xef\xbf\xbd"},                     /* overlong */
      {"a\xe2\x82" "b", "a\xef\xbf\xbd" "b"},
      {"\xf4\x90\x80\x80", "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd"}, /* past U+10FFFF */
      {"\xff\xfe", "\xef\xbf\xbd\xef\xbf\xbd"},
      {"\xe0\x80\x80", "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd"},
      {"caf\xc3\xa9 \xf0\x9f\x98\x80", "caf\xc3\xa9 \xf0\x9f\x98\x80"},
      {"\xf0\x9f\x98\x80\xf0", "\xf0\x9f\x98\x80\xef\xbf\xbd"}};
  for (size_t i = 0; i < sizeof bytes / sizeof *bytes; i++)
  {
    JsonBuf b = {0};
    json_dump_string(&b, bytes[i][0], strlen(bytes[i][0]));
    Json *j = json_parse(b.p, b.n, err);
    assert(j && j->type == JSON_STRING && j->len == strlen(bytes[i][1]) && !memcmp(j->s, bytes[i][1], j->len));
    json_free(j), free(b.p);
  }
  unsigned seed = 34;
  for (int t = 0; t < 20000; t++) /* any bytes: JSON json_parse reads back */
  {
    char r[16];
    const size_t n = (size_t)(t % 16);
    for (size_t i = 0; i < n; i++)
      seed = seed * 1103515245u + 12345u, r[i] = (char)(seed >> 16);
    JsonBuf b = {0};
    json_dump_string(&b, r, n);
    Json *j = json_parse(b.p, b.n, err);
    assert(j && j->type == JSON_STRING);
    json_free(j), free(b.p);
  }
  printf("  invalid UTF-8 written as Python's replace reads it (%zu cases, 20000 random byte strings read back)\n",
         sizeof bytes / sizeof *bytes);
  printf("  %d inputs written back as python writes them, %d refused as it refuses them\n", good, bad);
  printf("[test_json] ok\n");
  return 0;
}
