#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
  char err[OSMOS_ERR] = "";
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
  /* past a double's range: Python reads inf and writes Infinity, which no JSON reader (ours either) takes back - a
     client's 1e400 refused by name, not its tools dropped or its message "out of memory" (0.7.1) */
  assert(!json_parse("1e400", 5, err) && strstr(err, "range"));
  assert(!json_parse("[-1.5e999]", 10, err) && strstr(err, "range"));
  Json *edge = json_parse("[1e308, 1e-400, 123456789012345678901234567890]", 47, err);
  assert(edge && edge->n == 3 && edge->item[0]->f == 1e308 && edge->item[1]->f == 0.0 &&
         edge->item[2]->type == JSON_INT); /* the largest double, an underflow to 0 (Python's), an int any length */
  json_free(edge);
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
  /* an object's duplicate keys (Python's dict): a key in its first place, its last value - on random objects of keys
     from a small set (repeated often), against a reference written out by hand */
  for (int t = 0; t < 2000; t++)
  {
    char text[4096], *w = text;
    int order[8], last[8], seen = 0, at[8];
    for (int k = 0; k < 8; k++) at[k] = -1;
    const int n = 1 + (int)(seed % 40);
    *w++ = '{';
    for (int i = 0; i < n; i++)
    {
      seed = seed * 1103515245u + 12345u;
      const int k = (int)((seed >> 16) % 8);
      w += sprintf(w, "%s\"k%d\":%d", i ? "," : "", k, i);
      if (at[k] < 0) at[k] = seen, order[seen++] = k;
      last[k] = i;
    }
    *w++ = '}', *w = 0;
    Json *j = json_parse(text, (size_t)(w - text), err);
    assert(j && j->type == JSON_OBJECT && j->n == (size_t)seen);
    for (int i = 0; i < seen; i++)
    {
      char want[8];
      snprintf(want, sizeof want, "k%d", order[i]);
      assert(!strcmp(j->key[i], want) && j->item[i]->type == JSON_INT && atoi(j->item[i]->s) == last[order[i]]);
    }
    json_free(j);
  }
  /* the work an object's keys cost grows as n log n, not n^2 (0.15.2: each key compared with every earlier one - one
     1 MB body of 80,000 keys held a core 67 s, a 32 MB one hours): 8x the keys, under 24x the time (n log n ~9.5x,
     n^2 64x) */
  {
    double took[2];
    const int ns[2] = {25000, 200000};
    for (int r = 0; r < 2; r++)
    {
      char *b = malloc((size_t)ns[r] * 16 + 16), *w = b;
      *w++ = '{';
      for (int i = 0; i < ns[r]; i++) w += sprintf(w, "%s\"k%07d\":0", i ? "," : "", i);
      *w++ = '}', *w = 0;
      struct timespec a, z;
      clock_gettime(CLOCK_MONOTONIC, &a);
      Json *j = json_parse(b, (size_t)(w - b), err);
      clock_gettime(CLOCK_MONOTONIC, &z);
      assert(j && j->n == (size_t)ns[r]);
      took[r] = (double)(z.tv_sec - a.tv_sec) + (double)(z.tv_nsec - a.tv_nsec) / 1e9;
      json_free(j), free(b);
    }
    printf("  an object's keys: %d in %.3f s, %d in %.3f s (%.1fx)\n", ns[0], took[0], ns[1], took[1], took[1] / took[0]);
    assert(took[1] < 24 * took[0] + 0.05);
  }
  printf("  invalid UTF-8 written as Python's replace reads it (%zu cases, 20000 random byte strings read back)\n",
         sizeof bytes / sizeof *bytes);
  printf("  %d inputs written back as python writes them, %d refused as it refuses them\n", good, bad);
  printf("[test_json] ok\n");
  return 0;
}
