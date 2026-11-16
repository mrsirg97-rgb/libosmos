#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/chat_ink2.h"
#include "osmos/text/json.h"

/* Inkling-2's chat template against its own (tests/fixtures/chat_ink2.jsonl: transformers' apply_chat_template over
   the checkpoint's tokenizer - tools/gen_chat_fixtures_ink2.py): each chat's prompt byte for byte or its refusal (an
   image refused here too: this build reads text alone); each completion's parse the same message (its JSON as Python
   writes it) or its refusal - the template's assistant turns read back to the messages they render */

static char *slurp_line(FILE *f, size_t *n)
{
  size_t cap = 1 << 16, len = 0;
  char *b = malloc(cap);
  int c;
  while (b && (c = fgetc(f)) != EOF && c != '\n')
  {
    if (len + 1 == cap)
      b = realloc(b, cap *= 2);
    if (b)
      b[len++] = (char)c;
  }
  if (!b || (c == EOF && !len))
    return free(b), NULL;
  *n = len;
  return b;
}

static void expect(int i, const char *got, size_t gn, const char *want, size_t wn)
{
  if (gn == wn && memcmp(got, want, wn) == 0)
    return;
  size_t k = 0;
  while (k < gn && k < wn && got[k] == want[k])
    k++;
  printf("  render %d: differs at byte %zu of %zu (ours %zu)\n   ours: ...%.*s\n   want: ...%.*s\n", i, k, wn, gn,
         (int)(gn - k < 200 ? gn - k : 200), got + k, (int)(wn - k < 200 ? wn - k : 200), want + k);
  exit(1);
}

int main(void)
{
  printf("[test_chat_ink2] start\n");
  FILE *f = fopen("tests/fixtures/chat_ink2.jsonl", "r");
  assert(f);
  char err[JSON_ERR], cerr[CHAT_ERR];
  int renders = 0, refused = 0, parses = 0, parse_refused = 0, i = 0;
  size_t n;
  for (char *line; (line = slurp_line(f, &n)); i++)
  {
    Json *c = json_parse(line, n, err);
    free(line);
    assert(c);
    const Json *tools = json_get(c, "tools"), *want = json_get(c, "out"), *wrong = json_get(c, "error");
    JsonBuf b = {0};
    cerr[0] = 0;
    if (!strcmp(json_str(c, "kind"), "parse"))
    {
      const Json *text = json_get(c, "text");
      const bool ok = ichat_parse(text->s, text->len, &b, cerr);
      if (want)
      {
        JsonBuf w = {0};
        json_dump(&w, want);
        if (!ok)
          printf("  parse %d: refused (%s), Python reads it\n", i, cerr), exit(1);
        if (b.n != w.n || memcmp(b.p, w.p, w.n))
          printf("  parse %d:\n   ours: %.*s\n   want: %.*s\n", i, (int)b.n, b.p, (int)w.n, w.p), exit(1);
        free(w.p), parses++;
      }
      else if (ok)
        printf("  parse %d: read, Python refuses it (%s)\n", i, wrong ? wrong->s : "?"), exit(1);
      else
        parse_refused++;
      free(b.p), json_free(c);
      continue;
    }
    const bool ok = ichat_render(json_get(c, "messages"), tools && tools->type != JSON_NULL ? tools : NULL,
                                 json_get(c, "effort"), &b, cerr);
    if (want)
    {
      if (!ok)
        printf("  render %d: refused (%s), the template's renders it\n", i, cerr), exit(1);
      expect(i, b.p, b.n, want->s, want->len);
      renders++;
    }
    else
    {
      if (ok)
        printf("  render %d: rendered, the template refuses it (%s)\n", i, wrong ? wrong->s : "?"), exit(1);
      assert(cerr[0]);
      refused++;
    }
    free(b.p);
    json_free(c);
  }
  fclose(f);
  printf("  %d chats rendered the template's bytes, %d refused as it refuses them\n", renders, refused);
  printf("  %d completions read as Python reads them (the template's assistant turns among them), %d refused\n",
         parses, parse_refused);
  printf("[test_chat_ink2] ok\n");
  return 0;
}
