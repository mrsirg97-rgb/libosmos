#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"

/* the chat encoding against DeepSeek's own (tests/fixtures/chat.jsonl: its encoding.py over the cases
   tools/gen_chat_fixtures.py lists) - each chat's prompt byte for byte, each completion's parse the same message (its
   JSON as Python writes it) or refused where the reference raises; and refusals. */

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

static void expect(const char *what, const char *got, size_t gn, const char *want, size_t wn)
{
  if (gn == wn && memcmp(got, want, wn) == 0)
    return;
  size_t i = 0;
  while (i < gn && i < wn && got[i] == want[i])
    i++;
  printf("  %s: differs at byte %zu of %zu (ours %zu)\n   ours: ...%.*s\n   want: ...%.*s\n", what, i, wn, gn,
         (int)(gn - i < 160 ? gn - i : 160), got + i, (int)(wn - i < 160 ? wn - i : 160), want + i);
  exit(1);
}

int main(void)
{
  printf("[test_chat] start\n");
  FILE *f = fopen("tests/fixtures/chat.jsonl", "r");
  assert(f);
  char err[JSON_ERR], cerr[CHAT_ERR];
  int renders = 0, parses = 0, refusals = 0;
  size_t n;
  for (char *line; (line = slurp_line(f, &n)); free(line))
  {
    Json *c = json_parse(line, n, err);
    assert(c);
    const ChatMode mode = strcmp(json_str(c, "thinking"), "thinking") == 0 ? CHAT_THINKING : CHAT_PLAIN;
    JsonBuf b = {0};
    if (strcmp(json_str(c, "kind"), "render") == 0)
    {
      const int effort = chat_effort(json_get(c, "effort"), cerr);
      assert(effort > 0);
      if (!chat_render(json_get(c, "messages"), mode, effort, &b, cerr))
        printf("  case %d refused: %s\n", renders, cerr), exit(1);
      const Json *want = json_get(c, "out");
      char what[32];
      snprintf(what, sizeof what, "render %d", renders);
      expect(what, b.p, b.n, want->s, want->len);
      renders++;
    }
    else if (strcmp(json_str(c, "kind"), "render_refuse") == 0) /* the reference's encoding raises on the chat */
    {
      const int effort = chat_effort(json_get(c, "effort"), cerr);
      if (chat_render(json_get(c, "messages"), mode, effort, &b, cerr))
        printf("  render refusal %d rendered: %.*s\n", refusals, (int)b.n, b.p), exit(1);
      printf("  refused as the reference's encoding does: %s\n", cerr);
      refusals++;
    }
    else if (strcmp(json_str(c, "kind"), "refuse") == 0) /* the reference raises on it: so must the port */
    {
      const Json *text = json_get(c, "text");
      if (chat_parse(text->s, text->len, mode, &b, cerr))
        printf("  refusal %d parsed: %.*s\n", refusals, (int)b.n, b.p), exit(1);
      printf("  refused as the reference does: %s\n", cerr);
      refusals++;
    }
    else
    {
      const Json *text = json_get(c, "text");
      if (!chat_parse(text->s, text->len, mode, &b, cerr))
        printf("  parse %d refused: %s\n", parses, cerr), exit(1);
      Json *got = json_parse(b.p, b.n, err);
      assert(got);
      JsonBuf g = {0}, w = {0};
      json_dump(&g, got), json_dump(&w, json_get(c, "out"));
      expect("parse", g.p, g.n, w.p, w.n);
      free(g.p), free(w.p), json_free(got);
      parses++;
    }
    free(b.p);
    json_free(c);
  }
  fclose(f);
  /* refusals: an image, an unknown role, an effort out of range, a completion with no EOS */
  const char *bad[] = {
      "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"x\"}}]}]",
                       "[{\"role\": \"wizard\", \"content\": \"x\"}]", "[{\"role\": \"user\", \"content\": null}]"};
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
  {
    Json *m = json_parse(bad[i], strlen(bad[i]), err);
    JsonBuf b = {0};
    assert(m && !chat_render(m, CHAT_THINKING, 75, &b, cerr) && cerr[0]);
    printf("  refused: %s\n", cerr);
    free(b.p), json_free(m);
  }
  /* a string's text whole past a NUL (\u0000), as encoding.py keeps it - in every place a client's text lands: a user's
     string and its parts, a system's parts, a tool's result parts, an assistant's reasoning (0.7.1: cut at the NUL) */
  const char nul[] = "[{\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": \"s1\\u0000t1\"}]},"
                     "{\"role\": \"user\", \"content\": \"u2\\u0000v2\"},"
                     "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"u3\\u0000v3\"}]},"
                     "{\"role\": \"tool\", \"tool_call_id\": \"c\", \"content\": [{\"type\": \"text\", \"text\": "
                     "\"r4\\u0000s4\"}]},"
                     "{\"role\": \"assistant\", \"content\": \"ok\", \"reasoning_content\": \"p5\\u0000q5\"}]";
  Json *nm = json_parse(nul, strlen(nul), err);
  JsonBuf nb = {0};
  assert(nm && chat_render(nm, CHAT_THINKING, 75, &nb, cerr));
  for (int k = 1; k <= 5; k++)
  {
    const char *const ab[] = {"s1", "u2", "u3", "r4", "p5"}, *const cd[] = {"t1", "v2", "v3", "s4", "q5"};
    char want[8];
    const size_t wn = (size_t)snprintf(want, sizeof want, "%s#%s", ab[k - 1], cd[k - 1]);
    want[2] = 0; /* the NUL between */
    assert(memmem(nb.p, nb.n, want, wn));
  }
  free(nb.p), json_free(nm);
  printf("  a string's text whole past a NUL: system and user parts, a user's string, a tool's parts, reasoning\n");
  Json *e = json_parse("101", 3, err);
  assert(chat_effort(e, cerr) == -1), json_free(e);
  JsonBuf b = {0};
  assert(!chat_parse("no eos", 6, CHAT_PLAIN, &b, cerr));
  assert(!chat_parse("x<｜end▁of▁sentence｜>", 25, CHAT_THINKING, &b, cerr)); /* thinking: no </think> */
  free(b.p);
  printf("  %d chats rendered, %d completions parsed and %d refused as DeepSeek's encoding does\n", renders, parses,
         refusals);
  printf("[test_chat] ok\n");
  return 0;
}
