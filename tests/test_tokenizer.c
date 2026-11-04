#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/gguf.h"
#include "osmos/text/tokenizer.h"

/* the tokenizer against llama.cpp's own (tests/fixtures/tokenizer.txt: its llama-tokenize over UD-IQ1_M's first
   shard, tools/gen_tok_fixtures.py) - every case's ids, special tokens parsed and not; each case's pieces joined back
   to its bytes. skipped without the shard. */

#define MAX_TEXT 4096
#define MAX_IDS 2048

static int hexval(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }

int main(void)
{
  printf("[test_tokenizer] start\n");
  char path[512], err[GGUF_ERR], terr[TOKENIZER_ERR] = "";
  snprintf(path, sizeof path, "%s/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf",
           getenv("HOME") ? getenv("HOME") : "");
  Gguf *g = gguf_create(path, err);
  FILE *f = fopen("tests/fixtures/tokenizer.txt", "r");
  if (!g || !f)
  {
    printf("[test_tokenizer] skipped: no UD-IQ1_M first shard or no fixtures\n");
    return 0;
  }
  Tokenizer *t = tokenizer_create(g, terr);
  if (!t)
    printf("  refused: %s\n", terr);
  assert(t);
  gguf_destroy(g); /* the tokenizer's own copy: the Gguf may go */
  assert(tokenizer_vocab(t) == 129280 && tokenizer_find(t, "<｜User｜>") == 128803);
  assert(tokenizer_find(t, "zzzq") == -1);
  static char line[3 * MAX_TEXT], text[MAX_TEXT], joined[MAX_TEXT + TOKENIZER_PIECE_MAX];
  static int32_t want[MAX_IDS], got[MAX_IDS];
  int cases = 0, special = 0;
  size_t len = 0;
  while (fgets(line, sizeof line, f))
  {
    if (line[0] == 'S')
      special = line[2] == '1';
    else if (line[0] == 'T')
    {
      len = 0;
      for (const char *h = line + 2; h[0] && h[0] != '\n'; h += 2)
        text[len++] = (char)(hexval(h[0]) << 4 | hexval(h[1]));
    }
    else if (line[0] == 'I')
    {
      int n = 0;
      for (char *p = line + 2, *e; *p && *p != '\n'; p = e)
      {
        want[n++] = (int32_t)strtol(p, &e, 10);
        if (e == p)
          break;
      }
      const int64_t m = tokenizer_encode(t, text, len, special, got, MAX_IDS);
      if (m != n || memcmp(got, want, (size_t)n * sizeof *got) != 0)
      {
        printf("  case %d (special %d) \"%.*s\": %lld ids, llama.cpp's %d\n   ours:", cases, special, (int)len, text,
               (long long)m, n);
        for (int i = 0; i < (int)m && i < 24; i++)
          printf(" %d", got[i]);
        printf("\n   want:");
        for (int i = 0; i < n && i < 24; i++)
          printf(" %d", want[i]);
        printf("\n");
        exit(1);
      }
      size_t at = 0; /* back to its bytes: every token's piece, a control token's text where special */
      for (int i = 0; i < n; i++)
      {
        char piece[TOKENIZER_PIECE_MAX];
        const int k = tokenizer_piece(t, got[i], true, piece);
        assert(k >= 0 && at + (size_t)k <= sizeof joined);
        memcpy(joined + at, piece, (size_t)k), at += (size_t)k;
      }
      assert(at == len && memcmp(joined, text, len) == 0);
      cases++;
    }
  }
  fclose(f);
  assert(cases > 0);
  int32_t few[2];
  assert(tokenizer_encode(t, "Hello world!", 12, false, few, 2) == 3); /* past cap: the full count */
  char piece[TOKENIZER_PIECE_MAX];
  assert(tokenizer_piece(t, 128803, false, piece) == 0 && tokenizer_piece(t, 129280, true, piece) == -1);
  tokenizer_destroy(t);
  printf("  %d cases: every id llama.cpp's, every case's pieces its bytes\n", cases);
  printf("[test_tokenizer] ok\n");
  return 0;
}
