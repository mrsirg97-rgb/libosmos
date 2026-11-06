#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/gguf.h"
#include "osmos/text/tokenizer.h"

/* the tokenizer against llama.cpp's own (tools/gen_tok_fixtures.py: its llama-tokenize over a model's first shard) -
   DeepSeek's (tests/fixtures/tokenizer.txt, UD-IQ1_M) and Qwen3.5's pre-tokenizer (tokenizer_qwen35.txt, Qwen3.8's
   UD-IQ3_XXS): every case's ids, special tokens parsed and not; each case's pieces joined back to its bytes. a family
   skipped without its shard. */

#define MAX_TEXT 4096
#define MAX_IDS 2048

static int hexval(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }

/* a fixtures file's cases against t: how many, exiting on the first that differs */
static int cases_of(const Tokenizer *t, const char *fixtures)
{
  FILE *f = fopen(fixtures, "r");
  assert(f);
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
  return cases;
}

static Tokenizer *of(const char *shard)
{
  char path[512], err[GGUF_ERR], terr[TOKENIZER_ERR] = "";
  snprintf(path, sizeof path, "%s/Models/%s", getenv("HOME") ? getenv("HOME") : "", shard);
  Gguf *g = gguf_header(path, err);
  if (!g)
    return NULL;
  Tokenizer *t = tokenizer_create(g, terr);
  if (!t)
    printf("  refused: %s\n", terr);
  assert(t);
  gguf_destroy(g); /* the tokenizer's own copy: the Gguf may go */
  return t;
}

int main(void)
{
  printf("[test_tokenizer] start\n");
  char piece[TOKENIZER_PIECE_MAX];
  Tokenizer *t = of("ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf");
  if (t)
  {
    assert(tokenizer_vocab(t) == 129280 && tokenizer_find(t, "<｜User｜>") == 128803);
    assert(tokenizer_find(t, "zzzq") == -1);
    const int n = cases_of(t, "tests/fixtures/tokenizer.txt");
    int32_t few[2];
    assert(tokenizer_encode(t, "Hello world!", 12, false, few, 2) == 3); /* past cap: the full count */
    assert(tokenizer_piece(t, 128803, false, piece) == 0 && tokenizer_piece(t, 129280, true, piece) == -1);
    tokenizer_destroy(t);
    printf("  DeepSeek: %d cases, every id llama.cpp's, every case's pieces its bytes\n", n);
  }
  else
    printf("  DeepSeek skipped: no UD-IQ1_M first shard\n");
  t = of("qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf");
  if (t)
  {
    const int32_t im_end = tokenizer_find(t, "<|im_end|>");
    assert(im_end == 248046 && tokenizer_find(t, "<|endoftext|>") == 248044);
    const int n = cases_of(t, "tests/fixtures/tokenizer_qwen35.txt");
    assert(tokenizer_piece(t, im_end, false, piece) == 0 && tokenizer_piece(t, tokenizer_vocab(t), true, piece) == -1);
    tokenizer_destroy(t);
    printf("  Qwen3.5's: %d cases, every id llama.cpp's, every case's pieces its bytes\n", n);
  }
  else
    printf("  Qwen skipped: no Qwen3.8 UD-IQ3_XXS first shard\n");
  printf("[test_tokenizer] ok\n");
  return 0;
}
