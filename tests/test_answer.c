#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/gguf.h"
#include "osmos/text/answer.h"
#include "osmos/text/tokenizer.h"

/* the answer rule (osmos/text/answer.h): an answer is not begun while the turn's tokens since the reasoning's end are
   whitespace alone; the best token outside a set. on a vocabulary written by hand, then on the real tokenizers
   (DeepSeek's UD-IQ1_M first shard, Qwen3.8's): their blank tokens and </think>, skipped without the shards. */

enum
{
  END = 1, /* </think> */
  NL = 2,  /* "\n\n" */
  SP = 3,  /* " " */
  HI = 4,  /* "Hi" */
  EOT = 5, /* the end of turn */
  V = 6
};

static void hand(void)
{
  bool blank[V] = {false, false, true, true, false, false};
  const AnswerRule r = {blank, V, END};
  const int32_t closed[] = {HI, END}, spaced[] = {HI, END, NL, SP}, said[] = {END, NL, HI}, none[] = {HI, NL};
  assert(answer_unbegun(&r, closed, 2, NULL, 0));
  assert(answer_unbegun(&r, spaced, 4, NULL, 0));
  assert(!answer_unbegun(&r, said, 3, NULL, 0));    /* a word past it: begun */
  assert(!answer_unbegun(&r, none, 2, NULL, 0));    /* no reasoning closed */
  assert(!answer_unbegun(&r, NULL, 0, NULL, 0));
  /* a and b one sequence: b's own, else back into a */
  const int32_t a[] = {HI, END}, b_blank[] = {NL, SP}, b_word[] = {NL, HI}, b_end[] = {HI, END, NL};
  assert(answer_unbegun(&r, a, 2, b_blank, 2));
  assert(!answer_unbegun(&r, a, 2, b_word, 2));
  assert(answer_unbegun(&r, said, 3, b_end, 3));    /* b's own close, past a's answer */
  const int32_t odd[] = {END, -7}, big[] = {END, 99};
  assert(!answer_unbegun(&r, odd, 2, NULL, 0) && !answer_unbegun(&r, big, 2, NULL, 0)); /* out of range: not blank */

  const float row[V] = {0.5f, 1.0f, 3.0f, 3.0f, 2.0f, 9.0f};
  const int32_t ends[] = {EOT};
  assert(answer_best(row, V, ends, 1) == NL); /* 3.0 twice: the lower id */
  assert(answer_best(row, V, NULL, 0) == EOT);
  const float nan_row[V] = {NAN, NAN, NAN, NAN, NAN, 1.0f};
  assert(answer_best(nan_row, V, ends, 1) == -1);
  const float with_nan[V] = {NAN, 0.0f, NAN, NAN, -1.0f, 5.0f};
  assert(answer_best(with_nan, V, ends, 1) == END);
  printf("  by hand: unbegun past blanks, begun at a word, a and b one sequence; the best outside a set\n");
}

static Tokenizer *of(const char *shard, Gguf **g)
{
  char path[512], err[GGUF_ERR], terr[TOKENIZER_ERR] = "";
  snprintf(path, sizeof path, "%s/Models/%s", getenv("HOME") ? getenv("HOME") : "", shard);
  *g = gguf_header(path, err);
  return *g ? tokenizer_create(*g, terr) : NULL;
}

static void real(const char *name, const char *shard)
{
  Gguf *g = NULL;
  Tokenizer *t = of(shard, &g);
  if (!t)
  {
    printf("  %s skipped: no first shard\n", name);
    gguf_destroy(g);
    return;
  }
  const int32_t end = tokenizer_find(t, "</think>");
  assert(end >= 0);
  AnswerRule r;
  char err[ANSWER_ERR] = "";
  assert(!answer_rule_create(&r, t, -1, err) && err[0]);
  assert(!answer_rule_create(&r, t, tokenizer_vocab(t), err));
  assert(answer_rule_create(&r, t, end, err));
  int32_t ids[16];
  const char *blank_texts[] = {"\n\n", " ", "\n"}, *words[] = {"Hi", ".", "<|im_end|>"};
  for (int i = 0; i < 3; i++)
  {
    assert(tokenizer_encode(t, blank_texts[i], strlen(blank_texts[i]), true, ids, 16) == 1 && r.blank[ids[0]]);
    const int64_t n = tokenizer_encode(t, words[i], strlen(words[i]), true, ids, 16);
    assert(n >= 1 && !r.blank[ids[n - 1]]);
  }
  assert(!r.blank[end]);
  size_t n_blank = 0;
  for (int32_t i = 0; i < r.vocab; i++)
    n_blank += r.blank[i];
  const int32_t turn[] = {end};
  assert(answer_unbegun(&r, turn, 1, NULL, 0));
  printf("  %s: </think> %d, %zu blank tokens of %d\n", name, end, n_blank, r.vocab);
  answer_rule_destroy(&r);
  tokenizer_destroy(t);
  gguf_destroy(g);
}

int main(void)
{
  printf("[test_answer] start\n");
  hand();
  real("DeepSeek", "ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf");
  real("Qwen3.8", "qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf");
  printf("[test_answer] ok\n");
  return 0;
}
