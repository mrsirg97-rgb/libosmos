#ifndef OSMOS_TEXT_ANSWER_H
#define OSMOS_TEXT_ANSWER_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/text/tokenizer.h"

#define ANSWER_ERR 160

typedef struct AnswerRule
{
  bool *blank;   /* [vocab]: a token whose text is whitespace alone */
  int32_t vocab;
  int32_t after; /* the reasoning's end (</think>): an answer begins at the first token past it not blank */
} AnswerRule;

bool answer_rule_create(AnswerRule *r, const Tokenizer *t, int32_t after, char err[ANSWER_ERR]);
void answer_rule_destroy(AnswerRule *r);
bool answer_unbegun(const AnswerRule *r, const int32_t *a, uint32_t na, const int32_t *b, uint32_t nb);
int32_t answer_best(const float *row, uint32_t vocab, const int32_t *except, uint32_t n_except);

#endif

/*
AnswerRule
  a turn's answer not yet begun: osmos-serve's rule that a turn which closed its reasoning does not end before it
  has said anything (DESIGN.md, "0.5.10"). Qwen3.8 at small talk closed its thinking and ended the turn at once,
  greedily, so every retry of rig's was the same empty turn.

answer_rule_create
  the rule for the tokenizer's vocabulary: a token blank when its text (special tokens as their text) is not empty
  and only spaces, tabs, newlines and carriage returns. false with why: no tokenizer, `after` not one of its ids,
  out of memory.

answer_unbegun
  a sequence's tokens, a's then b's (either may be empty): walking back from the last over blank tokens, the first
  other token is `after` - the reasoning closed and nothing but whitespace since. false for a sequence of blanks
  alone, or none.

answer_best
  the row's greatest logit among the tokens not in `except`, ties to the lower id; NaNs never taken. -1 when every
  token is excepted or NaN.
*/
