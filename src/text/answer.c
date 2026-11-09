#include "osmos/text/answer.h"

#include <stdio.h>
#include <stdlib.h>

static bool blank_text(const char *s, int n)
{
  if (n <= 0)
    return false;
  for (int i = 0; i < n; i++)
    if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
      return false;
  return true;
}

bool answer_rule_create(AnswerRule *r, const Tokenizer *t, int32_t after, char err[ANSWER_ERR])
{
  *r = (AnswerRule){0};
  const int32_t v = t ? tokenizer_vocab(t) : 0;
  if (v <= 0 || after < 0 || after >= v)
    return snprintf(err, ANSWER_ERR, "no tokenizer, or the reasoning's end %d not one of its %d tokens", after, v),
           false;
  if (!(r->blank = calloc((size_t)v, sizeof *r->blank)))
    return snprintf(err, ANSWER_ERR, "out of memory"), false;
  char piece[TOKENIZER_PIECE_MAX];
  for (int32_t id = 0; id < v; id++)
    r->blank[id] = blank_text(piece, tokenizer_piece(t, id, true, piece));
  r->vocab = v, r->after = after;
  return true;
}

void answer_rule_destroy(AnswerRule *r)
{
  free(r->blank);
  *r = (AnswerRule){0};
}

/* the last of x's tokens that is not blank (one out of the vocabulary counts as not blank), via *t; false: none */
static bool last_other(const AnswerRule *r, const int32_t *x, uint32_t n, int32_t *t)
{
  for (uint32_t i = n; x && i-- > 0;)
    if (x[i] < 0 || x[i] >= r->vocab || !r->blank[x[i]])
      return *t = x[i], true;
  return false;
}

bool answer_unbegun(const AnswerRule *r, const int32_t *a, uint32_t na, const int32_t *b, uint32_t nb)
{
  int32_t t;
  return (last_other(r, b, nb, &t) || last_other(r, a, na, &t)) && t == r->after;
}

int32_t answer_best(const float *row, uint32_t vocab, const int32_t *except, uint32_t n_except)
{
  int32_t best = -1;
  for (uint32_t id = 0; id < vocab; id++)
  {
    bool out = row[id] != row[id]; /* NaN */
    for (uint32_t k = 0; !out && k < n_except; k++)
      out = except[k] == (int32_t)id;
    if (!out && (best < 0 || row[id] > row[best]))
      best = (int32_t)id;
  }
  return best;
}
