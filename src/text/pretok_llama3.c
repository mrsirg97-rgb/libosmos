#include "pretok.h"
#include "unicode_data.h"

/* Llama-3's pre-tokens (pretok.h: LFM3's), its expression branch for branch in its order - Qwen3.5's matcher
   (pretok_qwen.c) with the expression's own classes: letters \p{L} alone, digits in runs of three, the symbols
   [^\s\p{L}\p{N}] (a mark among them). a position past the text has no class */

typedef struct Text
{
  const uint32_t *cp;
  size_t n;
} Text;

static bool in(Text t, size_t i) { return i < t.n; }
static bool is(Text t, size_t i, uint16_t c) { return in(t, i) && (pretok_flags(t.cp[i]) & c); }
static bool letter(Text t, size_t i) { return is(t, i, UC_LETTER); }
static bool number(Text t, size_t i) { return is(t, i, UC_NUMBER); }
static bool space(Text t, size_t i) { return in(t, i) && pretok_space(t.cp[i]); }
static bool nl(Text t, size_t i) { return in(t, i) && (t.cp[i] == '\r' || t.cp[i] == '\n'); }

/* a code point lowered as the contractions' (?i:) compares it: only ASCII lowers to their letters */
static uint32_t lower(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; }

static size_t contraction(Text t, size_t i)
{
  if (t.cp[i] != '\'' || !in(t, i + 1))
    return i;
  const uint32_t a = lower(t.cp[i + 1]);
  if (a == 's' || a == 't' || a == 'm' || a == 'd')
    return i + 2;
  if (!in(t, i + 2))
    return i;
  const uint32_t b = lower(t.cp[i + 2]);
  return (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') ? i + 3 : i;
}

/* the piece from i: its end, past i */
static size_t match(Text t, size_t i)
{
  size_t j = contraction(t, i);
  if (j > i)
    return j;
  /* [^\r\n\p{L}\p{N}]?\p{L}+ */
  if (letter(t, i) || (!nl(t, i) && !number(t, i) && letter(t, i + 1)))
  {
    for (j = i + 1; letter(t, j);)
      j++;
    return j;
  }
  if (number(t, i)) /* \p{N}{1,3} */
  {
    for (j = i + 1; j < i + 3 && number(t, j);)
      j++;
    return j;
  }
  /* ' '?[^\s\p{L}\p{N}]+[\r\n]* */
  j = t.cp[i] == ' ' ? i + 1 : i;
  if (in(t, j) && !space(t, j) && !letter(t, j) && !number(t, j))
  {
    while (in(t, j) && !space(t, j) && !letter(t, j) && !number(t, j))
      j++;
    while (nl(t, j))
      j++;
    return j;
  }
  size_t run = 0, last_nl = 0;
  for (; space(t, i + run); run++)
    if (nl(t, i + run))
      last_nl = i + run + 1;
  if (last_nl) /* \s*[\r\n]+ */
    return last_nl;
  if (run > 1 && in(t, i + run)) /* \s+(?!\S) */
    return i + run - 1;
  return i + (run ? run : 1); /* \s+, or a code point none takes */
}

size_t pretok_split_llama3(const uint32_t *cp, size_t n, size_t *cuts)
{
  const Text t = {cp, n};
  size_t k = 0;
  cuts[0] = 0;
  for (size_t i = 0; i < n; i = cuts[k])
    cuts[++k] = match(t, i);
  return k;
}
