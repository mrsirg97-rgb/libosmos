#include "pretok.h"
#include "unicode_case.h"
#include "unicode_data.h"

/* o200k's pre-tokens (pretok.h: Inkling-2's tokenizer.json Split, the GPT-4o expression), its alternatives in the
   expression's order, each with the regex's own backtracking:
     [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
     [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
     \p{N}{1,3} |  ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
   a code point none of them takes a piece of its own (the Split's Isolated text between matches) */

typedef struct Text
{
  const uint32_t *cp;
  size_t n;
} Text;

static uint8_t case_of(uint32_t c)
{
  size_t lo = 0, hi = UC_CASE_RANGES;
  while (hi - lo > 1)
  {
    const size_t mid = (lo + hi) / 2;
    if (uc_case_start[mid] <= c)
      lo = mid;
    else
      hi = mid;
  }
  return uc_case_class[lo];
}

static bool in(Text t, size_t i) { return i < t.n; }
static bool upper(Text t, size_t i) { return in(t, i) && (case_of(t.cp[i]) & UC_CASE_UPPER); }
static bool lower(Text t, size_t i) { return in(t, i) && (case_of(t.cp[i]) & UC_CASE_LOWER); }
static bool letter(Text t, size_t i) { return in(t, i) && (pretok_flags(t.cp[i]) & UC_LETTER); }
static bool number(Text t, size_t i) { return in(t, i) && (pretok_flags(t.cp[i]) & UC_NUMBER); }
static bool space(Text t, size_t i) { return in(t, i) && pretok_space(t.cp[i]); }
static bool nl(Text t, size_t i) { return in(t, i) && (t.cp[i] == '\r' || t.cp[i] == '\n'); }
/* [^\r\n\p{L}\p{N}]: the letter branches' one leading code point */
static bool lead(Text t, size_t i) { return in(t, i) && !nl(t, i) && !letter(t, i) && !number(t, i); }
/* [^\s\p{L}\p{N}] */
static bool other(Text t, size_t i) { return in(t, i) && !space(t, i) && !letter(t, i) && !number(t, i); }

static uint32_t ascii_lower(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; }

/* (?i:'s|'t|'re|'ve|'m|'ll|'d)? at i: its end (i for none) */
static size_t contraction(Text t, size_t i)
{
  if (!in(t, i + 1) || t.cp[i] != '\'')
    return i;
  const uint32_t a = ascii_lower(t.cp[i + 1]);
  if (a == 's' || a == 't' || a == 'm' || a == 'd')
    return i + 2;
  if (!in(t, i + 2))
    return i;
  const uint32_t b = ascii_lower(t.cp[i + 2]);
  return (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') ? i + 3 : i;
}

/* the first branch's letters from j: the uppers greedy, given back one at a time until a lower follows (the lowers
   then greedy) - its end, or 0 for none */
static size_t lower_word(Text t, size_t j)
{
  size_t u = j;
  while (upper(t, u))
    u++;
  for (size_t k = u;; k--)
  {
    if (lower(t, k))
    {
      size_t e = k + 1;
      while (lower(t, e))
        e++;
      return contraction(t, e);
    }
    if (k == j)
      return 0;
  }
}

/* the second's: one upper at least, then the lowers - its end, or 0 */
static size_t upper_word(Text t, size_t j)
{
  if (!upper(t, j))
    return 0;
  size_t e = j + 1;
  while (upper(t, e))
    e++;
  while (lower(t, e))
    e++;
  return contraction(t, e);
}

/* the piece from i: its end, past i */
static size_t match(Text t, size_t i)
{
  size_t (*const words[2])(Text, size_t) = {lower_word, upper_word};
  for (int w = 0; w < 2; w++) /* each branch with its leading code point, then without */
  {
    size_t e = lead(t, i) ? words[w](t, i + 1) : 0;
    if (e || (e = words[w](t, i)))
      return e;
  }
  if (number(t, i))
  {
    size_t e = i + 1;
    while (e < i + 3 && number(t, e))
      e++;
    return e;
  }
  size_t j = t.cp[i] == ' ' ? i + 1 : i; /*  ?[^\s\p{L}\p{N}]+[\r\n/]* */
  if (other(t, j))
  {
    while (other(t, j))
      j++;
    while (nl(t, j) || (in(t, j) && t.cp[j] == '/'))
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

size_t pretok_split_o200k(const uint32_t *cp, size_t n, size_t *cuts)
{
  const Text t = {cp, n};
  size_t k = 0;
  cuts[0] = 0;
  for (size_t i = 0; i < n; i = cuts[k])
    cuts[++k] = match(t, i);
  return k;
}
