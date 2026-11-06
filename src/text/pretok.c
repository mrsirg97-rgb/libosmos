#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "pretok.h"
#include "unicode_data.h"

/* the classes as llama.cpp's regex sees them: a code point below 128 by the ASCII lists its collapse maps the
   categories to; above, by its Unicode category - whitespace first (collapsed to \v: \s), then N L P M S, any other
   in none (unicode.cpp's k_ucat_map, its text_collapsed) */

uint16_t pretok_flags(uint32_t c)
{
  size_t lo = 0, hi = UC_RANGES; /* the last range starting at or before c */
  while (hi - lo > 1)
  {
    const size_t mid = (lo + hi) / 2;
    if (uc_range_start[mid] <= c)
      lo = mid;
    else
      hi = mid;
  }
  return uc_range_flags[lo];
}

bool pretok_space(uint32_t c)
{
  if (c < 128)
    return c == ' ' || (c >= '\t' && c <= '\r');
  size_t lo = 0, hi = UC_SPACES;
  while (lo < hi)
  {
    const size_t mid = (lo + hi) / 2;
    if (uc_space[mid] == c)
      return true;
    if (uc_space[mid] < c)
      lo = mid + 1;
    else
      hi = mid;
  }
  return false;
}

static int category(uint32_t c) { return pretok_space(c) ? 0 : pretok_flags(c) & UC_CATEGORIES; }

static bool ascii_letter(uint32_t c) { return (c | 0x20) >= 'a' && (c | 0x20) <= 'z'; }

static bool ascii_symbol(uint32_t c) { return c && strchr("$+<=>^`|~", (int)c); }

static bool ascii_punct(uint32_t c) /* ASCII punctuation less the symbols: !-#%-*,-/:-;?-@[-]_{} */
{
  return c >= 33 && c < 127 && !ascii_letter(c) && !(c >= '0' && c <= '9') && !ascii_symbol(c);
}

static bool is_n(uint32_t c) { return c < 128 ? c >= '0' && c <= '9' : category(c) == UC_NUMBER; }
static bool is_l(uint32_t c) { return c < 128 ? ascii_letter(c) : category(c) == UC_LETTER; }
static bool is_p(uint32_t c) { return c < 128 ? ascii_punct(c) : category(c) == UC_PUNCT; }
static bool is_s(uint32_t c) { return c < 128 ? ascii_symbol(c) : category(c) == UC_SYMBOL; }
static bool is_m(uint32_t c) { return c >= 128 && category(c) == UC_MARK; }
static bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

static bool cjk(uint32_t c) /* [一-龥぀-ゟ゠-ヿ] */
{
  return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}

/* each expression's match at i of a piece ending at e: its end, or i for none (none of them matches empty) */
static size_t numbers(const uint32_t *cp, size_t i, size_t e)
{
  size_t j = i;
  while (j < e && j - i < 3 && is_n(cp[j]))
    j++;
  return j;
}

static size_t ideographs(const uint32_t *cp, size_t i, size_t e)
{
  size_t j = i;
  while (j < e && cjk(cp[j]))
    j++;
  return j;
}

/* a run from j of letters and marks, or of punctuation and symbols: its end, or 0 for none (an end is past j) */
static size_t letters(const uint32_t *cp, size_t j, size_t e)
{
  const size_t k = j;
  while (j < e && (is_l(cp[j]) || is_m(cp[j])))
    j++;
  return j > k ? j : 0;
}

static size_t punctuation(const uint32_t *cp, size_t j, size_t e)
{
  const size_t k = j;
  while (j < e && (is_p(cp[j]) || is_s(cp[j])))
    j++;
  while (j > k && j < e && is_nl(cp[j])) /* [\r\n]* after */
    j++;
  return j > k ? j : 0;
}

static size_t words(const uint32_t *cp, size_t i, size_t e)
{
  const uint32_t c = cp[i];
  size_t j;
  if (c < 128 && (ascii_punct(c) || ascii_symbol(c)) && i + 1 < e && ascii_letter(cp[i + 1]))
  { /* [ASCII punctuation][A-Za-z]+ */
    j = i + 1;
    while (j < e && ascii_letter(cp[j]))
      j++;
    return j;
  }
  /* [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+: the optional character taken first, then without it */
  if (!is_nl(c) && !is_l(c) && !is_p(c) && !is_s(c) && (j = letters(cp, i + 1, e)))
    return j;
  if ((j = letters(cp, i, e)))
    return j;
  /* ' '?[\p{P}\p{S}]+[\r\n]*: the space taken first, then without it */
  if (c == ' ' && (j = punctuation(cp, i + 1, e)))
    return j;
  if ((j = punctuation(cp, i, e)))
    return j;
  size_t run = i, nl = i; /* \s*[\r\n]+: the run's last newline ends it */
  while (run < e && pretok_space(cp[run]))
    if (is_nl(cp[run++]))
      nl = run;
  if (nl > i)
    return nl;
  if (run == i)
    return i;
  if (run == e) /* \s+(?!\S): to the piece's end, or all but the space before a non-space */
    return run;
  return run - i > 1 ? run - 1 : run; /* (\s+ itself for a single space before a non-space) */
}

/* one expression over every piece: matches, and the text between them, each a piece */
static size_t pass(const uint32_t *cp, const size_t *in, size_t k, size_t *out,
                   size_t (*match)(const uint32_t *, size_t, size_t))
{
  size_t m = 0;
  out[m] = 0;
  for (size_t p = 0; p < k; p++)
  {
    const size_t e = in[p + 1];
    size_t gap = in[p];
    for (size_t i = in[p]; i < e;)
    {
      const size_t j = match(cp, i, e);
      if (j == i)
      {
        i++;
        continue;
      }
      if (i > gap)
        out[++m] = i;
      out[++m] = j;
      gap = i = j;
    }
    if (e > gap)
      out[++m] = e;
  }
  return m;
}

size_t pretok_split(const uint32_t *cp, size_t n, size_t *cuts)
{
  if (!n)
    return cuts[0] = 0, 0;
  size_t *tmp = malloc((n + 1) * sizeof *tmp);
  if (!tmp)
    return 0;
  size_t k = 1;
  cuts[0] = 0, cuts[1] = n;
  k = pass(cp, cuts, k, tmp, numbers);
  k = pass(cp, tmp, k, cuts, ideographs);
  k = pass(cp, cuts, k, tmp, words);
  memcpy(cuts, tmp, (k + 1) * sizeof *cuts);
  free(tmp);
  return k;
}

uint32_t pretok_utf8(const unsigned char *s, size_t left, size_t *used)
{
  const unsigned char b = s[0];
  *used = 1;
  if (b < 0x80)
    return b;
  int len = (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
  const uint32_t least[5] = {0, 0, 0x80, 0x800, 0x10000};
  if (!len || (size_t)len > left)
    return 0xFFFD;
  uint32_t c = b & (0x7F >> len);
  for (int i = 1; i < len; i++)
  {
    if ((s[i] & 0xC0) != 0x80)
      return 0xFFFD;
    c = c << 6 | (s[i] & 0x3F);
  }
  if (c < least[len])
    return 0xFFFD;
  *used = (size_t)len;
  return c;
}
