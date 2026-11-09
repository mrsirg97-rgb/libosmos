#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bpe.h"
#include "osmos/text/tokenizer.h"
#include "pretok.h"

/* llama.cpp's token types (tokenizer.ggml.token_type) */
enum { TYPE_NORMAL = 1, TYPE_UNKNOWN = 2, TYPE_CONTROL = 3, TYPE_USER = 4 };

struct Tokenizer
{
  Bpe *bpe;
  int32_t n;
  uint8_t *type;
  char **text;     /* each token's text (its own copies) */
  int32_t *special; /* the special tokens, the longest text first (ties: the lower id) */
  int32_t n_special;
  size_t (*split)(const uint32_t *cp, size_t n, size_t *cuts); /* the pre-tokenizer's (pretok.h) */
};

static int longest_first(const void *a, const void *b, void *arg)
{
  const Tokenizer *t = arg;
  const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
  const size_t lx = strlen(t->text[x]), ly = strlen(t->text[y]);
  return lx != ly ? (lx < ly ? 1 : -1) : (x > y) - (x < y);
}

static bool fail(char *err, const char *why)
{
  snprintf(err, TOKENIZER_ERR, "tokenizer: %s", why);
  return false;
}

static const GgufKv *list(const Gguf *g, const char *key, GgufValueType item)
{
  const GgufKv *kv = gguf_kv(g, key);
  return kv && kv->type == GGUF_ARRAY && kv->item == item && kv->n ? kv : NULL;
}

Tokenizer *tokenizer_create(const Gguf *g, char err[TOKENIZER_ERR])
{
  const GgufKv *model = g ? gguf_kv(g, "tokenizer.ggml.model") : NULL;
  const GgufKv *pre = g ? gguf_kv(g, "tokenizer.ggml.pre") : NULL;
  const GgufKv *tok = g ? list(g, "tokenizer.ggml.tokens", GGUF_STRING) : NULL;
  const GgufKv *typ = g ? gguf_kv(g, "tokenizer.ggml.token_type") : NULL;
  const GgufKv *mer = g ? list(g, "tokenizer.ggml.merges", GGUF_STRING) : NULL;
  if (!model || model->type != GGUF_STRING || strcmp(model->v.s, "gpt2"))
    return fail(err, "not a byte-level BPE (tokenizer.ggml.model gpt2)"), NULL;
  const bool qwen = pre && pre->type == GGUF_STRING && !strcmp(pre->v.s, "qwen35");
  if (!pre || pre->type != GGUF_STRING ||
      (strcmp(pre->v.s, "deepseek-v3") && strcmp(pre->v.s, "joyai-llm") && !qwen))
    return fail(err, "a pre-tokenizer this build does not split by (deepseek-v3, joyai-llm, qwen35)"), NULL;
  if (!tok || !mer || !typ || typ->type != GGUF_ARRAY || typ->n != tok->n || tok->n > INT32_MAX)
    return fail(err, "no tokens, token types or merges, or of unequal lengths"), NULL;
  Tokenizer *t = calloc(1, sizeof *t);
  bool ok = t != NULL;
  if (ok)
  {
    t->split = qwen ? pretok_split_qwen35 : pretok_split;
    t->n = (int32_t)tok->n;
    ok = (t->type = calloc(tok->n, 1)) && (t->text = calloc(tok->n, sizeof *t->text)) &&
         (t->special = malloc(tok->n * sizeof *t->special));
  }
  char **texts = ok ? (char **)tok->v.items : NULL;
  for (int32_t i = 0; ok && i < t->n; i++)
  {
    int64_t ty;
    unsigned char piece[TOKENIZER_PIECE_MAX];
    ok = gguf_int(typ, (size_t)i, &ty) && (t->text[i] = strdup(texts[i])) &&
         strlen(texts[i]) < TOKENIZER_PIECE_MAX;
    t->type[i] = ok ? (uint8_t)ty : 0;
    if (ok && (ty == TYPE_CONTROL || ty == TYPE_USER || ty == TYPE_UNKNOWN))
      t->special[t->n_special++] = i;
    else if (ok && ty == TYPE_NORMAL)
      ok = bpe_decode(texts[i], piece, sizeof piece) >= 0; /* every normal token decodes to bytes */
  }
  if (!ok)
    fail(err, "a token type, text or byte it does not read");
  else if (!(t->bpe = bpe_create(texts, tok->n, (char **)mer->v.items, mer->n)))
    ok = fail(err, "out of memory");
  if (ok)
  {
    qsort_r(t->special, (size_t)t->n_special, sizeof *t->special, longest_first, t);
  }
  if (!ok)
  {
    tokenizer_destroy(t);
    return NULL;
  }
  return t;
}

void tokenizer_destroy(Tokenizer *t)
{
  if (!t)
    return;
  for (int32_t i = 0; t->text && i < t->n; i++)
    free(t->text[i]);
  free(t->text), free(t->type), free(t->special);
  bpe_destroy(t->bpe);
  free(t);
}

typedef struct Frag /* a stretch of the text: raw (id < 0) or a special token's */
{
  size_t off, len;
  int32_t id;
} Frag;

/* the text's fragments: each eligible special token, the longest first, cut out of the raw ones left to right */
static Frag *partition(const Tokenizer *t, const char *s, size_t len, bool parse_special, size_t *n_out)
{
  size_t n = 1, cap = 16;
  Frag *f = malloc(cap * sizeof *f);
  if (!f)
    return NULL;
  f[0] = (Frag){0, len, -1};
  for (int32_t k = 0; k < t->n_special; k++)
  {
    const int32_t id = t->special[k];
    if (!parse_special && t->type[id] != TYPE_USER)
      continue;
    const char *x = t->text[id];
    const size_t xl = strlen(x);
    for (size_t i = 0; i < n; i++)
    {
      if (f[i].id >= 0 || f[i].len < xl)
        continue;
      const char *hit = memmem(s + f[i].off, f[i].len, x, xl);
      if (!hit)
        continue;
      if (n + 2 > cap)
      {
        Frag *g = realloc(f, (cap *= 2) * sizeof *f);
        if (!g)
          return free(f), NULL;
        f = g;
      }
      const size_t at = (size_t)(hit - s), end = f[i].off + f[i].len;
      const size_t left = at - f[i].off, right = end - (at + xl);
      Frag mid = {at, xl, id}, rest = {at + xl, right, -1};
      memmove(f + i + 1 + (left > 0) + (right > 0), f + i + 1, (n - i - 1) * sizeof *f);
      size_t w = i;
      if (left)
        f[w++] = (Frag){f[i].off, left, -1};
      f[w++] = mid;
      if (right)
        f[w] = rest; /* scanned next: a later match of the same token in it */
      n += (left > 0) + (right > 0);
      i = w - 1;
    }
  }
  *n_out = n;
  return f;
}

/* a raw fragment's ids: code points (an invalid byte U+FFFD), its pre-tokens, each merged */
static size_t raw(const Tokenizer *t, const unsigned char *s, size_t len, int32_t *ids, size_t cap)
{
  uint32_t *cp = malloc((len + 1) * sizeof *cp);
  size_t *cuts = malloc((len + 2) * sizeof *cuts), m = 0, n = 0;
  unsigned char *word = malloc(4 * len + 4);
  if (!cp || !cuts || !word)
    return free(cp), free(cuts), free(word), 0;
  for (size_t i = 0, used; i < len; i += used)
    cp[m++] = pretok_utf8(s + i, len - i, &used);
  const size_t k = t->split(cp, m, cuts);
  for (size_t p = 0; p < k; p++)
  {
    size_t w = 0; /* the piece's code points as UTF-8 again (U+FFFD's three bytes for an invalid byte) */
    for (size_t i = cuts[p]; i < cuts[p + 1]; i++)
    {
      const uint32_t c = cp[i];
      if (c < 0x80)
        word[w++] = (unsigned char)c;
      else if (c < 0x800)
        word[w++] = (unsigned char)(0xC0 | c >> 6), word[w++] = (unsigned char)(0x80 | (c & 0x3F));
      else if (c < 0x10000)
        word[w++] = (unsigned char)(0xE0 | c >> 12), word[w++] = (unsigned char)(0x80 | (c >> 6 & 0x3F)),
        word[w++] = (unsigned char)(0x80 | (c & 0x3F));
      else
        word[w++] = (unsigned char)(0xF0 | c >> 18), word[w++] = (unsigned char)(0x80 | (c >> 12 & 0x3F)),
        word[w++] = (unsigned char)(0x80 | (c >> 6 & 0x3F)), word[w++] = (unsigned char)(0x80 | (c & 0x3F));
    }
    n += bpe_word(t->bpe, word, w, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
  }
  free(cp), free(cuts), free(word);
  return n;
}

int64_t tokenizer_encode(const Tokenizer *t, const char *text, size_t len, bool parse_special, int32_t *ids,
                         size_t cap)
{
  if (!t || (!text && len))
    return -1;
  size_t nf = 0, n = 0;
  Frag *f = len ? partition(t, text, len, parse_special, &nf) : NULL;
  if (len && !f)
    return -1;
  for (size_t i = 0; i < nf; i++)
  {
    if (f[i].id >= 0)
    {
      if (n < cap)
        ids[n] = f[i].id;
      n++;
    }
    else
      n += raw(t, (const unsigned char *)text + f[i].off, f[i].len, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
  }
  free(f);
  return (int64_t)n;
}

size_t tokenizer_defuse(const Tokenizer *t, char *s, size_t len)
{
  size_t changed = 0;
  for (int32_t k = 0; t && s && k < t->n_special; k++)
  {
    const char *x = t->text[t->special[k]];
    const size_t xl = strlen(x);
    for (char *at = xl ? memmem(s, len, x, xl) : NULL; at; at = memmem(at, len - (size_t)(at - s), x, xl))
    {
      const unsigned char lead = (unsigned char)at[0];
      const size_t w = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1; /* its first character */
      at[w <= xl ? w - 1 : 0] ^= 1;
      changed++;
    }
  }
  return changed;
}

int64_t tokenizer_encode_shaped(const Tokenizer *t, const char *text, const char *shape, size_t len, int32_t *ids,
                                size_t cap)
{
  if (!t || (len && (!text || !shape)))
    return -1;
  for (size_t i = 0; i < len; i++) /* the shape is the text defused: a byte apart by one bit at most */
    if (((unsigned char)text[i] ^ (unsigned char)shape[i]) > 1)
      return -1;
  size_t nf = 0, n = 0;
  Frag *f = len ? partition(t, shape, len, true, &nf) : NULL;
  if (len && !f)
    return -1;
  for (size_t i = 0; i < nf; i++)
  {
    if (f[i].id >= 0)
    {
      if (memcmp(text + f[i].off, shape + f[i].off, f[i].len)) /* a special token the text doesn't have */
        return free(f), -1;
      if (n < cap)
        ids[n] = f[i].id;
      n++;
    }
    else
      n += raw(t, (const unsigned char *)text + f[i].off, f[i].len, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
  }
  free(f);
  return (int64_t)n;
}

bool tokenizer_special(const Tokenizer *t, int32_t id)
{
  return t && id >= 0 && id < t->n &&
         (t->type[id] == TYPE_CONTROL || t->type[id] == TYPE_USER || t->type[id] == TYPE_UNKNOWN);
}

int tokenizer_piece(const Tokenizer *t, int32_t id, bool special, char out[TOKENIZER_PIECE_MAX])
{
  if (!t || id < 0 || id >= t->n)
    return -1;
  const char *x = t->text[id];
  if (t->type[id] == TYPE_NORMAL)
    return bpe_decode(x, (unsigned char *)out, TOKENIZER_PIECE_MAX);
  if (t->type[id] == TYPE_CONTROL && !special)
    return 0;
  const size_t n = strlen(x); /* a control token's, a user-defined one's (`<think>`), an unknown's: its text */
  memcpy(out, x, n);
  return (int)n;
}

int32_t tokenizer_find(const Tokenizer *t, const char *text) { return t ? bpe_id(t->bpe, text, strlen(text)) : -1; }

int32_t tokenizer_vocab(const Tokenizer *t) { return t ? t->n : 0; }
