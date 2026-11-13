#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "bpe.h"
#include "pretok.h"

#define BYTE_CPS 324 /* bytes_to_unicode's code points: 0 .. 255 and 256 + the 68 bytes not printable as themselves */

typedef struct Map /* text -> int, open addressing, keys borrowed from the arena */
{
  const char **key;
  uint32_t *len;
  int32_t *val;
  size_t mask;
} Map;

struct Bpe
{
  char *arena;
  Map vocab, merge;
};

static uint16_t byte_cp[256];
static int16_t cp_byte[BYTE_CPS];
static pthread_once_t byte_once = PTHREAD_ONCE_INIT;

static void byte_map(void) /* GPT-2's: the printable bytes themselves, the others 256 + n in byte order */
{
  int n = 0;
  for (int b = 0; b < 256; b++)
  {
    const int own = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    byte_cp[b] = (uint16_t)(own ? b : 256 + n++);
  }
  for (int c = 0; c < BYTE_CPS; c++)
    cp_byte[c] = -1;
  for (int b = 0; b < 256; b++)
    cp_byte[byte_cp[b]] = (int16_t)b;
}

static uint64_t hash(const char *s, size_t n)
{
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++)
    h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
  return h;
}

static bool map_init(Map *m, size_t n)
{
  size_t cap = 1;
  while (cap < 2 * n + 2)
    cap <<= 1;
  m->mask = cap - 1;
  m->key = calloc(cap, sizeof *m->key), m->len = calloc(cap, sizeof *m->len), m->val = calloc(cap, sizeof *m->val);
  return m->key && m->len && m->val;
}

static void map_free(Map *m) { free(m->key), free(m->len), free(m->val); }

static void map_put(Map *m, const char *k, size_t n, int32_t v) /* the first of a text kept (a later duplicate not) */
{
  for (size_t i = hash(k, n) & m->mask;; i = (i + 1) & m->mask)
  {
    if (!m->key[i])
    {
      m->key[i] = k, m->len[i] = (uint32_t)n, m->val[i] = v;
      return;
    }
    if (m->len[i] == n && memcmp(m->key[i], k, n) == 0)
      return;
  }
}

static int32_t map_get(const Map *m, const char *k, size_t n)
{
  for (size_t i = hash(k, n) & m->mask; m->key[i]; i = (i + 1) & m->mask)
    if (m->len[i] == n && memcmp(m->key[i], k, n) == 0)
      return m->val[i];
  return -1;
}

Bpe *bpe_create(char **tokens, size_t n_tokens, char **merges, size_t n_merges)
{
  pthread_once(&byte_once, byte_map);
  Bpe *b = calloc(1, sizeof *b);
  size_t bytes = 0;
  for (size_t i = 0; i < n_tokens; i++)
    bytes += strlen(tokens[i]) + 1;
  for (size_t i = 0; i < n_merges; i++)
    bytes += strlen(merges[i]) + 1;
  if (!b || !(b->arena = malloc(bytes + 1)) || !map_init(&b->vocab, n_tokens) || !map_init(&b->merge, n_merges))
    return bpe_destroy(b), NULL;
  char *at = b->arena;
  for (size_t i = 0; i < n_tokens; i++)
  {
    const size_t n = strlen(tokens[i]);
    memcpy(at, tokens[i], n + 1);
    map_put(&b->vocab, at, n, (int32_t)i);
    at += n + 1;
  }
  for (size_t i = 0; i < n_merges; i++)
  {
    const size_t n = strlen(merges[i]);
    memcpy(at, merges[i], n + 1);
    map_put(&b->merge, at, n, (int32_t)i); /* "left right": the rank its place */
    at += n + 1;
  }
  return b;
}

void bpe_destroy(Bpe *b)
{
  if (!b)
    return;
  map_free(&b->vocab), map_free(&b->merge);
  free(b->arena);
  free(b);
}

int32_t bpe_id(const Bpe *b, const char *text, size_t len) { return map_get(&b->vocab, text, len); }

static size_t put_cp(char *out, uint16_t c) /* a stand-in's UTF-8: one byte below 128, else two (all below 0x800) */
{
  if (c < 0x80)
    return out[0] = (char)c, 1;
  out[0] = (char)(0xC0 | c >> 6), out[1] = (char)(0x80 | (c & 0x3F));
  return 2;
}

/* a pair of adjacent symbols that is a merge: its rank, its left symbol, both symbols' sizes when it was queued (a
   pair whose either side has since grown is stale: skipped when it comes up) */
typedef struct Pair
{
  int32_t rank;
  long left, right;
  size_t lsize, rsize;
} Pair;

static bool before(const Pair *a, const Pair *b) /* the lowest rank first, the leftmost among equals */
{
  return a->rank != b->rank ? a->rank < b->rank : a->left < b->left;
}

static void heap_push(Pair *h, size_t *n, Pair p)
{
  size_t i = (*n)++;
  for (; i && before(&p, &h[(i - 1) / 2]); i = (i - 1) / 2)
    h[i] = h[(i - 1) / 2];
  h[i] = p;
}

static Pair heap_pop(Pair *h, size_t *n)
{
  const Pair top = h[0], last = h[--*n];
  size_t i = 0;
  for (;;)
  {
    size_t c = 2 * i + 1;
    if (c >= *n)
      break;
    if (c + 1 < *n && before(&h[c + 1], &h[c]))
      c++;
    if (!before(&h[c], &last))
      break;
    h[i] = h[c], i = c;
  }
  if (*n)
    h[i] = last;
  return top;
}

/* the pair (i, j) queued if it is a merge */
static void queue(const Bpe *b, const char *buf, const size_t *start, const size_t *size, char *key, long i, long j,
                  Pair *h, size_t *n)
{
  if (i < 0 || j < 0)
    return;
  memcpy(key, buf + start[i], size[i]);
  key[size[i]] = ' ';
  memcpy(key + size[i] + 1, buf + start[j], size[j]);
  const int32_t r = map_get(&b->merge, key, size[i] + 1 + size[j]);
  if (r >= 0)
    heap_push(h, n, (Pair){r, i, j, size[i], size[j]});
}

size_t bpe_word(const Bpe *b, const unsigned char *word, size_t len, int32_t *ids, size_t cap)
{
  char *buf = malloc(2 * len + 1), *key = malloc(4 * len + 2);
  size_t *start = malloc(len * sizeof *start), *size = malloc(len * sizeof *size);
  long *next = malloc(len * sizeof *next), *prev = malloc(len * sizeof *prev);
  /* a merge queues at most two pairs, the bytes len - 1 at the start: 3 len bounds the queue */
  Pair *heap = malloc((3 * len + 1) * sizeof *heap);
  if (!buf || !key || !start || !size || !next || !prev || !heap)
  {
    free(buf), free(key), free(start), free(size), free(next), free(prev), free(heap);
    return 0;
  }
  size_t at = 0, nh = 0;
  for (size_t i = 0; i < len; i++) /* a symbol a byte: its stand-in's UTF-8, the symbols adjacent in buf */
  {
    start[i] = at, size[i] = put_cp(buf + at, byte_cp[word[i]]), at += size[i];
    next[i] = i + 1 < len ? (long)(i + 1) : -1, prev[i] = (long)i - 1;
  }
  for (size_t i = 0; i + 1 < len; i++)
    queue(b, buf, start, size, key, (long)i, (long)i + 1, heap, &nh);
  /* the lowest-ranked pair merged, the leftmost among equals, until no pair is a merge - llama.cpp's queue: each
     merge's new pairs queued, a stale one skipped (0.7.1: the word rescanned each merge, quadratic in its length) */
  while (nh)
  {
    const Pair p = heap_pop(heap, &nh);
    if (size[p.left] != p.lsize || next[p.left] != p.right || size[p.right] != p.rsize)
      continue; /* a side merged since: not this pair any more */
    size[p.left] += size[p.right], next[p.left] = next[p.right];
    if (next[p.right] >= 0)
      prev[next[p.right]] = p.left;
    size[p.right] = 0; /* gone: a pair still queued with it on either side is stale */
    queue(b, buf, start, size, key, prev[p.left], p.left, heap, &nh);
    queue(b, buf, start, size, key, p.left, next[p.left], heap, &nh);
  }
  size_t n = 0;
  for (long i = 0; i >= 0; i = next[i])
  {
    const int32_t id = map_get(&b->vocab, buf + start[i], size[i]);
    if (id >= 0)
    {
      if (n < cap)
        ids[n] = id;
      n++;
      continue;
    }
    for (size_t k = 0; k < size[i]; k++) /* llama.cpp's fallback: each byte of the symbol's text alone, if a token */
    {
      const int32_t one = map_get(&b->vocab, buf + start[i] + k, 1);
      if (one >= 0)
      {
        if (n < cap)
          ids[n] = one;
        n++;
      }
    }
  }
  free(buf), free(key), free(start), free(size), free(next), free(prev), free(heap);
  return n;
}

int bpe_decode(const char *text, unsigned char *out, size_t cap)
{
  pthread_once(&byte_once, byte_map);
  size_t n = 0, len = strlen(text);
  for (size_t i = 0; i < len;)
  {
    size_t used;
    const uint32_t c = pretok_utf8((const unsigned char *)text + i, len - i, &used);
    if (c >= BYTE_CPS || cp_byte[c] < 0 || n >= cap)
      return -1;
    out[n++] = (unsigned char)cp_byte[c];
    i += used;
  }
  return (int)n;
}
