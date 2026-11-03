#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/model/engram.h"
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"

#define T 32
#define LAYERS 2
#define COLS 24

/* the hash in the model's metadata (llama.cpp's ple.*: unsloth's UD-IQ1_M): every row of 64 positions the
   oracle's (oracle/engram_rows.py: the primes, offsets and multipliers from DeepSeek's recipe, not the file's lists);
   a map entry past the compressed vocabulary and an even multiplier refused */
static void test_own(void)
{
  char p[512], err[GGUF_ERR], werr[WEIGHTS_ERR], eerr[ENGRAM_ERR] = "";
  const char *home = getenv("HOME");
  snprintf(p, sizeof p, "%s/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf", home);
  Gguf *g = gguf_create(p, err);
  snprintf(p, sizeof p, "%s/substrate-oracle/engram_rows_ud.bin", home);
  FILE *ref = fopen(p, "rb");
  if (!g || !ref)
  {
    printf("  UD-IQ1_M or its reference rows (oracle/engram_rows.py) not on this machine, skipped\n");
    if (g)
      gguf_destroy(g);
    return;
  }
  char magic[4];
  uint32_t head[4];
  assert(fread(magic, 4, 1, ref) == 1 && memcmp(magic, "SUBE", 4) == 0 && fread(head, 4, 4, ref) == 4);
  const uint32_t n = head[1], layers = head[2], cols = head[3];
  assert(head[0] == 1 && layers == LAYERS && cols == COLS);
  int32_t *ids = malloc(n * sizeof *ids);
  int64_t *want = malloc((size_t)n * LAYERS * COLS * sizeof *want);
  assert(ids && want && fread(ids, sizeof *ids, n, ref) == n &&
         fread(want, sizeof *want, (size_t)n * LAYERS * COLS, ref) == (size_t)n * LAYERS * COLS);
  fclose(ref);
  Weights *w = weights_create(g, NULL, werr);
  assert(w && w->geo.engram_own);
  int fds[8];
  for (int f = 0; f < g->n_files; f++)
    assert((fds[f] = open(g->paths[f], O_RDONLY)) >= 0);
  Engram *e = engram_create(w, fds, eerr);
  if (!e)
    printf("  refused: %s\n", eerr);
  assert(e && engram_cols(e) == COLS && engram_ngram(e) == 4);
  for (uint32_t t = 0; t < n; t++)
  {
    int32_t last[4];
    int k = 0;
    for (int s = 0; s < 4 && (int)t - s >= 0; s++, k++)
      last[s] = ids[t - s];
    int64_t rows[LAYERS * COLS];
    assert(engram_rows(e, last, k, rows));
    if (memcmp(rows, want + (size_t)t * LAYERS * COLS, sizeof rows) != 0)
    {
      printf("  position %u: the rows differ from the oracle's\n", t);
      exit(1);
    }
  }
  engram_destroy(e);
  printf("  UD-IQ1_M: %u positions, %d rows each, every one the oracle's (the recipe's primes and multipliers)\n", n,
         LAYERS * COLS);

  GgufKv *map = (GgufKv *)gguf_kv(g, "deepseek41.engram.token_map");
  int32_t *m = map->v.items, was = m[1000];
  m[1000] = (int32_t)w->geo.engram_vocab;
  assert(!engram_create(w, fds, eerr) && strstr(eerr, "map"));
  printf("    a map entry past the compressed vocabulary -> %s\n", eerr);
  m[1000] = was;
  GgufKv *mult = (GgufKv *)gguf_kv(g, "deepseek41.ple.layer_multipliers");
  uint64_t *mu = mult->v.items, mwas = mu[5];
  mu[5] = mwas + 1;
  assert(!engram_create(w, fds, eerr) && strstr(eerr, "multiplier"));
  printf("    an even multiplier -> %s\n", eerr);
  mu[5] = mwas;
  assert((e = engram_create(w, fds, eerr)));
  engram_destroy(e);
  for (int f = 0; f < g->n_files; f++)
    close(fds[f]);
  weights_destroy(w);
  gguf_destroy(g);
  free(ids);
  free(want);
}

/* the oracle's hashes of the 32 real ids (oracle/ref_gguf.py's Hasher, ~/substrate-oracle/engram_hash32.i64,
   [T][layers][cols]) against engram_rows decoding them a token at a time, as the decode does */
int main(void)
{
  printf("[test_engram] start\n");
  test_own();
  char p[512], err[GGUF_ERR], werr[WEIGHTS_ERR], eerr[ENGRAM_ERR] = "";
  const char *home = getenv("HOME");
  snprintf(p, sizeof p, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf", home);
  Gguf *g = gguf_create(p, err);
  snprintf(p, sizeof p, "%s/models/ds1017/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", home);
  Gguf *eg = gguf_create(p, err);
  snprintf(p, sizeof p, "%s/substrate-oracle/engram_hash32.i64", home);
  FILE *ref = fopen(p, "rb");
  snprintf(p, sizeof p, "%s/substrate-oracle/ids32.txt", home);
  FILE *idf = fopen(p, "r");
  if (!g || !eg || !ref || !idf)
  {
    printf("  ds1017 or the oracle's hashes not on this machine, skipped\n[test_engram] ok\n");
    return 0;
  }
  Weights *w = weights_create(g, eg, werr);
  assert(w);
  int fds[8];
  for (int f = 0; f < g->n_files; f++)
    assert((fds[f] = open(g->paths[f], O_RDONLY)) >= 0);
  assert(engram_create(NULL, fds, eerr) == NULL && eerr[0]);
  Engram *e = engram_create(w, fds, eerr);
  if (!e)
    printf("  refused: %s\n", eerr);
  assert(e && engram_cols(e) == COLS && engram_ngram(e) == 4);

  int32_t ids[T];
  for (int i = 0; i < T; i++)
    assert(fscanf(idf, "%d,", &ids[i]) == 1);
  static int64_t want[T][LAYERS][COLS];
  assert(fread(want, sizeof want, 1, ref) == 1);
  for (int t = 0; t < T; t++)
  {
    int32_t last[4];
    int n = 0;
    for (int s = 0; s < 4 && t - s >= 0; s++, n++)
      last[s] = ids[t - s];
    int64_t rows[LAYERS * COLS];
    assert(engram_rows(e, last, n, rows));
    assert(memcmp(rows, want[t], sizeof rows) == 0);
  }
  printf("  32 positions, a token at a time: %d rows each, every one the oracle's\n", LAYERS * COLS);
  const int32_t bad[1] = {129280};
  int64_t rows[LAYERS * COLS];
  assert(!engram_rows(e, bad, 1, rows));
  printf("  a token past the vocabulary refused\n");
  engram_destroy(e);
  engram_destroy(NULL);
  for (int f = 0; f < g->n_files; f++)
    close(fds[f]);
  fclose(ref);
  fclose(idf);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  printf("[test_engram] ok\n");
  return 0;
}
