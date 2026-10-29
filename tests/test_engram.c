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

/* the oracle's hashes of the 32 real ids (oracle/ref_gguf.py's Hasher, ~/substrate-oracle/engram_hash32.i64,
   [T][layers][cols]) against engram_rows decoding them a token at a time, as the decode does */
int main(void)
{
  printf("[test_engram] start\n");
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
