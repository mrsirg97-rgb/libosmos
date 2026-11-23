/* Hy4-flash's Engram hash (hy4_engram_shape -> engram_create_shape -> engram_rows) against the reference's
   (oracle/hy4_engram_rows.py --modeling: modeling_hy_v4_flash.py's own hash code on the release's lookup table and config, never the
   GGUF's lists): every row of 512 positions as one sequence, the pad before its start, the tokens past the release's
   table (llama.cpp's identity extension) read as the pad. then the map edited: an entry no compressed token refused.
   OSMOS_TEST_HY4 (or ~/Models/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf; a sparse header copy is enough) and
   ~/osmos-oracle/hy4_engram_rows.bin; skipped without */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/err.h"
#include "osmos/hy4/weights.h"
#include "osmos/model/engram.h"
#include "osmos/model/gguf.h"

#define MODEL "%s/Models/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf"
#define ROWS "%s/osmos-oracle/hy4_engram_rows.bin"
#define TABLE_TOKENS 120818  /* the release's lookup_table (model.layers.N.engram_layer.lookup_table: I32 [120818]) */
#define COMPRESSED 92620     /* its greatest entry + 1; the multipliers' bound (draw_hash_multipliers) */

typedef struct Ref
{
  uint32_t n, layers, cols;
  int32_t *ids;
  int64_t *rows;
} Ref;

static bool load(const char *path, Ref *r)
{
  FILE *f = fopen(path, "rb");
  if (!f)
    return false;
  char magic[4];
  uint32_t head[4];
  assert(fread(magic, 4, 1, f) == 1 && memcmp(magic, "HY4E", 4) == 0 && fread(head, 4, 4, f) == 4 && head[0] == 1);
  r->n = head[1], r->layers = head[2], r->cols = head[3];
  r->ids = malloc(r->n * sizeof *r->ids);
  r->rows = malloc((size_t)r->n * r->layers * r->cols * sizeof *r->rows);
  assert(r->ids && r->rows && fread(r->ids, sizeof *r->ids, r->n, f) == r->n);
  assert(fread(r->rows, sizeof *r->rows, (size_t)r->n * r->layers * r->cols, f) == (size_t)r->n * r->layers * r->cols);
  fclose(f);
  return true;
}

static void test_shape(const EngramShape *s)
{
  assert(s->n_layers == 4 && s->ngram == 3 && s->heads == 8 && s->pad == 120002 && s->n_vocab == 120832);
  assert(s->covered == TABLE_TOKENS && s->compressed == COMPRESSED && s->rows);
  printf("  the map covers %llu tokens (%d compressed); the last %llu read as the pad\n",
         (unsigned long long)s->covered, (int)s->compressed, (unsigned long long)(s->n_vocab - s->covered));
}

static void test_rows(const EngramShape *s, const Ref *r)
{
  char err[ENGRAM_ERR] = "";
  Engram *e = engram_create_shape(s, err);
  if (!e)
    printf("  refused: %s\n", err);
  assert(e && engram_cols(e) == (int)r->cols && engram_ngram(e) == 3);
  int64_t *got = malloc((size_t)r->layers * r->cols * sizeof *got);
  uint32_t bad = 0;
  for (uint32_t p = 0; p < r->n; p++)
  {
    int32_t last[3];
    const int n_last = p + 1 < 3 ? (int)p + 1 : 3;
    for (int k = 0; k < n_last; k++)
      last[k] = r->ids[p - (uint32_t)k];
    assert(engram_rows(e, last, n_last, got));
    const int64_t *want = r->rows + (size_t)p * r->layers * r->cols;
    if (memcmp(got, want, (size_t)r->layers * r->cols * sizeof *got) && bad++ < 4)
      printf("  position %u (token %d): layer 0 col 0 row %lld, the reference's %lld\n", p, r->ids[p],
             (long long)got[0], (long long)want[0]);
  }
  assert(!bad);
  printf("  %u positions x %u layers x %u columns: every row the reference's\n", r->n, r->layers, r->cols);
  free(got);
  engram_destroy(e);
}

static void test_refused(Gguf *g, const Hy4Weights *w)
{
  printf("  refused:\n");
  GgufKv *map = (GgufKv *)gguf_kv(g, "hy_v4_flash.engram.token_map");
  assert(map);
  int32_t *items = map->v.items, was = items[5];
  items[5] = 200000; /* covered: its product with a multiplier past int64 */
  EngramShape s;
  char err[OSMOS_ERR] = "", eerr[ENGRAM_ERR] = "";
  osmos_clear(err);
  Engram *e = hy4_engram_shape(w, &s, err) ? engram_create_shape(&s, eerr) : NULL;
  assert(!e && (err[0] || eerr[0]));
  printf("    %-52s -> %s\n", "a covered entry whose hash overflows int64", err[0] ? err : eerr);
  items[5] = was;
  s = (EngramShape){0};
  osmos_clear(err);
  assert(hy4_engram_shape(w, &s, err));
  s.covered = s.n_vocab + 1;
  assert(!(e = engram_create_shape(&s, eerr)));
  printf("    %-52s -> %s\n", "the map covering past its entries", eerr);
  assert(!engram_create_shape(NULL, eerr));
}

int main(void)
{
  printf("[test_hy4_engram] start\n");
  const char *home = getenv("HOME") ? getenv("HOME") : "";
  char path[512], gerr[GGUF_ERR] = "", err[OSMOS_ERR] = "";
  if (getenv("OSMOS_TEST_HY4"))
    snprintf(path, sizeof path, "%s", getenv("OSMOS_TEST_HY4"));
  else
    snprintf(path, sizeof path, MODEL, home);
  Gguf *g = gguf_create(path, gerr);
  char rpath[512];
  snprintf(rpath, sizeof rpath, ROWS, home);
  Ref r;
  if (!g || !load(rpath, &r))
  {
    printf("  Hy4-flash's GGUF or the reference rows (oracle/hy4_engram_rows.py) not on this machine, skipped\n");
    if (g)
      gguf_destroy(g);
    printf("[test_hy4_engram] ok\n");
    return 0;
  }
  osmos_clear(err);
  Hy4Weights *w = hy4_weights_create(g, err);
  if (!w)
    printf("  refused: %s\n", err);
  assert(w);
  EngramShape s;
  osmos_clear(err);
  assert(hy4_engram_shape(w, &s, err) || (printf("  refused: %s\n", err), 0));
  test_shape(&s);
  test_rows(&s, &r);
  test_refused(g, w);
  hy4_weights_destroy(w);
  gguf_destroy(g);
  free(r.ids), free(r.rows);
  printf("[test_hy4_engram] ok\n");
  return 0;
}
