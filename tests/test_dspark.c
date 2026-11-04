#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "substrate/model/dspark.h"

/* the index on the real drafter (~/models/dspark: skipped without it): its tensors as the pack has them; and
   malformed indexes refused, each with why */

static char dir[] = "/tmp/test_dspark_XXXXXX";

static void write(const char *name, const char *text, size_t bin_bytes)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  if (text)
    fputs(text, f);
  else
    for (size_t i = 0; i < bin_bytes; i++)
      fputc(0, f);
  fclose(f);
}

static void refused(const char *json, const char *why)
{
  write("dspark.json", json, 0);
  char err[DSPARK_ERR] = "";
  assert(dspark_index(dir, err) == NULL && err[0]);
  printf("  refused (%s): %s\n", why, err);
}

static void test_malformed(void)
{
  assert(mkdtemp(dir));
  write("dspark.bin", NULL, 4096);
  refused("{\"tensors\": [[\"a\", \"Q4_K\", 0, 144, [256]]]}", "a type it does not know");
  refused("{\"tensors\": [[\"a\", \"Q8_0\", 0, 35, [32]]]}", "bytes not the type's for the dims");
  refused("{\"tensors\": [[\"a\", \"IQ2_XS\", 0, 75, [256]]]}", "bytes not an IQ2_XS block's");
  refused("{\"tensors\": [[\"a\", \"F32\", 4000, 400, [100]]]}", "past the end of the bin");
  refused("{\"tensors\": [[\"a\", \"F32\", 0, 400, [100]]", "truncated");
  refused("{\"other\": 1}", "no tensors");
  write("dspark.json", "{\"tensors\": [[\"a.w\", \"MXFP4\", 17, 34, [2, 32]], [\"b\", \"BF16\", 64, 8, [4]]]}", 0);
  char err[DSPARK_ERR] = "";
  DsparkIndex *x = dspark_index(dir, err);
  assert(x && x->n == 2 && dspark_tensor(x, "b")->type == 30 && dspark_tensor(x, "a.w")->dims[1] == 32);
  assert(!dspark_tensor(x, "c"));
  dspark_index_destroy(x);
  /* a drafter's experts requantized to the trunk's types: the type table's sizes */
  write("dspark.json", "{\"tensors\": [[\"e\", \"IQ2_XS\", 0, 74, [1, 256]]]}", 0);
  x = dspark_index(dir, err);
  assert(x && x->n == 1 && dspark_tensor(x, "e")->type == 17);
  dspark_index_destroy(x);
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("  a well-formed one: its two tensors, by name\n");
}

static void test_real(void)
{
  char path[512], err[DSPARK_ERR] = "";
  snprintf(path, sizeof path, "%s/models/dspark", getenv("HOME") ? getenv("HOME") : "");
  struct stat st;
  char json[600];
  snprintf(json, sizeof json, "%s/dspark.json", path);
  if (stat(json, &st) != 0)
  {
    printf("  the real index: skipped (no ~/models/dspark)\n");
    return;
  }
  DsparkIndex *x = dspark_index(path, err);
  if (!x)
    printf("  %s\n", err);
  assert(x && x->n == 1224);
  for (uint32_t i = 0; i < x->n; i++)
    for (uint32_t j = i + 1; j < x->n; j++)
      assert(strcmp(x->t[i].name, x->t[j].name));
  for (int b = 0; b < 3; b++)
    for (int e = 0; e < 128; e++)
      for (int w = 1; w <= 3; w++)
      {
        char n[DSPARK_NAME_MAX];
        snprintf(n, sizeof n, "mtp.%d.ffn.experts.%d.w%d.weight", b, e, w);
        const DsparkTensor *t = dspark_tensor(x, n);
        assert(t && t->type == 39 && t->n_dims == 2 && t->dims[0] * t->dims[1] == 2304ull * 5120);
      }
  const DsparkTensor *mp = dspark_tensor(x, "mtp.0.main_proj.weight");
  assert(mp && mp->type == 8 && mp->dims[0] == 5120 && mp->dims[1] == 15360);
  const DsparkTensor *hm = dspark_tensor(x, "mtp.2.markov_head.head.weight");
  assert(hm && hm->type == 30 && hm->dims[0] == 129280 && hm->dims[1] == 256);
  assert(dspark_tensor(x, "mtp.2.confidence_head.proj.weight")->dims[1] == 5376);
  dspark_index_destroy(x);
  printf("  the real index: 1224 tensors, names unique, 3 x 128 x 3 MXFP4 experts, the heads' shapes\n");
}

int main(void)
{
  printf("[test_dspark] start\n");
  test_malformed();
  test_real();
  printf("[test_dspark] ok\n");
  return 0;
}
