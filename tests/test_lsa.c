/* latent sparse attention's read (osmos/lsa.h): its invariants on random keys, then - with OSMOS_LSA_CASES, a folder
   of tools/convert/lsa_cases.py's cases (real keys and queries of -vibes under its trained maps) - how its buckets and
   reads agree with lsa.py's, whose F32 ran on a GPU (the measured agreement printed; a block or read off it named) */
#include <assert.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/lsa.h"

static uint32_t rng = 655;
static float uniform(void)
{
  rng = rng * 1664525u + 1013904223u;
  return (float)(rng >> 8) / 16777216.0f - 0.5f;
}

static void synthetic(void)
{
  enum { B = 3, T = B * LSA_WINDOW + 300 };
  float *ik = malloc((size_t)T * LSA_DIM * 4), *cent = malloc((size_t)B * LSA_BUCKETS * LSA_DIM * 4);
  int32_t *mem = malloc((size_t)B * LSA_BUCKETS * LSA_CAP * 4);
  uint32_t *read = malloc(LSA_READ_MAX * 4);
  for (int t = 0; t < T; t++)
  {
    double n = 0;
    for (int d = 0; d < LSA_DIM; d++)
    {
      const float v = uniform();
      ik[t * LSA_DIM + d] = v, n += (double)v * v;
    }
    for (int d = 0; d < LSA_DIM; d++)
      ik[t * LSA_DIM + d] /= (float)sqrt(n);
  }
  for (int b = 0; b < B; b++)
    assert(lsa_block(ik + (size_t)b * LSA_WINDOW * LSA_DIM, (uint32_t)b, cent + (size_t)b * LSA_BUCKETS * LSA_DIM,
                     mem + (size_t)b * LSA_BUCKETS * LSA_CAP));
  int seen[B * LSA_WINDOW] = {0};
  for (int k = 0; k < B * LSA_BUCKETS; k++)
  {
    int32_t last = -1, n = 0;
    for (int s = 0; s < LSA_CAP; s++)
    {
      const int32_t m = mem[k * LSA_CAP + s];
      if (m < 0)
        continue;
      assert(m > last && m / LSA_WINDOW == k / LSA_BUCKETS); /* ascending, within its own block */
      seen[m]++, last = m, n++;
    }
    assert(n >= 1);
  }
  for (int i = 0; i < B * LSA_WINDOW; i++)
    assert(seen[i] == 1);
  for (uint32_t p = 0; p < T; p += 97)
  {
    const uint32_t n = lsa_read(ik + (size_t)p * LSA_DIM, p, cent, mem, B, read);
    static unsigned char hit[T];
    memset(hit, 0, sizeof hit);
    for (uint32_t i = 0; i < n; i++)
      assert(read[i] <= p && !hit[read[i]]++);
    const uint32_t w0 = p + 1 > LSA_WINDOW ? p + 1 - LSA_WINDOW : 0;
    for (uint32_t q = w0; q <= p; q++)
      assert(hit[q]);
    for (uint32_t q = 0; q < LSA_SINK && q <= p; q++)
      assert(hit[q]);
  }
  printf("  random keys, %d blocks: every key in one bucket of at most %d, ascending; every read its window and sinks, "
         "no position twice\n", B, LSA_CAP);
  free(ik), free(cent), free(mem), free(read);
}

static void *slurp(const char *path, size_t *n)
{
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void *b = malloc(*n);
  if (b && fread(b, 1, *n, f) != *n)
    free(b), b = NULL;
  fclose(f);
  return b;
}

static int cmp_u32(const void *a, const void *b)
{
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return (x > y) - (x < y);
}

static void one_case(const char *dir, const char *name)
{
  char path[1024];
  size_t n = 0, nw = 0;
  snprintf(path, sizeof path, "%s/%s", dir, name);
  unsigned char *b = slurp(path, &n);
  snprintf(path, sizeof path, "%s/%s.W", dir, name);
  float *W = slurp(path, &nw);
  assert(b && W && !memcmp(b, "LSA1", 4));
  uint32_t T, Q, G;
  memcpy(&T, b + 4, 4), memcpy(&Q, b + 8, 4), memcpy(&G, b + 12, 4);
  const float *ik = (const float *)(b + 16), *k0 = ik + (size_t)T * LSA_DIM, *ik0 = k0 + LSA_IN;
  const float *cent = ik0 + LSA_DIM;
  const int32_t *mem = (const int32_t *)(cent + (size_t)G * LSA_BUCKETS * LSA_DIM);
  const unsigned char *qs = (const unsigned char *)(mem + (size_t)G * LSA_BUCKETS * LSA_CAP);
  /* lsa_index: the first key, against lsa.py's */
  float mine[LSA_DIM];
  lsa_index(k0, LSA_IN, W, true, mine);
  int same_bits = 0;
  for (int d = 0; d < LSA_DIM; d++)
    same_bits += mine[d] == ik0[d];
  /* the blocks: lsa_block on lsa.py's index vectors, its buckets against lsa.py's */
  float *c = malloc((size_t)G * LSA_BUCKETS * LSA_DIM * 4);
  int32_t *m = malloc((size_t)G * LSA_BUCKETS * LSA_CAP * 4);
  uint32_t blocks_same = 0;
  double cmax = 0;
  for (uint32_t g = 0; g < G; g++)
  {
    assert(lsa_block(ik + (size_t)g * LSA_WINDOW * LSA_DIM, g, c + (size_t)g * LSA_BUCKETS * LSA_DIM,
                     m + (size_t)g * LSA_BUCKETS * LSA_CAP));
    const size_t mo = (size_t)g * LSA_BUCKETS * LSA_CAP, co = (size_t)g * LSA_BUCKETS * LSA_DIM;
    const bool same = !memcmp(m + mo, mem + mo, LSA_BUCKETS * LSA_CAP * 4);
    blocks_same += same;
    for (size_t i = 0; same && i < LSA_BUCKETS * LSA_DIM; i++)
      cmax = fmax(cmax, fabs((double)c[co + i] - cent[co + i]));
  }
  /* the reads: lsa_read over its own buckets, against lsa.py's served reads, as ordered lists and as sets */
  uint32_t *mine_r = malloc(LSA_READ_MAX * 4), *a = malloc(LSA_READ_MAX * 4), *bb = malloc(LSA_READ_MAX * 4);
  uint32_t exact = 0;
  double jac = 0, jmin = 1;
  for (uint32_t i = 0; i < Q; i++)
  {
    uint32_t p, nr;
    memcpy(&p, qs, 4);
    const float *iq = (const float *)(qs + 4);
    memcpy(&nr, qs + 4 + LSA_DIM * 4, 4);
    const uint32_t *ref = (const uint32_t *)(qs + 8 + LSA_DIM * 4);
    qs += 8 + LSA_DIM * 4 + (size_t)nr * 4;
    const uint32_t nm = lsa_read(iq, p, c, m, G, mine_r);
    exact += nm == nr && !memcmp(mine_r, ref, nr * 4);
    memcpy(a, mine_r, nm * 4), memcpy(bb, ref, nr * 4);
    qsort(a, nm, 4, cmp_u32), qsort(bb, nr, 4, cmp_u32);
    uint32_t x = 0, y = 0, both = 0;
    while (x < nm && y < nr)
      a[x] == bb[y] ? (both++, x++, y++) : a[x] < bb[y] ? x++ : y++;
    const double j = (double)both / (nm + nr - both);
    jac += j, jmin = fmin(jmin, j);
  }
  printf("  %s: T %u, %u blocks - index %d / %d of lsa.py's bits; blocks' buckets the same %u / %u (centroids within "
         "%.2g); reads the same lists %u / %u, sets' Jaccard mean %.5f, least %.5f\n",
         name, T, G, same_bits, LSA_DIM, blocks_same, G, cmax, exact, Q, jac / Q, jmin);
  assert(blocks_same * 10 >= G * 9 && jac / Q >= 0.99);
  free(c), free(m), free(mine_r), free(a), free(bb), free(b), free(W);
}

int main(void)
{
  printf("[test_lsa] start\n");
  synthetic();
  const char *dir = getenv("OSMOS_LSA_CASES");
  DIR *d = dir ? opendir(dir) : NULL;
  if (!d)
    printf("  lsa.py's cases: no OSMOS_LSA_CASES, not measured\n");
  for (struct dirent *e; d && (e = readdir(d));)
  {
    const size_t l = strlen(e->d_name);
    if (l > 4 && !strcmp(e->d_name + l - 4, ".bin"))
      one_case(dir, e->d_name);
  }
  if (d)
    closedir(d);
  printf("[test_lsa] ok\n");
  return 0;
}
