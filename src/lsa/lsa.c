/* latent sparse attention's read, the host's reference (osmos/lsa.h). Ward's costs and the centroids' sums in double:
   lsa.py's F32 ran on a GPU, its reduction orders the device's, so no F32 order here is its bits - double makes this
   read one answer, the one the GPU's read is held to; its agreement with lsa.py's sets is measured (tests/test_lsa) */
#include "osmos/lsa.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define POOLS (LSA_WINDOW / LSA_POOL)

static float bf16(float v)
{
  uint32_t u;
  memcpy(&u, &v, 4);
  if ((u & 0x7F800000u) != 0x7F800000u) /* finite: to nearest, ties to even */
    u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  memcpy(&v, &u, 4);
  return v;
}

void lsa_index(const float *x, uint32_t in, const float *W, bool round_bf16, float *out)
{
  double y[LSA_DIM] = {0}, n = 0;
  for (uint32_t i = 0; i < in; i++)
    for (int d = 0; d < LSA_DIM; d++)
      y[d] += (double)x[i] * W[(uint64_t)i * LSA_DIM + d];
  for (int d = 0; d < LSA_DIM; d++)
    n += y[d] * y[d];
  n = sqrt(n);
  n = n > 1e-12 ? n : 1e-12; /* F.normalize's eps */
  for (int d = 0; d < LSA_DIM; d++)
    out[d] = round_bf16 ? bf16((float)(y[d] / n)) : (float)(y[d] / n);
}

static double dot(const double *a, const double *b)
{
  double s = 0;
  for (int d = 0; d < LSA_DIM; d++)
    s += a[d] * b[d];
  return s;
}

/* Ward over a block's pools (lsa.py's ward): each round every pool's cheapest partner within the cap (cost |a| + |b|
   - |a + b|, a tie the lower index); the mutual pairs merged cheapest first (a tie the lower pool), as many as the
   bucket count still needs; a pool's label its root's rank among the pools left, in pool order */
static bool ward(double sums[POOLS][LSA_DIM], int *label)
{
  int sz[POOLS], parent[POOLS], best[POOLS];
  bool alive[POOLS];
  double cost[POOLS], an[POOLS];
  for (int i = 0; i < POOLS; i++)
    sz[i] = LSA_POOL, parent[i] = i, alive[i] = true;
  int left = POOLS;
  for (int round = 0; round < 256 && left > LSA_BUCKETS; round++)
  {
    const int need = left - LSA_BUCKETS;
    for (int i = 0; i < POOLS; i++)
      an[i] = alive[i] ? sqrt(fmax(dot(sums[i], sums[i]), 0)) : 0;
    for (int i = 0; i < POOLS; i++)
    {
      best[i] = -1, cost[i] = INFINITY;
      for (int j = 0; alive[i] && j < POOLS; j++)
      {
        if (j == i || !alive[j] || sz[i] + sz[j] > LSA_CAP)
          continue;
        const double c = an[i] + an[j] - sqrt(fmax(an[i] * an[i] + an[j] * an[j] + 2 * dot(sums[i], sums[j]), 0));
        if (c < cost[i])
          cost[i] = c, best[i] = j;
      }
    }
    int pairs[POOLS], np = 0; /* the mutual pairs, by their lower pool */
    for (int i = 0; i < POOLS; i++)
      if (best[i] > i && best[best[i]] == i)
        pairs[np++] = i;
    if (!np)
      break;
    for (int a = 1; a < np; a++) /* cheapest first, a tie the lower pool (insertion: stable) */
      for (int b = a; b > 0 && cost[pairs[b]] < cost[pairs[b - 1]]; b--)
      {
        const int t = pairs[b];
        pairs[b] = pairs[b - 1], pairs[b - 1] = t;
      }
    for (int k = 0; k < np && k < need; k++)
    {
      const int i = pairs[k], j = best[i];
      for (int d = 0; d < LSA_DIM; d++)
        sums[i][d] += sums[j][d];
      sz[i] += sz[j], alive[j] = false, parent[j] = i, left--;
    }
  }
  if (left != LSA_BUCKETS)
    return false;
  int rank[POOLS], r = 0;
  for (int i = 0; i < POOLS; i++)
    rank[i] = alive[i] ? r++ : -1;
  for (int i = 0; i < POOLS; i++)
  {
    int root = i;
    while (parent[root] != root)
      root = parent[root];
    label[i] = rank[root];
  }
  return true;
}

bool lsa_block(const float *ik, uint32_t block, float *cent, int32_t *members)
{
  static __thread double sums[POOLS][LSA_DIM];
  for (int p = 0; p < POOLS; p++)
    for (int d = 0; d < LSA_DIM; d++)
    {
      double s = 0;
      for (int k = 0; k < LSA_POOL; k++)
        s += ik[(uint64_t)(p * LSA_POOL + k) * LSA_DIM + d];
      sums[p][d] = s;
    }
  int label[POOLS];
  if (!ward(sums, label))
    return false;
  double c[LSA_BUCKETS][LSA_DIM];
  int fill[LSA_BUCKETS] = {0};
  memset(c, 0, sizeof c);
  for (int b = 0; b < LSA_BUCKETS * LSA_CAP; b++)
    members[b] = -1;
  for (int key = 0; key < LSA_WINDOW; key++) /* ascending: a bucket's keys in position order */
  {
    const int b = label[key / LSA_POOL];
    for (int d = 0; d < LSA_DIM; d++)
      c[b][d] += ik[(uint64_t)key * LSA_DIM + d];
    if (fill[b] >= LSA_CAP)
      return false;
    members[b * LSA_CAP + fill[b]++] = (int32_t)(block * LSA_WINDOW + key);
  }
  for (int b = 0; b < LSA_BUCKETS; b++)
  {
    double n = sqrt(dot(c[b], c[b]));
    n = n > 1e-12 ? n : 1e-12;
    for (int d = 0; d < LSA_DIM; d++)
      cent[b * LSA_DIM + d] = (float)(c[b][d] / n);
  }
  return true;
}

typedef struct
{
  double s;
  uint32_t b;
} Pick;

static int by_score(const void *x, const void *y) /* best first, a tie the lower bucket (lsa.py's stable sort) */
{
  const Pick *a = x, *b = y;
  return a->s > b->s ? -1 : a->s < b->s ? 1 : (a->b > b->b) - (a->b < b->b);
}

uint32_t lsa_read(const float *iq, uint32_t p, const float *cent, const int32_t *members, uint32_t blocks,
                  uint32_t *out)
{
  const uint32_t win0 = p + 1 > LSA_WINDOW ? p + 1 - LSA_WINDOW : 0;
  uint32_t visible = p / LSA_WINDOW * LSA_BUCKETS, n = 0;
  visible = visible < blocks * LSA_BUCKETS ? visible : blocks * LSA_BUCKETS;
  if (visible)
  {
    Pick *pk = malloc(visible * sizeof *pk);
    if (!pk)
      return 0;
    for (uint32_t b = 0; b < visible; b++)
    {
      double s = 0;
      for (int d = 0; d < LSA_DIM; d++)
        s += (double)cent[(uint64_t)b * LSA_DIM + d] * iq[d];
      pk[b] = (Pick){s, b};
    }
    qsort(pk, visible, sizeof *pk, by_score);
    for (uint32_t i = 0; i < visible && i < LSA_TOP_P && n < LSA_TOP_K; i++)
      for (int k = 0; k < LSA_CAP && n < LSA_TOP_K; k++)
      {
        const int32_t m = members[(uint64_t)pk[i].b * LSA_CAP + k];
        if (m >= LSA_SINK && (uint32_t)m < win0)
          out[n++] = (uint32_t)m;
      }
    free(pk);
  }
  for (uint32_t q = win0; q <= p; q++)
    out[n++] = q;
  const uint32_t sinks = LSA_SINK < win0 ? LSA_SINK : win0 < p + 1 ? win0 : p + 1;
  for (uint32_t q = 0; q < sinks; q++)
    out[n++] = q;
  return n;
}
