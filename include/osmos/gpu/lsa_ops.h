#ifndef OSMOS_GPU_LSA_OPS_H
#define OSMOS_GPU_LSA_OPS_H

/* latent sparse attention's read on the GPU (kernels/lsa.hip): osmos/lsa.h's lsa_read, the same list - the buckets
   the host's lsa_block made (centroids F32, members), each closed bucket's score against the query in double, its
   dims in order (the host's sums, bit for bit), the top LSA_TOP_P by (score, then the lower bucket) - a radix
   select over the scores' ordered bits, the ties at the threshold the lowest buckets - those ordered, their keys from
   LSA_SINK on and before the window in that order cut at LSA_TOP_K, then the window, then the sinks */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/lsa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LsaScratch LsaScratch;
/* the scratch for reads over at most max_buckets closed buckets */
LsaScratch *lsa_scratch_create(struct OsmosLedger *l, uint32_t max_buckets); /* its buffers l's (NULL counts nothing) */
void lsa_scratch_destroy(LsaScratch *s);

/* the read at position p over the first `blocks` closed blocks' buckets (device cent [blocks*64 x 64] F32, members
   [blocks*64 x 16]): its positions into out (device, LSA_READ_MAX), their count into *n (device) - queued on the
   stream, never waited for (the radix's choices made on the device) */
bool lsa_read_gpu(const float *iq, uint32_t p, const float *cent, const int32_t *members, uint32_t blocks,
                  LsaScratch *s, uint32_t *out, uint32_t *n, void *stream);

/* a head's position-free dims, the index's input: [lo0, hi0) then [lo1, hi1) - Qwen's past its rotary prefix
   ({rot, head_dim, 0, 0}), LFM3's zero tail ({32, 64, 96, 128}: pairs (i, i + 64) rotated for i < 32) */
typedef struct LsaDims
{
  uint32_t lo0, hi0, lo1, hi1;
} LsaDims;

/* index vectors of n tokens (osmos/lsa.h's lsa_index, its bits: double sums in order, BF16 for keys): token t's input
   the kv_heads heads' free dims (dims's), side by side, each the sum of `group` consecutive heads (a key: group 1; a
   query: a KV head's query heads) of src's row t (row_stride floats: heads of head_dim); W [in x LSA_DIM], in =
   kv_heads x the free dims' count, at most LSA_IN_MAX; out n x LSA_DIM */
#define LSA_IN_MAX LSA_IN

/* a prompt's reads, a batch at once (tokens at p0 .. p0 + n - 1, their index vectors iq [n x LSA_DIM]): lsa_read's
   lists - token t's at lists + t LSA_READ_MAX, its count at counts[t] - over the first `blocks` closed blocks (a
   token's its own block's before it) - but each bucket's score F32 (its dims in order, each product and sum rounded
   on its own: not lsa_read_gpu's double), a tie the lower bucket. n_sub tokens' scores at a time in scratch
   (lsa_batch_scratch's floats, for at most max_buckets closed buckets); queued on the stream */
uint64_t lsa_batch_scratch(uint32_t n_sub, uint32_t max_buckets);
bool lsa_read_batch_gpu(const float *iq, uint32_t n, uint32_t p0, const float *cent, const int32_t *members,
                        uint32_t blocks, float *scratch, uint32_t n_sub, uint32_t max_buckets, uint32_t *lists,
                        int32_t *counts, void *stream);
bool lsa_index_gpu(const float *src, uint32_t n, uint32_t row_stride, uint32_t kv_heads, uint32_t group,
                   uint32_t head_dim, const LsaDims *dims, const float *W, bool bf16, float *out, void *stream);

#ifdef __cplusplus
}
#endif

#endif
