#ifndef OSMOS_GPU_LSA_OPS_H
#define OSMOS_GPU_LSA_OPS_H

/* latent sparse attention's read on the GPU (kernels/lsa.hip): osmos/lsa.h's lsa_read, the same list - the buckets
   the host's lsa_block made (centroids F32, members), each closed bucket's score against the query in double, its
   dims in order (the host's sums, bit for bit), the top LSA_TOP_P by (score, then the lower bucket) - a radix
   select over the scores' ordered bits, the ties at the threshold the lowest buckets - those ordered, their keys from
   LSA_SINK on and before the window in that order cut at LSA_TOP_K, then the window, then the sinks */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LsaScratch LsaScratch;
/* the scratch for reads over at most max_buckets closed buckets */
LsaScratch *lsa_scratch_create(uint32_t max_buckets);
void lsa_scratch_destroy(LsaScratch *s);

/* the read at position p over the first `blocks` closed blocks' buckets (device cent [blocks*64 x 64] F32, members
   [blocks*64 x 16]): its positions into out (device, LSA_READ_MAX), their count into *n (the host's). waits for the
   stream (the radix passes read their histograms back) */
bool lsa_read_gpu(const float *iq, uint32_t p, const float *cent, const int32_t *members, uint32_t blocks,
                  LsaScratch *s, uint32_t *out, uint32_t *n, void *stream);

/* index vectors of n tokens (osmos/lsa.h's lsa_index, its bits: double sums in order, BF16 for keys): token t's input
   the kv_heads heads' dims past rot, side by side, each the sum of `group` consecutive heads (a key: group 1; a query:
   a KV head's query heads) of src's row t (row_stride floats: heads of head_dim); W [LSA_IN x LSA_DIM]; out n x
   LSA_DIM. kv_heads (head_dim - rot) must be LSA_IN */
bool lsa_index_gpu(const float *src, uint32_t n, uint32_t row_stride, uint32_t kv_heads, uint32_t group,
                   uint32_t head_dim, uint32_t rot, const float *W, bool bf16, float *out, void *stream);

#ifdef __cplusplus
}
#endif

#endif
