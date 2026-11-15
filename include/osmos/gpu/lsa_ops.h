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

#ifdef __cplusplus
}
#endif

#endif
