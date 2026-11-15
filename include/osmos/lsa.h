#ifndef OSMOS_LSA_H
#define OSMOS_LSA_H

/* latent sparse attention's read (DESIGN.md, "the 10M agent"; arXiv 2610.42147's D-LSA, the maps trained on -vibes:
   qwen/lsa.py), the host's reference - the GPU's read is checked against it, its sets lsa.py's. per attention layer
   and token one read, shared by the layer's heads:
     index   a key's index vector: normalize(key_in W_K), rounded to BF16 (key_in: the layer's normed k past its rotary
             prefix, every KV head's, LSA_IN values); a query's normalize(q_in W_Q) (q_in: each KV group's query heads
             summed)
     blocks  every closed LSA_WINDOW keys: Ward over its pools of LSA_POOL keys to LSA_BUCKETS buckets of at most
             LSA_CAP keys - a bucket its centroid (its keys' index vectors summed, normalized) and its keys ascending
     read    at position p: the buckets of the blocks closed before p's window, scored by the query, best first (ties
             by bucket), LSA_TOP_P of them; their keys from LSA_SINK on and before the window, in that order, the first
             LSA_TOP_K; then the window (the last LSA_WINDOW positions to p) and the sinks */

#include <stdbool.h>
#include <stdint.h>

#define LSA_IN 768 /* an index input: 4 KV heads x 192 (Qwen3.8-27B: 256 dims a head, its first 64 rotary) */
#define LSA_DIM 64 /* an index vector */
#define LSA_WINDOW 512
#define LSA_POOL 4
#define LSA_BUCKETS 64 /* a closed block's (LSA_WINDOW / 8) */
#define LSA_CAP 16
#define LSA_SINK 4
#define LSA_TOP_P 1024
#define LSA_TOP_K 4096
#define LSA_READ_MAX (LSA_TOP_K + LSA_WINDOW + LSA_SINK) /* a read's most positions */

/* out = normalize(x W) (W: in x LSA_DIM, row major), BF16-rounded when bf16 (a key's, as lsa.py keeps them) */
void lsa_index(const float *x, uint32_t in, const float *W, bool bf16, float *out);

/* one closed block's buckets: ik its LSA_WINDOW index vectors; cent LSA_BUCKETS x LSA_DIM, members LSA_BUCKETS x
   LSA_CAP (positions block * LSA_WINDOW + i ascending, -1 past a bucket's keys). false when Ward stops short of the
   bucket count (a block the cap leaves no merge for) */
bool lsa_block(const float *ik, uint32_t block, float *cent, int32_t *members);

/* the read at position p over the first `blocks` closed blocks' buckets (cent, members: lsa_block's, block after
   block): its positions into out (LSA_READ_MAX), their count returned - the buckets' keys in score order, then the
   window ascending, then the sinks */
uint32_t lsa_read(const float *iq, uint32_t p, const float *cent, const int32_t *members, uint32_t blocks,
                  uint32_t *out);

#endif
