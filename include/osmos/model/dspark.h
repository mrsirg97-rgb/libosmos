#ifndef DSPARK_H
#define DSPARK_H

#include <stdbool.h>
#include <stdint.h>

#define DSPARK_ERR 256
#define DSPARK_NAME_MAX 64
#define DSPARK_MAX_DIMS 4

typedef struct DsparkTensor
{
  char name[DSPARK_NAME_MAX];
  uint32_t type; /* the GGUF type id: F32 0, BF16 30, Q8_0 8, MXFP4 39 */
  uint32_t n_dims;
  uint64_t dims[DSPARK_MAX_DIMS]; /* outermost first, as dspark.json gives them: [out, in] for a matrix */
  uint64_t offset, bytes;         /* in dspark.bin */
} DsparkTensor;

typedef struct DsparkIndex
{
  char *bin; /* dspark.bin's path */
  DsparkTensor *t;
  uint32_t n;
} DsparkIndex;

DsparkIndex *dspark_index(const char *dir, char err[DSPARK_ERR]);
void dspark_index_destroy(DsparkIndex *x);
const DsparkTensor *dspark_tensor(const DsparkIndex *x, const char *name);

#endif

/*
DsparkIndex
  DeepSeek's DSpark drafter's tensors as Maya's drafter_pack.py wrote them: dir/dspark.json lists each one ([name,
  type, offset, bytes, [dims]]) in dir/dspark.bin. the blocks mtp.0 .. mtp.2: their attention, hyper-connections and
  128 routed experts (MXFP4) beside a shared one (Q8_0); mtp.0's main_proj and main_norm (the trunk's taps into the
  blocks' keys); mtp.2's final norm, markov and confidence heads.

dspark_index
  the index read from dir/dspark.json; NULL, with why in err, for a file it cannot read or parse, a type it does not
  know, a tensor past the end of dspark.bin, or bytes that do not match its type and dims.

dspark_index_destroy
  everything it allocated. NULL is nothing to do.

dspark_tensor
  the tensor of that name, or NULL.
*/
