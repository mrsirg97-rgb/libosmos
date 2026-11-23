#ifndef OSMOS_MODEL_TENSOR_H
#define OSMOS_MODEL_TENSOR_H
/* a tensor read from its file on the host: every family's one copy */
#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/gguf.h"

/* t's bytes whole into out (t->bytes of it), from its file (g's path) or an open descriptor of it: false for a file
   not opened or shorter than the tensor */
bool tensor_read(const Gguf *g, const GgufTensor *t, void *out);
bool tensor_read_fd(int fd, const GgufTensor *t, void *out);
/* a small tensor's values as F32 (F32, F16 or BF16 in the file: each the exact value), in memory of its own (free);
   *n its values. NULL for another type or a failed read */
float *tensor_f32(const Gguf *g, const GgufTensor *t, uint64_t *n);

#endif
