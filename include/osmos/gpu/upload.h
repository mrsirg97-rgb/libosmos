#ifndef OSMOS_GPU_UPLOAD_H
#define OSMOS_GPU_UPLOAD_H

/* a model's tensor read into GPU memory of its own (the families that hold their weights whole: q4x, q35) */

#include <stdint.h>

#include "osmos/model/gguf.h"

/* t's bytes into a new device allocation (*dev; a 0-byte tensor one byte), read from its file through the pinned
   host buffer pin (piece bytes) a piece at a time: NULL, or which step failed (for the caller's message; *dev then
   holds whatever was allocated, for the caller's free) */
const char *gpu_read_tensor(const Gguf *g, const GgufTensor *t, void *pin, uint64_t piece, void **dev);

#endif
