#ifndef OSMOS_MODEL_SAFETENSORS_H
#define OSMOS_MODEL_SAFETENSORS_H

#include "osmos/err.h"
#include "osmos/model/gguf.h"

/* a safetensors file read as a Gguf: one file, no metadata, every tensor of the header's under its name - dims in
   ggml's order (the row's values first), BF16 or F32, its bytes where data_offsets put them past the header. so the
   GGUF's readers read it (gguf_tensor, gpu_read_tensor; gguf_destroy frees it). NULL, err said, for a header that is
   not a JSON object, a dtype other than BF16 or F32, a tensor's bytes not its shape's, past the file or overlapping
   another (0.11.0: Hy4-flash's DFly2 drafter, the release's model-00148-of-00148.safetensors, read as it ships) */
Gguf *st_create(const char *path, char err[OSMOS_ERR]);

#endif
