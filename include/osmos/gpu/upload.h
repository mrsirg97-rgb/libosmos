#ifndef OSMOS_GPU_UPLOAD_H
#define OSMOS_GPU_UPLOAD_H

/* a model's tensor read into GPU memory of its own (the families that hold their weights whole: q4x, q35) */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/mem/ledger_hip.h"
#include "osmos/model/gguf.h"

/* t's bytes into a new device allocation (*dev; a 0-byte tensor one byte), read from its file through the pinned
   host buffer pin (piece bytes) a piece at a time: NULL, or which step failed (for the caller's message; *dev then
   holds whatever was allocated, for the caller's free). every allocation here through l (osmos/mem/ledger.h: the
   caller's build's; NULL counts nothing) - a refusal its ("refused by the ledger: ...") */
const char *gpu_read_tensor(OsmosLedger *l, const Gguf *g, const GgufTensor *t, void *pin, uint64_t piece, void **dev);

/* every tensor of g a skip does not pass over (NULL: every one) into GPU memory of its own, its pointer at its index
   in a new table (*dev, n_tensors long; a skipped one NULL), read piece bytes at a time through one pinned buffer -
   the families' upload (0.11.2: four copies). false with why ("NAME: not read into GPU memory (WHY)", or out of
   memory) in err (OSMOS_ERR bytes: osmos/err.h); *dev then holds what was read, for gpu_free_table */
typedef bool (*gpu_upload_skip)(const GgufTensor *t, void *arg);
bool gpu_upload_table(OsmosLedger *l, const Gguf *g, uint64_t piece, gpu_upload_skip skip, void *arg, void ***dev,
                      char *err);
/* a table's tensors freed (a shared one - shared[i], NULL none - left: its bytes another buffer's), then the table */
void gpu_free_table(OsmosLedger *l, const Gguf *g, void **dev, const bool *shared);
/* the bytes a table holds (each tensor read, a shared one - shared[i] - not its): a build's count of its upload */
uint64_t gpu_table_bytes(const Gguf *g, void *const *dev, const bool *shared);

/* a head's Q8_0 copy: V rows of D values (type, row_bytes apart on the device) dequantized and quantized again by
   ggml's Q8_0 (q4x_kv_put: blocks of 32, d = amax / 127) 4,096 rows at a time, into a new allocation *out - Qwen3.5's
   and LFM3's head (0.11.2: two copies). false (*out NULL) where the GPU refused or a launch failed; *out l's (freed
   with ledger_free) */
bool gpu_q8_rows(OsmosLedger *l, uint32_t type, const void *src, uint64_t row_bytes, uint32_t V, uint32_t D, void **out);

#endif
