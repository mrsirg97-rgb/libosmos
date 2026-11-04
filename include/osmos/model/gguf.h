#ifndef GGUF_H
#define GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GGUF_ERR 256
#define GGUF_MAX_DIMS 4

typedef enum GgufValueType
{
  GGUF_UINT8 = 0,
  GGUF_INT8 = 1,
  GGUF_UINT16 = 2,
  GGUF_INT16 = 3,
  GGUF_UINT32 = 4,
  GGUF_INT32 = 5,
  GGUF_FLOAT32 = 6,
  GGUF_BOOL = 7,
  GGUF_STRING = 8,
  GGUF_ARRAY = 9,
  GGUF_UINT64 = 10,
  GGUF_INT64 = 11,
  GGUF_FLOAT64 = 12,
} GgufValueType;

typedef struct GgufKv
{
  char *key;
  GgufValueType type;
  GgufValueType item;
  uint64_t n;
  union
  {
    uint64_t u;
    int64_t i;
    double f;
    char *s;
    void *items;
  } v;
} GgufKv;

typedef struct GgufTensor
{
  char *name;
  uint32_t type;
  uint32_t n_dims;
  uint64_t dims[GGUF_MAX_DIMS];
  int file;
  uint64_t offset;
  uint64_t bytes;
} GgufTensor;

typedef struct Gguf
{
  int n_files;
  char **paths;
  uint64_t *sizes;
  GgufKv *kv;
  size_t n_kv;
  GgufTensor *tensors;
  size_t n_tensors;
} Gguf;

Gguf *gguf_create(const char *path, char err[GGUF_ERR]);
Gguf *gguf_header(const char *path, char err[GGUF_ERR]);
void gguf_destroy(Gguf *g);

const GgufTensor *gguf_tensor(const Gguf *g, const char *name);
const GgufKv *gguf_kv(const Gguf *g, const char *key);
uint64_t gguf_bytes(uint32_t type, uint64_t values);
bool gguf_int(const GgufKv *kv, size_t i, int64_t *out);

#endif

/*
Gguf
  a model's GGUF files as data: their metadata and their tensors' places. nothing is mapped or read past the headers;
  the bytes stay where they are, for the pool to read in place. a split model (`-00001-of-0000N`) is one Gguf over
  all its shards; `n_files` and `paths` in shard order, `sizes` each file's bytes.

gguf_create
  `path`: a file, or a split model's first shard (its siblings found by name). the files are untrusted: every count,
  length and offset is checked against the file's size, a string or array past the end, a tensor whose bytes leave its
  file, an unknown tensor type, a duplicate name or a shard missing refuses the whole model - NULL, the reason in
  `err` (file, offset, what). everything allocated on the way is released.

gguf_header
  one file alone, as gguf_create reads it: a split model's first shard without its siblings (none opened, none
  required) - its metadata whole (the tokenizer's: tokenizer.h), its own tensors only. for what one file's header
  holds; a model to run is gguf_create's.

gguf_destroy
  the Gguf and everything it owns: names, keys, strings, arrays, paths. NULL is nothing to do.

GgufKv
  one metadata entry. scalars in `v.u` / `v.i` / `v.f` (unsigned, signed, float, widened), `v.s` a string (NUL
  terminated, its own copy). an array: `item` its element type, `n` its length, `v.items` a copy of them - for strings
  a `char **` of `n`, else `n` packed values of the element's own width.

GgufTensor
  `type` the ggml type id (IQ1_M 29, IQ2_XXS 16, IQ2_XS 17, Q5_K 13, Q6_K 14, Q8_0 8, BF16 30, F32 0, ...), `dims`
  innermost first as in the file, `file` its shard, `offset` its first byte's absolute offset in that file (the data
  section's aligned start plus the tensor's own), `bytes` its size from its type and dims.

gguf_tensor / gguf_kv
  by exact name; NULL when absent. pointers into the Gguf, valid until gguf_destroy.

gguf_bytes
  the bytes `values` values take in `type` (a whole number of blocks); 0 for a type this build does not know or a
  count that is not whole blocks - which gguf_create refuses.

gguf_int
  an integer at its own width (any of the eight integer types): a scalar's value (i ignored), or an array's item i
  (i < n). false for another type, or a value past int64's range (an unsigned 64-bit one above it).
*/
