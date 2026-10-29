#ifndef GGUF_READ_H
#define GGUF_READ_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "substrate/model/gguf.h"

typedef struct
{
  FILE *f;
  const char *path;
  uint64_t pos;
  uint64_t size;
  char *err;
} Reader;

bool gguf_fail(Reader *r, const char *what);
bool gguf_take(Reader *r, void *dst, uint64_t n);
bool gguf_u32(Reader *r, uint32_t *v);
bool gguf_u64(Reader *r, uint64_t *v);
char *gguf_string(Reader *r);
bool gguf_value(Reader *r, GgufKv *kv);
void gguf_kv_free(GgufKv *kv);

#endif

/*
Reader
  one file's header, read in order, every read checked against the file's size: a read past it fails with the file,
  the offset and what was being read, in `err` (the first failure kept). gguf.c's helper, not part of the API.

gguf_value
  a metadata value after its key: a scalar, a string or an array (counts checked against the bytes left before
  anything is allocated). on failure what it allocated stays in `kv` for gguf_kv_free.
*/
