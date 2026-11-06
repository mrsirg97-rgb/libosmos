/* a model's metadata read and checked, key by key: each read fails with the key and why in err (GGUF_META_ERR). a
   family's geometry reader (weights_geo.c, q4x_geo.c) builds on these; its own spellings of a key, if any, as
   aliases: a key found in either, both present and unequal refused */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/model/gguf.h"

#define GGUF_META_ERR 256
#define GGUF_META_LIST_MAX 256 /* a list's most entries (a layer list, a ratio a layer) */

typedef struct GgufMeta
{
  const Gguf *g;
  char *err;                     /* GGUF_META_ERR bytes */
  const char *const (*aliases)[2]; /* pairs of spellings of one key, or NULL */
  size_t n_aliases;
} GgufMeta;

bool meta_fail(GgufMeta *m, const char *key, const char *why);
bool meta_present(GgufMeta *m, const char *key);
/* GGUF_UINT32 asks for an integer scalar of any width; another type, that type */
const GgufKv *meta_find(GgufMeta *m, const char *key, GgufValueType type);
bool meta_uint_in(GgufMeta *m, const char *key, int lo, int hi, int *out);
bool meta_uint_is(GgufMeta *m, const char *key, uint64_t want);
bool meta_flag_set(GgufMeta *m, const char *key);
bool meta_real(GgufMeta *m, const char *key, float *out); /* a positive float */
bool meta_text_is(GgufMeta *m, const char *key, const char *want);
/* an array of integers: n (at least min, at most cap) into out, each read at its own width */
bool meta_ints(GgufMeta *m, const char *key, size_t min, size_t cap, int64_t *out, size_t *n);
/* a list of layers, each below n_layers and once, as flags (count: how many, or NULL) */
bool meta_layers(GgufMeta *m, const char *key, int n_layers, bool *set, int *count);
