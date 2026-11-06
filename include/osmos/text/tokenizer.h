#ifndef OSMOS_TEXT_TOKENIZER_H
#define OSMOS_TEXT_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/model/gguf.h"

#define TOKENIZER_ERR 256
#define TOKENIZER_PIECE_MAX 512 /* the most bytes a token's piece takes (DeepSeek's longest token text: 256) */

typedef struct Tokenizer Tokenizer;

Tokenizer *tokenizer_create(const Gguf *g, char err[TOKENIZER_ERR]);
void tokenizer_destroy(Tokenizer *t);
int64_t tokenizer_encode(const Tokenizer *t, const char *text, size_t len, bool parse_special, int32_t *ids,
                         size_t cap);
int tokenizer_piece(const Tokenizer *t, int32_t id, bool special, char out[TOKENIZER_PIECE_MAX]);
int32_t tokenizer_find(const Tokenizer *t, const char *text);
int32_t tokenizer_vocab(const Tokenizer *t);

#endif

/*
Tokenizer
  a model's own tokenizer, from its GGUF (tokenizer.ggml.tokens / token_type / merges): byte-level BPE (`model`
  gpt2) after the pre-tokenizer `pre` names (deepseek-v3, or joyai-llm: the same split; qwen35, Qwen3.5 and 3.8's) - llama.cpp's
  tokenization of the same file, token for token (tests/test_tokenizer.c against its llama-tokenize). its own copy
  of the vocabulary: the Gguf may go after create.

tokenizer_create
  the tokenizer of g, or NULL with why: a model type, a pre-tokenizer, or tables it does not know (a missing or
  short list, a merge naming no pair, a token of more bytes than TOKENIZER_PIECE_MAX decodes).

tokenizer_destroy
  everything it allocated. NULL is nothing to do.

tokenizer_encode
  the ids of text (len bytes of UTF-8 - a byte that begins no valid sequence read as U+FFFD, as llama.cpp's), in
  order: the special tokens matched whole first - the longest first; user-defined ones (`<think>`) always, control ones
  (`<｜User｜>`) only when parse_special - and the text between them split and merged. returns how many; when more
  than cap, only cap written and the full count returned (call again with room). no BOS: the caller's template adds
  what it wants.

tokenizer_piece
  the bytes id stands for, into out: its decoded bytes (a byte-level token's UTF-8 may be a partial character -
  joined across tokens by the caller); a control token's text when `special`, else nothing. returns the byte count,
  or -1 for an id past the vocabulary.

tokenizer_find
  the id of the token whose text is exactly `text` (a special token by its name: "<｜User｜>"), or -1.

tokenizer_vocab
  the vocabulary's size.
*/
