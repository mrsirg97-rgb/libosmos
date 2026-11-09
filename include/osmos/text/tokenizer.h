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
size_t tokenizer_defuse(const Tokenizer *t, char *s, size_t len);
int64_t tokenizer_encode_shaped(const Tokenizer *t, const char *text, const char *shape, size_t len, int32_t *ids,
                                size_t cap);
int tokenizer_piece(const Tokenizer *t, int32_t id, bool special, char out[TOKENIZER_PIECE_MAX]);
int32_t tokenizer_find(const Tokenizer *t, const char *text);
bool tokenizer_special(const Tokenizer *t, int32_t id);
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

tokenizer_defuse
  s (len bytes, in place) with no special token's text left in it: at each match (the longest first, as encode
  matches them) one bit of its first character's last byte flipped - '<' to '=', '｜' to '｝' - so s keeps its
  length, its characters' lengths and (for the special tokens these models have) its JSON escaping. returns how
  many bytes changed: 0, s as it was. (a flip that made another token's text - none can with these models' - is
  refused by tokenizer_encode_shaped: the shape's token not the text's.) a client's strings defused, the prompt rendered from them is the
  shape tokenizer_encode_shaped takes.

tokenizer_encode_shaped
  the ids of text (len bytes) with a special token only where shape (the same len bytes) has one: shape cut into
  special tokens as tokenizer_encode cuts it (parse_special), each stretch between them text's own bytes split and
  merged, no special token matched in them - so a client's string holding a special token's text is that text, not
  the token. shape the same bytes as text: tokenizer_encode's ids (parse_special). returns and caps as encode; -1
  when shape is not text defused (a byte apart by more than its lowest bit, or a special token text doesn't hold).

tokenizer_special
  whether id is a special token (a control, user-defined or unknown one: matched whole by encode); false past the
  vocabulary.

tokenizer_find
  the id of the token whose text is exactly `text` (a special token by its name: "<｜User｜>"), or -1.

tokenizer_vocab
  the vocabulary's size.
*/
