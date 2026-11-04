#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* byte-level BPE (GPT-2's): a word's bytes, each as its printable stand-in (bytes_to_unicode), merged pair by pair -
   the lowest-ranked merge first, the leftmost among equals (llama.cpp's llm_tokenizer_bpe_session) - and each
   symbol left looked up whole. its own copies of the vocabulary and the merges */
typedef struct Bpe Bpe;

Bpe *bpe_create(char **tokens, size_t n_tokens, char **merges, size_t n_merges);
void bpe_destroy(Bpe *b);
/* the id of a token's exact text, or -1 */
int32_t bpe_id(const Bpe *b, const char *text, size_t len);
/* the ids of a word (len raw bytes) appended to ids (at most cap written, the count returned as if all were) */
size_t bpe_word(const Bpe *b, const unsigned char *word, size_t len, int32_t *ids, size_t cap);
/* a byte-level token's text decoded back to its bytes; -1 for a character no byte stands for */
int bpe_decode(const char *text, unsigned char *out, size_t cap);
