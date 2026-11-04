#pragma once
#include <stddef.h>
#include <stdint.h>

/* a text's code points (n of them) cut into DeepSeek-V3's pre-tokens: cuts[0] = 0 .. cuts[k] = n, k pieces;
   `cuts` has room for n + 1. returns k. its three expressions in turn, each over the last one's pieces (llama.cpp's
   unicode_regex_split): \p{N}{1,3}; [CJK ideographs, hiragana, katakana]+; then
   [ASCII punctuation][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ' '?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ |
   \s+(?!\S) | \s+ - a piece's text between matches a piece of its own */
size_t pretok_split(const uint32_t *cp, size_t n, size_t *cuts);

/* a code point decoded from s (at most `left` bytes): 0xFFFD for a byte that begins no valid sequence (llama.cpp's
   unicode_cpts_from_utf8: the byte skipped). *used the bytes taken */
uint32_t pretok_utf8(const unsigned char *s, size_t left, size_t *used);
