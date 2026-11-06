#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* a text's code points (n of them) cut into DeepSeek-V3's pre-tokens: cuts[0] = 0 .. cuts[k] = n, k pieces;
   `cuts` has room for n + 1. returns k. its three expressions in turn, each over the last one's pieces (llama.cpp's
   unicode_regex_split): \p{N}{1,3}; [CJK ideographs, hiragana, katakana]+; then
   [ASCII punctuation][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ' '?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ |
   \s+(?!\S) | \s+ - a piece's text between matches a piece of its own */
size_t pretok_split(const uint32_t *cp, size_t n, size_t *cuts);

/* the same, by Qwen3.5's one expression - llama.cpp's own code for it (unicode_regex_split_custom_qwen35), its
   classes the code point's own (not the regex path's collapsed ASCII): (?i:'s|'t|'re|'ve|'m|'ll|'d) |
   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+ - and a
   code point none of them takes, a piece of its own */
size_t pretok_split_qwen35(const uint32_t *cp, size_t n, size_t *cuts);

/* a code point's flags (unicode_data.h's UC_*: llama.cpp's unicode_cpt_flags, whitespace aside), and whether it is
   whitespace (its \s) */
uint16_t pretok_flags(uint32_t c);
bool pretok_space(uint32_t c);

/* a code point decoded from s (at most `left` bytes): 0xFFFD for a byte that begins no valid sequence (llama.cpp's
   unicode_cpts_from_utf8: the byte skipped). *used the bytes taken */
uint32_t pretok_utf8(const unsigned char *s, size_t left, size_t *used);
