/* Qwen's chat template's shared pieces (osmos/text/chat_qwen.h): chat_qwen.c renders, chat_qwen_parse.c parses */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "osmos/text/chat_qwen.h"

typedef struct Span
{
  const char *p;
  size_t n;
} Span;

/* s with what Python's str.strip takes off each end (str.isspace's code points: the template's `trim`) */
Span qchat_trim(Span s);
/* a call's function name or parameter key as the format can hold it: not empty, no newline, no '<' - the parse's rule,
   and the render's (0.7.1: a client's "f<|im_end|" and the template's '>' after it made a special token) */
static inline bool qchat_name_ok(const char *p, size_t n) { return n && !memchr(p, '\n', n) && !memchr(p, '<', n); }

bool qchat_fail(char *err, const char *why);
