/* Qwen's chat template's shared pieces (osmos/text/chat_qwen.h): chat_qwen.c renders, chat_qwen_parse.c parses */
#pragma once
#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/chat_qwen.h"

typedef struct Span
{
  const char *p;
  size_t n;
} Span;

/* s with what Python's str.strip takes off each end (str.isspace's code points: the template's `trim`) */
Span qchat_trim(Span s);

bool qchat_fail(char *err, const char *why);
