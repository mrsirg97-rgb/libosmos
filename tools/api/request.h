#pragma once
#include <stdbool.h>

#include "api.h"
#include "loop.h"

struct Dialect;

/* a chat request as the model family's template takes it (request.c, then the dialect's reader) */
typedef struct Request
{
  Json *messages; /* the template's messages (owned) */
  Json *tools;    /* OpenAI function tools apart from them (owned; NULL: none, or the template's own place) */
  const Json *branch; /* the messages the template's choices read (NULL: messages) - the text's, for the shape's render */
  ChatMode mode;
  int effort;          /* DeepSeek's reasoning budget */
  char level[16];      /* Qwen's reasoning effort as asked ("": the template's default) */
  bool preserve;       /* Qwen's preserve_thinking (its default: true) */
  unsigned max_tokens; /* 0: as many as the context leaves */
  bool stream, usage;
  Loop loop; /* the history's loop signals (loop.h): reported beside the answer */
} Request;

bool request_read(const Json *body, const struct Dialect *d, bool fill_orphans, Request *r, char err[API_ERR]);
void request_free(Request *r);

/* shared by the readers: the effort asked (reasoning_effort, else chat_template_kwargs'), a failure, the tools as
   OpenAI function tools (each in its {"type": "function", "function"} wrapper; NULL: none) */
const Json *request_effort(const Json *body);
bool request_fail(char *err, const char *what);
Json *request_tools(const Json *body);
bool request_format(const Json *body, JsonBuf *out, char err[API_ERR]);
