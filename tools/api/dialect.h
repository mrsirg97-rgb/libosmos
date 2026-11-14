/* a model family's chat as osmos-api speaks it (dialect.c): its request read into the template's inputs, the prompt
   rendered, a completion parsed, and the markers its stream is cut at - chosen once by the model's
   general.architecture */
#pragma once

#include "request.h"

#define DIALECT_REFUSED_MAX 8 /* the most refused texts a dialect names (DeepSeek's five, Qwen's three) */

typedef struct Dialect
{
  const char *arch;       /* general.architecture */
  const char *eos;        /* the end of turn's token text (osmos-serve's --eos) */
  const char *think_end;  /* the reasoning's end */
  const char *calls_open; /* a tool calls block's start, as the parse cuts the content there */
  bool trims;             /* its parse trims the reasoning and the content: their streams start past whitespace */
  const char *const *refused; /* special tokens' text its parse refuses in a completion: never streamed */
  bool (*read)(const Json *body, Request *r, char err[API_ERR]);
  bool (*render)(const Request *r, JsonBuf *out, char err[CHAT_ERR]);
  bool (*parse)(const Request *r, const char *text, const char *shape, size_t n, JsonBuf *out,
                char err[CHAT_ERR]); /* markers read from shape (the said special tokens' text defused) */
  /* the template drops this prompt's latest reasoning when a user message follows (0.7.5: a point kept before it) */
  bool (*drops)(const Request *r);
} Dialect;

/* the dialect of a model family, or NULL (one naming more than DIALECT_REFUSED_MAX refused texts too) */
const Dialect *dialect_of(const char *arch);

/* request_ds.c, request_qwen.c: a request's messages, tools and effort as the family's template takes them */
bool request_ds(const Json *body, Request *r, char err[API_ERR]);
bool request_qwen(const Json *body, Request *r, char err[API_ERR]);
