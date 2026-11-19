/* a model family's chat as osmos-api speaks it (dialect.c): its request read into the template's inputs, the prompt
   rendered, a completion parsed, and the markers its stream is cut at - chosen once by the model's
   general.architecture */
#pragma once

#include "request.h"

#define DIALECT_REFUSED_MAX 12 /* the most refused texts a dialect names (DeepSeek's five, Qwen's three, Inkling's ten) */

/* the field a block-shaped dialect's next block streams into (its header hook's) */
typedef enum DialectField
{
  DIALECT_REASONING,
  DIALECT_CONTENT,
  DIALECT_DONE, /* a call, the end of turn, or a block the stream stops at: the parse at DONE says the rest */
} DialectField;

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
  const char *answer_after; /* osmos-serve's --answer-after (no end while the answer past it is blank), or NULL */
  /* a block-shaped turn (Inkling's): each block's header says its field - header at a boundary returns the header's
     length and its field, or -1 while more bytes could change it; block_end ends a block (the next a boundary);
     loose reads a completion out of the format (cut by max_tokens). NULL: the reasoning to think_end, the content
     to calls_open */
  long (*header)(const char *p, size_t n, DialectField *field);
  const char *block_end;
  void (*loose)(const char *text, const char *shape, size_t n, JsonBuf *out);
} Dialect;

/* the dialect of a model family, or NULL (one naming more than DIALECT_REFUSED_MAX refused texts too) */
const Dialect *dialect_of(const char *arch);

/* request_ds.c, request_qwen.c: a request's messages, tools and effort as the family's template takes them */
bool request_ds(const Json *body, Request *r, char err[API_ERR]);
bool request_qwen(const Json *body, Request *r, char err[API_ERR]);
bool request_ink2(const Json *body, Request *r, char err[API_ERR]);
bool request_lfm3(const Json *body, Request *r, char err[API_ERR]);
