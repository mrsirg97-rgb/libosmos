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
  bool preserve;       /* Qwen's preserve_thinking (its default: true); Hy4's preserved_thinking where given */
  bool preserve_given; /* Hy4's: preserved_thinking given (else its template's default: on with tools) */
  bool preserve_default; /* the pack's preserve_thinking (LFM3's reader: a request's unset, this) */
  Json *effort_value;  /* Inkling's reasoning effort as asked - a name or a number (owned; NULL: the template's 0.9) */
  unsigned max_tokens; /* 0: as many as the context leaves */
  double temperature, top_p; /* 0.14.0: as asked, else the pack's (temperature 0 greedy) */
  unsigned long long seed;   /* as asked, else drawn (sampled: returned in x-osmos-seed) */
  bool sampled;              /* temperature above 0 */
  bool stream, usage;
  Loop loop; /* the history's loop signals (loop.h): reported beside the answer */
} Request;

bool request_read(const Json *body, const struct Dialect *d, bool fill_orphans, bool preserve_default,
                  double temperature, double top_p, Request *r, char err[API_ERR]);
void request_free(Request *r);

/* shared by the readers: the effort asked (reasoning_effort, else chat_template_kwargs'), a failure, the tools as
   OpenAI function tools (each in its {"type": "function", "function"} wrapper; NULL: none) */
const Json *request_effort(const Json *body);
bool request_fail(char *err, const char *what);
Json *request_tools(const Json *body);
bool request_format(const Json *body, JsonBuf *out, char err[API_ERR]);
/* the messages as Qwen's and Inkling's templates take them (request_messages.c): `reasoning` as reasoning_content,
   a call's arguments given as a JSON string parsed into an object - *out owned, NULL with why on a refusal */
bool request_messages(const Json *msgs, Json **out, char err[API_ERR]);
