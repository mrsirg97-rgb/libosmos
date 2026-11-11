#pragma once
#include "api.h"
#include "dialect.h"
#include "osmos/text/tokenizer.h"

/* an answer as it streams (answer.c): the completion's text, how much of it went out, and in which field */
typedef enum Phase
{
  PHASE_REASONING, /* before </think> (thinking mode) */
  PHASE_CONTENT,
  PHASE_DONE, /* at a calls block or the EOS: nothing more streams; the parse at DONE says the rest */
} Phase;

typedef struct Answer
{
  int fd;
  bool stream;
  const char *model;
  const Dialect *d; /* the markers it is cut at */
  char id[48];
  long long created;
  JsonBuf text; /* every piece so far, special tokens' text included (chat_parse reads it through the EOS) */
  JsonBuf shape; /* text with the special tokens' text its plain pieces spell defused: its markers are read here */
  size_t run;    /* where shape's run of plain pieces starts (past the last special token) */
  size_t sent;  /* text's bytes streamed or passed over */
  Phase phase;
  Phase cut;  /* the phase DONE was reached in (a flush sends what was held in its field) */
  bool begun; /* the phase has streamed text (until then a trimming dialect's whitespace is passed over) */
  bool gone; /* the client stopped reading */
} Answer;

bool answer_take(Answer *s, const Tokenizer *t, int32_t id);
bool answer_chunk(Answer *s, const char *delta_json, const char *finish, const char *extra);
void answer_advance(Answer *s);
void answer_flush(Answer *s);
