#pragma once
#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/json.h"

/* the loop signals (issue #47): what a request's own history says about the agent sending it - a pure function of its
   messages (no model, no state kept), reported beside the answer (its `osmos` field, the x-osmos-loop header). an
   agent's loops are a trajectory's property, invisible a step at a time: these are the cheap ones a server can see */

#define LOOP_TURNS 3        /* assistant turns since its last tool call that make `stuck` (loop 1: "Let me try.") */
#define LOOP_REPEATS 3      /* one call's name and arguments this many times among the recent ones (loop 2) */
#define LOOP_WINDOW 16      /* the recent calls looked at */
#define LOOP_IDS 16         /* orphaned ids listed (past it, counted) */
#define LOOP_ID_MAX 64      /* an id's bytes listed */
#define LOOP_NAME_MAX 64    /* a repeated call's name's bytes listed */
/* the result osmos puts where a call's was lost (fill_orphans: the request's osmos.fill_orphans, or the pack's): says
   what happened, nothing of what the result was */
#define LOOP_ORPHAN_TEXT "[tool result dropped from the conversation's history - run the call again if its result is needed]"

typedef struct Loop
{
  unsigned turns_without_call; /* assistant messages since the last one with a tool call (all of them, with none) */
  unsigned repeat_count;       /* the recent calls' most repeated (name, arguments): its count (0: no calls) */
  char repeat_name[LOOP_NAME_MAX];
  unsigned n_orphan_calls, n_orphan_results; /* a call with no result before the next assistant; a result whose id no
                                                call of the assistant before it has */
  char orphan_calls[LOOP_IDS][LOOP_ID_MAX], orphan_results[LOOP_IDS][LOOP_ID_MAX];
  bool filled; /* the orphaned calls given LOOP_ORPHAN_TEXT as their result */
  bool stuck;  /* turns_without_call >= LOOP_TURNS or repeat_count >= LOOP_REPEATS */
} Loop;

/* the signals of messages (an OpenAI request's: roles, tool_calls with ids, tool messages' tool_call_id) */
void loop_read(const Json *messages, Loop *l);
/* body with each orphaned call's result put after its call (and its other results): a new body's JSON into out */
bool loop_fill(const Json *body, JsonBuf *out);
/* ", \"osmos\": {\"loop\": {...}}" - fields after a response's choices */
void loop_json(JsonBuf *b, const Loop *l);
/* the x-osmos-loop header's value: "turns-without-call=N; repeated-call=K name; orphan-calls=C; orphan-results=R;
   orphans-filled=0|1; stuck=0|1" */
void loop_header(char *out, size_t cap, const Loop *l);
