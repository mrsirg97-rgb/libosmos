/* --parallel's lanes and loop (multi.c), shared with its rounds (round.c) */
#pragma once

#include "libevt/scheduler/engine.h"
#include "serve.h"

enum { PRIO_TURN = 1, PRIO_LINE = 2 };
typedef enum { IDLE, READING, ANSWERING } LaneState;

typedef struct Lane
{
  Conv c;
  LaneState st;
  Gen g;
  uint32_t reused;
  int32_t t; /* an answering lane's next token: printed at the next round, fed after */
  long long produced, limit, accepted, drafted;
  double t0, t1;
  uint64_t used; /* when it last took a request (the loop's count): the least recently used is set aside first */
} Lane;

typedef struct Multi
{
  Server *sv;
  Lane *lane;
  int n;
  Engine *evt;
  Gen *wait; /* requests no idle lane could take, in arrival order */
  int nwait, capwait;
  bool turn_queued, last_piece;
  int rr;
  uint64_t clock;
} Multi;

void lane_finish(Multi *m, Lane *ln, const char *why);
void lane_fail(Multi *m, Lane *ln);
void round_all(Multi *m);

/*
round_all
  a round: every answering lane's next token - a greedy lane's printed, then fed (not once it ends); a drafting lane's
  drafts made (oseq_draft) and verified with its token, the accepted printed, the state rolled back to all but the
  last printed and the drafter seeded from the verify - all of them in one osmos_batch: a decode step, at most
  Server.round_tokens tokens (more would be a prompt's step: other bits), the drafts cut to fit, each lane its token.
*/
