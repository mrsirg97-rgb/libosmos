/* osmos-serve's shared shapes: the server (the engine, its drafter and slots), a conversation (a sequence and the
   tokens it holds), a GEN's fields. one conversation (single.c) or several at once (multi.c) over the same ones */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/run/engine.h"
#include "osmos/serve/slots.h"
#include "osmos/text/answer.h"

#define SERVE_MAX_EOS 16

typedef struct Ids
{
  int32_t *v;
  uint32_t n, cap;
} Ids;

typedef struct Server
{
  Osmos *e;       /* the model's engine: its drafter when osmos_block is not 0 */
  Slots *slots;    /* or NULL (OSMOS_SLOT_KEEP off) */
  uint32_t max_tokens, max_ctx;
  uint32_t round_tokens; /* a decode step's most tokens: (experts - 1) / experts a token; more, a prompt's step */
  uint32_t window;       /* a drafter seed's window: the positions it reads back from a prompt's end */
  uint64_t ram_kept;     /* the system's RAM kept free (--fill; FIT_RAM_FREE by default): the build's and the slots' */
  int32_t eos[SERVE_MAX_EOS];
  int n_eos;
  AnswerRule answer; /* --answer-after's: no end before a turn's answer has begun (blank NULL: off) */
} Server;

/* a place the conversation goes back to (0.7.5): the sequence's mark (oseq_mark_save) at held.v's first n tokens - n 0:
   none. the sequence's own while it holds those tokens: a rollback under it, a point before it gone back to, a
   restart or a kept conversation taken back drops it */
typedef struct Point
{
  void *buf;
  uint64_t bytes, cap;
  uint32_t n;
} Point;

typedef struct Conv
{
  OsmosSeq *s;
  Ids held; /* the tokens the sequence holds: oseq_position(s) of them */
  /* its last prompt's point, one token short of its end: a family that reads apart (DeepSeek) goes back to it every
     turn (a decoded answer read again as the prompt's, never continued on: its answers the bits of their prompts
     read whole, 0.7.1); every family where a turn leaves the decoded answer (a client writes it back otherwise) */
  Point prompt;
  /* the last user turn's point (GEN mark=, osmos-api's): where a turn after a new user message leaves this prompt,
     the template dropping the reasoning since */
  Point user;
  char err[OSMOS_ERR];
} Conv;

typedef struct Gen
{
  long long max_new, rid;
  bool has_rid;
  int spec;     /* spec=0 / 1, or -1 (not given): as the server was started - drafting with a drafter (maya's) */
  int32_t *ids; /* malloc'd */
  uint32_t n;
  uint32_t mark; /* mark=: a point besides the prompt's (Conv.user), 0 none */
} Gen;

double now_ms(void);
void out(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
bool ids_push(Ids *a, int32_t x);
bool parse_gen(char *line, Gen *g, char *why, size_t wn);
bool server_eos(const Server *sv, int32_t t);
bool server_answer(Server *sv, const Ids *held, const int32_t *extra, uint32_t n, uint32_t row, int32_t *pick,
                   char err[OSMOS_ERR]);
bool server_spec(const Server *sv, const Gen *g);

bool conv_create(Conv *c, const Server *sv);
void conv_destroy(Conv *c);
bool conv_restart(Conv *c, const Server *sv);
void conv_failed(Conv *c, const Server *sv);
void conv_set_aside(Conv *c, const Server *sv);
uint32_t conv_reach(const Conv *c, const Server *sv, const int32_t *p, uint32_t n);
bool conv_continue(Conv *c, const Server *sv, const int32_t *p, uint32_t n);
uint32_t conv_prompt_end(const Conv *c, uint32_t n, uint32_t mark);
bool conv_read(Conv *c, const Server *sv, const int32_t *p, uint32_t k, uint32_t n, uint32_t mark, bool seed,
               uint32_t *row);

bool serve_build(Server *sv, Conv *conv, int lanes, const OsmosModel *m, OsmosOptions o, char err[OSMOS_ERR]);
void serve_unbuild(Server *sv, Conv *conv, int lanes);
int serve_single(Server *sv, Conv *c);
int serve_multi(Server *sv, Conv *lanes, int n);

/*
conv_reach
  what the conversation would reuse of prompt p (n tokens, one left to read): its tokens when they are a prefix of
  p[0 .. n - 1), their common part when the rest is within the sequence's rollback reach (oseq_reach), else the
  furthest of its points (Conv.prompt, Conv.user) p continues; else 0. a family that reads apart
  (osmos_reads_apart) counts its tokens to its prompt's point alone.

conv_continue
  the conversation made a prefix of p: continued, rolled back to the common part, gone back to the furthest point p
  continues (0.7.5: the prompt's, past an answer the client wrote back otherwise; the last user turn's, before
  reasoning the template drops), or set aside and the longest kept conversation p continues taken back (else
  emptied). held.n: the tokens reused. a family that reads apart first goes back to its prompt's point: a turn's
  earlier answer read again as the prompt's, so the answer is the bits of its prompt read whole, whatever came
  before it (0.7.1). conv_set_aside goes back to it too, every family: what the slots keep a prompt's state, never a
  decoded answer's (the next turn's history writes it otherwise).

conv_prompt_end
  the prompt's end as its first piece reads it (oseq_prompt): mark when a mark lies ahead - so the state there is
  what a prompt ending at mark leaves (DeepSeek's decoder skip) - else n; conv_read moves it to n past the mark.

conv_read
  k tokens of p (from held.n) through osmos_read, into the sequence and held; *row the last read's row of the last
  token (its logits the answer's first). the read stops at the prompt's points (0.7.5), each kept: mark (the last
  user turn's, when given: Conv.user) and n - 1 (Conv.prompt: the generation prompt's last token is one a later turn
  renders otherwise - DeepSeek's <think> becomes </think>), the rest then read - a prompt in pieces the bits of one
  read whole. with seed, the drafter's rings seeded from each piece's step - its part of the prompt's last window
  (n - window .. n - 1): the positions a step taps past the decoder's compaction are those near the prompt's end,
  and a later piece's step would not hold an earlier piece's.
*/
