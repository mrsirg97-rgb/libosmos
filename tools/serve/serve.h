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

typedef struct Conv
{
  OsmosSeq *s;
  Ids held; /* the tokens the sequence holds: oseq_position(s) of them */
  /* a family that reads apart (osmos_reads_apart: DeepSeek): the state at its last prompt's end, held.v's first
     snap_n tokens - a turn's earlier answer (decoded, not a prompt's bits) read again from it, never continued on, so
     a turn's answer the bits of its prompt read whole (0.7.1). snap_n 0: none */
  void *snap;
  uint64_t snap_bytes, snap_cap;
  uint32_t snap_n;
  char err[OSMOS_ERR];
} Conv;

typedef struct Gen
{
  long long max_new, rid;
  bool has_rid;
  int spec;     /* spec=0 / 1, or -1 (not given): as the server was started - drafting with a drafter (maya's) */
  int32_t *ids; /* malloc'd */
  uint32_t n;
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
bool conv_read(Conv *c, const Server *sv, const int32_t *p, uint32_t k, uint32_t n, bool seed);

bool serve_build(Server *sv, Conv *conv, int lanes, const OsmosModel *m, OsmosOptions o, char err[OSMOS_ERR]);
void serve_unbuild(Server *sv, Conv *conv, int lanes);
int serve_single(Server *sv, Conv *c);
int serve_multi(Server *sv, Conv *lanes, int n);

/*
conv_reach
  what the conversation would reuse of prompt p (n tokens, one left to read): its tokens when they are a prefix of
  p[0 .. n - 1), or their common part when the rest is within the sequence's rollback reach (oseq_reach); else 0. a
  family that reads apart (osmos_reads_apart) counts its tokens to its last prompt's end alone (the snapshot's).

conv_continue
  the conversation made a prefix of p: continued, rolled back to the common part, or set aside and the longest kept
  conversation p continues taken back (else emptied). held.n: the tokens reused. a family that reads apart first goes
  back to its last prompt's end (its snapshot, or nothing): a turn's earlier answer read again as the prompt's, so
  the answer is the bits of its prompt read whole, whatever came before it (0.7.1). conv_set_aside the same: what
  the slots keep a prompt's state, never a decoded answer's.

conv_read
  k tokens of p (from held.n) through one osmos_read, into the sequence and held: the step's logits theirs; at the
  prompt's end (held.n == n) the state kept as the conversation's snapshot (a family that reads apart); with
  seed, the drafter's rings seeded from the step - the piece's part of the prompt's last window (n - window .. n - 1,
  n the prompt's tokens): the positions a step taps past the decoder's compaction are those near the prompt's end,
  and a later piece's step would not hold an earlier piece's.
*/
