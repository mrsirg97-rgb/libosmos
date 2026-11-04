/* osmos-serve's shared shapes: the server (the engine, its drafter and slots), a conversation (a sequence and the
   tokens it holds), a GEN's fields. one conversation (single.c) or several at once (multi.c) over the same ones */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/run/drafter.h"
#include "osmos/run/forward.h"
#include "osmos/serve/slots.h"

#define SERVE_MAX_EOS 16

typedef struct Ids
{
  int32_t *v;
  uint32_t n, cap;
} Ids;

typedef struct Server
{
  Forward *f;
  Drafter *d;      /* or NULL */
  Slots *slots;    /* or NULL (OSMOS_SLOT_KEEP off) */
  uint32_t max_tokens, max_ctx;
  uint32_t round_tokens; /* a decode step's most tokens: (experts - 1) / experts a token; more, a prompt's step */
  uint32_t window;       /* the attention's window: the drafter's ring of positions */
  int32_t eos[SERVE_MAX_EOS];
  int n_eos;
} Server;

typedef struct Conv
{
  Seq *s;
  Ids held; /* the tokens the sequence holds: seq_position(s) of them */
  char err[FORWARD_ERR];
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
bool server_spec(const Server *sv, const Gen *g);

bool conv_create(Conv *c, const Server *sv);
void conv_destroy(Conv *c);
bool conv_restart(Conv *c, const Server *sv);
void conv_set_aside(Conv *c, const Server *sv);
uint32_t conv_reach(const Conv *c, const int32_t *p, uint32_t n);
bool conv_continue(Conv *c, const Server *sv, const int32_t *p, uint32_t n);
bool conv_read(Conv *c, const Server *sv, const int32_t *p, uint32_t k, uint32_t n, bool seed);

bool serve_build(Server *sv, Conv *conv, int lanes, const Weights *w, ForwardOptions fo, const char *dspark,
                 char err[FORWARD_ERR]);
void serve_unbuild(Server *sv, Conv *conv, int lanes);
int serve_single(Server *sv, Conv *c);
int serve_multi(Server *sv, Conv *lanes, int n);

/*
conv_reach
  what the conversation would reuse of prompt p (n tokens, one left to read): its tokens when they are a prefix of
  p[0 .. n - 1), or their common part when the rest is within seq_rollback's reach; else 0.

conv_continue
  the conversation made a prefix of p: continued, rolled back to the common part, or set aside and the longest kept
  conversation p continues taken back (else emptied). held.n: the tokens reused.

conv_read
  k tokens of p (from held.n) through one forward_batch, into the sequence and held: the step's logits theirs; with
  seed, the drafter's rings seeded from the step - the piece's part of the prompt's last window (n - window .. n - 1,
  n the prompt's tokens): the positions a step taps past the decoder's compaction are those near the prompt's end,
  and a later piece's step would not hold an earlier piece's.
*/
