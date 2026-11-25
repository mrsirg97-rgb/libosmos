#ifndef OSMOS_GPU_ATTN_CORE_H
#define OSMOS_GPU_ATTN_CORE_H

/* the attention core (docs/attention-core.md): one attention for every family. a query's keys in units of
   ATTN_UNIT at absolute positions (a list's: ATTN_UNIT of its entries), each unit's online softmax from nothing,
   folded in order from the sink; ATTN_TILE keys a tile, ascending; f16 operands on the matrix cores, F32 sums - so a
   token's bits are its own whatever its step, its batch or its decode. a family supplies its rows' format, its logit
   term, its sink and its gate; rope, clips and norms come before it */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ATTN_UNIT 256 /* a unit's keys (Inkling's, every family's) */
#define ATTN_TILE 16  /* a tile's keys (WMMA's) */
#define ATTN_HEAD 128 /* Inkling's head (its records' rows); C1: 256 too (Qwen's) */
#define ATTN_HEAD_MAX 256
#define ATTN_GROUP_MAX 8 /* query heads a KV head at most (a prompt block's waves) */
#define ATTN_REL 16      /* ATTN_TERM_REL's states a head (its sum unrolled) */
#define ATTN_EXTENT_MAX 1024
/* Inkling's records (ink2_to_kv8, ink2_to_kv5): a head's row of ATTN_HEAD values */
#define ATTN_KV8_BLOCK 32 /* values a scale */
#define ATTN_KV8_RECORD (ATTN_HEAD + ATTN_HEAD / ATTN_KV8_BLOCK * 2) /* int8s, then f16 scales */
#define ATTN_KV5_RECORD (ATTN_HEAD / 2 + ATTN_HEAD / 8 + ATTN_HEAD / ATTN_KV8_BLOCK * 2) /* nibbles, fifth bits, scales */

/* a cache row's format, keys and values each their own */
typedef enum AttnRows
{
  ATTN_ROWS_F32,
  ATTN_ROWS_F16,  /* IEEE half bits (ink2_to_f16's) */
  ATTN_ROWS_KV8,  /* Inkling's Q8 records (ink2_to_kv8's): a head's int8s, then a f16 scale a 32 */
  ATTN_ROWS_KV5,  /* Inkling's Q5 records (ink2_to_kv5's), values alone */
  ATTN_ROWS_Q8_0, /* ggml's Q8_0 blocks (q4x_kv_put's) */
} AttnRows;

/* a logit's term past q . k scaling */
typedef enum AttnTerm
{
  ATTN_TERM_NONE,
  ATTN_TERM_REL, /* Inkling's relative bias: rel[t][h] . rel_proj[:, d] at distance d < extent, else 0 */
} AttnTerm;

/* a prompt block's rows: tokens of one query head (Inkling's: 16 tokens a tile, their keys shared), or one token's
   query heads (Qwen's: its picks its own - a tile's rows must read the same keys, the matrix cores not blind to a key
   its weight zeroes) */
typedef enum AttnBlock
{
  ATTN_BLOCK_TOKENS,
  ATTN_BLOCK_HEADS,
} AttnBlock;

/* what one gate value scales: none, a head's values, a value */
typedef enum AttnGate
{
  ATTN_GATE_NONE,
  ATTN_GATE_HEAD,
  ATTN_GATE_VALUE,
} AttnGate;

/* a layer kind's attention, checked whole at attn_core_create */
typedef struct AttnCoreDesc
{
  uint32_t heads, kv_heads; /* GQA: heads / kv_heads query heads a key head, consecutive */
  uint32_t dk, dv;          /* a key's and a value's dims */
  AttnRows k_rows, v_rows;
  uint32_t window;          /* the keys a token sees, itself the last (0: every one before it) */
  float scaling;            /* q . k's */
  AttnTerm term;
  uint32_t d_rel, extent;   /* ATTN_TERM_REL's */
  bool sink;                /* a sink logit a (token, head), no value */
  AttnGate gate;            /* sigmoid(gate) times the output */
  AttnBlock block;
  uint32_t ratio;           /* picks' blocks of keys (0: no picks; ATTN_BLOCK_HEADS alone) */
  uint32_t unit;            /* a unit's keys, a multiple of ATTN_TILE (0: ATTN_UNIT); a family's, fixed with its bits */
  uint32_t slots;           /* the cache's rows: 0 a row a position, else a ring (a position's row position % slots: at
                               least the window and a step's tokens past it) */
  bool apart;               /* a family whose prompt and decode read apart (osmos1.7b's, q35's): the step's read picks
                               the prompt's kernel or the decode's, each its own bits; else one kernel for both */
  uint32_t max_tokens;      /* a step's queries at most */
  uint32_t max_keys;        /* a query's keys at most (positions: the context; lists: their stride) */
} AttnCoreDesc;

typedef struct AttnCore AttnCore;

/* the device scratch a core for d borrows (a split decode's units' partials): its bytes */
uint64_t attn_core_scratch(const AttnCoreDesc *d);
/* the core for d over scratch (the device's, attn_core_scratch(d) bytes at least; borrowed - its owner frees it, and
   may use it between steps: a step's alone); NULL with err naming what it does not take - a dim, a group, a format,
   a term, a bound, the scratch short */
AttnCore *attn_core_create(const AttnCoreDesc *d, void *scratch, uint64_t scratch_bytes, char err[OSMOS_ERR]);
void attn_core_destroy(AttnCore *c);

/* a step's keys: the cache's rows from key0's position (row r the key at position key0 + r), n_keys of them; picks
   given (the desc's ratio), each query's keys its picked blocks (sel[t stride ..] ascending, npick[t] of them - npick
   equal to its visible blocks, (pos + 1) / ratio: every one, none read) then its tail (its visible blocks' end to
   itself); lists given, each query's own counts[t] positions (lists[t stride ..], any order) of the same rows */
typedef struct AttnKeys
{
  const void *k, *v;
  uint32_t key0, n_keys;  /* key0 a multiple of ATTN_TILE */
  const int32_t *sel, *npick; /* NULL: every key */
  const uint32_t *lists;  /* NULL: positions */
  const int32_t *counts;
  uint32_t stride;        /* sel's or lists' entries a query */
} AttnKeys;

/* a step: n_tok queries at positions pos0 .. pos0 + n_tok - 1 (with positions: each a key, pos0 + t - key0 < n_keys) */
typedef struct AttnStep
{
  const float *q; /* [n_tok][heads][dk] */
  uint32_t n_tok, pos0;
  AttnKeys keys;
  const float *rel, *rel_proj; /* ATTN_TERM_REL's: [n_tok][heads][d_rel], [d_rel][extent] */
  const float *sink;           /* [n_tok][heads] (the desc's sink) */
  const float *gate;           /* [n_tok][heads][dv] or [n_tok][heads] (the desc's gate) */
  float *out;                  /* [n_tok][heads][dv] */
  void *x16;                   /* out also as mm_x16 lays it (NULL: not): its rows x16_row0 .. */
  int32_t *overflow;           /* x16's flag (mm_x16's) */
  uint32_t x16_row0;           /* the step's first row in x16 (a sequence's among a step's) */
  bool read;                   /* a prompt's step (the desc's apart: its kernel; else unread) */
} AttnStep;

/* the step on stream; false (nothing written) when it is past the core's desc: more tokens or keys, a list with
   positions' term, a key0 off a tile */
bool attn_core_run(const AttnCore *c, const AttnStep *s, void *stream);

#ifdef __cplusplus
}
#endif

#endif
