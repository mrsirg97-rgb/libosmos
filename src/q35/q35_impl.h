/* the q35 forward's and sequence's insides (src/q35/hip_q35.c: the forward and its step; hip_q35_seq.c: a sequence -
   its rings, rollback, prompt, state and marks) */
#pragma once

#include "osmos/mem/ledger_hip.h"
#include "osmos/gpu/bufs.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/gpu/lsa_ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/lsa.h"
#include "osmos/q35/forward.h"
#include "osmos/gpu/overflow.h"

#define F sizeof(float)

struct Q35Forward
{
  OsmosLedger *ledger; /* borrowed (Q35Options'): every allocation of the forward and its sequences */
  const Q35Weights *w;
  uint32_t max_tokens, max_ctx, logit_rows;
  Q4xRope rope;
  void **dev;         /* the model's tensors, by index in its table */
  uint32_t head_type; /* the LM head's type on the GPU: its file's, or Q8_0 (head_q8) */
  float *h, *xn, *out;
  float *qkv, *z, *al, *be, *g, *b, *cv, *qkn, *ro, *fo;
  float *qf, *kr, *vr, *q, *gt, *kq, *ao, *fa, *da; /* da: the decode attention's scratch */
  float *ga, *up, *y;
  float *hx;                /* the kept rows' residuals, gathered for the head (logit_rows x n_embd) */
  float *logits;            /* the kept rows' logits, in the order kept (logit_rows x n_vocab) */
  float *pin_logits;        /* q35_step's host copy of its rows */
  int32_t *slot;            /* the last step's row -> its logits' slot, or -1 (host, max_tokens) */
  uint32_t rows;            /* the last step's rows */
  int32_t *ids, *argmax;    /* q35_greedy's ids (device, logit_rows) and its scratch */
  int32_t *tok, *pin_tok, *overflow;
  OverflowFlag ovf; /* the flag's reset (osmos/gpu/overflow.h): owed when one failed */
  void *x16;
  const float *x16_x; /* the activations x16 holds as f16 (the last matrix's, on the matmul), or NULL */
  q35_tap_fn tap;
  void *tap_arg;
  /* latent sparse attention (q35_forward_lsa): a layer's maps (device, LSA_IN x LSA_DIM: NULL off), the read's scratch,
     a query's index vector, its read; the host's room for a closing block's Ward */
  bool lsa;
  float *wq[Q35_MAX_LAYERS], *wk[Q35_MAX_LAYERS];
  LsaScratch *lsa_s;
  float *iq;
  uint32_t *list, *list_n; /* a read and its count (device) */
  float *h_ik, *h_cent;
  int32_t *h_mem;
  GpuBufs bufs; /* every GPU and pinned buffer of the forward, recorded as made (q35_alloc, the step's want[]) -
                   freed from the record at its destroy */
};

struct Q35Seq
{
  GpuBufs bufs; /* its GPU buffers, recorded as made (its forward's ledger's) */
  Q35Forward *f;
  int32_t pos;
  uint32_t cur, back;      /* the rings' current slot, and the tokens kept behind it (q35_seq_rollback's reach) */
  uint32_t prompt_end;     /* q35_seq_prompt's: rows before end - 1 not kept */
  bool broken;             /* a failed step's: refused until a load */
  float *conv[Q35_MAX_LAYERS], *ssm[Q35_MAX_LAYERS]; /* a DeltaNet's conv histories and states, Q35_SLOTS each */
  void *kc[Q35_MAX_LAYERS], *vc[Q35_MAX_LAYERS];      /* an attention's cache, Q8_0 rows (q4x_kv_put) */
  /* with LSA: an attention layer's keys' index vectors (max_ctx x LSA_DIM), its closed blocks' buckets */
  float *ik[Q35_MAX_LAYERS], *cent[Q35_MAX_LAYERS];
  int32_t *mem[Q35_MAX_LAYERS];
  uint32_t closed[Q35_MAX_LAYERS];
};

/* a sequence's per-layer bytes: a DeltaNet state slot, a conv history slot, an attention cache row */
typedef struct Q35SeqBytes
{
  uint64_t ssm, conv, kv_row;
} Q35SeqBytes;
Q35SeqBytes q35_seq_bytes(const Q35Geometry *g);

/* a GPU buffer, or a pinned one, recorded in b (its owner's: the forward's or a sequence's - b's ledger the forward's),
   freed from the record at the owner's destroy */
bool q35_alloc(GpuBufs *b, void *p, uint64_t bytes);
bool q35_pinned(GpuBufs *b, void *p, uint64_t bytes);

/* a tensor's bytes on the GPU (by its index in the model's table) */
static inline const void *dev(const Q35Forward *f, const GgufTensor *t) { return f->dev[t - f->w->model->tensors]; }
static inline const float *fv(const Q35Forward *f, const GgufTensor *t) { return (const float *)dev(f, t); }
