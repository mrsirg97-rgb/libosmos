/* the Qwen forward's shapes (osmos/q4x/forward.h): its weights on the device, a step's buffers, a sequence's state.
   hip_forward.c builds them, hip_layers.c runs a layer's blocks on them, hip_step.c a step */
#pragma once

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/mv.h"
#include "osmos/gpu/npu_pack.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/mem/rowread.h"
#include "osmos/npu/npu.h"
#include "osmos/q4x/dflash.h"
#include "osmos/q4x/forward.h"

/* a step's buffers, for max_tokens tokens */
typedef struct Q4xBuf
{
  float *h, *xn, *lo, *gate, *mixed, *inj, *out, *x;                  /* the residual, a mixer's, a block's output */
  float *qkv, *z, *al, *be, *g, *b, *cv, *qkn, *ro, *fo;              /* a DeltaNet's */
  float *qf, *kr, *kq, *vr, *q, *gt, *ao, *fa; /* an attention's (kq, vr: k and v to the cache; fa: its chunks') */
  float *iqr, *iq, *ikr, *ixs; /* QSA's indexer: q raw then normed and roped, the raw keys, a piece's scores */
  int32_t *sel, *npick;        /* its picks: a token's blocks (q4x_ix_pick), their count */
  float *rl, *w, *ga, *up, *y, *shg, *shu, *sh, *sl;                  /* the MoE's */
  int32_t *ids, *rows, *cnt, *gidx, *hidx, *where; /* the picks, mm_group's tables (q4x_expert_tiles) */
  int32_t *eoff;                                   /* their experts' offsets (q4x_expert_tiles' scratch) */
  uint32_t *xg, *yg, *xd, *yd;
  const void **ptr;                                /* Q4X_GROUP_TENSORS tables of entries */
  void *x16, *xe16;                                /* mm's f16 activations: a matrix's, the experts' tiles */
  void *x16b; /* a small step's mixer's down's silu as f16 for its up (mm_run_silu: x16 still read), MM_SILU_MAX_TOKENS rows */
  int32_t *overflow;                               /* an activation past f16's: the step refused */
  float *emb, *key, *val, *kn, *qn, *pg, *gated, *nrm, *conv;         /* the n-gram layer's */
  int32_t *ple_rows, *tok;                                            /* device: its rows, the step's tokens */
  uint8_t *ple_sbuf; /* device: a streamed n-gram table's step rows, a token's ple_heads (stream_ngram) */
  int32_t *ple_iota; /* their places 0 .. : q4x_rows' row indices over ple_sbuf */
  float *taps;       /* device: the step's taps, [token][n_taps][n_embd] (Q4xOptions' tap_layers; NULL: none) */
  void *top_part;                                                     /* q4x_greedy's slices' bests (ops_argmax) */
  int32_t *pin_rows, *pin_tok, *pin_overflow;                         /* their pinned host copies */
  float *logits;
  bool *kept; /* host, [max_tokens]: a row's logits its last step's (q4x_kept) */
} Q4xBuf;

/* the MTP head's buffers (with a head): a step's hidden rows, a draft's inputs and logits */
typedef struct Q4xMtpBuf
{
  float *hid;   /* [max_tokens][hc n_embd]: each step's final residual (the head's h) at its rows */
  float *hin;   /* [max_tokens][hc n_embd]: the head's h rows (a position's before it) */
  float *emb;   /* [max_tokens][n_embd]: the head's tokens' embedding, then normed */
  float *hn;    /* [max_tokens][hc n_embd]: h normed */
  float *join;  /* [max_tokens][hc][2 n_embd]: eh_proj's input */
  float *logits;    /* [min(max_tokens, Q4X_MAX_RUNS)][the shortlist]: a draft step's, a sequence a row */
  int32_t *top_id;
  float *top_p;
  int32_t *pin_id; /* pinned: a draft step's tokens and their probabilities, a sequence each */
  float *pin_p;
} Q4xMtpBuf;

/* the DFlash2 drafter's buffers (with a drafter): a seed's context rows, then a block's */
typedef struct Q4xDfBuf
{
  float *th;            /* [max_tokens][n_embd]: the context's taps through fc, then its hidden norm */
  float *k, *v;         /* [max(max_tokens, block)][kv]: a layer's keys and values (the context's, the block's) */
  float *h, *x, *xc, *dyn, *q, *att, *a, *gate, *up, *hn; /* the block's: residual, normed, convolved, .. its out */
  float *logits, *hp, *val; /* the head's at the block's drafts' places, the selector's projection, the top-k */
  float *top_p;             /* each place's greatest token's probability (q4x_top: a draft's confidence's) */
  int32_t *idx, *path, *top_id;
  int32_t *tok, *iota; /* device: the block's tokens (y, then masks); 0 .. max_ctx + block (a run's positions) */
  int32_t *pin_tok, *pin_path, *pin_idx; /* pinned: the block's anchor in; its path, the top-k's ids, values, p out */
  float *pin_val, *pin_p;
} Q4xDfBuf;

#define Q4X_SIDE_STEP MM_TILE /* a step this small or smaller: the second stream (decode, a draft's verify) */
#define Q4X_ROW_BUFS 96 /* b's buffers at most (hip_forward.c's table) */
#define Q4X_ROW_THREADS 64 /* stream_ngram's readers: DeepSeek's Engram measure (832,000 random rows/s; 128 no more) */

#define Q4X_EXPERT_READERS 12 /* the experts' membrane's readers: DeepSeek's (FWD_FETCH_THREADS: DESIGN.md, the fetcher) */

struct Q4xForward
{
  const Q4xWeights *w;
  const Q4xMtp *mtp; /* or NULL */
  const Q4xDflash *df; /* or NULL */
  void **df_dev;       /* its tensors' bytes on the device (its file's table) */
  uint32_t df_ring;    /* a sequence's drafter rings' rows: the drafter's window (max_ctx at most) */
  Q4xDfBuf d;
  void **dev, **mtp_dev, **ov_dev; /* a tensor's bytes on the device, by its index in its file's table (the model's -
                                      none where the overlay stands in for it -, the MTP head's, the overlay's) */
  uint32_t max_ctx, max_tokens;
  Q4xRope rope; /* every rotation's: plain to the native context, YaRN past it (q4x_rope) */
  Q4xBuf b;
  Q4xMtpBuf m;
  q4x_probe_fn probe;
  void *probe_arg;
  /* a small step's second stream (q4x_moe_shared: the shared expert beside the routed ones): the stream the matmuls
     launch on now (NULL, or s2), s2 itself (non-blocking: the null stream waits on none of it), its events - in, the
     mixer done; out, the shared expert done */
  void *st;
  hipStream_t s2;
  hipEvent_t e_in, e_out;
  hipEvent_t e_mix, e_inj; /* a small step's inject on s2: the norm's output ready, the inject's done */
  hipEvent_t e_gates;      /* a small step's DeltaNet gates on s2 (alpha, beta, their gates): done */
  hipEvent_t e_ix;         /* a small step's QSA indexer on s2 (its keys and picks): done */
  bool inj_side;           /* an inject on s2 not yet waited for (q4x_inj_wait) */
  hipEvent_t pin_free; /* the pinned buffers' last copy to the GPU done (q4x_pin_wait, q4x_pin_sent) */
  /* stream_ngram: the n-gram table's file and readers, a step's rows read into ple_stage (pinned) at its embed, copied
     to b.ple_sbuf at the n-gram layer; ple_copied that copy's end (the next step's reads wait for it) */
  int32_t tap_layer[Q4X_MAX_TAPS]; /* Q4xOptions' tap_layers, ascending; n_taps of them */
  uint32_t n_taps;
  bool stream_ngram, ple_pending;
  int ple_fd;
  /* the experts on a membrane (q4x_forward_whole): its memory, the model's files opened O_DIRECT, the device table
     [layer][gate, up, down][expert] of their addresses; mem NULL: the experts their tensors', uploaded whole */
  struct Membrane *mem;
  struct HostSlots *exp_slots, *exp_stream;
  int *exp_fds;
  const void **exp_tab;
  Rowread *ple_read;
  uint8_t *ple_stage;
  uint64_t *ple_off;
  uint32_t ple_n;
  hipEvent_t ple_copied;
  uint32_t side_spin; /* q4x_forward_side_spin: a test's delay before the shared expert, on s2 (0: none) */
  /* a step past MM_MAX_TOKENS (hip_step.c): b's buffers a token a row (each one's field and row bytes), moved to a
     sub-batch's rows; and its runs' sequences at a sub-batch's start (their rings' cur and positions moved on) */
  struct
  {
    void *p;
    uint64_t row;
  } rows[Q4X_ROW_BUFS];
  uint32_t n_rows;
  Q4xSeq *sub_seq; /* Q4X_MAX_RUNS of them */
  /* 0.7.0: the NPU beside the GPU (q4x_forward_npu, hip_npu.c): a prompt step's first whole 128-token entries of an
     eligible layer's - their gate, up and down the NPU's, the GPU's mm_group skipping them (their counts zeroed) */
  Npu *npu;
  float npu_share;                 /* the share of a layer's picks the NPU takes (a format's cost scaling it) */
  bool npu_layer[Q4X_MAX_LAYERS];  /* a kernel for each of its experts' formats */
  float npu_cost[Q4X_MAX_LAYERS];  /* its share's scale: a part's NPU time, IQ3_XXS's 1 (q4x_npu_want) */
  NpuPart *npu_sel;                /* mapped: the parts taken (npu_select_q4x) */
  uint32_t *npu_nsel, npu_want;    /* mapped: their count; the layer's wanted (packed) */
  int32_t *npu_tok, *npu_slot;     /* device: the parts' rows of b.mixed and of b.xe16 */
  hipEvent_t npu_packed;           /* the parts packed: the NPU may read them */
  const void **npu_w0, **npu_w1;
  void **npu_c0, **npu_c1;
  uint64_t npu_wanted, npu_taken;  /* summed: the parts wanted, and those whole entries gave */
};

#define Q4X_SLOTS (Q4X_MAX_BACK + 1) /* a ring's states: the latest and those behind it */

struct Q4xSeq
{
  Q4xForward *f;
  int32_t pos;
  int32_t prompt_end; /* q4x_seq_prompt's: no logits for its tokens before prompt_end - 1 (0: none) */
  float *ssm[Q4X_MAX_LAYERS], *conv[Q4X_MAX_LAYERS]; /* a DeltaNet layer's S and conv history, Q4X_SLOTS each */
  uint8_t *kc[Q4X_MAX_LAYERS], *vc[Q4X_MAX_LAYERS];  /* an attention layer's cache (Q8_0 rows: q4x_kv_put) */
  float *ixk[Q4X_MAX_LAYERS], *ixr[Q4X_MAX_LAYERS];  /* its indexer's pooled keys (a block's), raw keys' ring */
  float *ple_hist;                                   /* the n-gram layer's conv history, Q4X_SLOTS of them */
  uint32_t cur, back;                                /* the rings' latest slot, and the states kept behind it */
  int32_t last[Q4X_MAX_NGRAM + Q4X_MAX_BACK];        /* its last tokens, the latest first (a rollback's too) */
  int n_last;
  /* with an MTP head: its cache, the main model's h of position pend_pos (-1: none, zeros), its last step's place */
  uint8_t *mkc, *mvc;
  float *mixk, *mixr; /* the head's indexer's */
  float *pend;
  int32_t pend_pos, step_pos0;
  uint32_t step_row0, step_n, step_t0; /* step_t0: its first token's place in the step (its taps' row) */
  /* with a drafter: each layer's key and value rings (df_ring rows of kv floats, position p at p % df_ring), the
     context positions they hold (df_lo .. df_pos - 1: seeded since the last gap - a load, a seed past its end) */
  float *dk[Q4X_DFLASH_MAX_LAYERS], *dv[Q4X_DFLASH_MAX_LAYERS];
  int32_t df_lo, df_pos;
  int32_t *step_tok; /* [max_tokens], host */
  /* a step failed after its GPU work began (q4x_step): its DeltaNet and n-gram rings part written - refused by every
     step, rollback, save and draft until a load (q4x_seq_reset: refuse() and a new sequence clear it) */
  bool broken;
};

/* a sequence's buffers' bytes: a DeltaNet layer's state and conv history, an attention cache's row, the n-gram
   layer's conv history */
typedef struct Q4xSeqBytes
{
  uint64_t ssm, conv, kv_row, ple;
  uint64_t ix_block, ix_ring; /* QSA's: a pooled key's (a block's), the raw keys' ring (q4x_ix_ring rows) */
} Q4xSeqBytes;
/* the raw keys' ring: a partial block's members after a rollback's reach (q4x_ix_pool) */
#define Q4X_IX_RING(ratio) ((uint32_t)(ratio) - 1u + Q4X_MAX_BACK)

Q4xSeqBytes q4x_seq_bytes(const Q4xGeometry *g);
bool q4x_seq_reset(Q4xSeq *s); /* hip_state.c: empty, position 0 (its recurrent states and histories zeroed) */

bool q4x_fail(char *err, const char *why);
/* y = W x for n tokens (rows xs, ys apart): mm (f16 activations, WMMA) for a matrix it tiles, else the matvec - by the
   matrix alone, so a token's bits are its own whatever its batch (hip_layers.c). `ready`: b.x16 is this x's already
   (the call before made it); q4x_mat2: two matrices on one x, one launch when mm takes both (mm_run2: each the bits it
   takes alone) */
/* the pinned host buffers (b.pin_tok, b.pin_rows) are read by copies queued, not done: before the host writes them
   again, the last copy waited for (a seed's queued behind a busy GPU, the next draft's tokens written over its own -
   0.5.10's drafts a run apart); after a copy from them, its end marked */
bool q4x_pin_wait(Q4xForward *f);
bool q4x_pin_sent(Q4xForward *f);
/* the mixer's inject (hc rows: latency, not bandwidth) beside the rest of the mixer on s2, a small step's: b.inj its
   output, b.xn its input - waited for before either is touched again (the next mixer, an hc_combine) */
bool q4x_inj_wait(Q4xForward *f);
bool q4x_mat(Q4xForward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys, bool ready);
bool q4x_mat2(Q4xForward *f, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
              uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
              bool ready);
/* the step's mm activations all within f16: the flag queued to its pinned copy (q4x_overflow_read, before the sync a
   step makes anyway - no round trip of its own), then read after it (q4x_overflow_ok: false with why, the flag
   cleared, when set). the flag accumulates: a call that queues no read (the MTP head's seeding) is caught at the next */
bool q4x_overflow_read(Q4xForward *f);
bool q4x_overflow_ok(Q4xForward *f, char *err);
const void *q4x_dev(const Q4xForward *f, const GgufTensor *t);
/* the NPU's (hip_npu.c): a layer's share of n tokens' picks taken from its tables and packed (0: none); its gate and
   up, then its down, once the GPU's of each are queued */
uint32_t q4x_npu_take(Q4xForward *f, const Q4xLayer *L, uint32_t n, const Q4xExpertBounds *bd, uint32_t D, uint32_t ff);
bool q4x_npu_gate_up(Q4xForward *f, const Q4xLayer *L, uint32_t D, uint32_t ff);
bool q4x_npu_pack_down(Q4xForward *f, uint32_t ff);
bool q4x_npu_down(Q4xForward *f, const Q4xLayer *L, uint32_t D, uint32_t ff);
void q4x_npu_destroy(Q4xForward *f);
void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n);

/* hip_layers.c: a layer's blocks on n tokens of b.h (the residual), its output into b.h */
/* combine: the block before's output (b.out, b.inj) into b.h first, in its norm's launch (q4x_hc_combine_norm) */
bool q4x_mix(Q4xForward *f, const GgufTensor *norm, const GgufTensor *down, const GgufTensor *up,
             const GgufTensor *inject, uint32_t n, bool combine);
/* a step's sequence: its tokens t0 .. t0 + n - 1 of the step's (q4x_step); an attention's, its cache and position */
typedef struct Q4xRun
{
  Q4xSeq *s;
  uint32_t t0, n;
} Q4xRun;
typedef struct Q4xCache
{
  uint8_t *kc, *vc;
  float *ixk, *ixr; /* its indexer's pooled keys and ring (QSA) */
  int32_t pos;
  uint32_t t0, n;
} Q4xCache;
/* the blocks with a sequence's own: the step's n tokens through the weights at once, each run's through its own */
bool q4x_deltanet(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n);
bool q4x_attention(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t n);
bool q4x_attention_kv(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t n);
bool q4x_moe_shared(Q4xForward *f, const Q4xLayer *L, uint32_t n); /* b.sh, b.sl: before q4x_moe, which reads them */
bool q4x_moe(Q4xForward *f, const Q4xLayer *L, uint32_t n);
bool q4x_ple(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n);
bool q4x_ple_issue(Q4xForward *f, uint32_t n); /* stream_ngram: the step's n tokens' rows begun (b.pin_rows') */
