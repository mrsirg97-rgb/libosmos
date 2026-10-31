#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthread.h>

#include <hip/hip_runtime_api.h>

#include "substrate/mem/fetchq.h"
#include "substrate/mem/pool.h"
#include "substrate/mem/resident.h"
#include "substrate/mem/rowread.h"
#include "substrate/gpu/mm.h"
#include "substrate/gpu/quant.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/model/engram.h"
#include "substrate/run/forward.h"

#define FWD_MAX_FILES 16
#define FWD_MAX_BATCH MM_MAX_TOKENS                 /* a step's tokens: the matmul's */
#define FWD_MAX_SUBS FORWARD_SUBS
/* a ratio r source's pending positions a sequence: a step's and the r - 1 before them, each row's r all held */
#define FWD_PEND_ROWS(r) (FWD_MAX_BATCH + (r) - 1)
#define FWD_MAX_STEP FORWARD_MAX_TOKENS             /* the experts' tokens: a layer's every sub-batch at once */
#define FWD_MAX_PAIRS (FWD_MAX_BATCH * OPS_MAX_TOPK) /* a sub-batch's (token, routed expert) pairs */
#define FWD_MAX_GROUPS 64                           /* a layer's wo_a groups */
/* the resident set's following of the text (hip_adapt.c): an expert's heat halved every this many steps; a streamed
   one takes a resident one's place when this much hotter; at most this many a layer a step (each a copy in GPU
   memory) */
#define FWD_GUESS_K 6 /* a token's next-layer experts guessed (hip_guess.c): the router's top this many */
#define FWD_HEAT_HALF_LIFE 128.0 /* measured: 8 - 128 within ~2 ms of one another (DESIGN.md) */
#define FWD_SWAP_MARGIN 2.0      /* and 1.25 the same */
#define FWD_SWAPS_A_LAYER 2      /* 6 the same: a step's new picks are a few a layer */
/* a layer's routing tables at most for `pairs` pairs: 3 matrix pointers, 4 offsets, a count an entry (an expert, or
   a part of MM_MAX_TOKENS of one: at most pairs entries); an order and a pair a pair; 2 x 16 slots an expert - with
   each table's alignment (16) */
#define FWD_UP_BYTES(pairs) ((uint64_t)(pairs) * (3 * 8 + 4 * 4 + 4 + 2 * 4 + 2 * MM_TILE * 4) + 10 * 16)
/* the experts' slots at most: each its tokens padded to whole tiles - the pairs and a tile's padding an expert */
#define FWD_MAX_SLOTS(pairs, experts) ((pairs) + (uint64_t)(experts) * (MM_TILE - 1))

/* the fetcher's finished reads, handed from its workers to the forward (the pool's one consumer) */
typedef struct Read
{
  uint32_t unit;
  bool ok;
} Read;

typedef struct Inbox
{
  pthread_mutex_t mu;
  pthread_cond_t cv;
  Read *r;
  size_t n, cap;
  bool broken; /* a done the inbox could not hold: the forward refuses from then on */
} Inbox;

/* a sub-batch's own: all its tokens carry from one layer to the next - swapped into the Forward and its Scratch to
   run one of its layers (hip_step.c's enter and leave) */
typedef struct Sub
{
  float *h, *f_pre;
  int32_t *pos, *sel, *n_sel, *oa_cnt, *keep, *back;
  struct Pinned *pin;
  int64_t *rows;
  uint32_t n, n_all, n_keep;
  uint32_t xrow; /* its first row in the step's experts' input and output (Forward's xf0, moe0): a layer's */
  int32_t at;
  Seq *bs[FWD_MAX_BATCH];
  int32_t row[FWD_MAX_BATCH];
  bool want[FWD_MAX_BATCH];
} Sub;

/* the step's device buffers, each [FWD_MAX_BATCH][...] a token - but the experts' (the FFN's input xf and output
   moe, the router's, the shared and routed experts'), a layer's every sub-batch at once: [max_tokens][...] a token,
   [max_tokens * n_used][...] a pair; and mid, f_post, f_comb [n_subs][FWD_MAX_BATCH][...], a sub-batch's
   own across its layer's experts. xf, moe, mid, f_post and f_comb point at the running sub-batch's (Forward's xf0 ..
   the whole) */
typedef struct Scratch
{
  float *h, *hcp, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *pre0;
  float *x, *qa, *q, *kv, *o, *oa, *y, *mid, *xf;
  float *logits, *rw, *vocab, *ga, *upr, *hh, *ye, *moe;
  float *sga, *sup, *shh, *sye; /* the shared expert's, beside the routed ones' [n_used][...] */
  void *up;                     /* a layer's routing tables, sent in one copy from Pinned's up */
  float *rwp;                   /* a pair's routing weight, in its row's order */
  void *x16, *xe16;             /* f16 activations: a matmul's [16][cols]; the experts' [expert][16][cols] */
  float *lat, *kvc, *scg, *ikr, *qix, *wix, *score;
  float *att;                   /* the attention's scores, then their denominators (ops_sparse_attn) */
  float *erow_f, *ekv;
  uint8_t *erow;
  int32_t *ids, *pos, *rpos, *ridx, *sel, *n_sel, *overflow, *oa_cnt;
  int32_t *keep, *back; /* the decoder's tokens (hip_skip.c): kept, compacted; then each back at its row */
  int32_t *gidx;        /* the next layer's experts guessed (hip_guess.c), [FWD_MAX_BATCH][FWD_GUESS_K] */
  float *gw;            /* (their weights: the router's, unread) */
} Scratch;

/* host memory the GPU copies to and from on the stream (pinned: the copies are asynchronous). each is written only
   after the stream has passed the copy that read it last (the next routing event, or the step's end) */
typedef struct Pinned
{
  int32_t tok[FWD_MAX_BATCH]; /* the step's tokens */
  int32_t pos[FWD_MAX_BATCH]; /* and their positions */
  int32_t overflow;               /* an activation past f16: the step refused */
  int32_t oa_cnt[FWD_MAX_GROUPS]; /* each wo_a group's tokens: the step's */
  int32_t rpos[WEIGHTS_MAX_LAYERS][FWD_MAX_BATCH];
  int32_t keep[FWD_MAX_BATCH], back[FWD_MAX_BATCH]; /* hip_skip.c's: the kept tokens' rows, and back */
} Pinned;

struct Forward
{
  const Weights *w;
  const Geometry *g;
  int n_layers;
  uint32_t max_ctx;
  uint32_t score_stride; /* a token's indexer scores: the most rows visible at max_ctx (the smallest ratio) */
  uint32_t slots;
  int fds[FWD_MAX_FILES];
  Pool *pool;
  Fetchq *fq;
  Inbox in;
  Resident *res;            /* the layers' tensors and a share of the experts, in GPU memory (resident.h) */
  int32_t *rix;             /* an expert's place in res (unit: layer * n_experts + expert), -1: streamed by the pool */
  uint32_t n_expert_units;  /* the pool's units: layer * n_experts + expert */
  uint64_t *picks;          /* each unit's picks by the router so far (forward_picks) */
  PoolUnit *units;          /* each expert unit's file ranges (a swap's read: hip_adapt.c) */
  double *heat;             /* each unit's heat at heat_at, its picks halved every FWD_HEAT_HALF_LIFE steps */
  uint64_t *heat_at, *seen; /* the step its heat was last updated at; its picks folded into the heat so far */
  uint64_t steps, swaps;    /* the steps ended; the experts swapped into the resident set */
  const uint8_t **tdev;     /* a model tensor's device pointer (index: its place in the Gguf), resident ones only */
  uint32_t *ttype;          /* its type as the GPU reads it when not the file's (QUANT_Q5_KR: repacked), else 0 */
  const void **oa_tab;      /* each layer's wo_a groups for mm_group: [layer][o_groups], on the device */
  uint32_t *oa_xoff, *oa_yoff; /* group k: x at k * its 4,096 (of a token's attention), y at k * o_lora */
  int eg_fd;
  Rowread *eg_read; /* Engram's rows, read many at once */
  uint64_t *eg_off; /* a layer's rows' places in the Engram file: [token][col] */
  Engram *hash;
  float *freqs_win, *freqs_comp;
  Scratch b;
  float *xf0, *moe0, *mid0, *f_post0, *f_comb0; /* Scratch's xf .. f_comb whole (the sub-batches' views of them) */
  /* the experts' routing, a layer's every sub-batch at once: pinned, the picks [token][slot] and the tables
     (hip_ffn.c's carve, one copy to Scratch's up); on the host, the pairs' by-expert order and places */
  int32_t *r_ids;
  int32_t *r_guess;       /* pinned: the next layer's guessed experts, [token][FWD_GUESS_K] */
  uint64_t *guessed;      /* a unit's guess: the visit it was guessed for (hip_guess.c) */
  uint64_t g_need, g_hit; /* a decode's streamed picks, and of them the ones guessed a layer ahead */
  uint8_t *r_up;
  uint32_t *m_by, *m_row, *m_base, *m_slot0;
  /* forward_lane's: on, its CPU threads (0: every expert on the GPU, in the order the CPU would sum it), and a small
     step's tables - pinned, [gate, up, down][pair] pointers then each token's pairs in ascending expert id */
  bool lane;
  int lane_threads;
  uint8_t *lane_up;
  void *lane_dev;
  float *lane_x, *lane_w; /* a small step's rows' FFN inputs and weights, gathered by expert */
  uint8_t *stage; /* pinned: the Engram rows, [sub-batch][Engram layer][token]: a sub-batch's copy may still be
                     queued when the next one's rows are read (a layer's sub-batches run to its experts unwaited) */
  int n_eg;       /* the Engram layers (stage's) */
  /* DSpark's taps (fwd_taps_on, the drafter's): the streams' mean entering each target layer, [row][tap][D], for the
     last sub-batch's rows of the step (a sequence's last ones: the drafter's ring holds 128); their positions and
     sequences. NULL: none taken */
  float *taps;
  int n_tap, tap_of[WEIGHTS_MAX_LAYERS];
  uint32_t tap_n;
  int32_t tap_pos[FWD_MAX_BATCH];
  Seq *tap_bs[FWD_MAX_BATCH];
  uint32_t cur;   /* the running sub-batch (enter's) */
  Pinned *pin;
  hipEvent_t routed;                /* the router's picks are on the host */
  uint32_t held[FWD_MAX_PAIRS];     /* the last layer's experts: released once the stream is past them */
  uint32_t n_held;
  int held_layer;                   /* the layer they are (-1: none): its sub-batches share them */
  bool *is_held;                    /* an expert unit among them */
  Sub sub[FWD_MAX_SUBS];            /* the step's sub-batches; the Forward's own fields the one running */
  uint32_t n_subs, max_tokens;      /* the most sub-batches a step takes (ForwardOptions' max_tokens), and its tokens */
  uint32_t n_sub;
  uint64_t visits; /* the layers' expert steps so far: the held ones wanted again n_layers on (pool_release) */
  int64_t *rows;  /* each token's Engram rows, every Engram layer's: [token][layer][col] */
  uint32_t n;                    /* the step's tokens */
  Seq *bs[FWD_MAX_BATCH];        /* token i's sequence */
  int32_t at;                    /* the step's first position (the probes' index) */
  int dec;                       /* the decoder's first layer: its KV the only one past it (-1: none) */
  uint32_t reach;                /* how far back the decoder past dec reads: (layers past dec) (window - 1) + 1 */
  uint32_t n_all, n_keep;        /* the step's tokens before the decoder's compaction, and those it keeps */
  int32_t row[FWD_MAX_BATCH];    /* a kept token's row in the step (compacted row i: row[i]) */
  bool want[FWD_MAX_BATCH];      /* a token whose logits the caller may read */
  forward_probe_fn probe;
  void *probe_arg;
};

struct Seq
{
  Forward *f;
  int32_t pos;
  int32_t prompt_end; /* seq_prompt's end (0: none) */
  int32_t staged;     /* its tokens placed so far in the step being placed */
  int32_t dec_from;   /* the first position whose decoder state is kept: every one from it to pos */
  int32_t last[ENGRAM_MAX_NGRAM + SEQ_MAX_BACK]; /* last[0] the newest: the n-gram's, and seq_rollback's depth */
  int n_last;
  float *ring[WEIGHTS_MAX_LAYERS];
  float *ckv[WEIGHTS_MAX_LAYERS], *ik[WEIGHTS_MAX_LAYERS];
  float *pend_kvc[WEIGHTS_MAX_LAYERS], *pend_sc[WEIGHTS_MAX_LAYERS];
  float *dring; /* a drafter's rings, [block][window][head_dim] (drafter_seed's; NULL until then) */
};

/* a resident tensor's bytes as the GPU reads them (its copy in the resident set) */
static inline const uint8_t *fwd_dev(const Forward *f, const GgufTensor *t)
{
  return f->tdev[t - f->w->model->tensors];
}

/* the type the kernels read a resident tensor as (the file's, or QUANT_Q5_KR once repacked), and its bytes so */
static inline uint32_t fwd_type(const Forward *f, const GgufTensor *t)
{
  const uint32_t k = f->ttype[t - f->w->model->tensors];
  return k ? k : t->type;
}

static inline uint64_t fwd_bytes(const Forward *f, const GgufTensor *t)
{
  return fwd_type(f, t) == QUANT_Q5_KR ? t->bytes / QUANT_Q5_K_BYTES * QUANT_Q5_KR_BYTES_A_BLOCK : t->bytes;
}

bool fwd_fail(char *err, const char *why);
void fwd_probe(const Forward *f, const char *step, int layer, int index, void *dev, uint64_t n, bool ints);
bool fwd_put(void *dev, const void *host, uint64_t bytes);
bool fwd_put_async(void *dev, const void *pinned, uint64_t bytes);
bool fwd_copy(void *dst, const void *src, uint64_t bytes); /* device to device, on the stream */
void fwd_release_held(Forward *f);
void fwd_on_read(uint32_t unit, bool ok, void *arg); /* fetchq's done: the unit to the inbox */
bool fwd_alloc(void *dev, uint64_t bytes); /* *(void **)dev = zeroed device memory */
bool fwd_wait_read(Forward *f, char *err); /* a read's end waited for, the finished ones to the pool */
bool fwd_pinned(Forward *f, uint32_t unit, const void *dev[POOL_PARTS], char *err);
bool fwd_drain(Forward *f, char *err); /* the finished reads to the pool, no wait */
bool fwd_taps_on(Forward *f); /* the taps taken from the next step on (a drafter's): false if they cannot be held */

bool fwd_attention(Forward *f, int l, char *err);
/* a weight's product over n tokens (token t's x at x + t * xs, its y at y + t * ys; n up to FWD_MAX_STEP): mm for
   the quantized types (MM_MAX_TOKENS a launch), the matvec for F32 and BF16 - a row the same sums either way */
bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys);
bool fwd_mat2(Forward *f, const GgufTensor *t0, const void *w0, uint32_t rows0, float *y0, uint32_t ys0,
              const GgufTensor *t1, const void *w1, uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols,
              const float *x, uint32_t xs, uint32_t n, bool ready);
bool fwd_mat_ready(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x,
                   uint32_t xs, uint32_t n, float *y, uint32_t ys, bool ready);
bool fwd_engram(Forward *f, int l, int nth, char *err);
/* hip_skip.c: the decoder skip (seq_prompt) - the step's plan, its compaction past layer dec, the head, the stream */
void fwd_decoder(Forward *f);
void fwd_plan(Forward *f);
bool fwd_compact(Forward *f, char *err);
bool fwd_head(Forward *f, char *err);
bool fwd_moe(Forward *f, int l, char *err);
/* hip_guess.c: a small step's next-layer experts guessed from this layer's FFN input - the guess queued before the
   routing event (its copy beside the picks'), its streamed experts wanted after it, the picks scored against it */
bool fwd_guess_queue(Forward *f, int l);
void fwd_guess_want(Forward *f, int l);
void fwd_guess_score(Forward *f, int l, const int32_t *ids, uint32_t picks);
bool fwd_swap_layer(Forward *f, int l, char *err); /* a layer's experts queued: its streamed ones' swaps in */
/* hip_lane.c: a small step's routed experts in the matvec's order (forward_lane), into b->moe (the shared one added) */
bool fwd_lane_experts(Forward *f, int l, const int32_t *ids, uint32_t picks, char *err);
void fwd_heat(Forward *f);                          /* a step's end: its picks into the heat */

#endif
