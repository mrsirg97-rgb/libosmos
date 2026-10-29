#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthread.h>

#include <hip/hip_runtime_api.h>

#include "substrate/mem/fetchq.h"
#include "substrate/mem/pool.h"
#include "substrate/gpu/mm.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/model/engram.h"
#include "substrate/run/forward.h"

#define FWD_MAX_FILES 16
#define FWD_MAX_BATCH MM_MAX_TOKENS                 /* a step's tokens: the matmul's */
/* a ratio r source's pending positions a sequence: a step's and the r - 1 before them, each row's r all held */
#define FWD_PEND_ROWS(r) (FWD_MAX_BATCH + (r) - 1)
#define FWD_MAX_PAIRS (FWD_MAX_BATCH * OPS_MAX_TOPK) /* a step's (token, routed expert) pairs */
#define FWD_MAX_GROUPS 64                           /* a layer's wo_a groups */
/* a layer's routing tables at most: 3 matrix pointers, 4 offsets, a count an expert; an order and a pair a pair; 2 x 16
   slots an expert - with each table's alignment (16) */
#define FWD_UP_BYTES (FWD_MAX_PAIRS * (3 * 8 + 4 * 4 + 4 + 2 * 4 + 2 * MM_TILE * 4) + 10 * 16)
/* the experts' slots at most: each its tokens padded to whole tiles - the pairs and a tile's padding an expert */
#define FWD_MAX_SLOTS(pairs, experts) ((pairs) + (uint64_t)(experts) * (MM_TILE - 1))

/* the fetcher's finished reads, handed from its workers to the forward (the pool's one consumer) */
typedef struct Inbox
{
  pthread_mutex_t mu;
  pthread_cond_t cv;
  uint32_t *unit;
  size_t n, cap;
  bool broken; /* a done the inbox could not hold: the forward refuses from then on */
} Inbox;

/* the step's device buffers, each [FWD_MAX_BATCH][...] a token (the routed experts' [FWD_MAX_PAIRS][...] a pair) */
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
} Scratch;

/* host memory the GPU copies to and from on the stream (pinned: the copies are asynchronous). each is written only
   after the stream has passed the copy that read it last (the next routing event, or the step's end) */
typedef struct Pinned
{
  int32_t tok[FWD_MAX_BATCH]; /* the step's tokens */
  int32_t pos[FWD_MAX_BATCH]; /* and their positions */
  int32_t ids[FWD_MAX_PAIRS];     /* the routed experts, [token][slot] */
  int32_t overflow;               /* an activation past f16: the step refused */
  int32_t oa_cnt[FWD_MAX_GROUPS]; /* each wo_a group's tokens: the step's */
  uint8_t up[FWD_UP_BYTES];       /* a layer's routing tables (hip_ffn.c's carve), one copy to Scratch's up */
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
  uint32_t n_expert_units;  /* units 0 .. : layer * n_experts + expert; then the resident tensors */
  const uint8_t **tdev;     /* a model tensor's device pointer (index: its place in the Gguf), resident ones only */
  const void **oa_tab;      /* each layer's wo_a groups for mm_group: [layer][o_groups], on the device */
  uint32_t *oa_xoff, *oa_yoff; /* group k: x at k * its 4,096 (of a token's attention), y at k * o_lora */
  int eg_fd;
  Engram *hash;
  float *freqs_win, *freqs_comp;
  Scratch b;
  uint8_t *stage; /* pinned: a position's Engram rows, each Engram layer its own */
  Pinned *pin;
  hipEvent_t routed;                /* the router's picks are on the host */
  uint32_t held[FWD_MAX_PAIRS];     /* the last layer's experts: released once the stream is past them */
  uint32_t n_held;
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
  int32_t dec_from;   /* the first position whose decoder state is kept: every one from it to pos */
  int32_t last[ENGRAM_MAX_NGRAM]; /* last[0] the newest */
  int n_last;
  float *ring[WEIGHTS_MAX_LAYERS];
  float *ckv[WEIGHTS_MAX_LAYERS], *ik[WEIGHTS_MAX_LAYERS];
  float *pend_kvc[WEIGHTS_MAX_LAYERS], *pend_sc[WEIGHTS_MAX_LAYERS];
};

/* a resident tensor's bytes as the GPU reads them (pinned by the pool: the mapping's own address) */
static inline const uint8_t *fwd_dev(const Forward *f, const GgufTensor *t)
{
  return f->tdev[t - f->w->model->tensors];
}

bool fwd_fail(char *err, const char *why);
void fwd_probe(const Forward *f, const char *step, int layer, int index, void *dev, uint64_t n, bool ints);
bool fwd_put(void *dev, const void *host, uint64_t bytes);
bool fwd_put_async(void *dev, const void *pinned, uint64_t bytes);
bool fwd_copy(void *dst, const void *src, uint64_t bytes); /* device to device, on the stream */
void fwd_release_held(Forward *f);
void fwd_on_read(uint32_t unit, bool ok, void *arg); /* fetchq's done: the unit to the inbox */
bool fwd_alloc(void *dev, uint64_t bytes); /* *(void **)dev = zeroed device memory */
bool fwd_pinned(Forward *f, uint32_t unit, const void *dev[POOL_PARTS], char *err);

bool fwd_attention(Forward *f, int l, char *err);
/* a weight's product over n tokens (token t's x at x + t * xs, its y at y + t * ys): mm for the quantized types (one
   kernel for 1 .. 16 tokens), the matvec 8 tokens at a time for F32 and BF16 - a row the same sums either way */
bool fwd_mat(Forward *f, const GgufTensor *t, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys);
bool fwd_engram(Forward *f, int l, int nth, char *err);
/* hip_skip.c: the decoder skip (seq_prompt) - the step's plan, its compaction past layer dec, the head, the stream */
void fwd_decoder(Forward *f);
void fwd_plan(Forward *f);
bool fwd_compact(Forward *f, char *err);
bool fwd_head(Forward *f, char *err);
bool fwd_moe(Forward *f, int l, char *err);

#endif
