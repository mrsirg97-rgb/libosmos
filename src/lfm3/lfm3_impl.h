#ifndef OSMOS_LFM3_IMPL_H
#define OSMOS_LFM3_IMPL_H

/* the lfm3 forward's state (osmos/lfm3/forward.h), shared by hip_lfm3_make.c (the weights, buffers, sequences) and
   hip_lfm3.c (the step) */

#include "osmos/gpu/lsa_ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/lsa.h"
#include "osmos/lfm3/forward.h"
#include "osmos/q4x/dflash.h"

#define F 4ull               /* a float's bytes */
#define LFM3_CONV_SLOTS (LFM3_MAX_BACK + 1) /* a convolution's states a sequence keeps: the current, and a rollback's */

/* the drafter's buffers (q4x's Q4xDfBuf): a seed's context rows, a block's, the selector's, pinned copies out */
typedef struct Lfm3DfBuf
{
  float *th, *k, *v, *h, *x, *xc, *dyn, *q, *att, *a, *gate, *up, *hn, *logits, *hp, *val, *top_p;
  int32_t *idx, *path, *top_id, *tok, *iota;
  int32_t *pin_tok, *pin_path, *pin_idx;
  float *pin_val, *pin_p;
} Lfm3DfBuf;

struct Lfm3Forward
{
  const Lfm3Weights *w;
  void **dev; /* each tensor's device copy, by the model's index */
  uint32_t max_tokens, max_ctx, logit_rows;
  uint32_t ring; /* a sliding layer's cache rows: window + max_tokens + LFM3_MAX_BACK (DESIGN.md, step 2) */
  Q4xRope rope_full[LFM3_MAX_STAGES], rope_swa; /* the full layers' a stage's: plain, then YaRN x2, x4, ... */
  uint32_t stages, yarn_from;                    /* the stages made, YaRN's original positions */
  Lfm3Stage stage[LFM3_MAX_STAGES];              /* their table (Lfm3Options' or lfm3_stages_default) */
  uint32_t yarn_static_ctx;                      /* Lfm3Options' (0: the stages) */
  const void *head;   /* the tied head as the step reads it: head_q8 (LFM3_HEAD_Q8_0) or the embedding's table */
  uint32_t head_type;
  void *head_q8;      /* its Q8_0 copy (owned), NULL for the file's */
  /* a one-token step's joints: an attention layer's q, k, v and gate one matrix, every layer's FFN gate and up one -
     their rows concatenated at load (a row's sums its own: the same bits), each tensor's device pointer into it */
  const void *j_attn[LFM3_MAX_LAYERS], *j_ffn[LFM3_MAX_LAYERS]; /* NULL: not joined (types unlike) */
  void *joints[2 * LFM3_MAX_LAYERS];
  uint32_t n_joints;
  bool *in_joint;     /* a tensor's bytes inside a joint (not freed alone) */
  float *jo;          /* a joint's outputs (a token's) */
  void *x16;          /* a read's matrices' f16 input (MM_MAX_TOKENS rows of the widest) */
  float *fa_scratch;  /* a read's attention's (q4x_attn: the matrix cores'), q4x_attn_scratch_dim floats */
  /* latent sparse attention (Lfm3Options' lsa): a full layer's maps (device, lsa_in x LSA_DIM), the read's scratch,
     a query's index and read (device), a block's index vectors, centroids and buckets on the host (Ward's) */
  bool lsa;
  uint64_t lsa_id; /* the maps file's (fnv_file): a state's index vectors and buckets theirs (lfm3_identity - 0.11.2) */
  uint32_t lsa_in;
  float *wq[LFM3_MAX_LAYERS], *wk[LFM3_MAX_LAYERS];
  LsaScratch *lsa_s;
  float *iq;
  uint32_t *list, *list_n;
  /* a prompt's sparse reads: from position sparse_from, its queries' index vectors, lists and counts (max_tokens
     each), their scores sub_n tokens at a time (lsa_batch_scratch) */
  uint32_t sparse_from, sub_n, max_buckets;
  float *iqb, *lsa_sc;
  uint32_t *lists;
  int32_t *counts;
  float *h_ik, *h_cent;
  int32_t *h_mem;
  int32_t *overflow;  /* an activation past f16's range in a read (mm_x16's): the step refused */
  void *st;
  /* a step's buffers: max_tokens rows each */
  int32_t *tok, *pin_tok;
  float *h, *xn, *y, *bcx, *u, *qf, *kf, *vf, *qr, *kr, *g, *att, *ga, *up, *act, *attn_scratch;
  float *xf, *logits, *pin_logits; /* the kept rows' final norm and logits */
  float *hk;                        /* the kept rows' residuals, gathered */
  int32_t *slot;                    /* each row of the last step its logits' row, -1 not kept (host) */
  uint32_t rows;                    /* the last step's rows (0 after a failed one) */
  int32_t *ids;                     /* lfm3_greedy's: rows' picks (device) */
  void *argmax;                     /* ops_argmax's scratch */
  lfm3_tap_fn tap;
  void *tap_arg;
  /* a DFlash2 drafter (Lfm3Options' dflash), or none: its tensors on the device (its file's table), its taps' rows
     of every step ([token][n_taps][n_embd]), a layer's place among them (-1 none), its buffers, a sequence's ring rows */
  const Q4xDflash *df;
  void **df_dev;
  float *taps;
  int tap_of[LFM3_MAX_LAYERS];
  uint32_t df_ring;
  uint64_t steps; /* the steps run: a sequence's taps its last step's (Lfm3Seq.step_id) */
  Lfm3DfBuf d;
};

struct Lfm3Seq
{
  Lfm3Forward *f;
  int32_t pos;
  uint32_t stage, turned; /* its YaRN stage; the position its last crossing turned the keys at (a rollback's floor) */
  uint32_t back, prompt_end, cur; /* the reach; the prompt's end (0: none); the convolutions' current slot */
  bool broken;
  float *conv[LFM3_MAX_LAYERS];         /* a convolution layer's state ring (LFM3_CONV_SLOTS x (taps - 1) x D) */
  void *kc[LFM3_MAX_LAYERS], *vc[LFM3_MAX_LAYERS]; /* an attention layer's Q8_0 cache: a full layer's a row a
                                                       position, a sliding layer's a ring of f->ring rows */
  /* with LSA a full layer's index vectors (a row a position), its closed blocks' centroids and buckets; the closed
     blocks' count (every full layer's the same) */
  float *ik[LFM3_MAX_LAYERS], *cent[LFM3_MAX_LAYERS];
  int32_t *mem[LFM3_MAX_LAYERS];
  uint32_t closed;
  /* the last step's rows (its taps': positions step_pos0 .., rows step_t0 .. of step step_id), and with a drafter
     each of its layers' key and value rings (df_ring rows, position p at p % df_ring) holding df_lo .. df_pos - 1 */
  int32_t step_pos0;
  uint32_t step_n, step_t0;
  uint64_t step_id;
  float *dk[Q4X_DFLASH_MAX_LAYERS], *dv[Q4X_DFLASH_MAX_LAYERS];
  int32_t df_lo, df_pos;
};

/* a step's runs: each sequence's tokens at rows off .. off + len - 1 of the step */
typedef struct Lfm3Run
{
  Lfm3Seq *s;
  uint32_t off, len;
} Lfm3Run;

/* a head's position-free dims, LSA's index input: the zero tail's (the rotated pairs (i, i + head_dim / 2), i <
   rope_pairs; the rest the free) */
static inline LsaDims lfm3_free_dims(const Lfm3Geometry *g)
{
  const uint32_t half = (uint32_t)g->head_dim / 2, p = (uint32_t)g->rope_pairs;
  return (LsaDims){p, half, half + p, (uint32_t)g->head_dim};
}

bool lfm3_fail(char *err, const char *why);
bool lfm3_alloc(void *p, uint64_t bytes);
static inline const void *lfm3_dev(const Lfm3Forward *f, const GgufTensor *t)
{
  return f->dev[t - f->w->model->tensors];
}
static inline const float *lfm3_fv(const Lfm3Forward *f, const GgufTensor *t) { return (const float *)lfm3_dev(f, t); }

#endif
