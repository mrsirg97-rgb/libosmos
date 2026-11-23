/* the hy4 forward's and sequence's insides (hip_hy4_make.c: the forward made; hip_hy4.c: its step; hip_hy4_layer.c: a
   layer's blocks; hip_hy4_seq.c: a sequence) */
#pragma once
#include "osmos/mem/ledger_hip.h"
#include "osmos/mem/experts.h"
#include "osmos/model/moe_route.h"

#include "osmos/gpu/bufs.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/hy4/forward.h"
#include "osmos/mem/rowread.h"
#include "osmos/model/engram.h"
#include "osmos/gpu/overflow.h"

#define F sizeof(float)
#define HY4_ROW_THREADS 8   /* the Engram rows' readers (rowread's threads) */
#define HY4_MAX_HEADS 128   /* a layer's heads at most (the per-head tables' room) */
#define HY4_STATE_HISTORY 2 /* the tokens before the position a saved state keeps (a trigram's) */
#define HY4_SCORE_ROWS 256 /* the indexer's score scratch's tokens: a longer run's selection in sub-batches of these (a step
                              of 4,096 at 1M was 4.3 GB of scratch; a block a token, the same bits) */
#define HY4_DFLY_KEEP (HY4_MAX_BACK + 1) /* a step's last rows whose drafting snapshot a sequence keeps (a verify's) */

/* matrices that read the same input (a layer's normed attention or FFN input), for a one-token step: each type's
   members one matrix (their rows concatenated at load: a row's sums its own, the same bits) and one launch, every
   member's output at its offset of one buffer */
#define HY4_JOINT_MAX 6
typedef struct Hy4Joint
{
  uint32_t n_runs, total;
  struct
  {
    uint32_t type, rows, off;
    const void *w;
  } run[HY4_JOINT_MAX];
  uint32_t off[HY4_JOINT_MAX]; /* member k's output offset (members in the builder's order) */
} Hy4Joint;

struct Hy4Forward
{
  OsmosLedger *ledger; /* borrowed (Hy4Options'): every allocation of the forward, its sequences and its drafter */
  const Hy4Weights *w;
  uint32_t max_tokens, max_ctx, logit_rows, ring; /* ring: a sliding layer's rows (its window, a step, the reach) */
  bool kv8;                                       /* MLA's cache rows Q8_0 (Hy4Options' kv_f32 not set) */
  void *head8;                                    /* Hy4Options' head_q8: the head's Q8_0 copy (NULL: the file's) */
  const void **kb_tab, **vb_tab;                  /* a read's k_b / v_b: each layer's heads' matrices (mm_group's) */
  uint32_t *kb_xo, *kb_yo, *vb_xo, *vb_yo;        /* a head's x16 and y offsets (head h's slice of the rows) */
  int32_t *hcnt;                                  /* every head's token count (a chunk's) */
  uint32_t kring;    /* a sparse layer's token keys a ring of this many (a step's, a pool's and the reach: a key is
                        read only when its group completes - 0.11.0; was a row a position, 6 KB a token) */
  void **dev;                                     /* the model's tensors on the GPU, by index in its table */
  float *fn_attn[HY4_MAX_LAYERS], *fn_ffn[HY4_MAX_LAYERS], *fn_out; /* iHC's mixes, F32 (the file's, or decoded) */
  float *freqs_swa, *freqs_sparse;                                  /* the ropes' frequencies */
  /* Engram: the hash, the table's file, its rows' bytes and each Engram layer's first row; a step's offsets, rows */
  Engram *eg;
  int eg_fd;
  uint64_t eg_row_bytes, eg_first[HY4_MAX_ENGRAM];
  Rowread *eg_read;
  uint64_t *eg_off;  /* host: the step's rows' offsets, layer-major */
  uint8_t *eg_stage; /* pinned: the rows read */
  void *eg_raw;      /* device: the rows as stored */
  float *eg_rows;    /* device: the rows decoded ([layer][token][cols][dim]) */
  /* a step's buffers */
  float *h, *xn, *y, *post;            /* the streams, a sublayer's input, its output, iHC's post */
  float *hcs;                          /* iHC's dots' slices (hy4_hc_scratch) */
  void *table;                         /* MLA's per-head matvecs' table (hy4_mla_table_bytes) */
  float *attn_part;                    /* MLA's attention's partials (hy4_mla_attn_scratch) */
  /* the read math's: the f16 activations an mm reads (MM_MAX_TOKENS of the widest input), the experts' gathered
     slots, f16's overflow; the experts' read tables (mm_group's: entries, slots) */
  void *x16, *xe16;
  int32_t *overflow;
  OverflowFlag ovf; /* the flags' reset (osmos/gpu/overflow.h): owed when one failed */
  void *st;                   /* the step's stream (every launch) */
  Hy4Joint j_attn[HY4_MAX_LAYERS], j_ffn[HY4_MAX_LAYERS];
  float *jo_attn, *jo_ffn;    /* their outputs (a token's) */
  void **joints;              /* the concatenated matrices (freed whole), n_joints of them */
  uint32_t n_joints;
  bool *in_joint;             /* a tensor's device bytes inside one (not freed alone) */
  int32_t *fault;             /* a decode's routing saw a pick out of range or twice (hy4_route): the step refused */
  int32_t *pin_flags;         /* overflow and fault (one buffer on the GPU: fault = overflow + 1) read with one copy */
  uint32_t *rx, *ry, *rxd, *ryd, *h_rx, *h_ry, *h_rxd, *h_ryd;
  int32_t *rgidx, *rhidx, *h_rgidx, *h_rhidx, *rcnt, *h_rcnt;
  const void **rw, **h_rw;    /* the entries' gate, up, down (3 x entries) */
  uint64_t *rkey;             /* the entries' sort keys (count, then build order) */
  /* the routed experts behind the membrane (experts.h): units [MoE layer][expert], a MoE layer l - n_dense; their own
     O_DIRECT descriptors; every one resident (whole): their addresses' device table [MoE layer][gate, up, down][expert]
     (hy4_route's), else decode's tables the host's; the fixed bytes (GPU memory before them) and their pinned */
  Experts ex;
  int *xfds;
  bool whole;
  const void **exp_tab, **h_tab;        /* (h_tab: its host copy, a read's) */
  const void *(*h_ex)[POOL_PARTS];      /* a step's runs' experts' addresses (n_experts of them) */
  uint64_t fixed_bytes, pinned_bytes;
  /* a read's routing (moe_route): the sort's scratch (n_experts + 1), each sorted row's pick, the runs' expert /
     start / count / slot0 (4 x n_experts), the entries (moe_entries) */
  uint32_t *h_estart, *h_by, *h_run;
  MoeEntry *h_ent;
  void *h_stage;              /* the sorted tables' staging (host) */

  float *qr, *qres, *q, *kva, *gt, *qa, *olat, *ao, *out;
  float *ixk, *ixkn, *ixq, *ixw, *scores; /* the indexer (ixkn: the step's keys normed, before their rows) */
  void *ix_spread;                        /* its spread selection's scratch (hy4_ix_spread_bytes): the forward's own */
  int32_t *sel, *n_sel;
  float *emb, *ekv;                    /* Engram's projection's input and output */
  float *ga, *up, *act;                /* the MLPs */
  float *logit_r, *wtok, *rin, *xg, *eg_g, *eg_u, *eg_h, *eg_y, *wsel, *rsum, *sh;
  int32_t *idx, *gidx, *sidx, *cnt, *xrow; /* the router's picks; the experts' order and back; mv_group_tok's table */
  void **wptr;                             /* mv_group_tok's matrices (device), a table of three */
  int32_t *h_idx, *h_gidx, *h_sidx;                  /* host copies (a read's routing) */
  float *h_w, *h_wsel;                                /* the router's weights, and in the experts' order */
  int32_t *tok, *pos, *pin_tok, *pin_pos;
  float *hx, *logits, *pin_logits;
  int32_t *slot, *ids, *argmax;
  uint32_t rows;
  hy4_tap_fn tap;
  void *tap_arg;
  struct Hy4Dfly *dfly;                /* DFly2, when loaded (hip_hy4_dfly.c) */
  int dfly_slot[HY4_MAX_LAYERS];       /* a target layer's draft layer reading it, -1 none */
  GpuBufs bufs; /* every GPU and pinned buffer of the forward, recorded as made (hy4_alloc, the step's want[]) -
                   freed from the record at its destroy, the last made first */
};

struct Hy4Seq
{
  GpuBufs bufs; /* its GPU buffers, recorded as made (its forward's ledger's) */
  Hy4Forward *f;
  int32_t pos;
  uint32_t back, prompt_end;
  bool broken;
  int32_t *tokens;                                     /* host: every token so far (the n-grams' history) */
  float *cache[HY4_MAX_LAYERS];                        /* MLA's rows: a sparse layer's by position, a sliding one's ring */
  float *keys[HY4_MAX_LAYERS], *groups[HY4_MAX_LAYERS]; /* a sparse layer's token keys (a ring) and pooled groups */
  /* DFly2's snapshot of the last step's last rows (positions dpos0 .. dpos0 + dkeep - 1): each draft layer's target
     picks ([HY4_DFLY_LAYERS][HY4_DFLY_KEEP][top] and their counts) and the head's normed input (dhid_ok: kept) */
  int32_t *dsel, *dn;
  float *dhid;
  int32_t dpos0;
  uint32_t dkeep;
  bool dhid_ok[HY4_DFLY_KEEP];
};

/* a step's runs: each sequence's tokens at rows off .. off + len - 1 of the step */
typedef struct Hy4Run
{
  Hy4Seq *s;
  uint32_t off, len;
} Hy4Run;

/* a GPU buffer, or a pinned host one, recorded in b (its ledger's: the forward's, a sequence's or the drafter's
   record - each freed from it at its owner's destroy) */
bool hy4_alloc(GpuBufs *b, void *p, uint64_t bytes);
bool hy4_pinned(GpuBufs *b, void *p, uint64_t bytes);
/* y = W x for n tokens (W the tensor on the GPU, any type the matvec reads): a token's sums its own */
bool hy4_mat(Hy4Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n, float *y,
             bool read);
/* a read's two matrices on one x (t1 NULL: one) - one launch where mm takes both, x's f16 made once a run of them
   (ready: f->x16 holds x's; in and out); else each hy4_mat's. every matrix's sums its own: the same bits */
bool hy4_mat_pair(Hy4Forward *f, const GgufTensor *t0, uint32_t rows0, float *y0, const GgufTensor *t1, uint32_t rows1,
                  float *y1, const float *x, uint32_t cols, uint32_t n, bool read, bool *ready);
/* the step's Engram rows read and decoded (f->eg_rows), the runs' tokens already in their sequences' histories */
bool hy4_engram_read(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n);
/* layer l's blocks on the step's streams */
bool hy4_layer(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, int l, bool read);
/* a joint's launches over a token's x into out (its members at their offsets) */
bool hy4_joint(Hy4Forward *f, const Hy4Joint *j, uint32_t cols, const float *x, float *out);

/* y = W x for a matrix by its pointer and type (hy4_mat's math) */
bool hy4_mat_raw(Hy4Forward *f, uint32_t type, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                 float *y, bool read);
/* a BF16 tensor's values as F32 on the host (read from its file; NULL on failure), an F32 matrix [rows][cols] as Q8_0
   on the GPU (ggml's rounding) - hip_hy4_dfly.c's */
float *hy4_host_f32(const Gguf *g, const GgufTensor *t);
bool hy4_q8_up(GpuBufs *b, const float *x, uint64_t rows, uint32_t cols, void **dev);
/* DFly2's weights and buffers freed (hip_hy4_dfly.c; none loaded: nothing) */
void hy4_dfly_free(Hy4Forward *f);

/* the latent MoE (hip_hy4_moe.c): the router, the experts (the matvecs, or read: the matrix cores), the shared
   expert, the combine norm - into f->y */
bool hy4_moe(Hy4Forward *f, uint32_t n, int l, bool read);
/* the read tables' room: entries and slots for n tokens */
uint64_t hy4_read_entries(uint64_t n, const Hy4Geometry *g);
uint64_t hy4_read_slots(uint64_t n, const Hy4Geometry *g);

static inline const void *hy4_dev(const Hy4Forward *f, const GgufTensor *t)
{
  return f->dev[t - f->w->model->tensors];
}
static inline const float *hy4_fv(const Hy4Forward *f, const GgufTensor *t) { return (const float *)hy4_dev(f, t); }
