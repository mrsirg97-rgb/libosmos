/* the hy4 forward's and sequence's insides (hip_hy4_make.c: the forward made; hip_hy4.c: its step; hip_hy4_layer.c: a
   layer's blocks; hip_hy4_seq.c: a sequence) */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "osmos/hy4/forward.h"
#include "osmos/mem/rowread.h"
#include "osmos/model/engram.h"

#define F sizeof(float)
#define HY4_ROW_THREADS 8   /* the Engram rows' readers (rowread's threads) */
#define HY4_MAX_HEADS 128   /* a layer's heads at most (the per-head tables' room) */
#define HY4_STATE_HISTORY 2 /* the tokens before the position a saved state keeps (a trigram's) */

struct Hy4Forward
{
  const Hy4Weights *w;
  uint32_t max_tokens, max_ctx, logit_rows, ring; /* ring: a sliding layer's rows (its window, a step, the reach) */
  void **dev;                                     /* the model's tensors on the GPU, by index in its table */
  float *fn_attn[HY4_MAX_LAYERS], *fn_ffn[HY4_MAX_LAYERS], *fn_out; /* iHC's mixes, F32 (the file's Q8_0) */
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
  int32_t *fault;             /* a decode's routing saw a pick out of range or twice (hy4_route): the step refused */
  uint32_t *rx, *ry, *rxd, *ryd, *h_rx, *h_ry, *h_rxd, *h_ryd;
  int32_t *rgidx, *rhidx, *h_rgidx, *h_rhidx, *rcnt, *h_rcnt;
  const void **rw, **h_rw;    /* the entries' gate, up, down (3 x entries) */
  uint64_t *rkey;             /* the entries' sort keys (count, then build order) */
  void *h_stage;              /* the sorted tables' staging (host) */

  float *qr, *qres, *q, *kva, *gt, *qa, *olat, *ao, *out;
  float *ixk, *ixq, *ixw, *scores;     /* the indexer */
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
};

struct Hy4Seq
{
  Hy4Forward *f;
  int32_t pos;
  uint32_t back, prompt_end;
  bool broken;
  int32_t *tokens;                                     /* host: every token so far (the n-grams' history) */
  float *cache[HY4_MAX_LAYERS];                        /* MLA's rows: a sparse layer's by position, a sliding one's ring */
  float *keys[HY4_MAX_LAYERS], *groups[HY4_MAX_LAYERS]; /* a sparse layer's token keys and pooled groups */
};

/* a step's runs: each sequence's tokens at rows off .. off + len - 1 of the step */
typedef struct Hy4Run
{
  Hy4Seq *s;
  uint32_t off, len;
} Hy4Run;

bool hy4_fail(char *err, const char *why);
bool hy4_alloc(void *p, uint64_t bytes);
/* y = W x for n tokens (W the tensor on the GPU, any type the matvec reads): a token's sums its own */
bool hy4_mat(Hy4Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n, float *y,
             bool read);
/* the step's Engram rows read and decoded (f->eg_rows), the runs' tokens already in their sequences' histories */
bool hy4_engram_read(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n);
/* layer l's blocks on the step's streams */
bool hy4_layer(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, int l, bool read);
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
