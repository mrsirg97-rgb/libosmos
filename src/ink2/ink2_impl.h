/* Ink2Forward's and Ink2Seq's insides (osmos/ink2/forward.h): src/ink2/hip_forward.c the weights, the membrane and
   the step; src/ink2/hip_seq.c a sequence's tails, its saves and marks */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "osmos/ink2/forward.h"
#include "osmos/ink2/mtp.h"
#include "osmos/mem/host_slots.h"
#include "osmos/mem/membrane.h"

/* a conv tail's rows past its 2 keep: a step of at most these appended in place (a decode's, a drafted round's); a
   larger step's inputs read from the step's buffer, the tail keeping their last keep rows (ink2_tail_put) */
#define INK2_TAIL_STEP 64
#define INK2_STREAM_PASSES 8 /* experts()'s room and stream settled at most this many times (each smaller: a few) */

/* a layer's resident tensors on the device: its vectors as F32 (the norms, clips, convs, queries, the router's
   bias), its matrices as stored (mv_run reads their type) */
typedef struct Ink2Dev
{
  float *attn_res_q, *ffn_res_q, *attn_norm, *q_norm, *k_norm, *v_norm, *ffn_norm;
  void *q, *k, *v, *rel, *out, *gate, *sink, *router, *k_conv_proj, *v_conv_proj;
  float *q_clip, *rel_clip, *rel_proj, *router_bias;
  float *k_conv, *v_conv, *out_conv, *ffn_out_conv; /* the static convs: a full layer's k / v [taps][ch], tap 0 first */
  float *scale[3]; /* host: each expert's gate, up, down scale (1 where the file has none) */
} Ink2Dev;

/* a sequence's rows of one kind, oldest first: `held` of them at d, room for cap; the last `keep` (what a later
   token reads, and a rollback's) moved to the front when a step would pass cap (cap 2 keep + a step's: the move's
   source and target apart) */
typedef struct Ink2Tail
{
  void *d;
  uint32_t row, esize; /* elements a row; bytes an element (a conv's inputs F32, a window's keys and values f16, a
                          full layer's a key's KV heads each a record: keys INK2_KV8_RECORD, values INK2_KV5_RECORD) */
  uint32_t keep, cap, held;
} Ink2Tail;

/* the drafter's resident tensors and its step's buffers (hip_mtp.c) */
typedef struct Ink2MtpDev
{
  void *wq[INK2_MTP_MAX_LAYERS], *wk[INK2_MTP_MAX_LAYERS], *wv[INK2_MTP_MAX_LAYERS], *wkc[INK2_MTP_MAX_LAYERS],
      *wvc[INK2_MTP_MAX_LAYERS], *wo[INK2_MTP_MAX_LAYERS], *wgate[INK2_MTP_MAX_LAYERS], *wup[INK2_MTP_MAX_LAYERS],
      *wdown[INK2_MTP_MAX_LAYERS];
  float *in_norm[INK2_MTP_MAX_LAYERS], *post_norm[INK2_MTP_MAX_LAYERS], *dflare[INK2_MTP_MAX_LAYERS],
      *q_norm[INK2_MTP_MAX_LAYERS], *k_norm[INK2_MTP_MAX_LAYERS], *k_norm_ctx[INK2_MTP_MAX_LAYERS];
  float *alpha; /* [n_layers + 1][n_taps]: softmax(fuse_weight), a layer's mixture, the Markov head's last */
  float *mask, *norm, *m_dflare, *w_sp, *w_hf, *rnn_b, *fold_b;
  void *w1, *fold; /* [the head's rows][rank]: markov_w1's BF16, and lm_head . hidden_out Q8_0 */
  float *taps;     /* the step's taps [max_tokens][n_taps][n_embd] */
  float *mix, *kctx, *vctx, *x, *res, *nrm, *q, *kb, *vb, *att, *ob, *g, *u, *hf, *inv, *vec, *p, *state, *latent,
      *corr, *base, *lg, *probs, *scr, *f_rnn; /* lg: a Markov step's pick's scratch */
  int32_t *pos, *prev, *ids;
  void *x16;         /* the block's activations as mm reads them: a tile of its widest input */
  int32_t *overflow; /* an x16 value past f16's range (the draft's refusal, not the step's) */
} Ink2MtpDev;

enum
{
  INK2_CONV_K,
  INK2_CONV_V,
  INK2_CONV_OUT,
  INK2_CONV_FFN,
  INK2_CONVS
};

struct Ink2Seq
{
  Ink2Forward *f;
  int32_t pos;
  uint32_t reach, prompt_end;
  bool failed;
  Ink2Tail conv[INK2_MAX_LAYERS][INK2_CONVS]; /* each layer's short convs' inputs */
  Ink2Tail kw[INK2_MAX_LAYERS], vw[INK2_MAX_LAYERS]; /* a sliding layer's window, f16; a full layer's max_ctx rows, keys Q8, values Q5 */
  /* the drafter's: each layer's context keys and values (f16 rings of mrows: a local layer's window and a block, a
     full one's max_ctx), the last seeded position's taps (the Markov head's features), the context seeded mfrom ..
     mend - 1; the last step's taps' first row and its position */
  uint16_t *mk[INK2_MTP_MAX_LAYERS], *mv[INK2_MTP_MAX_LAYERS];
  uint32_t mrows[INK2_MTP_MAX_LAYERS];
  float *mlast;
  int32_t mfrom, mend, tap_pos0;
  uint32_t tap_row0;
  uint64_t tap_step; /* the forward's step that wrote its taps (Ink2Forward.steps; 0: none) - a seed reads its own */
};

struct Ink2Forward
{
  uint64_t steps; /* ink2_step's calls: the drafter's taps the last one's, every sequence's in it (Ink2Seq.tap_step) */
  const Ink2Weights *w;
  const Ink2Mtp *mtp; /* the drafter, or NULL */
  Ink2MtpDev md;
  uint32_t max_ctx, max_tokens, maxb, vocab;
  Ink2Dev L[INK2_MAX_LAYERS];
  void *embed, *output;
  float *embed_norm, *out_res_q, *out_norm;
  uint64_t dense_bytes, pinned_bytes;
  uint64_t fixed_bytes; /* the GPU memory the dense weights, buffers and drafter took (create's measure) */
  uint64_t *picks; /* each unit's picks so far (ink2_forward_picks) */
  /* the experts */
  int *fds;
  HostSlots *slots, *stream;
  Membrane *mem;
  /* the step's buffers (max_tokens rows) */
  int32_t *ids_d;
  float *h, *blocks, *a3, *xn, *kn, *vn, *q, *k, *v, *kc, *vc, *rel, *gate, *sink, *att, *dyn, *o, *r, *moe, *logit,
      *tmp, *res_sc, *eg, *eu, *ed, *head, *logits;
  /* the picks, their weights after them (pick + the step's pairs); the next layer's guessed (gw their weights) */
  int32_t *pick, *hpick, *gid, *gpick, *hgpick;
  float *gw;
  const float *m_a;     /* the layer's experts' weights a pick and the picks' rows (in m_dev): their sum's */
  const int32_t *m_row;
  void *x16, *xe16; /* mm's f16 activations: a dense matrix's input (MM_MAX_TOKENS rows), the experts' slots */
  int32_t *overflow; /* an activation past f16's range in the step (mm_x16's flag): the step refused */
  uint32_t head_rows; /* the head's rows a launch (f->head's) */
  float *fa_part; /* a split decode's attention units (ink2_fa_scratch) */
  /* moe's tables: the counting sort's, the entries' keys, the region built on the host and its device copy */
  uint32_t *m_at, *m_by;
  uint64_t *m_key;
  uint8_t *m_host, *m_build;
  void *m_dev;
  void *amax;
  bool *kept;
  int32_t *slot;       /* a step's row's logits slot, or -1 (none kept) */
  uint32_t logit_rows, n_slots; /* the logits' slots, and the step's taken */
  uint32_t n; /* the last step's rows */
};

bool ink2_fail(char *err, const char *why);
/* hip_forward.c's: device memory counted in the forward's bytes; a tensor's bytes from g's file, its F32 values on the
   host (F32 / F16 / BF16), uploaded; the drafter's y [n][rows] = W x (mv_run, MV_MAX_TOKENS a launch) */
bool ink2_dalloc(Ink2Forward *f, void **p, uint64_t bytes);
bool ink2_read_tensor(const Gguf *g, const GgufTensor *t, void *out);
float *ink2_host_f32(const Gguf *g, const GgufTensor *t, uint64_t *n);
bool ink2_up_host(Ink2Forward *f, const float *v, uint64_t n, float **out);
bool ink2_up_f32_of(Ink2Forward *f, const Gguf *g, const GgufTensor *t, float **out);
bool ink2_up_raw_of(Ink2Forward *f, const Gguf *g, const GgufTensor *t, void **out);
bool ink2_mv(const GgufTensor *t, const void *w, const float *x, uint32_t n, float *y);
uint32_t ink2_type(const GgufTensor *t); /* the type a resident matrix's bytes are (a Q5_K one's repacked: Q5_KR) */
/* the tails: room for n more rows (the move), the last k rows, n F32 rows appended (an f16 tail's converted; queued
   on the null stream); a tail's row r */
bool ink2_tail_room(Ink2Tail *t, uint32_t n);
const void *ink2_tail_last(const Ink2Tail *t, uint32_t k);
bool ink2_tail_put(Ink2Tail *t, const float *x, uint32_t n);
/* a tail's next n rows, held (a kernel writes them, in the tail's own format) - NULL past its room (an F32 tail's:
   ink2_tail_put's then) */
void *ink2_tail_claim(Ink2Tail *t, uint32_t n);
/* hip_mtp.c: the drafter's weights and buffers (its fold made), freed; a sequence's rings; a step's taps kept */
bool ink2_mtp_up(Ink2Forward *f, char *err);
void ink2_mtp_down(Ink2Forward *f);
bool ink2_mtp_seq_make(Ink2Seq *s);
void ink2_mtp_seq_free(Ink2Seq *s);
bool ink2_mtp_tap(Ink2Forward *f, int slot, const float *x, uint32_t n, uint32_t row0);
void *ink2_tail_row(const Ink2Tail *t, uint32_t r);
