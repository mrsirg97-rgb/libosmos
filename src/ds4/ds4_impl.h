#ifndef DS4_IMPL_H
#define DS4_IMPL_H

#include "osmos/gpu/bufs.h"
#include "osmos/mem/ledger_hip.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthread.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/kvfmt.h"
#include "osmos/mem/host_slots.h"
#include "osmos/mem/membrane.h"
#include "osmos/mem/pool.h"
#include "osmos/mem/hip_slots.h"
#include "osmos/mem/resident.h"
#include "osmos/mem/rowread.h"
#include "osmos/mem/rowstream.h"
#include "osmos/cpu/lane.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/quant.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/npu_pack.h"
#include "osmos/gpu/attn_core.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/engram.h"
#include "osmos/npu/npu.h"
#include "osmos/ds4/forward.h"
#include "osmos/gpu/overflow.h"
#include "osmos/model/dspark.h"

#define FWD_MAX_FILES WEIGHTS_MAX_FILES
#define FWD_SCORE_BYTES (512ull << 20) /* the indexer's scores at most: past it a step is scored in chunks of
                                          tokens (a token's scores a row of score_stride) */
#define FWD_CKV_F32_ROWS 32768 /* a prompt's compressed rows unpacked once at most (a 128K context at
                                  ratio 4); past it the attention reads them packed */
#define FWD_MAX_BATCH MM_MAX_TOKENS                 /* a step's tokens: the matmul's */
#define FWD_MAX_SUBS FORWARD_SUBS
/* a ratio r source's pending positions a sequence: a step's and the r - 1 before them, each row's r all held */
#define FWD_PEND_ROWS(step, r) ((step) + (r) - 1)
#define FWD_MAX_STEP FORWARD_MAX_TOKENS             /* the experts' tokens: a layer's every sub-batch at once */
#define FWD_MAX_PAIRS (FWD_MAX_BATCH * OPS_MAX_TOPK) /* a sub-batch's (token, routed expert) pairs */
#define FWD_MAX_GROUPS 64                           /* a layer's wo_a groups */
#define LANE_PAIRS OPS_MAX_EXPERTS             /* a small step's pairs (the lane's): fewer than a layer's experts */
#define LANE_ROWS (LANE_PAIRS + MV_MAX_TOKENS) /* the lane's rows' buffers: an entry's rows past it read, not used */
#define FWD_GUESS_K 6 /* a token's next-layer experts guessed (hip_ds4_guess.c): the router's top this many */
/* a layer's routing tables at most for `pairs` pairs: 3 matrix pointers, 4 offsets, a count an entry (an expert, or
   a part of MM_MAX_TOKENS of one: at most pairs entries); an order and a pair a pair; 2 x 16 slots an expert; plus
   both NPU rest_entries (gu: 2 ptrs + 3 ints; dn: 1 ptr + 3 ints) - with each table's alignment (16) */
#define FWD_UP_BYTES(pairs) \
  ((uint64_t)(pairs) * (3 * 8 + 4 * 4 + 4 + 2 * 4 + 2 * MM_TILE * 4 + 3 * 8 + 6 * 4) + 18 * 16)
/* the experts' slots at most: each its tokens padded to whole tiles - the pairs and a tile's padding an expert */
#define FWD_MAX_SLOTS(pairs, experts) ((pairs) + (uint64_t)(experts) * (MM_TILE - 1))

/* a sub-batch's own: all its tokens carry from one layer to the next - swapped into the Forward and its Scratch to
   run one of its layers (hip_ds4_step.c's enter and leave) */
typedef struct Sub
{
  float *h, *f_pre;
  int32_t *pos, *sel, *n_sel, *oa_cnt, *keep, *back;
  uint32_t *cand; /* the candidate blocks' bits a token (ops_cand_blocks; Forward.cand_words), or NULL */
  bool cand_on;   /* built at this step's candidate source layer: its later index sources masked by them */
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
  uint8_t *kv_p, *lat_p, *ikr_p; /* the step's rows packed as a sequence keeps them (kvfmt.h): keys, latents, ix keys */
  float *row_f;                  /* a kept row read back for a probe, and packed again after it */
  float *ckv_f;                  /* a prompt step's compressed rows unpacked once for its attention (hip_ds4_attn.c), up
                                    to ckv_f_rows of them */
  float *ring_f;                 /* and the window's ring (the forward's slots) */
  float *att;                   /* the attention's scores, then their denominators (the latent core borrows it) */
  float *erow_f, *ekv;
  uint8_t *erow;
  int32_t *ids, *pos, *rpos, *ridx, *sel, *n_sel, *overflow, *oa_cnt;
  uint32_t *cand;  /* the running sub-batch's candidate bits (Sub's) */
  uint32_t *clist; /* a scored chunk's tokens' candidate blocks, listed (ops_cand_list: score_rows x cand_blocks) */
  uint32_t *clist_n;
  float *bmax;    /* the candidate source's blocks' highest scores (ops_cand_blocks_max: score_rows x blocks) */
  int32_t *keep, *back; /* the decoder's tokens (hip_ds4_skip.c): kept, compacted; then each back at its row */
  float *gw;            /* the next layer's guessed experts' weights (hip_ds4_guess.c: the router's, unread) */
} Scratch;

/* host memory the GPU copies to and from on the stream (pinned: the copies are asynchronous). each is written only
   after the stream has passed the copy that read it last (the next routing event, or the step's end) */
typedef struct Pinned
{
  int32_t tok[FWD_MAX_BATCH];
  int32_t pos[FWD_MAX_BATCH];
  int32_t overflow;               /* an activation past f16: the step refused */
  int32_t oa_cnt[FWD_MAX_GROUPS]; /* each wo_a group's tokens: the step's */
  int32_t rpos[WEIGHTS_MAX_LAYERS][FWD_MAX_BATCH];
  int32_t keep[FWD_MAX_BATCH], back[FWD_MAX_BATCH]; /* hip_ds4_skip.c's: the kept tokens' rows, and back */
  int32_t greedy[FWD_MAX_BATCH];                    /* forward_greedy's picks */
} Pinned;

struct Forward
{
  OsmosLedger *ledger; /* borrowed (ForwardOptions'): every allocation of the forward, its sequences and drafter */
  GpuBufs bufs;        /* every GPU and pinned buffer of the forward, recorded as made (fwd_alloc, fwd_pinned) -
                          freed from the record at its destroy, the views put back first */
  const Weights *w;
  const Geometry *g;
  int n_layers;
  uint32_t max_ctx;
  uint64_t budget;       /* the expert budget built with (forward_budget: ForwardOptions' or fitted) */
  uint32_t score_stride; /* a token's indexer scores: the most rows visible at max_ctx (the smallest ratio) */
  uint32_t score_rows;   /* the tokens scored at once: the step's, at most FWD_SCORE_BYTES of scores */
  uint32_t cand_words;   /* a token's candidate bits' words, when max_ctx can pass the candidate blocks; else 0 */
  uint32_t ckv_f_rows;   /* b.ckv_f's rows: score_stride, at most FWD_CKV_F32_ROWS */
  uint32_t slots;
  int fds[FWD_MAX_FILES];
  int dfds[FWD_MAX_FILES]; /* the same files O_DIRECT: the pool's reads, straight into host_slots */
  HostSlots *hs;           /* the membrane's memory: registered host pages an O_DIRECT read DMAs into */
  HostSlots *hs_stream;    /* the stream's */
  Membrane *mem; /* every expert's cache and a prompt's stream (membrane.h) */
  Resident *res;            /* the layers' tensors and a share of the experts, in GPU memory (resident.h) */
  HipSlots *gpu_slots;      /* res's memory (hip_slots.h): made before it, destroyed after */
  uint32_t n_expert_units;  /* the pool's units: layer * n_experts + expert */
  uint64_t *picks;          /* each unit's picks by the router so far (forward_picks) */
  PoolUnit *units;          /* each expert unit's file ranges (the CPU lane's parts) */
  const uint8_t **tdev;     /* a tensor's device pointer (index: weights_tensor_index), resident ones only */
  uint32_t *ttype;          /* its type as the GPU reads it when not the file's (QUANT_Q5_KR: repacked), else 0 */
  const void **oa_tab;      /* each layer's wo_a groups for mm_group: [layer][o_groups], on the device */
  uint32_t *oa_xoff, *oa_yoff; /* group k: x at k * its 4,096 (of a token's attention), y at k * o_lora */
  int eg_fd;
  Rowread *eg_read; /* Engram's rows, read many at once */
  uint64_t *eg_off; /* a step's rows' places in the Engram file: [sub-batch][Engram layer][token][col] */
  uint64_t eg_at[FWD_MAX_SUBS][WEIGHTS_MAX_LAYERS]; /* where each sub-batch's nth Engram layer's rows start (rows) */
  RowStream eg_rs;                                   /* the step's rows read under the GPU's work (osmos/mem/rowstream.h) */
  Engram *hash;
  float *eg_gate[WEIGHTS_MAX_LAYERS]; /* each Engram layer's gate, F32 q k on the device (ops_engram_gate) */
  uint16_t *guess_router[WEIGHTS_MAX_LAYERS]; /* an F32 router's BF16 copy for the guess alone (hip_ds4_guess.c), or NULL */
  bool no_guess;                              /* ForwardOptions': nothing guessed, no router copy made */
  float *freqs_win, *freqs_comp;
  Scratch b;
  /* DS4_PROFILE=2: a step's parts each timed to a device sync (the lane's and the GPU's work both), summed by the
     step's kind (0: a small step - a decode or a verify; 1: a prompt's) - printed at destroy */
  struct
  {
    bool on;
    int kind;
    uint64_t steps[2], tokens[2];
    double s[2][12], step_s[2];
  } pf;
  AttnCore *latent; /* the attention (the core's latent: the window's FP8 ring, the selected FP4 rows) */
  OverflowFlag ovf; /* b.overflow's reset (osmos/gpu/overflow.h): owed when one failed */
  float *xf0, *moe0, *mid0, *f_post0, *f_comb0; /* Scratch's xf .. f_comb whole (the sub-batches' views of them) */
  /* the experts' routing, a layer's every sub-batch at once: pinned, the picks [token][slot] and the tables
     (hip_ds4_ffn.c's carve, one copy to Scratch's up); on the host, the pairs' by-expert order and places */
  int32_t *r_ids;         /* mapped and coherent: the router writes the picks here (r_ids_dev) - no copy kernel */
  int32_t *r_guess;       /* mapped and coherent: the next layer's guessed experts, [token][FWD_GUESS_K] */
  int32_t *r_ids_dev, *r_guess_dev;
  uint64_t ram_kept;       /* the system's RAM a build leaves free: ForwardOptions' fill, else FIT_RAM_FREE */
  uint64_t pool_room;      /* the membrane's room: the budget's experts' bytes, taken at create */
  uint8_t *r_up;
  uint32_t *m_by, *m_row, *m_base, *m_slot0; /* a layer's routing (moe_route): sorted rows' pairs, pairs' rows, runs */
  int32_t *m_gidx, *m_hidx;                  /* its slots' tokens and rows (FWD_MAX_SLOTS), then carved */
  /* the lane's (ForwardOptions' lane): on, its CPU threads (0: every expert on the GPU, in the order the CPU
     would sum it), and a small step's tables - pinned, [gate, up, down][pair] pointers then each token's pairs in
     ascending expert id */
  /* the NPU beside the GPU (npu.h; forward_npu): a prompt step's hottest experts' gate and up, a fixed share of an
     eligible layer's pairs in whole NPU_TOKENS parts (hip_ds4_ffn.c's npu_pick) - by routing alone, so the same
     prompt the same bits; on, another forward_identity */
  Npu *npu;
  float npu_share;   /* the share of a layer's pairs the NPU takes */
  uint32_t npu_max;  /* its buffers' parts */
  int32_t *npu_tok;  /* mapped and coherent: the parts' tokens' rows of xf ([part][NPU_TOKENS]), then their first
                        rows of ga / upr / ye - written by the host, read by npu_pack_a and fwd_npu_run */
  uint32_t *npu_row;
  const void **npu_wg, **npu_wu; /* the parts' weights: gate (or the down's) and up */
  void **npu_c0, **npu_c1;       /* where their rows go: ga's and upr's, or ye's */
  hipEvent_t npu_packed;         /* the layer's parts packed: the NPU may read them */
  hipEvent_t npu_gpu_done, npu_resumed; /* timed: the stream past the GPU's own rows, and the work after the NPU's
                                           queued - between them the GPU idle on the NPU (read at the next call) */
  bool npu_timed;                        /* a pair recorded, not yet read */
  double npu_idle_ms, npu_gpu_ahead_ms;  /* the GPU's idle on the NPU, summed; and its work left when the NPU began */
  uint64_t ahead_limit; /* ForwardOptions': the membrane's bound on reads ahead in flight */
  uint32_t logit_rows;  /* ForwardOptions': a step's last rows whose logits are kept (0: every row) */
  bool lane;
  bool reading; /* forward_reading: a prompt's read - its steps the large steps' math whatever their size */
  int lane_threads;
  bool no_int8; /* the experts' int8 decode off (ForwardOptions'; the lane's CPU half its own: lane_q8_types) */
  uint8_t *lane_up;
  void *lane_dev; /* lane_up's device address (mapped: the kernels read the tables where the host wrote them) */
  float *lane_x, *lane_w; /* a small step's rows' FFN inputs and weights, gathered by expert */
  int8_t *lane_q;         /* the rows' activations to int8 (q8_round), and their blocks' scales (down's input) */
  float *lane_qd;
  int8_t *lane_xq;        /* the step's tokens' FFN inputs to int8, a row a token - once, however many experts it
                             picked; gate and up read a row's by its token (xin) - and their scales */
  float *lane_xqd;
  /* the lane's CPU half (hip_ds4_cpu_lane.c; lane_threads > 0): the experts of a small step that are not in GPU memory
     computed by a pool of threads from the model's files mapped read-only, their rows handed back in host memory the
     GPU maps, cpu_seq posted with cpu_value when they are there (the stream waits on it) */
  CpuLane *cpu;
  CpuLaneExpert cpu_e[LANE_PAIRS]; /* the job's experts (the pool reads them till it posts) */
  uint32_t cpu_unit[LANE_PAIRS];   /* and their units */
  const uint8_t *map[FWD_MAX_FILES];
  uint64_t map_len[FWD_MAX_FILES];
  float *cpu_xf, *cpu_rw;  /* pinned: a small step's FFN inputs and routing weights, copied with its picks */
  float *cpu_x, *cpu_w;    /* the job's rows (the lane's: by expert) and their weights */
  float *cpu_y, *cpu_y_dev;          /* host memory the GPU maps: the job's rows out, the lane's rows */
  uint32_t *cpu_seq, *cpu_seq_dev;   /* and the job's posted value */
  uint32_t cpu_value;
  uint64_t *on_cpu;        /* a unit's last visit computed on the CPU (the visit it is picked in: this one) */
  uint64_t cpu_picks;      /* the routed picks the CPU has computed */
  uint8_t *stage; /* pinned: the Engram rows, [sub-batch][Engram layer][token]: a sub-batch's copy may still be
                     queued when the next one's rows are read (a layer's sub-batches run to its experts unwaited) */
  int n_eg;       /* the Engram layers (stage's) */
  /* DSpark's taps (fwd_taps_on, the drafter's): the streams' mean entering each target layer, [row][tap][D], for the
     layer's last FWD_MAX_BATCH rows of the step, whichever sub-batches hold them (a sequence's last ones: the
     drafter's ring holds 128); their positions and sequences. NULL: none taken. tapped: the target layers the drafter
     gave; tap_rows: the layer's rows, every sub-batch's (the cut's) */
  void *argmax_part; /* forward_greedy's slices' bests (ops_argmax_scratch of a step's rows) */
  float *taps;
  int n_tap, tap_of[WEIGHTS_MAX_LAYERS];
  struct Drafter *drafter; /* ForwardOptions' drafter: the forward's own, destroyed first in forward_destroy */
  bool tapped[WEIGHTS_MAX_LAYERS];
  uint32_t tap_n, tap_rows;
  int32_t tap_pos[FWD_MAX_BATCH];
  Seq *tap_bs[FWD_MAX_BATCH];
  uint32_t cur;   /* the running sub-batch (enter's) */
  Pinned *pin;
  hipEvent_t routed;                /* the router's picks are on the host */
  hipEvent_t passed;                /* fwd_wait's: the stream past the work queued so far */
  uint32_t step_n;                  /* the step's tokens (forward_batch's n): a wait's spin or sleep (hip_ds4_wait.c) */
  Sub sub[FWD_MAX_SUBS];            /* the step's sub-batches; the Forward's own fields the one running */
  uint32_t n_subs, max_tokens;      /* the most sub-batches a step takes (ForwardOptions' max_tokens), and its tokens */
  uint32_t step;                    /* a sub-batch's most tokens: FWD_MAX_BATCH, or max_tokens (to 16s) below it - every
                                       per-token device buffer's rows */
  uint32_t n_sub;
  bool xf_x16; /* the layer's FFN input's x16 in b.x16 (layer_in's mix wrote it: a one-sub-batch step) */
  bool xf_q8;  /* and its int8 in lane_xq / lane_xqd (the same launch: the lane's q8_round of it skipped) */
  bool hcp_next; /* the next layer's attention hc_dot partial in b.hcp (layer_out's ops_hc_post_dot wrote it) */
  int64_t *rows;  /* each token's Engram rows, every Engram layer's: [token][layer][col] */
  uint32_t n;
  Seq *bs[FWD_MAX_BATCH];        /* token i's sequence */
  int32_t at;                    /* the step's first position (the probes' index) */
  int dec;                       /* the decoder's first layer: its KV the only one past it (-1: none) */
  uint32_t reach;                /* how far back the decoder past dec reads: (layers past dec) (window - 1) + 1 */
  uint32_t n_all, n_keep;        /* the step's tokens before the decoder's compaction, and those it keeps */
  int32_t row[FWD_MAX_BATCH];    /* a kept token's row in the step (compacted row i: row[i]) */
  bool want[FWD_MAX_BATCH];      /* a token whose logits the caller may read */
  uint32_t vocab_rows;           /* the last step's logits rows (its head's; 0 when it ran none) */
  /* host, [max_tokens]: a batch row's slot in b.vocab, or -1 - the rows its head computed, not a step's last
     vocab_rows (osmos/gpu/kept.h) */
  int32_t *slot;
  forward_probe_fn probe;
  void *probe_arg;
};

struct Seq
{
  Forward *f;
  GpuBufs bufs; /* its GPU buffers, recorded as made (its forward's ledger's) */
  int32_t pos;
  int32_t prompt_end; /* seq_prompt's end (0: none) */
  int32_t staged;     /* its tokens placed so far in the step being placed */
  int32_t dec_from;   /* the first position whose decoder state is kept: every one from it to pos */
  int32_t last[ENGRAM_MAX_NGRAM + SEQ_MAX_BACK]; /* last[0] the newest: the n-gram's, and seq_rollback's depth */
  int n_last;
  /* the furthest position its steps wrote (one past it): hi - pos the positions a rollback left behind, still in the
     window's ring and the compressors' pending rows - seq_rollback's depth since then, so rollbacks back to back count
     whole */
  int32_t hi;
  bool broken; /* a step that failed once placed: last[] its tokens, pos not - refused until reset or loaded */
  uint8_t *ring[WEIGHTS_MAX_LAYERS];                             /* KV_FP8 rows (kvfmt.h) */
  uint8_t *ckv[WEIGHTS_MAX_LAYERS], *ik[WEIGHTS_MAX_LAYERS];     /* KV_FP4_E4M3, KV_FP4_POW2 rows */
  float *pend_kvc[WEIGHTS_MAX_LAYERS], *pend_sc[WEIGHTS_MAX_LAYERS];
  uint8_t *dring; /* a drafter's rings, [block][window] KV_FP8 rows - made with the sequence */
  bool dseeded;   /* dring holds this sequence's context (drafter_seed's, or a state's): a reset or a load without
                     rings leaves the buffer, not its rows - a draft never reads another conversation's */
};

/* a small step's routed experts the lane's (int8, the matvec's order): a step of `picks` under n_experts, the lane
   set, and not a prompt's read - which takes the large steps' (mm's) math at any size, so a prompt's bits are its
   own however it is split and wherever its read began */
static inline bool fwd_small(const Forward *f, uint32_t picks)
{
  return f->lane && !f->reading && picks < (uint32_t)f->g->n_experts;
}

/* a resident tensor's bytes as the GPU reads them (its copy in the resident set) */
static inline const uint8_t *fwd_dev(const Forward *f, const GgufTensor *t)
{
  return f->tdev[weights_tensor_index(f->w, t)];
}

/* the type the kernels read a resident tensor as (the file's, or QUANT_Q5_KR once repacked), and its bytes so */
static inline uint32_t fwd_type(const Forward *f, const GgufTensor *t)
{
  const uint32_t k = f->ttype[weights_tensor_index(f->w, t)];
  return k ? k : t->type;
}

static inline uint64_t fwd_bytes(const Forward *f, const GgufTensor *t)
{
  return fwd_type(f, t) == QUANT_Q5_KR ? t->bytes / QUANT_Q5_K_BYTES * QUANT_Q5_KR_BYTES_A_BLOCK : t->bytes;
}

void fwd_probe(const Forward *f, const char *step, int layer, int index, void *dev, uint64_t n, bool ints);
bool fwd_put(void *dev, const void *host, uint64_t bytes);
bool fwd_put_async(void *dev, const void *pinned, uint64_t bytes);
bool fwd_copy(void *dst, const void *src, uint64_t bytes); /* device to device, on the stream */
/* *(void **)dev = zeroed device memory, recorded in b (its owner's: the forward's, a sequence's, the drafter's - b's
   ledger the forward's), freed from it at the owner's destroy; fwd_pinned / fwd_mapped pinned host memory alike */
bool fwd_alloc(GpuBufs *b, void *dev, uint64_t bytes);
bool fwd_pinned(GpuBufs *b, void *host, uint64_t bytes);
bool fwd_mapped(GpuBufs *b, void *host, uint64_t bytes);
bool fwd_wait(Forward *f, hipEvent_t ev); /* the GPU past `ev`: spun on in a decode-sized step, slept on in a prompt's */
bool fwd_settle(Forward *f);              /* the GPU past everything queued (fwd_wait on `passed`) */
bool fwd_settle_spin(Forward *f);         /* short-wait spin settle (drafter rounds after large prompt steps) */
/* the taps at target layers `at` taken from the next step on (a drafter's): false if none, or they cannot be held,
   or taps are on already at other layers */
/* the taps planned (the target layers at[]: f->tapped, tap_of, n_tap), before the step's buffers are made (taps' with
   them) - ForwardOptions' drafter's */
void fwd_taps_plan(Forward *f, const bool *at);
/* hip_ds4_lane.c, hip_ds4_npu.c, hip_ds4_draft.c: what forward_create_with makes last (ForwardOptions' lane, npu, drafter) */
bool fwd_lane_make(Forward *f, int cpu_threads, const int *cpu_cores, char *err);
/* whether layer l's routed experts take the int8 path: with the lane's CPU half, the types both sides decode
   (lane_q8_types: the CPU's bits the GPU's); without it, every type the GPU's lane takes (q8_takes), unless OSMOS_Q8 0 */
bool fwd_lane_q8(const Forward *f, int l);
bool fwd_npu_make(Forward *f, const char *dir, float share, char *err);
/* the drafter's plan, read before the forward's GPU memory (its index, its target layers checked: at[]), and the
   drafter made from it at the forward's end (ix its from then on); destroyed with the forward */
bool fwd_drafter_plan(const Forward *f, const char *dir, DsparkIndex **ix, bool *at, char *err);
struct Drafter *fwd_drafter_make(Forward *f, DsparkIndex *ix, char *err);
void fwd_drafter_destroy(struct Drafter *d);
uint64_t fwd_drafter_bytes(const struct Drafter *d); /* what it holds (its buffers, its resident weights): 0 for none */

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
/* every Engram row a step reads, begun at its start (the forward's thread goes on); fwd_engram waits for them */
bool fwd_engram_issue(Forward *f, char *err);
/* hip_ds4_skip.c: the decoder skip (seq_prompt) - the step's plan, its compaction past layer dec, the head, the stream */
void fwd_decoder(Forward *f);
void fwd_plan(Forward *f);
bool fwd_compact(Forward *f, char *err);
bool fwd_head(Forward *f, char *err);
bool fwd_moe(Forward *f, int l, char *err);
/* hip_ds4_guess.c: a small step's next-layer experts guessed from this layer's FFN input - the guess queued before the
   routing event (its copy beside the picks'), its streamed experts wanted after it, the picks scored against it */
bool fwd_guess_queue(Forward *f, int l);
bool fwd_prompt_ahead(const Forward *f, int l); /* a prompt's step: its read ahead the guess's union */
void fwd_guess_want(Forward *f, int l);
/* hip_ds4_lane.c: a small step's routed experts in the matvec's order (the lane's), into b->moe (the shared one added) */
bool fwd_lane_experts(Forward *f, int l, const int32_t *ids, uint32_t picks, char *err);
/* hip_ds4_cpu_lane.c: the CPU half's pool, maps and buffers (the lane's CPU half), their end; whether a small step's unit goes to
   the CPU (not in GPU memory: neither resident nor read into the pool); a unit's parts in the mapped files */
bool fwd_cpu_create(Forward *f, int threads, const int *cores, char *err);
void fwd_cpu_destroy(Forward *f);
/* the NPU's (hip_ds4_npu.c) */
void fwd_npu_destroy(Forward *f);
uint32_t fwd_npu_layer(const Forward *f, int l, uint32_t pairs);
uint32_t fwd_npu_pick(const Forward *f, uint32_t pairs, const int32_t *cnt, uint32_t nd, uint32_t *nq);
bool fwd_npu_pack(Forward *f, NpuKind kind, uint32_t parts, const uint32_t *nq, uint32_t nd, const uint32_t *base,
                  const uint32_t *slot0, const int32_t *gidx, const void *const *w0, const void *const *w1);
bool fwd_npu_run(Forward *f, NpuKind kind, int l, uint32_t parts, char *err);
bool fwd_cpu_takes(const Forward *f, uint32_t unit, uint32_t picks);
bool fwd_cpu_parts(const Forward *f, uint32_t unit, const void *part[POOL_PARTS]);
/* the layer's CPU experts (cpu_e, cpu_unit: nc of them, their rows among the step's picks rows by expert, row r's
   token tok[r] and pair pair[r]) given to the pool, cpu_value its post */
bool fwd_cpu_submit(Forward *f, int l, uint32_t picks, uint32_t nc, const int32_t *tok, const int32_t *pair,
                    char *err);


/* hip_ds4_resident.c's, for hip_ds4_make.c: the dense part, the experts' pool, the tables, Engram, the gates and the
   guess's routers made resident; the build's bytes so far */
bool fwd_dense(Forward *f, char *err);
int fwd_engram_fd(const Forward *f, const Weights *w);
size_t fwd_engram_layers(const Weights *w, int n_layers);
bool fwd_experts(Forward *f, uint64_t budget, const uint64_t *picks, char *err);
bool fwd_gates(Forward *f, const Weights *w, char *err);
bool fwd_guess_routers(Forward *f, const Weights *w, char *err);
bool fwd_oa_tables(Forward *f, char *err);
/* hip_ds4_make.c's, for the experts' pool (hip_ds4_resident.c): the step buffers' sub-views, made again */
bool fwd_subs(Forward *f);
/* DS4_PROFILE: the part just queued, to a device sync, into its sum (its kind's) */
#include <time.h>
static inline double fwd_now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static inline bool fwd_part(Forward *f, int k, double *t)
{
  if (!f->pf.on)
    return true;
  const bool ok = hipDeviceSynchronize() == hipSuccess;
  const double now = fwd_now();
  f->pf.s[f->pf.kind][k] += now - *t, *t = now;
  return ok;
}

#endif
