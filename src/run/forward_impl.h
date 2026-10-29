#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthread.h>

#include "substrate/mem/fetchq.h"
#include "substrate/mem/pool.h"
#include "substrate/model/engram.h"
#include "substrate/run/forward.h"

#define FWD_MAX_FILES 16

/* the fetcher's finished reads, handed from its workers to the forward (the pool's one consumer) */
typedef struct Inbox
{
  pthread_mutex_t mu;
  pthread_cond_t cv;
  uint32_t *unit;
  size_t n, cap;
  bool broken; /* a done the inbox could not hold: the forward refuses from then on */
} Inbox;

/* the step's device buffers, one token's (the decode's) */
typedef struct Scratch
{
  float *h, *mix, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *pre0;
  float *x, *qa, *q, *kv, *o, *oa, *y, *mid, *xf;
  float *logits, *rw, *vocab, *ga, *up, *hh, *ye, *moe;
  float *sga, *sup, *shh, *sye; /* the shared expert's, beside the routed ones' [n_used][...] */
  const void **wtab;            /* the routed experts' matrices for mv_group: gates, ups, downs */
  float *lat, *ikr, *qix, *wix, *score;
  float *erow_f, *ekv;
  uint8_t *erow;
  int32_t *ids, *pos, *rpos, *ridx, *sel, *n_sel;
} Scratch;

struct Forward
{
  const Weights *w;
  const Geometry *g;
  int n_layers;
  uint32_t max_ctx;
  uint32_t slots;
  int fds[FWD_MAX_FILES];
  Pool *pool;
  Fetchq *fq;
  Inbox in;
  uint32_t n_expert_units;  /* units 0 .. : layer * n_experts + expert; then the resident tensors */
  const uint8_t **tdev;     /* a model tensor's device pointer (index: its place in the Gguf), resident ones only */
  const void **oa_tab;      /* each layer's wo_a groups for mv_group: [layer][o_groups], on the device */
  int eg_fd;
  Engram *hash;
  float *freqs_win, *freqs_comp;
  Scratch b;
  uint8_t *stage; /* pinned: a position's Engram rows */
  int64_t *rows;  /* the position's Engram rows, every Engram layer's */
  int32_t at; /* the step's position */
  forward_probe_fn probe;
  void *probe_arg;
};

struct Seq
{
  Forward *f;
  int32_t pos;
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
bool fwd_alloc(void *dev, uint64_t bytes); /* *(void **)dev = zeroed device memory */
bool fwd_pinned(Forward *f, uint32_t unit, const void *dev[POOL_PARTS], char *err);

bool fwd_attention(Forward *f, Seq *s, int l, char *err);
bool fwd_engram(Forward *f, int l, int nth, char *err);
bool fwd_moe(Forward *f, int l, char *err);

#endif
