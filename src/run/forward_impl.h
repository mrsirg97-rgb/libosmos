#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "substrate/model/engram.h"
#include "substrate/run/forward.h"

#define FWD_MAX_FILES 16
#define FWD_MAX_SPANS 4096

typedef struct Span
{
  void *at;
  uint64_t bytes;
} Span;

/* the step's device buffers, one token's (the decode's) */
typedef struct Scratch
{
  float *h, *mix, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *pre0;
  float *x, *qa, *q, *kv, *o, *oa, *y, *mid, *xf;
  float *logits, *rw, *ga, *up, *hh, *ye, *moe;
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
  uint8_t *base[FWD_MAX_FILES];
  uint64_t size[FWD_MAX_FILES];
  Span span[FWD_MAX_SPANS];
  int n_span; /* collected */
  int n_reg;  /* of them registered, the first n_reg: destroy unregisters these */
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

/* a tensor's bytes as the GPU reads them (registered: the host's address) */
static inline const uint8_t *fwd_dev(const Forward *f, const GgufTensor *t) { return f->base[t->file] + t->offset; }

bool fwd_fail(char *err, const char *why);
void fwd_probe(const Forward *f, const char *step, int layer, int index, void *dev, uint64_t n, bool ints);
bool fwd_put(void *dev, const void *host, uint64_t bytes);
bool fwd_alloc(void *dev, uint64_t bytes); /* *(void **)dev = zeroed device memory */

bool fwd_attention(Forward *f, Seq *s, int l, char *err);
bool fwd_engram(Forward *f, int l, int nth, char *err);
bool fwd_moe(Forward *f, int l, char *err);

#endif
