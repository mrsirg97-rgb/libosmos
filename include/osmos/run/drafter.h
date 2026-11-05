#ifndef DRAFTER_H
#define DRAFTER_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/run/forward.h"

/* DeepSeek's DSpark (its config.json): a round's block, the token y and DRAFTER_BLOCK - 1 noise tokens after it, each
   row a draft; the noise token; the routed experts a row picks */
#define DRAFTER_BLOCK 5
#define DRAFTER_NOISE 128799
#define DRAFTER_TOPK 3

typedef struct Drafter Drafter;

Drafter *drafter_create(Forward *f, const char *dir, char err[FORWARD_ERR]);
void drafter_destroy(Drafter *d);
bool drafter_seed(Drafter *d, Seq *s, int32_t from, int32_t to, char err[FORWARD_ERR]);
/* a draft kept while each confidence is at least this and their product too (drafter_draft's conf_min / cum_min): swept
   for energy at low-power (DESIGN.md, "0.3.x: drafting for joules") - 0.1 the least J a token, 0.5 (maya's) 6% more,
   0.0 as much (the early stop gone). the bits are the trunk's whatever it is */
#define DRAFTER_CONF_MIN 0.1f
#define DRAFTER_CUM_MIN 0.1f

int drafter_draft(Drafter *d, Seq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[DRAFTER_BLOCK],
                  float conf[DRAFTER_BLOCK], char err[FORWARD_ERR]);

#endif

/*
Drafter
  DeepSeek-V4.1-Flash's DSpark drafter beside a Forward (DeepSeek's inference/model.py DSparkBlock / forward_spec, as
  maya's tools/ds4/dspark_ref.py writes it): three blocks - each the trunk's layer in small (hyper-connections, window
  attention, 128 routed experts of which a row picks DRAFTER_TOPK, a shared one) - whose window is a ring of keys made
  from the trunk's taps (the streams' mean entering its target layers, through main_proj and main_norm); a round runs
  the block [y, noise x 4] at the sequence's next positions through them, then the trunk's head and a rank-256 markov
  bias of each row's previous token give each row's draft (greedy), and a confidence head its confidence. its weights
  (dir/dspark.bin, ~4.7 GB as the packs ship it: experts IQ2_XS) in GPU memory of their own, outside the Forward's
  budget; its products the matvec's (F32 sums). drafts change no bit of the trunk's: they are only proposals, which a batched forward verifies.

drafter_create
  the drafter read from dir (dspark.json, dspark.bin) for f; f takes its taps from its next step on - at the target
  layers the model's GGUF names, else those dspark.json does. NULL with why in err for a drafter it cannot read, target
  layers neither names (or both, unequal), taps its main_proj does not take, or GPU memory refused.

drafter_destroy
  its memory (its rings' are the Seqs'). NULL is nothing to do; before its Forward.

drafter_seed
  s's ring rows for its positions from .. to - 1 of f's last step (each one's taps, keys for every block): after a
  prompt (its last 128 positions), a step, or a verify (its accepted rows only: a rejected row's would overwrite a
  position still in the window). only to - window .. to - 1 are seeded (the ring holds a window). false with why for
  a position the last step did not tap.

drafter_draft
  a round at s's next position, y the token there (not yet stepped): up to DRAFTER_BLOCK drafts, each with its
  confidence, the longest run whose confidences are each at least conf_min and whose product is at least cum_min (a
  row's head and markov step only run once its confidence passes: conf[i] depends on draft i - 1, not on draft i).
  the count, or -1 with why in err. s's own state is not touched.
*/
