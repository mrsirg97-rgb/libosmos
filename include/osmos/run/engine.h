#ifndef OSMOS_RUN_ENGINE_H
#define OSMOS_RUN_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/state_io.h"

#define OSMOS_ERR 256
#define OSMOS_MAX_BLOCK 8 /* a drafter's most drafts a round, any family's */
/* a draft kept while each confidence is at least this and their product too (oseq_draft's floors): swept for energy at
   low-power (DESIGN.md, "0.3.x: drafting for joules") - 0.1 the least J a token, 0.5 (maya's) 6% more, 0.0 as much
   (the early stop gone). the bits are the model's whatever they are */
#define OSMOS_DRAFT_CONF_MIN 0.1f
#define OSMOS_DRAFT_CUM_MIN 0.1f

typedef struct OsmosModel OsmosModel;
typedef struct Osmos Osmos;
typedef struct OsmosSeq OsmosSeq;

/* the files a model is read from: its first shard, and the family's own extras (NULL when not given) */
typedef struct OsmosFiles
{
  const char *model;
  const char *table;   /* DeepSeek's Engram table: a file of its own beside the shards */
  const char *overlay; /* a patch's tensors in place of the model's (an abliteration) */
} OsmosFiles;

/* what the server sizes itself by, before any build */
typedef struct OsmosFacts
{
  const char *family;     /* general.architecture: "deepseek41" */
  uint32_t vocab;         /* the logits' row */
  uint32_t native_ctx;    /* the model's own longest context */
  uint32_t round_tokens;  /* a decode step's most tokens: past it, a prompt's step */
  uint32_t seed_window;   /* the positions a drafter's seed reads back from a prompt's end */
  bool streams;           /* experts streamed under a budget (sized to the GPU); false: every weight resident */
} OsmosFacts;

typedef struct OsmosOptions
{
  uint32_t max_ctx;
  uint32_t max_tokens;  /* a step's most tokens: 0 the family's most */
  uint64_t budget;      /* the GPU memory streamed experts may hold: bytes, given (the server sizes it) */
  const char *picks;    /* a ranking file for the resident set, or NULL */
  bool no_guess;        /* no experts guessed a layer ahead */
  const char *drafter;  /* the drafter's directory, or NULL: no drafting */
  bool fill_set;        /* fill given; else the RAM a sizing keeps free is FIT_RAM_FREE (osmos/run/fit.h) */
  uint64_t fill;        /* the system's RAM a sizing leaves free, bytes (osmos-serve's --fill; 0: all of it the cache's) */
  bool stream_ngram;    /* Qwen's n-gram table read from its file a step at a time, not held in GPU memory (DeepSeek's
                           Engram always is) */
  const char *npu;      /* 0.7.0: the NPU's kernels' directory (forward_npu), or NULL: the GPU alone */
  float npu_share;      /* its share of a prompt step's layer's pairs: 0 the default (OSMOS_NPU_SHARE) */
} OsmosOptions;

#define OSMOS_NPU_SHARE 0.15f /* the NPU's default share: its best measured (DESIGN.md, "0.7.0") */

/* the counters a bench and the server's stats read: zeros where a family has none */
typedef struct OsmosStats
{
  uint64_t budget, cpu_picks;
  uint64_t streamed, guessed, guess_read;
  uint64_t waits[2], wait_ns[2]; /* reads waited for: [not guessed, guessed] */
} OsmosStats;

OsmosModel *osmos_open(const OsmosFiles *files, char err[OSMOS_ERR]);
void osmos_close(OsmosModel *m);
OsmosFacts osmos_facts(const OsmosModel *m);

Osmos *osmos_create(const OsmosModel *m, const OsmosOptions *o, char err[OSMOS_ERR]);
void osmos_destroy(Osmos *e);
uint64_t osmos_identity(const Osmos *e);
bool osmos_room(const Osmos *e, uint64_t *bytes);
void osmos_stats(const Osmos *e, OsmosStats *s);

bool osmos_batch(Osmos *e, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR]);
const float *osmos_logits(const Osmos *e, uint32_t row);
bool osmos_greedy(Osmos *e, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR]);
bool osmos_greedy_except(Osmos *e, uint32_t row, const int32_t *except, uint32_t n, int32_t *id, char err[OSMOS_ERR]);

OsmosSeq *oseq_create(Osmos *e, char err[OSMOS_ERR]);
void oseq_destroy(OsmosSeq *s);
bool oseq_prompt(OsmosSeq *s, uint32_t end);
bool oseq_rollback(OsmosSeq *s, uint32_t n);
uint32_t oseq_reach(const OsmosSeq *s);
int32_t oseq_position(const OsmosSeq *s);
uint64_t oseq_state_bytes(const OsmosSeq *s);
bool oseq_save(const OsmosSeq *s, void *out, uint64_t bytes, char err[OSMOS_ERR]);
bool oseq_load(OsmosSeq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR]);
bool oseq_save_to(const OsmosSeq *s, StateSink sink, void *ctx, char err[OSMOS_ERR]);
bool oseq_load_from(OsmosSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR]);

uint32_t osmos_block(const Osmos *e);
bool oseq_seed(OsmosSeq *s, int32_t from, int32_t to, char err[OSMOS_ERR]);
int oseq_draft(OsmosSeq *s, int32_t y, float conf_min, float cum_min, int32_t drafts[OSMOS_MAX_BLOCK],
               float conf[OSMOS_MAX_BLOCK], char err[OSMOS_ERR]);
bool oseq_draft_all(OsmosSeq *const *s, const int32_t *y, uint32_t n, float conf_min, float cum_min,
                    int32_t (*drafts)[OSMOS_MAX_BLOCK], float (*conf)[OSMOS_MAX_BLOCK], int *counts,
                    char err[OSMOS_ERR]);

#endif

/*
engine
  the one shape the server, the bench and the tools run a model through, whatever its family (DESIGN.md, "0.4.0: one
  engine interface"): a family is chosen by the file's general.architecture, and each call is that family's - its
  bits, its errors. a family's own headers (osmos/run/forward.h, drafter.h) stay for its tests; this header names no
  family's types.

osmos_open
  the model's files read and checked (no GPU memory taken): NULL with why for a missing file, an architecture no
  family runs (named in err), or a family's refusal of an extra it has no use for. the model outlives every engine
  built from it.

osmos_facts
  the family, its vocabulary, its native context, a decode step's most tokens, a drafter seed's window and whether
  its experts stream under a budget: the server's sizing, known before a build (a family that does not stream builds
  once, its budget 0).

osmos_create
  the forward built from the model: its GPU memory, its resident set (picks ranked, else the family's order), its
  drafter when o->drafter names one. NULL with why for a build that does not fit, or an option the family refuses.
  every engine call after it is the family's own, unchanged: the bits of the family's forward called directly.

osmos_identity
  a 64-bit key of what a saved sequence depends on (the model's files and the state's format): oseq_load refuses a
  state saved under another.

osmos_room
  the GPU memory left after the build (the server's sizing reads it); false when it cannot be read.

osmos_stats
  the build's budget and the counters a bench reads; zeros for a family without them.

osmos_batch
  one step: token i fed to seqs[i] at its next position (a sequence's tokens in order, consecutive), the logits of
  every row after it - or of its last rows only: a family may keep a step's last part (DeepSeek: the last
  round_tokens rows of its last sub-batch of MM_MAX_TOKENS - a decode round's most), always the batch's last row. rows
  are the batch's own numbers: osmos_logits(row) its vocab floats (device memory), NULL for a row not kept;
  osmos_greedy: rows row0 .. row0 + rows - 1's greatest token each, the lowest id among equals, false for a row not
  kept.

osmos_greedy_except
  row's greatest token among those not in `except` (osmos/text/answer.h's answer_best: ties to the lower id, NaNs
  never), from its logits copied to the host - osmos-serve's pick where an end is not taken (the answer rule). false
  with why: a row not kept, a GPU call failed, every token excepted.

oseq_create .. oseq_load
  a sequence: its position (tokens it holds), oseq_prompt (its prompt ends at end: the family may skip work no later
  logit reads), oseq_rollback (its last n tokens dropped: n at most oseq_reach), oseq_reach (how far back a rollback
  can go now: a cache a position indexes reaches SEQ_MAX_BACK; a recurrent state, the snapshots it holds), its saved
  state's bytes, save and load (load refuses another identity, size or a damaged state - the sequence then empty,
  position 0, never part one conversation and part another). oseq_save_to / oseq_load_from: the same bytes in
  pieces of at most STATE_CHUNK (osmos/model/state_io.h) - a state of any size through that much host memory;
  oseq_save / oseq_load are them over one buffer.

osmos_block, oseq_seed, oseq_draft, oseq_draft_all
  the drafter, when the engine has one (osmos_block 0 when not: oseq_seed and oseq_draft then refuse). oseq_seed:
  the drafter's own state from the sequence's positions from .. to (a step's rows, read after the step that ran
  them); oseq_draft: up to osmos_block drafts after token y, each kept while its confidence is at least conf_min
  and their product at least cum_min - the count, or -1 with why. drafts are proposals: a batched step verifies
  them, and they change no bit of the sequence's. oseq_draft_all: n sequences of one engine (each once), sequence i's
  after y[i] at drafts[i] / conf[i], its count counts[i] - each its oseq_draft's, bit for bit; the family's drafter on
  all of them at once where it has one (Qwen's MTP head: a draft's depth one step for those still drafting), else one
  after another. false with why.
*/
