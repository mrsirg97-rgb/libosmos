/* the engine's families (osmos/run/engine.h): one table of a family's calls, its model, engine and sequence behind
   the public shapes. hip_engine.c dispatches through it; hip_engine_<family>.c fills one */
#pragma once

#include "osmos/mem/ledger.h"
#include "osmos/gpu/kept.h"
#include "osmos/run/engine.h"

/* a build's step: max_tokens, or the family's own at 0 - never past the context (every family's one rule; Qwen alone
   refused a step past it, and at 0 sized its step to the whole context) */
static inline uint32_t osmos_step_tokens(const OsmosOptions *o, uint32_t family_default)
{
  const uint32_t t = o->max_tokens ? o->max_tokens : family_default;
  return t < o->max_ctx ? t : o->max_ctx;
}

typedef struct OsmosFamily
{
  const char *arch; /* general.architecture */
  void *(*open)(const OsmosFiles *files, OsmosFacts *facts, char err[OSMOS_ERR]);
  void (*close)(void *model);
  void *(*create)(const void *model, const OsmosOptions *o, OsmosLedger *ledger, char err[OSMOS_ERR]);
  void (*destroy)(void *eng);
  uint64_t (*identity)(const void *eng);
  void (*stats)(const void *eng, OsmosStats *s);
  /* a step: runs[r] (a family sequence) the next len[r] tokens, consecutive - the engine's checks made, its runs
     built (OSMOS_MAX_RUNS at most: hip_engine.c); read a prompt's (the large steps' math), where reads_apart */
  bool (*step)(void *eng, void *const *runs, const uint32_t *len, uint32_t n_runs, const int32_t *tokens, bool read,
               char err[OSMOS_ERR]);
  void (*kept)(const void *eng, KeptView *k); /* the last step's kept rows (osmos/gpu/kept.h): the engine reads and
                                                 picks every family's rows through it - 0.16.0: six copies */
  uint32_t (*block)(const void *eng);
  void *(*seq_create)(void *eng, char err[OSMOS_ERR]);
  void (*seq_destroy)(void *seq);
  bool (*seq_reset)(void *seq, char err[OSMOS_ERR]);
  bool (*prompt)(void *seq, uint32_t end);
  bool (*rollback)(void *seq, uint32_t n);
  uint32_t (*reach)(const void *seq);
  int32_t (*position)(const void *seq);
  uint64_t (*state_bytes)(const void *seq);
  bool (*save_to)(const void *seq, StateSink sink, void *ctx, char err[OSMOS_ERR]);
  bool (*load_from)(void *seq, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR]);
  bool (*seed)(void *eng, void *seq, int32_t from, int32_t to, char err[OSMOS_ERR]);
  int (*draft)(void *eng, void *seq, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
               char err[OSMOS_ERR]);
  /* several sequences' drafts at once (oseq_draft_all), or NULL: draft one after another */
  bool (*draft_all)(void *eng, OsmosSeq *const *seqs, const int32_t *y, uint32_t n, float conf_min, float cum_min,
                    int32_t (*drafts)[OSMOS_MAX_BLOCK], float (*conf)[OSMOS_MAX_BLOCK], int *counts,
                    char err[OSMOS_ERR]);
  bool reads_apart; /* a prompt's read another math than a decode's (step's read): osmos_reads_apart; false one math
                       (Qwen's, Inkling's: read ignored) */
  /* a mark (oseq_mark_save), or NULL: the whole state */
  uint64_t (*mark_bytes)(const void *seq);
  bool (*mark_save)(const void *seq, void *out, uint64_t bytes, char err[OSMOS_ERR]);
  bool (*mark_load)(void *seq, const void *in, uint64_t bytes, char err[OSMOS_ERR]);
} OsmosFamily;

/* a sequence's family half (a family's batch reads its sequences' through it) */
void *oseq_impl(const OsmosSeq *s);

/* the families' shared glue (hip_engine_runs.c). a batch's runs: osmos_runs_consecutive - every sequence's tokens
   together (false with why: "a sequence's tokens not consecutive in the batch"); osmos_runs - from token at, the
   next OSMOS_FAMILY_RUNS runs at most (each a sequence's consecutive tokens: first[r] its first, len[r] its count),
   their count in *runs, the tokens they take returned (n - at when every run fit). a family's step its own: refused
   past its runs, or split over several steps (q4x's). osmos_gpu_room: hipMemGetInfo's free bytes */
#define OSMOS_FAMILY_RUNS OSMOS_MAX_RUNS /* every family's step's sequences at most (LFM3_MAX_RUNS, Q4X_MAX_RUNS, ...) */
bool osmos_runs_consecutive(OsmosSeq *const *seqs, uint32_t n, char err[OSMOS_ERR]);
uint32_t osmos_runs(OsmosSeq *const *seqs, uint32_t n, uint32_t at, uint32_t first[OSMOS_FAMILY_RUNS],
                    uint32_t len[OSMOS_FAMILY_RUNS], uint32_t *runs);
bool osmos_gpu_room(uint64_t *bytes);

extern const OsmosFamily osmos_ds4, osmos_q4x, osmos_ink2, osmos_q35, osmos_lfm3, osmos_hy4;

