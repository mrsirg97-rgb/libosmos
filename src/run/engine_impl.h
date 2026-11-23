/* the engine's families (osmos/run/engine.h): one table of a family's calls, its model, engine and sequence behind
   the public shapes. hip_engine.c dispatches through it; hip_engine_<family>.c fills one */
#pragma once

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
  void *(*create)(const void *model, const OsmosOptions *o, char err[OSMOS_ERR]);
  void (*destroy)(void *eng);
  uint64_t (*identity)(const void *eng);
  bool (*room)(const void *eng, uint64_t *bytes);
  void (*stats)(const void *eng, OsmosStats *s);
  bool (*batch)(void *eng, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR]);
  const float *(*logits)(const void *eng, uint32_t row);
  bool (*greedy)(void *eng, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR]);
  uint32_t (*block)(const void *eng);
  void *(*seq_create)(void *eng, char err[OSMOS_ERR]);
  void (*seq_destroy)(void *seq);
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
  /* a prompt's read (osmos_read), or NULL: the family's batch, its prompt and decode steps one math (Qwen's) */
  bool (*read)(void *eng, OsmosSeq *const *seqs, const int32_t *tokens, uint32_t n, char err[OSMOS_ERR]);
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
#define OSMOS_FAMILY_RUNS 64 /* every family's step's sequences at most (LFM3_MAX_RUNS, Q4X_MAX_RUNS, ...) */
bool osmos_runs_consecutive(OsmosSeq *const *seqs, uint32_t n, char err[OSMOS_ERR]);
uint32_t osmos_runs(OsmosSeq *const *seqs, uint32_t n, uint32_t at, uint32_t first[OSMOS_FAMILY_RUNS],
                    uint32_t len[OSMOS_FAMILY_RUNS], uint32_t *runs);
bool osmos_gpu_room(uint64_t *bytes);

extern const OsmosFamily osmos_ds4, osmos_q4x, osmos_ink2, osmos_q35, osmos_lfm3, osmos_hy4;

