/* the engine's families (osmos/run/engine.h): one table of a family's calls, its model, engine and sequence behind
   the public shapes. hip_engine.c dispatches through it; hip_engine_<family>.c fills one */
#pragma once

#include "osmos/run/engine.h"

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
  bool (*save)(const void *seq, void *out, uint64_t bytes, char err[OSMOS_ERR]);
  bool (*load)(void *seq, const void *in, uint64_t bytes, char err[OSMOS_ERR]);
  bool (*seed)(void *eng, void *seq, int32_t from, int32_t to, char err[OSMOS_ERR]);
  int (*draft)(void *eng, void *seq, int32_t y, float conf_min, float cum_min, int32_t *drafts, float *conf,
               char err[OSMOS_ERR]);
  /* several sequences' drafts at once (oseq_draft_all), or NULL: draft one after another */
  bool (*draft_all)(void *eng, OsmosSeq *const *seqs, const int32_t *y, uint32_t n, float conf_min, float cum_min,
                    int32_t (*drafts)[OSMOS_MAX_BLOCK], float (*conf)[OSMOS_MAX_BLOCK], int *counts,
                    char err[OSMOS_ERR]);
} OsmosFamily;

/* a sequence's family half (a family's batch reads its sequences' through it) */
void *oseq_impl(const OsmosSeq *s);

extern const OsmosFamily osmos_ds4, osmos_q4x;
