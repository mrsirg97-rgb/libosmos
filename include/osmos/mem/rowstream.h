#ifndef ROWSTREAM_H
#define ROWSTREAM_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/rowread.h"

/* a step's rows from a table in a file (Engram's, Qwen's n-gram stream), read on rowread's workers while the GPU works:
   issued at the step's start, waited for once where a layer first needs them. the caller owns its parts (the readers,
   the file, the offsets, the pinned stage) and fills `off` before an issue */
typedef struct RowStream
{
  Rowread *read;
  int fd;
  uint64_t *off;  /* the step's rows' places in the file, filled by the caller */
  uint8_t *stage; /* pinned: row i at stage + i * row_bytes */
  uint64_t row_bytes;
  uint64_t at;    /* the batch issued: rows at .. at + n */
  uint32_t n;
  bool pending;   /* issued, not yet waited for */
} RowStream;

bool rowstream_issue(RowStream *s, uint64_t at, uint32_t n, uint64_t row_bytes);
bool rowstream_wait(RowStream *s);
const uint8_t *rowstream_rows(const RowStream *s, uint64_t at);

#endif

/*
RowStream
  the one way a forward streams a step's table rows (DeepSeek's Engram, Hy4's Engram, Qwen's stream_ngram): the
  step's every row begun at once on the readers, so they read under the GPU's work on the layers before the first one
  that needs them; that layer waits, the rest take their rows from the stage.

rowstream_issue
  rows at .. at + n of row_bytes (off[at ..], the caller's) read into the stage at row at, begun and back at once - a
  step's every row in one batch (at 0), or a layer's at a time, each issued when the one before it is waited for (Hy4:
  its first Engram layer early, so it waits on its own rows alone). a batch still pending (a step that failed or
  stopped before its layer) is waited for first. n 0: nothing pending. false if the readers refused it (nothing
  pending then).

rowstream_wait
  the issued batch done, once: later calls (the step's later layers) return at once. false if a row could not be read.

rowstream_rows
  the stage at row `at`.
*/
