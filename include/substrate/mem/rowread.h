#ifndef ROWREAD_H
#define ROWREAD_H

#include <stdbool.h>
#include <stdint.h>

#define ROWREAD_MAX_THREADS 256

typedef struct Rowread Rowread;

Rowread *rowread_create(int threads);
void rowread_destroy(Rowread *r);

bool rowread(Rowread *r, int fd, const uint64_t *offset, uint32_t n, uint32_t bytes, uint8_t *dst);
bool rowread_start(Rowread *r, int fd, const uint64_t *offset, uint32_t n, uint32_t bytes, uint8_t *dst);
bool rowread_wait(Rowread *r);

#endif

/*
Rowread
  many small reads at once: Engram's rows, 144 B each, ~49,000 a 2,048-token step an Engram layer, at random places
  in a 110 GB file the page cache mostly does not hold. one at a time they are the SSD's latency each: 16,700 reads/s,
  ~5.9 s a step. measured on this box (DESIGN.md, "Engram's reads"): 64 threads 832,000 reads/s, 128 no more. the
  workers live for the reader's life; a call hands them its rows and waits for all of them.

rowread_create
  `threads` workers (1 to ROWREAD_MAX_THREADS), started now. NULL when one cannot start (those started are stopped).

rowread_destroy
  stops and joins the workers. NULL is nothing to do.

rowread
  reads row i (`bytes` at `offset[i]` of `fd`) into dst + i * bytes, for every i < n, the workers taking rows as they
  go; returns once all are read. false if any read failed or came short (the rest still read). one caller at a time
  (the forward's thread): rowread_start then rowread_wait.

rowread_start
  the same batch handed to the workers, and back at once: offset and dst must stay till rowread_wait. false, nothing
  begun, for no reader or a batch still pending (one at a time). no rows: nothing pending.

rowread_wait
  the pending batch done (the caller reads its rows too, rather than only wait); false if any of its reads failed or
  came short. none pending: true. rowread_destroy waits for a pending batch before it stops the workers.
*/
