#ifndef POOL_H
#define POOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define POOL_ERR 256
#define POOL_PARTS 3
#define POOL_ALIGN 256 /* where a part starts in its unit's memory: any GPU load aligned, as the allocator's own */

typedef enum PoolState
{
  POOL_COLD = 0,
  POOL_READING = 1,
  POOL_PINNED = 2,
} PoolState;

typedef struct PoolRange
{
  int fd;
  uint64_t offset;
  uint64_t bytes;
} PoolRange;

typedef struct PoolUnit
{
  PoolRange part[POOL_PARTS];
  int n_parts;
} PoolUnit;

typedef struct Slots
{
  void *(*alloc)(uint64_t bytes, void *ctx);
  void (*free)(void *dev, void *ctx);
  void *ctx;
} Slots;

typedef struct Fetcher
{
  void (*fetch)(uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], bool urgent, void *ctx);
  void *ctx;
  void (*hurry)(uint32_t unit, void *ctx); /* NULL: none */
} Fetcher;

typedef enum PoolPin
{
  POOL_OK = 0,
  POOL_WAIT = 1,
  POOL_FULL = -1,
} PoolPin;

typedef enum PoolEventKind
{
  POOL_EV_READ = 1,
  POOL_EV_PIN = 2,
  POOL_EV_UNPIN = 3,
  POOL_EV_REFUSED = 4,
  POOL_EV_FAILED = 5,
} PoolEventKind;

typedef struct PoolEvent
{
  PoolEventKind kind;
  uint32_t unit;
  uint64_t bytes;
  uint64_t held; /* pool_bytes after it */
  int64_t at_ns;
} PoolEvent;

typedef void (*pool_event_fn)(const PoolEvent *ev, void *arg);

typedef struct Pool Pool;

Pool *pool_create(const PoolUnit *units, size_t n_units, uint64_t budget, Slots slots, Fetcher fetcher,
                  pool_event_fn on_event, void *event_arg, char err[POOL_ERR]);
void pool_destroy(Pool *p);

PoolPin pool_pin(Pool *p, uint32_t unit, const void *dev[POOL_PARTS]);
void pool_want(Pool *p, uint32_t unit, uint64_t next);
void pool_ready(Pool *p, uint32_t unit, bool ok);
bool pool_dev(const Pool *p, uint32_t unit, const void *dev[POOL_PARTS]);
void pool_release(Pool *p, uint32_t unit, uint64_t next);

PoolState pool_state(const Pool *p, uint32_t unit);
uint64_t pool_bytes(const Pool *p);

#endif

/*
Pool
  where the model's experts are: copies in GPU memory, a unit's own, read from the files straight into it (DESIGN.md,
  "the memory"). a unit is cold (in its files), reading (its memory taken, its read under way) or pinned (its memory
  holds its bytes; the GPU reads them there). the pool knows units and file ranges, not experts or layers: the model
  builds the units, the pool decides which are in GPU memory. nothing in it blocks: a read is the fetcher's, and its
  end comes back as pool_ready. the scheduler's consumer is its only caller (libevt: one consumer, so nothing locks).

pool_create
  `units`: what can be pinned, a unit up to POOL_PARTS ranges of open, read-only files (an expert's gate, up, down);
  copied. a unit's memory holds its parts back to back, each from a POOL_ALIGN boundary. `budget`: the bytes of GPU
  memory the pool may hold - its units' and the spare memory it keeps to reuse - the caller's plan of the machine.
  `slots`: the memory, injected: alloc(bytes) GPU memory the CPU can write and the GPU reads (NULL refused), free(dev)
  at once, with no wait for the GPU (hip_slots.h in the binary; malloc in the tests). `fetcher`: begins reading a
  unit's parts into `dst` (a part's place in its memory, in part order) and returns at once; when the read is done,
  whoever wired it arranges pool_ready(unit, ok) on the consumer (the binary: fetchq's workers, then the forward's
  inbox; the tests: by hand). a pin's read is `urgent`, a want's not; `hurry(unit)`, when given, is told a reading
  unit is now pinned - waited on - so a read begun ahead goes before the others ahead of it. `on_event`: every state
  change as a PoolEvent, in order (the event log), or NULL.
  refuses (NULL, the reason in `err`) a unit with no parts or more than POOL_PARTS, a range of 0 bytes or past its
  file, or a unit larger than the budget.

pool_destroy
  every unit's memory and every spare freed (an UNPIN event for each pinned unit), then itself. the caller's part: the
  GPU is past every unit, and the fetcher's workers are stopped (fetchq_destroy) - none still writes. NULL is nothing
  to do.

pool_pin
  a hold on the unit: it stays pinned until every hold is released. never blocks:
  - POOL_OK: pinned now; `dev` filled, in part order.
  - POOL_WAIT: cold or reading - its read is begun if it was cold (its memory taken now: a READ event, then the
    fetcher), and a PIN or FAILED event for it follows pool_ready. the hold is taken; `dev` untouched.
  - POOL_FULL: a cold unit with no room even with every unheld pinned unit gone - no hold, nothing changed - or
    memory the allocator refused - no hold, though room may have been made for it (a REFUSED event either way).
  room: spare memory of the unit's size is reused; else spares are freed, then unheld pinned units go (UNPIN), the
  one wanted furthest ahead first (pool_release's next), of those the least recently released, their memory reused
  when it is the size, kept spare when not; a held or reading one never goes.

pool_want
  the unit, no hold on it, is wanted at `next` (pool_release's clock): a cold one's read begins (its memory taken: a
  READ event) if room can be made from units wanted later than it - spares, the allocator's, unheld units wanted
  after `next` - and nothing happens if not; a reading or pinned one is wanted at `next` from now on (an old guess
  the caller corrects). a held unit is nothing to do: its release says when. the prefetch: the units the caller
  expects to pin soon read before it does, never in place of ones wanted sooner.

pool_ready
  a read is done. ok: the unit is pinned (PIN), the waiters' cue to take its pointers with pool_dev. not ok: its
  memory goes spare and it is cold again (FAILED); its holds stay, for their pool_release (a pin then begins the read
  again). a ready for a unit that is not reading is nothing to do.

pool_dev
  a pinned unit's pointers; false, `dev` untouched, when it is not pinned.

pool_release
  one hold ends. at none, a pinned unit stays pinned, a candidate for room: `next` is when the caller expects to want
  it again, on the caller's own clock (any order it keeps - the forward's: the layer's next visit), and the unit
  wanted furthest ahead goes first. the forward's layers come round in a fixed order, so least recently used would
  free exactly the units wanted soonest; furthest ahead is the best a cycle allows. the caller's part: the GPU is past
  the unit when its last hold ends - its memory may be written again at once. a release with no hold is nothing to do.

pool_state / pool_bytes
  a unit's state; the bytes of GPU memory held now (units' and spares').
*/
