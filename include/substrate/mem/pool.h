#ifndef POOL_H
#define POOL_H

#include <stddef.h>
#include <stdint.h>

#define POOL_ERR 256
#define POOL_PARTS 3

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

typedef struct Pinner
{
  int (*pin)(void *host, size_t bytes, void **dev, void *ctx);
  void (*unpin)(void *host, void *ctx);
  void *ctx;
} Pinner;

typedef enum PoolPin
{
  POOL_OK = 0,
  POOL_WAIT = 1,
  POOL_FULL = -1,
} PoolPin;

typedef void (*pool_post_fn)(uint32_t unit, void *arg);

typedef enum PoolEventKind
{
  POOL_EV_READ = 1,
  POOL_EV_PIN = 2,
  POOL_EV_UNPIN = 3,
  POOL_EV_REFUSED = 4,
} PoolEventKind;

typedef struct PoolEvent
{
  PoolEventKind kind;
  uint32_t unit;
  uint64_t bytes;
  uint64_t pinned;
  int64_t at_ns;
} PoolEvent;

typedef void (*pool_event_fn)(const PoolEvent *ev, void *arg);

typedef struct Pool Pool;

Pool *pool_create(const PoolUnit *units, size_t n_units, uint64_t budget, Pinner pinner, pool_post_fn post,
                  void *post_arg, pool_event_fn on_event, void *event_arg, char err[POOL_ERR]);
void pool_destroy(Pool *p);

void pool_want(Pool *p, const uint32_t *units, size_t n);
void pool_ready(Pool *p, uint32_t unit);
PoolPin pool_pin(Pool *p, uint32_t unit, const void *dev[POOL_PARTS]);
void pool_release(Pool *p, uint32_t unit);

PoolState pool_state(const Pool *p, uint32_t unit);
uint64_t pool_pinned(const Pool *p);

#endif

/*
Pool
  where the model's experts are, on one APU whose GPU reads the page cache in place (DESIGN.md, "the memory"): a unit
  is cold (on the SSD), reading (asked into the page cache) or pinned (its pages registered: the GPU reads them where
  they lie, at its memory's speed). nothing is copied. the pool knows units and file ranges, not experts or layers: the
  model builds the units, the pool decides what is pinned. nothing in it blocks: a read is the SSD's and its end comes
  back as an event. the scheduler's consumer is its only caller (libevt: one consumer, so nothing in it locks); the
  one thing that happens elsewhere - a read ending - reaches it only through `post`.

pool_create
  `units`: what can be pinned, a unit up to POOL_PARTS ranges of open, read-only files (an expert's gate, up, down);
  copied, so the caller's array may go. `budget`: the bytes the pool may hold pinned - RAM, not GTT (registrations do
  not count against ttm's limit) - the caller's plan of the machine. `pinner`: the registration, injected - HIP's
  hipHostRegister / hipHostUnregister in the binary, a fake in the tests (the seam). `post`: called from wherever a
  read ends (any thread), with the unit - the binary's posts an event to the engine (engine_add) that calls
  pool_ready on the consumer; the tests' runs it in place. `on_event`: every state change
  as a PoolEvent, in order (the event log), or NULL. refuses (NULL, the reason in `err`) a unit with no parts or more
  than POOL_PARTS, a range of 0 bytes, or a unit larger than the budget. the files stay the caller's.

pool_destroy
  unpins everything it pinned (each with the pinner, an UNPIN event each), then frees itself. NULL is nothing to do.

pool_want
  the scheduler knows what is coming (a prompt's routes, a decode's next layers): the cold ones among `units` begin to
  read into the page cache, at the SSD's speed (a READ event each); each read's end comes back through `post`. pinned
  or reading ones are left as they are. returns at once.

pool_ready
  the consumer's half of a read's end (the event `post` queued): the unit's pages are in the page cache. if a
  pool_pin waits for it, it is pinned now (~0.03 ms) and a PIN event says so - the waiter's cue to go on.

pool_pin
  the unit's parts as the GPU's pointers, in `dev` (part order), pinned until it is released. never blocks:
  - POOL_OK: pinned (it was, or its pages were in the page cache and pinning took ~0.03 ms); `dev` filled.
  - POOL_WAIT: its pages are not read yet - the read is begun (as pool_want), and a PIN event for this unit follows
    once pool_ready has run; ask again then. `dev` untouched.
  - POOL_FULL: it does not fit even with every released unit unpinned (a REFUSED event; nothing changed).
  room is made by unpinning released units, the least recently released first; a pinned unit that is not released is
  never unpinned.
  registrations cannot share a page, and a unit's ranges need not be page-aligned: the pool pins page-aligned runs and
  merges neighbours whose pages touch - an internal matter, the caller sees units.

pool_release
  the caller is done with the unit for now: it stays pinned, but may be unpinned for room. releasing one that is not
  pinned is nothing to do.

pool_state / pool_pinned
  a unit's state; the bytes pinned now (whole pages).
*/
