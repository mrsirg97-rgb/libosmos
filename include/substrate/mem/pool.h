#ifndef POOL_H
#define POOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define POOL_ERR 256
#define POOL_PARTS 3

typedef enum PoolState
{
  POOL_COLD = 0,
  POOL_READING = 1,
  POOL_READ = 2,
  POOL_PINNED = 3,
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
  bool (*pin)(void *host, size_t bytes, void *ctx);
  void (*unpin)(void *host, void *ctx);
  void *ctx;
} Pinner;

typedef struct Fetcher
{
  void (*fetch)(uint32_t unit, const void *const *host, const uint64_t *bytes, int n, void *ctx);
  void *ctx;
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
  POOL_EV_READY = 2,
  POOL_EV_PIN = 3,
  POOL_EV_UNPIN = 4,
  POOL_EV_REFUSED = 5,
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

Pool *pool_create(const PoolUnit *units, size_t n_units, uint64_t budget, Pinner pinner, Fetcher fetcher,
                  pool_event_fn on_event, void *event_arg, char err[POOL_ERR]);
void pool_destroy(Pool *p);

void pool_want(Pool *p, const uint32_t *units, size_t n);
void pool_ready(Pool *p, uint32_t unit);
PoolPin pool_pin(Pool *p, uint32_t unit, const void *dev[POOL_PARTS]);
bool pool_dev(const Pool *p, uint32_t unit, const void *dev[POOL_PARTS]);
void pool_release(Pool *p, uint32_t unit);

PoolState pool_state(const Pool *p, uint32_t unit);
uint64_t pool_pinned(const Pool *p);

#endif

/*
Pool
  where the model's experts are, on one APU whose GPU reads the page cache in place at its memory's speed (DESIGN.md,
  "the memory"). a unit is cold (on the SSD), reading, read (in the page cache, not pinned: the kernel may drop it) or
  pinned (its pages registered; the GPU reads them where they lie - its pointer is the mapping's own address, which
  this APU's registration keeps). nothing is copied. the pool knows units and file ranges, not experts or layers: the
  model builds the units, the pool decides what is pinned. nothing in it blocks: a read is the fetcher's, and its end
  comes back as pool_ready. the scheduler's consumer is its only caller (libevt: one consumer, so nothing locks).

pool_create
  `units`: what can be pinned, a unit up to POOL_PARTS ranges of open, read-only files (an expert's gate, up, down);
  copied. the pool maps each file read-only, whole, and owns the mappings; the descriptors stay the caller's.
  `budget`: the bytes the pool may hold pinned (whole pages) - RAM, not GTT (registrations do not count against
  ttm's limit) - the caller's plan of the machine. `pinner`: the registration, injected (HIP's hipHostRegister /
  hipHostUnregister in the binary, which refuses unless the device's pointer is the host's; a fake in the tests).
  `fetcher`: begins reading a unit's parts into the page cache and returns at once; when the read is done, whoever
  wired it arranges pool_ready(unit) on the consumer (the binary: an engine event; the tests: by hand). `on_event`:
  every state change as a PoolEvent, in order (the event log), or NULL. refuses (NULL, the reason in `err`) a unit
  with no parts or more than POOL_PARTS, a range of 0 bytes or past its file, or a unit larger than the budget.

pool_destroy
  unpins every pinned unit (an UNPIN event each), unregisters every page, unmaps the files, frees itself. NULL is
  nothing to do.

pool_want
  the cold ones among `units` begin to read (a READ event each, then the fetcher). others are left as they are.

pool_ready
  a read is done (READY): the unit is read. if a hold waits on it, it is pinned now - a PIN event, the waiter's cue
  to take its pointers with pool_dev - or, without room, a REFUSED event: its holds stay, for their pool_release.
  a ready for a unit that is not reading is nothing to do.

pool_pin
  a hold on the unit: it stays pinned until every hold is released. never blocks:
  - POOL_OK: pinned now (it was, or it was read and pinning takes ~0.03 ms); `dev` filled, in part order.
  - POOL_WAIT: not read yet - its read is begun (as pool_want) and a PIN or REFUSED event for it follows pool_ready.
    the hold is taken; `dev` untouched.
  - POOL_FULL: it does not fit even with every unheld pinned unit unpinned - no hold, nothing changed (a REFUSED
    event).
  room: unheld pinned units are unpinned, the least recently released first; a held one never is.
  pages: registrations cannot share a page and a unit's ranges need not be page-aligned, so a unit registers only
  the pages no registration covers yet, and a registration goes only when no pinned unit touches its pages - nothing
  a unit reads is ever unregistered under it.

pool_dev
  a pinned unit's pointers (after its PIN event); false, `dev` untouched, when it is not pinned.

pool_release
  one hold ends. at none, a pinned unit stays pinned but is the newest candidate for room. a release with no hold is
  nothing to do.

pool_state / pool_pinned
  a unit's state; the bytes registered now (whole pages, each counted once).
*/
