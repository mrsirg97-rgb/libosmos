#ifndef OSMOS_MEM_LEDGER_H
#define OSMOS_MEM_LEDGER_H
/* the ledger: the memory an engine holds, one count every family's build shares (DESIGN.md, "0.15.3: the ledger, a
   component"). an engine owns one for its life (osmos_create to osmos_destroy); each family's forward borrows it, its
   sequences through their forward. every allocation of the engine's tree goes through it - counted, and one that would
   leave under FIT_HEADROOM of the room refused before it is taken; every free given back. the room: this machine's
   (fit_room - the GPU's free memory, or the system's RAM past ram_kept, the less), read at the ledger's making and
   again at a build's start (what others took since counted), and once it stands still before a refusal is final.
   a NULL ledger counts nothing (a library caller sizing its own build). one thread's at a time: the engine's.
   HIP's allocations through it: osmos/mem/ledger_hip.h (this header host code's too) */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OsmosLedger OsmosLedger;

/* made, its room read (NULL, why said: no memory for it, or the GPU not read) */
OsmosLedger *ledger_create(uint64_t ram_kept, char err[OSMOS_ERR]);
/* made with this room, never read (a test's: a ledger's rules on any machine) */
OsmosLedger *ledger_create_room(uint64_t room, uint64_t ram_kept);
/* gone (what it still counts: its owner's leak - the engine's tests hold it at zero) */
void ledger_destroy(OsmosLedger *l);

/* the room read again, what this ledger holds added back - a build's start: what others took since counted */
void ledger_read(OsmosLedger *l);
/* what an allocation may still take past FIT_HEADROOM (UINT64_MAX: a NULL ledger); _settled: read once it stands
   still first (a decision sized to the room: experts_fit_now's) */
uint64_t ledger_left(const OsmosLedger *l);
uint64_t ledger_left_settled(OsmosLedger *l);
/* the bytes it holds now */
uint64_t ledger_taken(const OsmosLedger *l);
/* its last refusal, until an allocation after it holds ("" none): the bytes asked, the room left, the RAM kept */
const char *ledger_why(const OsmosLedger *l);
/* an allocator not HIP's (mapped host slots, HSA's pools) counted alike: reserve the bytes before (false: refused, why
   kept), then keep them with the pointer it gave - or unreserve, the allocation failed - and release the pointer when
   it is freed */
bool ledger_reserve(OsmosLedger *l, size_t bytes);
void ledger_unreserve(OsmosLedger *l, size_t bytes);
void ledger_keep(OsmosLedger *l, void *p, size_t bytes);
void ledger_release(OsmosLedger *l, void *p);

#ifdef __cplusplus
}
#endif

#endif
