#ifndef SLOTS_H
#define SLOTS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/state_io.h"

#define SLOTS_ERR 256
#define SLOTS_META_MAGIC 0x31544f4c53425553ull /* "SUBSLOT1" (little-endian): a .meta's first bytes */

typedef struct Slots Slots;

typedef struct SlotsOptions
{
  const char *dir;     /* the folder past RAM (NULL: RAM alone, nothing kept past the process) */
  uint32_t cap;        /* conversations in RAM; past it the least recently used goes to the folder */
  uint64_t ram_bytes;  /* their states' bytes in RAM at most, likewise (0: none in RAM); one past it alone straight
                          to the folder (with none, not kept) */
  double gb;           /* the folder's .bin bytes at most, in GiB (<= 0: no cap) */
  uint32_t min_tokens; /* a conversation shorter is not kept */
  double days;         /* a folder's conversation unused longer is removed at start (<= 0: never) */
  uint64_t identity;   /* the model and build a state is for (forward_identity): another's is removed at start */
} SlotsOptions;

Slots *slots_create(const SlotsOptions *o, char err[SLOTS_ERR]);
void slots_destroy(Slots *s);
/* room in RAM for a state of `bytes` about to be saved (and put): RAM's least recently used to the folder (or dropped
   without one) until it fits under ram_bytes; false past ram_bytes alone (nothing moved), or a write that failed */
bool slots_room(Slots *s, uint64_t bytes, char err[SLOTS_ERR]);
/* state: malloc'd, the store's from the call (kept, or freed: not kept or a failure) */
bool slots_put(Slots *s, const int32_t *tokens, uint32_t n, void *state, uint64_t bytes, bool *kept,
               char err[SLOTS_ERR]);
/* a state given in pieces to a sink (a sequence's save: oseq_save_to), or taken from a source (its load) */
typedef bool (*SlotsSave)(void *ctx, StateSink sink, void *sink_ctx);
typedef bool (*SlotsLoad)(void *ctx, uint64_t bytes, StateSource src, void *src_ctx);
/* a conversation straight into the folder from its save - never whole in RAM (STATE_CHUNK at a time): its bytes
   `bytes`, written under a temporary name then renamed, its .meta after. not kept (true, *kept false) with no folder
   or under min_tokens; false for a save or a write that failed (nothing left of it) */
bool slots_put_stream(Slots *s, const int32_t *tokens, uint32_t n, uint64_t bytes, SlotsSave save, void *ctx,
                      bool *kept, char err[SLOTS_ERR]);
int64_t slots_find(const Slots *s, const int32_t *prompt, uint32_t n, uint32_t beyond, uint32_t *len);
/* conversation id given to load in pieces of at most STATE_CHUNK (the folder's read from its file, RAM's from
   memory), then gone from the store (its files removed) whether or not load took it */
bool slots_take_stream(Slots *s, int64_t id, SlotsLoad load, void *ctx, char err[SLOTS_ERR]);
bool slots_flush(Slots *s, char err[SLOTS_ERR]);
uint32_t slots_in_ram(const Slots *s);
uint64_t slots_ram_bytes(const Slots *s);
uint32_t slots_in_dir(const Slots *s);
uint64_t slots_dir_bytes(const Slots *s);

#endif

/*
Slots
  conversations set aside (maya #356's, the operator's): each a sequence's state (seq_save's bytes, opaque here) and
  the tokens it holds, kept so a request that continues one takes it back and reads only the rest of its prompt.
  RAM first; past `cap` the least recently used is written to the folder and dropped from RAM; the folder's .bin bytes
  kept under `gb` (its least recently used removed - never the one just written, so one larger than the cap stays).
  a kept conversation is two files, slot-<hash>.bin (the state) and slot-<hash>.meta (SLOTS_META_MAGIC, the identity,
  the state's bytes, when it was last used, the tokens, a checksum), each written under a temporary name and renamed.
  one thread (the server's loop): no locks.

slots_create
  the store, the folder (if any) created and indexed: a .meta that is damaged, of another identity or magic, older
  than `days`, or whose .bin is missing or not its size is removed with its .bin; a .bin without a .meta too. NULL,
  the reason in err, for no options, cap 0 or a folder that cannot be made or read.

slots_destroy
  RAM's conversations freed - not written (slots_flush first to keep them); the folder's stay. NULL is nothing.

slots_put
  a conversation set aside: its n tokens and its state (both copied). n under min_tokens: not kept (*kept false,
  true returned). a kept conversation with the same tokens is replaced. then RAM past cap: the least recently used
  to the folder (without a folder, dropped); the folder past gb: its least recently used removed, not the one just
  written. false for a write that failed (the conversation then not kept).

slots_find
  the kept conversation (RAM or folder) whose tokens are a prefix of prompt[0 .. n) and longer than `beyond` (what
  the caller holds already), the longest; its id and its length in *len. -1 when none.

slots_take
  the conversation slots_find named, out of the store: its state in *state (malloc'd: the caller frees it) and its
  bytes; a folder's read back and its files removed. false for an id not in the store or a .bin that cannot be read
  (the conversation then dropped).

slots_flush
  every conversation in RAM to the folder (the end of a run), then the folder under gb. false without a folder or for
  a write that failed.

slots_in_ram / slots_in_dir / slots_dir_bytes
  the conversations in RAM, in the folder, and the folder's .bin bytes.
*/
