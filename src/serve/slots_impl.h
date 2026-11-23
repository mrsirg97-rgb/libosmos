#pragma once
/* the kept conversations' insides (osmos/serve/slots.h): slots.c the store - its files, index and trims; slots_put.c
   a conversation put, found and streamed (one file before 0.16.0) */
#include <stdbool.h>
#include <stdint.h>

#include "osmos/serve/slots.h"

/* an array of entries, each in RAM (its state) or in the folder (its files), every
   one's tokens in memory for the prefix match. a few dozen at most: linear scans */

#define FNV_SEED 1469598103934665603ull
#define FNV_PRIME 1099511628211ull
#define NAME 32 /* "slot-" + 16 hex digits + an extension */

typedef struct Entry
{
  int32_t *tokens;
  uint32_t n;
  uint8_t *state; /* RAM's (NULL: in the folder) */
  uint64_t bytes;
  uint64_t used;  /* when last used (ns, the wall clock: kept in the .meta across runs) */
  char name[NAME]; /* the folder's files' stem (slot-<hash>), its tokens' hash */
} Entry;

struct Slots
{
  SlotsOptions o;
  char *dir;
  Entry *e;
  uint32_t n, cap;
};

typedef struct Meta /* a .meta's fixed part; then the tokens (int32 x n), then the FNV-1a of everything before it */
{
  uint64_t magic, identity, bytes, used;
  uint32_t n, pad;
} Meta;

/* slots.c's, for slots_put.c */
void slots_drop(Slots *s, uint32_t i);
bool slots_fail(char *err, const char *why);
uint64_t slots_fnv(const void *p, uint64_t n, uint64_t h);
bool slots_grow(Slots *s);
int64_t slots_lru(const Slots *s, bool dir, int64_t keep);
void slots_path(const Slots *s, const char *name, const char *ext, char *out, size_t cap);
uint64_t slots_stamp(void);
bool slots_to_dir(Slots *s, uint32_t i);
void slots_trim_dir(Slots *s, int64_t keep);
bool slots_trim_ram(Slots *s);
void slots_unlink_entry(const Slots *s, const char *name);
bool slots_write_meta(const Slots *s, const Entry *e);
