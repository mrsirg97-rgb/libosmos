#ifndef OSMOS_SERVE_GEN_H
#define OSMOS_SERVE_GEN_H
/* osmos-serve's GEN line (src/serve/gen.c): GEN <max_new> [rid= spec= mark= temp= top_p= top_k= seed=] <id,id,...>
   - every number its field's whole (0.16.0: each read as its prefix - "12x" as 12, an id of -5 a huge index,
   99999999999 wrapped): digits to its end, no sign where none belongs, within its range; a float finite */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/run/engine.h"

typedef struct Gen
{
  long long max_new, rid;
  bool has_rid;
  int spec;     /* spec=0 / 1, or -1 (not given): as the server was started - drafting with a drafter (maya's) */
  int32_t *ids; /* malloc'd (gen_free) */
  uint32_t n;
  uint32_t mark;      /* mark=: a point besides the prompt's (Conv.user), 0 none */
  OsmosSampling samp; /* temp=, top_p=, top_k= (0.15.2), seed= (0.14.0): temperature 0 (not given) greedy */
} Gen;

/* line (written into: its tokens cut) parsed into g; false with why (wn bytes) - g's ids then none */
bool gen_parse(char *line, Gen *g, char *why, size_t wn);
void gen_free(Gen *g);
/* a request id (GEN's rid=, STOP's): a whole number, 0 or more */
bool gen_rid(const char *s, long long *rid);

#endif
