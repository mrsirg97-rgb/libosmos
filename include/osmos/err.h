#ifndef OSMOS_ERR_H
#define OSMOS_ERR_H
/* the error shape every module shares (DESIGN.md, "0.16.0: the consolidation"): a failure is false with its cause in
   a caller's OSMOS_ERR buffer, handed down empty - the first cause in it kept, so a call given a buffer with an
   earlier failure's in it would report that one: the engine's entry points (osmos/run/engine.h) empty theirs on entry,
   and a caller of a family's own calls passes an empty one (osmos_clear) */
#include <stdbool.h>
#include <stddef.h>

#define OSMOS_ERR 256

#ifdef __cplusplus
extern "C" {
#endif

/* the cause written, formatted - unless one is there already (the first is the closest to why: a later failure on
   the way out adds nothing over it); a message past the buffer ends "...". err may be NULL. always false */
bool osmos_fail(char err[OSMOS_ERR], const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/* "who: " before the cause, once (the same who again: nothing); no cause there: "who: failed, no cause given". err
   may be NULL */
void osmos_context(char err[OSMOS_ERR], const char *who);

/* the buffer emptied for the next call (NULL allowed) */
static inline void osmos_clear(char err[OSMOS_ERR])
{
  if (err)
    err[0] = 0;
}

#ifdef __cplusplus
}
#endif

#endif
