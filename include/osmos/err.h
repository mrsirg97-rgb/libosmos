#ifndef OSMOS_ERR_H
#define OSMOS_ERR_H
/* the error shape every module shares (DESIGN.md, "0.15.3: the consolidation"): a failure is false with its cause in
   a caller's OSMOS_ERR buffer */
#include <stdbool.h>
#include <stddef.h>

#define OSMOS_ERR 256

/* the cause written, formatted - unless one is there already (the first is the closest to why: a later failure on
   the way out adds nothing over it); a message past the buffer ends "...". err may be NULL. always false */
bool osmos_fail(char err[OSMOS_ERR], const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/* "who: " before the cause, once (the same who again: nothing); no cause there: "who: failed, no cause given". err
   may be NULL */
void osmos_context(char err[OSMOS_ERR], const char *who);

#endif
