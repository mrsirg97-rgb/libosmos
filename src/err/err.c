/* the error shape (osmos/err.h) */
#include "osmos/err.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void mark_cut(char err[OSMOS_ERR], int wanted)
{
  if (wanted >= OSMOS_ERR)
    memcpy(err + OSMOS_ERR - 4, "...", 4);
}

bool osmos_fail(char err[OSMOS_ERR], const char *fmt, ...)
{
  if (!err || err[0])
    return false;
  va_list ap;
  va_start(ap, fmt);
  mark_cut(err, vsnprintf(err, OSMOS_ERR, fmt, ap));
  va_end(ap);
  return false;
}

void osmos_context(char err[OSMOS_ERR], const char *who)
{
  if (!err || !who)
    return;
  const size_t n = strlen(who);
  if (!strncmp(err, who, n) && err[n] == ':' && err[n + 1] == ' ')
    return;
  char cause[OSMOS_ERR];
  snprintf(cause, sizeof cause, "%s", err[0] ? err : "failed, no cause given");
  mark_cut(err, snprintf(err, OSMOS_ERR, "%s: %s", who, cause));
}
