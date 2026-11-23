/* the error shape (osmos/err.h): one buffer size; osmos_fail keeps the first cause (a later failure on the way out
   adds nothing over it), formats, and marks a cut message; osmos_context names who failed, once; a NULL buffer is
   allowed; a context on no cause says so, loudly */
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "osmos/err.h"

static bool cut(const char *e) { return strlen(e) == OSMOS_ERR - 1 && !strcmp(e + OSMOS_ERR - 4, "..."); }

int main(void)
{
  printf("[test_err] start\n");
  char err[OSMOS_ERR] = "";
  assert(!osmos_fail(NULL, "nowhere %d", 1)); /* (no buffer: false all the same) */
  osmos_context(NULL, "inkling2");

  /* the first cause kept: the one closest to it is the one that says why */
  assert(!osmos_fail(err, "GPU memory refused (the sequence's caches: %.2f GiB)", 0.12));
  assert(!osmos_fail(err, "a sequence could not be made"));
  assert(!strcmp(err, "GPU memory refused (the sequence's caches: 0.12 GiB)"));

  /* the context once, before the cause */
  osmos_context(err, "inkling2");
  assert(!strcmp(err, "inkling2: GPU memory refused (the sequence's caches: 0.12 GiB)"));
  osmos_context(err, "inkling2");
  assert(!strcmp(err, "inkling2: GPU memory refused (the sequence's caches: 0.12 GiB)"));
  osmos_context(err, "osmos-serve"); /* another's context: before it */
  assert(!strcmp(err, "osmos-serve: inkling2: GPU memory refused (the sequence's caches: 0.12 GiB)"));

  /* a cut message marked, never silently short: by the cause's length, and by a context past the room */
  char big[2 * OSMOS_ERR];
  memset(big, 'x', sizeof big - 1), big[sizeof big - 1] = 0;
  char e2[OSMOS_ERR] = "";
  assert(!osmos_fail(e2, "%s", big) && cut(e2));
  char e3[OSMOS_ERR] = "";
  big[OSMOS_ERR - 3] = 0; /* a cause that fits alone */
  assert(!osmos_fail(e3, "%s", big) && strlen(e3) == OSMOS_ERR - 3 && !cut(e3));
  osmos_context(e3, "qwen4exp");
  assert(cut(e3) && !strncmp(e3, "qwen4exp: ", 10));

  /* a context on no cause: the failure says it gave none (a false with no why is a bug, not a quiet path) */
  char e4[OSMOS_ERR] = "";
  osmos_context(e4, "hy_v4_flash");
  assert(!strcmp(e4, "hy_v4_flash: failed, no cause given"));

  printf("  the first cause kept, a context once, a cut message marked, NULL allowed, no cause said\n");
  printf("[test_err] ok\n");
  return 0;
}
