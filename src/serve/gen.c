/* osmos-serve's GEN line (osmos/serve/gen.h) */
#include "osmos/serve/gen.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* s a whole unsigned number at most max: digits alone to its end (no sign, no space, no trailing byte) */
static bool whole(const char *s, unsigned long long max, unsigned long long *v)
{
  if (*s < '0' || *s > '9')
    return false;
  char *end;
  errno = 0;
  const unsigned long long x = strtoull(s, &end, 10);
  if (errno || *end || x > max)
    return false;
  *v = x;
  return true;
}

/* s a whole finite float */
static bool real(const char *s, float *v)
{
  char *end;
  errno = 0;
  const float x = strtof(s, &end);
  if (end == s || *end || errno || !isfinite(x))
    return false;
  *v = x;
  return true;
}

bool gen_rid(const char *s, long long *rid)
{
  unsigned long long v;
  if (!whole(s, (unsigned long long)INT64_MAX, &v))
    return false;
  *rid = (long long)v;
  return true;
}

void gen_free(Gen *g)
{
  free(g->ids);
  g->ids = NULL, g->n = 0;
}

bool gen_parse(char *line, Gen *g, char *why, size_t wn)
{
  memset(g, 0, sizeof *g);
  g->spec = -1;
  g->samp.top_p = 1;
  char *save = NULL, *tok = strtok_r(line + 4, " ", &save);
  unsigned long long v;
  if (!tok)
    return snprintf(why, wn, "expected: GEN <max_new> [keys] <id,id,...>"), false;
  if (!whole(tok, (unsigned long long)INT64_MAX, &v))
    return snprintf(why, wn, "GEN's max_new not a whole number (%.24s)", tok), false;
  g->max_new = (long long)v;
  for (tok = strtok_r(NULL, " ", &save); tok; tok = strtok_r(NULL, " ", &save))
  {
    char *eq = strchr(tok, '=');
    if (!eq)
      break; /* the ids */
    *eq = 0;
    const char *x = eq + 1;
    bool ok;
    if (!strcmp(tok, "spec"))
      ok = whole(x, 1, &v), g->spec = (int)v;
    else if (!strcmp(tok, "rid"))
      ok = gen_rid(x, &g->rid), g->has_rid = true;
    else if (!strcmp(tok, "mark"))
      ok = whole(x, UINT32_MAX, &v), g->mark = (uint32_t)v;
    else if (!strcmp(tok, "temp"))
      ok = real(x, &g->samp.temperature);
    else if (!strcmp(tok, "top_p"))
      ok = real(x, &g->samp.top_p);
    else if (!strcmp(tok, "seed"))
      ok = whole(x, UINT64_MAX, &v), g->samp.seed = (uint64_t)v;
    else if (!strcmp(tok, "top_k")) /* 0: every token */
      ok = whole(x, UINT32_MAX, &v), g->samp.top_k = (uint32_t)v;
    else
      return snprintf(why, wn, "the key '%s' is not this engine's (rid, spec, mark, temp, top_p, top_k, seed)", tok),
             false;
    if (!ok)
      return snprintf(why, wn, "%s= not a whole value of its range (%.24s)", tok, x), false;
  }
  if (!(g->samp.temperature >= 0 && g->samp.temperature < 1e30f) || !(g->samp.top_p > 0 && g->samp.top_p <= 1))
    return snprintf(why, wn, "temp= finite and at least 0, top_p= in (0, 1]"), false;
  if (!tok)
    return snprintf(why, wn, "a GEN with no ids"), false;
  if (strtok_r(NULL, " ", &save))
    return snprintf(why, wn, "a GEN's ids one list (id,id,...), nothing after it"), false;
  uint32_t cap = 1;
  for (const char *c = tok; *c; c++)
    cap += *c == ',';
  if (!(g->ids = malloc((size_t)cap * sizeof *g->ids)))
    return snprintf(why, wn, "out of memory"), false;
  for (char *c = tok;;) /* each id whole, one comma between two: none empty, none after the last */
  {
    char *comma = strchr(c, ',');
    if (comma)
      *comma = 0;
    if (!whole(c, INT32_MAX, &v))
      return gen_free(g), snprintf(why, wn, "a GEN's id not a whole token id (%.24s)", c), false;
    g->ids[g->n++] = (int32_t)v;
    if (!comma)
      break;
    c = comma + 1;
  }
  return true;
}
