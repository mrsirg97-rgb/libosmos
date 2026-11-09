#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

/* osmos.json read whole and checked: a field of the wrong type, or one this reader does not know, refused by name
   (a misspelt setting is not silently its default) */

#define CONFIG_MAX (1u << 20)

static bool fail(char *err, const char *what, const char *key)
{
  snprintf(err, API_ERR, "osmos.json: %s%s%s", key ? key : "", key ? ": " : "", what);
  return false;
}

static bool str(const Json *v, char *out, size_t cap, const char *key, char *err)
{
  if (v->type != JSON_STRING || !v->len || v->len >= cap)
    return fail(err, "a non-empty path (string)", key);
  memcpy(out, v->s, v->len + 1);
  return true;
}

static bool num(const Json *v, double *out, const char *key, char *err)
{
  if (v->type != JSON_INT && v->type != JSON_FLOAT)
    return fail(err, "a number", key);
  *out = v->type == JSON_INT ? atof(v->s) : v->f;
  if (*out <= 0)
    return fail(err, "a positive number", key);
  return true;
}

static bool field(Config *c, const char *k, const Json *v, char *err)
{
  double x;
  if (!strcmp(k, "pack"))
    return str(v, c->pack, sizeof c->pack, k, err);
  if (!strcmp(k, "title"))
    return v->type == JSON_STRING || fail(err, "a string", k);
  if (!strcmp(k, "model"))
    return str(v, c->model, sizeof c->model, k, err);
  if (!strcmp(k, "picks"))
    return str(v, c->picks, sizeof c->picks, k, err);
  if (!strcmp(k, "drafter"))
    return str(v, c->drafter, sizeof c->drafter, k, err);
  if (!strcmp(k, "overlay"))
    return str(v, c->overlay, sizeof c->overlay, k, err);
  if (!strcmp(k, "budget_gib"))
    return num(v, &c->budget_gib, k, err);
  if (!strcmp(k, "drafted_budget_gib"))
    return num(v, &c->drafted_budget_gib, k, err);
  if (!strcmp(k, "fill_gib")) /* 0 or more: osmos-serve's --fill */
  {
    if (v->type != JSON_INT && v->type != JSON_FLOAT)
      return fail(err, "a number of GiB, 0 or more", k);
    c->fill_gib = v->type == JSON_INT ? atof(v->s) : v->f, c->fill_set = true;
    return c->fill_gib >= 0 || fail(err, "a number of GiB, 0 or more", k);
  }
  if (!strcmp(k, "no_guess"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->no_guess = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "max_context") || !strcmp(k, "parallel") || !strcmp(k, "max_tokens"))
  {
    if (v->type != JSON_INT || !num(v, &x, k, err) || x > 1e7)
      return fail(err, "a positive integer", k);
    *(!strcmp(k, "max_context") ? &c->max_context : !strcmp(k, "parallel") ? &c->parallel : &c->max_tokens) =
        (unsigned)x;
    return true;
  }
  return fail(err, "not a setting osmos-api knows", k);
}

bool config_read(const char *path, Config *c, char err[API_ERR])
{
  memset(c, 0, sizeof *c);
  c->parallel = 1;
  FILE *f = fopen(path, "rb");
  if (!f)
    return snprintf(err, API_ERR, "%s: could not be opened", path), false;
  char *text = malloc(CONFIG_MAX);
  const size_t n = text ? fread(text, 1, CONFIG_MAX, f) : 0;
  fclose(f);
  char jerr[JSON_ERR];
  Json *j = n && n < CONFIG_MAX ? json_parse(text, n, jerr) : NULL;
  free(text);
  if (!j || j->type != JSON_OBJECT)
    return json_free(j), fail(err, "not a JSON object", NULL);
  bool ok = true;
  for (size_t i = 0; ok && i < j->n; i++)
    ok = field(c, j->key[i], j->item[i], err);
  json_free(j);
  if (ok && (!c->pack[0] || !c->model[0] || !c->max_context))
    return fail(err, "pack, model and max_context are needed", NULL);
  return ok;
}

bool config_argv(Config *c, const char *engine, int eos, const char *answer_after, char *argv[CONFIG_ARGS_MAX],
                 char err[API_ERR])
{
  int n = 0;
  snprintf(c->eng, sizeof c->eng, "%s", engine);
  snprintf(c->num[0], sizeof c->num[0], "%u", c->max_context);
  const double b = c->drafter[0] ? c->drafted_budget_gib : c->budget_gib; /* 0: auto (osmos-serve sizes it) */
  snprintf(c->num[1], sizeof c->num[1], b > 0 ? "%g" : "auto", b);
  snprintf(c->num[2], sizeof c->num[2], "%d", eos);
  snprintf(c->num[3], sizeof c->num[3], "%u", c->parallel);
  argv[n++] = c->eng, argv[n++] = "--model", argv[n++] = c->model;
  if (c->picks[0])
    argv[n++] = "--picks", argv[n++] = c->picks;
  if (c->drafter[0])
    argv[n++] = "--dspark", argv[n++] = c->drafter;
  if (c->overlay[0])
    argv[n++] = "--overlay", argv[n++] = c->overlay;
  argv[n++] = "--max-context", argv[n++] = c->num[0];
  argv[n++] = "--budget", argv[n++] = c->num[1];
  argv[n++] = "--eos", argv[n++] = c->num[2];
  argv[n++] = "--answer-after", argv[n++] = (char *)answer_after;
  argv[n++] = "--parallel", argv[n++] = c->num[3];
  if (c->max_tokens)
  {
    snprintf(c->num[4], sizeof c->num[4], "%u", c->max_tokens);
    argv[n++] = "--max-tokens", argv[n++] = c->num[4];
  }
  if (c->no_guess)
    argv[n++] = "--no-guess";
  if (c->fill_set)
  {
    snprintf(c->num[5], sizeof c->num[5], "%g", c->fill_gib);
    argv[n++] = "--fill", argv[n++] = c->num[5];
  }
  argv[n] = NULL;
  (void)err;
  return true;
}
