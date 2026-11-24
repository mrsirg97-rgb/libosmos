#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/run/engine.h" /* (OSMOS_MAX_LANES: osmos-serve's lanes at most) */
#include "config.h"
#include "osmos/err.h"

/* osmos.json read whole and checked: a field of the wrong type, or one this reader does not know, refused by name
   (a misspelt setting is not silently its default) */

#define CONFIG_MAX (1u << 20)

static bool fail(char *err, const char *what, const char *key)
{
  osmos_fail(err, "osmos.json: %s%s%s", key ? key : "", key ? ": " : "", what);
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

/* osmos.json's slots object: keep (a bool), dir (a path), cap / min (whole counts), gb / ram_gb / days (numbers) -
   each its type, past 0; another key refused */
static bool slots(Config *c, const Json *v, char *err)
{
  if (v->type != JSON_OBJECT)
    return fail(err, "an object (keep, dir, cap, gb, ram_gb, min, days)", "slots");
  for (size_t i = 0; i < v->n; i++)
  {
    const char *k = v->key[i];
    const Json *x = v->item[i];
    double d = 0;
    if (!strcmp(k, "keep"))
    {
      if (x->type != JSON_TRUE && x->type != JSON_FALSE)
        return fail(err, "true or false", "slots.keep");
      c->slots_off = x->type == JSON_FALSE;
      continue;
    }
    if (!strcmp(k, "dir"))
    {
      if (!str(x, c->slot_dir, sizeof c->slot_dir, "slots.dir", err))
        return false;
      continue;
    }
    double *at = !strcmp(k, "cap") ? &c->slot_cap : !strcmp(k, "gb") ? &c->slot_gb : !strcmp(k, "ram_gb") ? &c->slot_ram_gb
               : !strcmp(k, "min") ? &c->slot_min : !strcmp(k, "days") ? &c->slot_days : NULL;
    char name[48];
    snprintf(name, sizeof name, "slots.%.32s", k);
    if (!at)
      return fail(err, "a key slots does not take (keep, dir, cap, gb, ram_gb, min, days)", name);
    const bool count = !strcmp(k, "cap") || !strcmp(k, "min");
    if ((count && x->type != JSON_INT) || !num(x, &d, name, err) || d > 1e7)
      return fail(err, count ? "a whole count above 0" : "a number above 0", name);
    *at = d;
  }
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
  if (!strcmp(k, "lsa"))
    return str(v, c->lsa, sizeof c->lsa, k, err);
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
  if (!strcmp(k, "preserve_thinking"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->preserve_thinking = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "fill_orphans"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->fill_orphans = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "no_guess"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->no_guess = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "npu")) /* the NPU beside the GPU (forward_npu) */
    return str(v, c->npu, sizeof c->npu, k, err);
  if (!strcmp(k, "npu_share"))
    return (v->type == JSON_INT || v->type == JSON_FLOAT) && (c->npu_share = v->type == JSON_INT ? atof(v->s) : v->f) > 0 &&
                   c->npu_share < 1
               ? true
               : fail(err, "a share between 0 and 1", k);
  if (!strcmp(k, "stream_ngram"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->stream_ngram = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "temperature")) /* 0: greedy */
    return (v->type == JSON_INT || v->type == JSON_FLOAT) &&
                   (c->temperature = v->type == JSON_INT ? atof(v->s) : v->f) >= 0 && c->temperature < 100
               ? true
               : fail(err, "a number, 0 or more", k);
  if (!strcmp(k, "top_p"))
    return (v->type == JSON_INT || v->type == JSON_FLOAT) && (c->top_p = v->type == JSON_INT ? atof(v->s) : v->f) > 0 &&
                   c->top_p <= 1
               ? true
               : fail(err, "a share above 0, 1 at most", k);
  if (!strcmp(k, "top_k")) /* 0: every token */
    return v->type == JSON_INT && v->s[0] != '-' && strlen(v->s) <= 9 ? (c->top_k = (unsigned)atol(v->s), true)
                                                                         : fail(err, "an integer, 0 (every token) or more", k);
  if (!strcmp(k, "kv_q8"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->kv_q8 = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "slots"))
    return slots(c, v, err);
  if (!strcmp(k, "stats"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->stats = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "profile"))
    return v->type == JSON_TRUE || v->type == JSON_FALSE ? c->profile = v->type == JSON_TRUE, true
                                                         : fail(err, "true or false", k);
  if (!strcmp(k, "rope_stages")) /* (its form osmos-serve's to refuse: lfm3_stages_parse) */
    return v->type == JSON_STRING && v->s[0] ? str(v, c->rope_stages, sizeof c->rope_stages, k, err)
                                             : fail(err, "a string, from:factor,...", k);
  if (!strcmp(k, "max_context") || !strcmp(k, "parallel") || !strcmp(k, "max_tokens"))
  {
    if (v->type != JSON_INT || !num(v, &x, k, err) || x > 1e7)
      return fail(err, "a positive integer", k);
    if (!strcmp(k, "parallel") && (x < 1 || x > OSMOS_MAX_LANES)) /* (osmos-serve's bound: refused here, at start) */
      return fail(err, "1 to 64 (osmos-serve's lanes at most)", k);
    *(!strcmp(k, "max_context") ? &c->max_context : !strcmp(k, "parallel") ? &c->parallel : &c->max_tokens) =
        (unsigned)x;
    return true;
  }
  return fail(err, "not a setting osmos-api knows", k);
}

bool config_read(const char *path, Config *c, char err[OSMOS_ERR])
{
  memset(c, 0, sizeof *c);
  c->parallel = 1, c->top_p = 1;
  FILE *f = fopen(path, "rb");
  if (!f)
    return osmos_fail(err, "%s: could not be opened", path), false;
  char *text = malloc(CONFIG_MAX);
  const size_t n = text ? fread(text, 1, CONFIG_MAX, f) : 0;
  fclose(f);
  char jerr[OSMOS_ERR] = "";
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
  if (ok && c->npu_share > 0 && !c->npu[0])
    return fail(err, "is nothing without npu (its kernels' directory)", "npu_share");
  return ok;
}

bool config_argv(Config *c, const char *engine, int eos, const char *answer_after, char *argv[CONFIG_ARGS_MAX],
                 char err[OSMOS_ERR])
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
  if (c->lsa[0])
    argv[n++] = "--lsa", argv[n++] = c->lsa;
  argv[n++] = "--max-context", argv[n++] = c->num[0];
  argv[n++] = "--budget", argv[n++] = c->num[1];
  argv[n++] = "--eos", argv[n++] = c->num[2];
  if (answer_after)
    argv[n++] = "--answer-after", argv[n++] = (char *)answer_after;
  argv[n++] = "--parallel", argv[n++] = c->num[3];
  if (c->max_tokens)
  {
    snprintf(c->num[4], sizeof c->num[4], "%u", c->max_tokens);
    argv[n++] = "--max-tokens", argv[n++] = c->num[4];
  }
  if (c->no_guess)
    argv[n++] = "--no-guess";
  if (c->stream_ngram)
    argv[n++] = "--stream-ngram";
  if (c->kv_q8)
    argv[n++] = "--kv-q8";
  if (c->profile)
    argv[n++] = "--profile";
  if (c->stats)
    argv[n++] = "--stats";
  if (c->slots_off)
    argv[n++] = "--no-slots";
  if (c->slot_dir[0])
    argv[n++] = "--slot-dir", argv[n++] = c->slot_dir;
  const struct
  {
    const char *flag;
    double v;
  } slot_nums[] = {{"--slot-cap", c->slot_cap}, {"--slot-gb", c->slot_gb}, {"--slot-ram-gb", c->slot_ram_gb},
                   {"--slot-min", c->slot_min}, {"--slot-days", c->slot_days}};
  for (size_t i = 0; i < sizeof slot_nums / sizeof *slot_nums; i++)
    if (slot_nums[i].v > 0)
    {
      snprintf(c->num[7 + i], sizeof c->num[7 + i], "%g", slot_nums[i].v);
      argv[n++] = (char *)slot_nums[i].flag, argv[n++] = c->num[7 + i];
    }
  if (c->rope_stages[0])
    argv[n++] = "--rope-stages", argv[n++] = c->rope_stages;
  if (c->npu[0])
    argv[n++] = "--npu", argv[n++] = c->npu;
  if (c->npu[0] && c->npu_share > 0)
  {
    snprintf(c->num[6], sizeof c->num[6], "%g", c->npu_share);
    argv[n++] = "--npu-share", argv[n++] = c->num[6];
  }
  if (c->fill_set)
  {
    snprintf(c->num[5], sizeof c->num[5], "%g", c->fill_gib);
    argv[n++] = "--fill", argv[n++] = c->num[5];
  }
  argv[n] = NULL;
  (void)err;
  return true;
}
