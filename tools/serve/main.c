/* osmos-serve: the engine over stdin / stdout in maya's DeepSeek serve protocol (its src/ds4/ds4_main.cpp), so its
   harnesses drive this engine as they drive maya's (DESIGN.md, "serving").

     osmos-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] [--max-context N]
                     [--budget GiB|auto] [--max-tokens N] [--eos id,id] [--parallel N] [--no-guess]
                     [--overlay <gguf>] [--serve]

   in:  GEN <max_new> [rid=<r>] [spec=<0|1>] [seed=<s>] <id,id,...>   STOP [<rid>]   QUIT
   out: READY <max_ctx> stop[ multi=N] / T [<rid>] <token> / PP [<rid>] <done> <total> <ms> <tok/s> <reused> /
        DONE [<rid>] <produced> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <drafted> <reused> 0 0 /
        ERR [<rid>] <reason> / NOTE <text>

   one conversation (single.c), or --parallel N at once (multi.c: GEN then needs rid=, every line carries it).
   with --dspark a GEN drafts unless it says spec=0 (maya's default: as the engine was started).
   --no-guess: no experts guessed a layer ahead (ForwardOptions' no_guess - a model whose misses are few).
   --overlay: a GGUF whose tensors stand in for the model's of their names (an abliteration patch: weights.h).
   --budget: GiB of experts and dense weights in GPU memory, or auto (the default): sized to this GPU (build.c).
   OSMOS_SLOT_KEEP=1 keeps conversations (maya #356): OSMOS_SLOT_CAP (4) in RAM, past it OSMOS_SLOT_DIR
   (<the model's folder>/slots) under OSMOS_SLOT_GB (16), OSMOS_SLOT_MIN (1024) tokens at least,
   OSMOS_SLOT_DAYS (7). greedy: maya's sampling, penalty and thinking keys are refused, named */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "serve.h"
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/version.h"

static const char *opt(int argc, char **argv, const char *name, const char *dflt)
{
  for (int i = 1; i + 1 < argc; i++)
    if (!strcmp(argv[i], name))
      return argv[i + 1];
  return dflt;
}

static bool has(int argc, char **argv, const char *name)
{
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], name))
      return true;
  return false;
}

static double env_num(const char *name, double dflt)
{
  const char *v = getenv(name);
  return v && *v ? atof(v) : dflt;
}

static int die(const char *what)
{
  fprintf(stderr, "osmos-serve: %s\n", what);
  return 2;
}

static bool open_slots(Server *sv, const char *model, char *err, size_t en)
{
  char dflt[4096];
  const char *slash = strrchr(model, '/');
  snprintf(dflt, sizeof dflt, "%.*s/slots", slash ? (int)(slash - model) : 1, slash ? model : ".");
  const char *dir = getenv("OSMOS_SLOT_DIR") ? getenv("OSMOS_SLOT_DIR") : dflt;
  const SlotsOptions so = {dir, (uint32_t)env_num("OSMOS_SLOT_CAP", 4), env_num("OSMOS_SLOT_GB", 16),
                           (uint32_t)env_num("OSMOS_SLOT_MIN", 1024), env_num("OSMOS_SLOT_DAYS", 7),
                           forward_identity(sv->f)};
  char serr[SLOTS_ERR] = "";
  if (!(sv->slots = slots_create(&so, serr)))
    return snprintf(err, en, "%s", serr), false;
  out("NOTE slots: %u conversations kept from an earlier run (%s)\n", slots_in_dir(sv->slots), dir);
  return true;
}

int main(int argc, char **argv)
{
  if (argc == 2 && !strcmp(argv[1], "--version"))
    return printf("osmos %s\n", OSMOS_VERSION), 0;
  const char *model = opt(argc, argv, "--model", NULL);
  if (!model)
    return die("usage: osmos-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] "
               "[--max-context N] [--budget GiB|auto] [--max-tokens N] [--eos id,id] [--parallel N] [--no-guess] "
               "[--overlay <gguf>] [--serve]");
  Server sv = {0};
  sv.max_ctx = (uint32_t)atol(opt(argc, argv, "--max-context", "1048576"));
  sv.max_tokens = (uint32_t)atol(opt(argc, argv, "--max-tokens", "2048"));
  const char *budget_arg = opt(argc, argv, "--budget", "auto");
  const double budget = strcmp(budget_arg, "auto") ? atof(budget_arg) : 0; /* 0: sized to this GPU (build.c) */
  if (strcmp(budget_arg, "auto") && budget <= 0)
    return die("--budget takes GiB or auto");
  const char *par = opt(argc, argv, "--parallel", NULL);
  const int lanes = par ? atoi(par) : 1;
  if (lanes < 1)
    return die("--parallel takes 1 or more");
  char *list = strdup(opt(argc, argv, "--eos", "1")), *sv_ = NULL;
  for (char *x = strtok_r(list, ",", &sv_); x && sv.n_eos < SERVE_MAX_EOS; x = strtok_r(NULL, ",", &sv_))
    sv.eos[sv.n_eos++] = (int32_t)atol(x);
  free(list);
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", err[FORWARD_ERR] = "";
  Gguf *g = gguf_create(model, gerr);
  const char *engram = opt(argc, argv, "--engram", NULL);
  Gguf *eg = engram ? gguf_create(engram, gerr) : NULL;
  const char *overlay = opt(argc, argv, "--overlay", NULL);
  Gguf *ov = overlay ? gguf_create(overlay, gerr) : NULL;
  if (!g || (engram && !eg) || (overlay && !ov))
    return die(gerr);
  Weights *w = weights_create_with(&(WeightsOptions){g, eg, ov}, werr);
  if (!w)
    return die(werr);
  sv.round_tokens = (uint32_t)((w->geo.n_experts - 1) / w->geo.n_used);
  sv.window = (uint32_t)w->geo.window;
  uint64_t *picks = NULL;
  if (opt(argc, argv, "--picks", NULL))
  {
    const size_t np = (size_t)w->geo.n_layers * w->geo.n_experts;
    FILE *pf = fopen(opt(argc, argv, "--picks", NULL), "rb");
    picks = malloc(np * sizeof *picks);
    if (!pf || !picks || fread(picks, sizeof *picks, np, pf) != np)
      return die("the picks file could not be read");
    fclose(pf);
  }
  const ForwardOptions fo = {(int)w->geo.n_layers, sv.max_ctx, (uint64_t)(budget * (double)(1ull << 30)), picks,
                             sv.max_tokens, has(argc, argv, "--no-guess")};
  Conv *conv = calloc((size_t)lanes, sizeof *conv);
  if (!conv)
    return die("out of memory");
  if (!serve_build(&sv, conv, lanes, w, fo, opt(argc, argv, "--dspark", NULL), err))
    return die(err);
  if (env_num("OSMOS_SLOT_KEEP", 0) != 0 && !open_slots(&sv, model, err, sizeof err))
    return die(err);
  const int rc = par ? serve_multi(&sv, conv, lanes) : serve_single(&sv, conv);
  if (sv.slots) /* the end: every conversation set aside, RAM's written to the folder */
  {
    char serr[SLOTS_ERR] = "";
    for (int i = 0; i < lanes; i++)
      conv_set_aside(&conv[i], &sv);
    if (!slots_flush(sv.slots, serr))
      out("NOTE slots: %s\n", serr);
    slots_destroy(sv.slots);
  }
  serve_unbuild(&sv, conv, lanes);
  free(conv);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  gguf_destroy(ov);
  free(picks);
  return rc;
}
