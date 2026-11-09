/* osmos-serve: the engine over stdin / stdout in maya's DeepSeek serve protocol (its src/ds4/ds4_main.cpp), so its
   harnesses drive this engine as they drive maya's (DESIGN.md, "serving").

     osmos-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] [--max-context N]
                     [--budget GiB|auto] [--fill GiB] [--max-tokens N] [--eos id,id] [--parallel N] [--no-guess]
                     [--stream-ngram] [--overlay <gguf>] [--answer-after <text>] [--serve]

   in:  GEN <max_new> [rid=<r>] [spec=<0|1>] [seed=<s>] <id,id,...>   STOP [<rid>]   QUIT
   out: READY <max_ctx> stop[ multi=N] / T [<rid>] <token> / PP [<rid>] <done> <total> <ms> <tok/s> <reused> /
        DONE [<rid>] <produced> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <drafted> <reused> 0 0 /
        ERR [<rid>] <reason> / NOTE <text>

   one conversation (single.c), or --parallel N at once (multi.c: GEN then needs rid=, every line carries it).
   with --dspark a GEN drafts unless it says spec=0 (maya's default: as the engine was started).
   --no-guess: no experts guessed a layer ahead (OsmosOptions' no_guess - a model whose misses are few).
   --overlay: a GGUF whose tensors stand in for the model's of their names (an abliteration patch: weights.h).
   --stream-ngram: Qwen's n-gram table (per_layer_token_embd) read from its file each step, not held in GPU memory
   (Q4xOptions' stream_ngram: the same bits, its 26.8 GiB of UD-IQ3_XXS left free); DeepSeek's Engram always is.
   --budget: GiB of experts and dense weights in GPU memory, or auto (the default): sized to this GPU (build.c).
   --fill: GiB of the system's RAM kept free - 8 by default (FIT_RAM_FREE: room for an editor, a browser, language
   servers and the agent harness beside the model), 0 or more: auto's sizing fills past it, and saved conversations'
   RAM stops at it. more than can be kept free: the smallest build (the dense weights alone, every expert streamed),
   with a NOTE. the output's bits the same at every value: only what stays in memory moves.
   --answer-after: the reasoning's end token's text (</think>), or #<id>: no --eos token is taken while the turn's
   answer past it is whitespace alone - the row's best other token is (osmos/text/answer.h; osmos-api passes its
   dialect's text; tools/serve/answer_test.py an id).
   conversations are kept (maya #356; OSMOS_SLOT_KEEP=0: not): OSMOS_SLOT_CAP (4) in RAM, past it OSMOS_SLOT_DIR
   (<the model's folder>/slots) under OSMOS_SLOT_GB (16), OSMOS_SLOT_MIN (1024) tokens at least, RAM's room the
   available above --fill's (OSMOS_SLOT_RAM_GB lowers it; past it a state streams to the folder in pieces),
   OSMOS_SLOT_DAYS (7). OSMOS_STATS=1: the forward's read counters on stderr at exit (the reads it waited on, their
   time; the guess's). greedy: maya's sampling, penalty and thinking keys are refused, named */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "serve.h"
#include "osmos/run/fit.h"
#include "osmos/model/gguf.h"
#include "osmos/run/gpu.h"
#include "osmos/text/tokenizer.h"
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

/* --fill's GiB as bytes: a number and nothing after it, 0 or more and finite (more than any machine's: the most) */
static bool fill_of(const char *s, uint64_t *bytes)
{
  char *end = NULL;
  const double gib = strtod(s, &end);
  if (!*s || end == s || *end || !(gib >= 0) || gib > 1e15)
    return false;
  const double b = gib * (double)(1ull << 30);
  *bytes = b >= 18e18 ? UINT64_MAX : (uint64_t)b;
  return true;
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
  const uint64_t avail = fit_ram_available(), room = avail > sv->ram_kept ? avail - sv->ram_kept : 0;
  const double cap_gb = env_num("OSMOS_SLOT_RAM_GB", -1); /* lower only: the room above --fill's is the most */
  const uint64_t ram = cap_gb >= 0 && (uint64_t)(cap_gb * (double)(1ull << 30)) < room ? (uint64_t)(cap_gb * (double)(1ull << 30)) : room;
  const SlotsOptions so = {dir, (uint32_t)env_num("OSMOS_SLOT_CAP", 4), ram, env_num("OSMOS_SLOT_GB", 16),
                           (uint32_t)env_num("OSMOS_SLOT_MIN", 1024), env_num("OSMOS_SLOT_DAYS", 7),
                           osmos_identity(sv->e)};
  char serr[SLOTS_ERR] = "";
  if (!(sv->slots = slots_create(&so, serr)))
    return snprintf(err, en, "%s", serr), false;
  out("NOTE slots: %u conversations kept from an earlier run (%s); RAM for them %.2f GiB (%.2f available, %.0f kept "
      "free)\n", slots_in_dir(sv->slots), dir, ram / (double)(1ull << 30), avail / (double)(1ull << 30),
      sv->ram_kept / (double)(1ull << 30));
  return true;
}

/* --answer-after's rule, from the model's own tokenizer (the token of its text, its blank tokens) */
static bool answer_of(Server *sv, const char *model, const char *after, char err[OSMOS_ERR])
{
  char gerr[GGUF_ERR] = "", terr[TOKENIZER_ERR] = "", aerr[ANSWER_ERR] = "";
  Gguf *g = gguf_header(model, gerr);
  Tokenizer *t = g ? tokenizer_create(g, terr) : NULL;
  char *rest = NULL;
  const long n = after[0] == '#' ? strtol(after + 1, &rest, 10) : -1;
  const int32_t id = after[0] == '#' ? (rest > after + 1 && !*rest ? (int32_t)n : -1) : t ? tokenizer_find(t, after) : -1;
  const bool ok = t && answer_rule_create(&sv->answer, t, id, aerr);
  if (!ok)
    snprintf(err, OSMOS_ERR, "--answer-after: %s", !g ? gerr : !t ? terr : aerr);
  else
    out("NOTE answer-after %d\n", sv->answer.after);
  tokenizer_destroy(t), gguf_destroy(g);
  return ok;
}

int main(int argc, char **argv)
{
  if (argc == 2 && !strcmp(argv[1], "--version"))
    return printf("osmos %s\n", OSMOS_VERSION), 0;
  if (!gpu_sleep_on_waits()) /* before the first kernel: after one it does not take (osmos/run/gpu.h) */
    return die("the GPU could not be set to sleep on waits (hipSetDeviceFlags)");
  const char *model = opt(argc, argv, "--model", NULL);
  if (!model)
    return die("usage: osmos-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] "
               "[--max-context N] [--budget GiB|auto] [--fill GiB] [--max-tokens N] [--eos id,id] [--parallel N] "
               "[--no-guess] [--stream-ngram] "
               "[--overlay <gguf>] [--serve]");
  Server sv = {0};
  sv.max_ctx = (uint32_t)atol(opt(argc, argv, "--max-context", "1048576"));
  sv.max_tokens = (uint32_t)atol(opt(argc, argv, "--max-tokens", "2048"));
  const char *budget_arg = opt(argc, argv, "--budget", "auto");
  const double budget = strcmp(budget_arg, "auto") ? atof(budget_arg) : 0; /* 0: sized to this GPU (build.c) */
  if (strcmp(budget_arg, "auto") && budget <= 0)
    return die("--budget takes GiB or auto");
  const char *fill_arg = opt(argc, argv, "--fill", NULL);
  if (fill_arg && !fill_of(fill_arg, &sv.ram_kept))
    return die("--fill takes GiB of RAM to keep free, 0 or more");
  if (!fill_arg)
    sv.ram_kept = FIT_RAM_FREE;
  const char *par = opt(argc, argv, "--parallel", NULL);
  const int lanes = par ? atoi(par) : 1;
  if (lanes < 1)
    return die("--parallel takes 1 or more");
  char *list = strdup(opt(argc, argv, "--eos", "1")), *sv_ = NULL;
  for (char *x = strtok_r(list, ",", &sv_); x && sv.n_eos < SERVE_MAX_EOS; x = strtok_r(NULL, ",", &sv_))
    sv.eos[sv.n_eos++] = (int32_t)atol(x);
  free(list);
  char err[OSMOS_ERR] = "";
  if (opt(argc, argv, "--answer-after", NULL) && !answer_of(&sv, model, opt(argc, argv, "--answer-after", ""), err))
    return die(err);
  const OsmosFiles files = {model, opt(argc, argv, "--engram", NULL), opt(argc, argv, "--overlay", NULL)};
  OsmosModel *m = osmos_open(&files, err);
  if (!m)
    return die(err);
  const OsmosFacts fx = osmos_facts(m);
  sv.round_tokens = fx.round_tokens;
  sv.window = fx.seed_window;
  const OsmosOptions o = {sv.max_ctx, sv.max_tokens, (uint64_t)(budget * (double)(1ull << 30)),
                          opt(argc, argv, "--picks", NULL), has(argc, argv, "--no-guess"),
                          opt(argc, argv, "--dspark", NULL), true, sv.ram_kept, has(argc, argv, "--stream-ngram")};
  Conv *conv = calloc((size_t)lanes, sizeof *conv);
  if (!conv)
    return die("out of memory");
  if (!serve_build(&sv, conv, lanes, m, o, err))
    return die(err);
  if (env_num("OSMOS_SLOT_KEEP", 1) != 0 && !open_slots(&sv, model, err, sizeof err))
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
  if (getenv("OSMOS_STATS") && sv.e) /* the engine's counters at the end (a bench's: where its time waits) */
  {
    OsmosStats st;
    osmos_stats(sv.e, &st);
    fprintf(stderr, "STATS waited %llu reads not guessed (%.0f ms), %llu guessed (%.0f ms); streamed picks %llu, guessed "
                    "%llu, the guess read %llu\n",
            (unsigned long long)st.waits[0], st.wait_ns[0] / 1e6, (unsigned long long)st.waits[1], st.wait_ns[1] / 1e6,
            (unsigned long long)st.streamed, (unsigned long long)st.guessed, (unsigned long long)st.guess_read);
  }
  serve_unbuild(&sv, conv, lanes);
  free(conv);
  osmos_close(m);
  answer_rule_destroy(&sv.answer);
  return rc;
}
