/* osmos-serve: the engine over stdin / stdout in maya's DeepSeek serve protocol (its src/ds4/ds4_main.cpp), so its
   harnesses drive this engine as they drive maya's (DESIGN.md, "serving").

     osmos-serve --model <first shard> [--engram <gguf>] [--picks <file>] [--dspark <dir>] [--max-context N]
                     [--budget GiB|auto] [--fill GiB] [--max-tokens N] [--eos id,id] [--parallel N] [--no-guess]
                     [--stream-ngram] [--kv-q8] [--no-int8] [--overlay <gguf>] [--lsa <maps>] [--answer-after <text>]
                     [--profile] [--rope-stages <from:factor,...>] [--npu <dir> [--npu-share S]] [--stats]
                     [--no-fit-hint]
                     [--no-slots | --slot-cap N --slot-dir <dir> --slot-gb GB --slot-min N --slot-ram-gb GB
                      --slot-days D] [--serve]
     osmos-serve --version

   in:  GEN <max_new> [rid=<r>] [spec=<0|1>] [mark=<n>] [temp=<t>] [top_p=<p>] [top_k=<k>] [seed=<s>] <id,id,...>
        STOP [<rid>]   QUIT
   out: READY <max_ctx> stop[ multi=N] / T [<rid>] <token> / PP [<rid>] <done> <total> <ms> <tok/s> <reused> /
        DONE [<rid>] <produced> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <drafted> <reused> 0 0 /
        ERR [<rid>] <reason> / NOTE <text> (before READY, `NOTE reads apart: ...` where the family's prompt and decode
        are two maths: a next turn then reuses its prompt's point, its earlier answer read again)

   one conversation (single.c), or --parallel N at once (multi.c: GEN then needs rid=, every line carries it).
   with --dspark a GEN drafts unless it says spec=0 (maya's default: as the engine was started).
   --budget auto (the default) sizes to this machine in whole builds, its first step the last start's budget for the
   same flags, model and version (a hint beside the model, .osmos-fit-<hash>: build.c); --no-fit-hint sizes from cold.
   --no-guess: no experts guessed a layer ahead (OsmosOptions' no_guess - a model whose misses are few).
   --overlay: a GGUF whose tensors stand in for the model's of their names (an abliteration patch: weights.h).
   --lsa: latent sparse attention's maps (LFM3's: osmos1.7b-lsa-maps.safetensors) - a decode's full layers read them.
   --stream-ngram: Qwen's n-gram table (per_layer_token_embd) read from its file each step, not held in GPU memory
   (Q4xOptions' stream_ngram: the same bits); DeepSeek's Engram always is.
   --kv-q8: Inkling's full layers' cached values Q8, as its keys, not Q5 (OsmosOptions' kv_q8: 17 KiB a token of
   Mini's cache, not 14 - another output; a pack with the room's choice); the other families refuse it.
   --no-int8: the routed experts' int8 decode off - each family's f16 / F32 path at every step (OsmosOptions'
   no_int8: a pack's "int8": false, where its drafted decode measured slower: Inkling-2-Mini's NVFP4).
   --npu: the NPU's kernels' directory, the NPU beside the GPU; --npu-share its share of a layer's pairs (0 to 1).
   --profile: a measurement's build - a step's sections timed (each after a synchronize: the bits the same), their
   totals on stderr at the engine's destroy (OsmosOptions' profile: Inkling's; the others refuse it).
   --rope-stages: osmos1.7b's YaRN stages, "from:factor,..." (OsmosOptions' rope_stages: a table tried without a build;
   malformed, or another family's, refused).
   --budget: GiB of experts and dense weights in GPU memory, or auto (the default): sized to this GPU (build.c).
   --fill: GiB of the system's RAM kept free - 8 by default (FIT_RAM_FREE: room for an editor, a browser, language
   servers and the agent harness beside the model), 0 or more: auto's sizing fills past it, and saved conversations'
   RAM stops at it. more than the machine can keep free beside the smallest build: refused, not started. the output's
   bits the same at every value: only what stays in memory moves.
   --answer-after: the reasoning's end token's text (</think>), or #<id>: no --eos token is taken while the turn's
   answer past it is whitespace alone - the row's best other token is (osmos/text/answer.h; osmos-api passes its
   dialect's text; tools/serve/answer_test.py an id).
   conversations are kept (--no-slots: not): --slot-cap (4) in RAM, past it --slot-dir (<the model's folder>/slots)
   under --slot-gb (16), --slot-min (1024) tokens at least, RAM's room the available above --fill's (--slot-ram-gb
   lowers it; past it a state streams to the folder in pieces), --slot-days (7) - osmos.json's slots object.
   --stats: the forward's read counters on stderr at exit (the reads it waited on, their time; the guess's).
   --serve: maya's, taken and ignored. maya's penalty and thinking keys are refused, named */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "serve.h"
#include "osmos/run/fit.h"
#include "osmos/model/gguf.h"
#include "osmos/run/gpu.h"
#include "osmos/text/tokenizer.h"
#include "osmos/version.h"
#include "osmos/err.h"

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

/* a flag's number whole (a decimal and nothing after it, finite) in [lo, hi], or dflt when not given: false for
   one given otherwise */
static bool flag_num(int argc, char **argv, const char *name, double dflt, double lo, double hi, double *v)
{
  const char *s = opt(argc, argv, name, NULL);
  if (!s)
    return *v = dflt, true;
  char *end = NULL;
  errno = 0;
  const double x = strtod(s, &end);
  if (end == s || *end || errno || !(x >= lo && x <= hi))
    return false;
  *v = x;
  return true;
}

/* the slots' options (--slot-*): their defaults the slots' own measures */
typedef struct SlotFlags
{
  const char *dir; /* NULL: the model's folder's slots/ */
  double cap, gb, ram_gb, min, days;
} SlotFlags;

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

static bool open_slots(Server *sv, const char *model, const SlotFlags *sf, char err[OSMOS_ERR])
{
  char dflt[4096];
  const char *slash = strrchr(model, '/');
  snprintf(dflt, sizeof dflt, "%.*s/slots", slash ? (int)(slash - model) : 1, slash ? model : ".");
  const char *dir = sf->dir ? sf->dir : dflt;
  const uint64_t avail = fit_ram_available(), room = avail > sv->ram_kept ? avail - sv->ram_kept : 0;
  const double cap_gb = sf->ram_gb; /* lower only: the room above --fill's is the most */
  const uint64_t ram = cap_gb >= 0 && (uint64_t)(cap_gb * (double)(1ull << 30)) < room ? (uint64_t)(cap_gb * (double)(1ull << 30)) : room;
  const SlotsOptions so = {dir, (uint32_t)sf->cap, ram, sf->gb, (uint32_t)sf->min, sf->days, osmos_identity(sv->e)};
  char serr[OSMOS_ERR] = "";
  if (!(sv->slots = slots_create(&so, serr)))
    return osmos_fail(err, "%s", serr);
  out("NOTE slots: %u conversations kept from an earlier run (%s); RAM for them %.2f GiB (%.2f available, %.0f kept "
      "free)\n", slots_in_dir(sv->slots), dir, ram / (double)(1ull << 30), avail / (double)(1ull << 30),
      sv->ram_kept / (double)(1ull << 30));
  return true;
}

/* --answer-after's rule, from the model's own tokenizer (the token of its text, its blank tokens) */
static bool answer_of(Server *sv, const char *model, const char *after, char err[OSMOS_ERR])
{
  char gerr[OSMOS_ERR] = "", terr[OSMOS_ERR] = "", aerr[OSMOS_ERR] = "";
  Gguf *g = gguf_header(model, gerr);
  Tokenizer *t = g ? tokenizer_create(g, terr) : NULL;
  char *rest = NULL;
  const long n = after[0] == '#' ? strtol(after + 1, &rest, 10) : -1;
  const int32_t id = after[0] == '#' ? (rest > after + 1 && !*rest ? (int32_t)n : -1) : t ? tokenizer_find(t, after) : -1;
  const bool ok = t && answer_rule_create(&sv->answer, t, id, aerr);
  if (!ok)
    osmos_fail(err, "--answer-after: %s", !g ? gerr : !t ? terr : aerr);
  else
    out("NOTE answer-after %d\n", sv->answer.after);
  tokenizer_destroy(t), gguf_destroy(g);
  return ok;
}

/* the sized build's hint's file (build.c): beside the model, named by what decides the build - this version, the
   model file's size and time, every flag - so another build of it, or another version, never takes this one's */
static void fit_hint_path(Server *sv, const char *model, int argc, char **argv)
{
  struct stat st;
  if (stat(model, &st))
    return;
  uint64_t h = 1469598103934665603ull;
  char head[96];
  snprintf(head, sizeof head, "%s %lld %lld", OSMOS_VERSION, (long long)st.st_size, (long long)st.st_mtime);
  for (int i = -1; i < argc; i++)
    for (const char *c = i < 0 ? head : argv[i];; c++)
    {
      h = (h ^ (unsigned char)*c) * 1099511628211ull;
      if (!*c)
        break;
    }
  const char *slash = strrchr(model, '/');
  snprintf(sv->fit_hint, sizeof sv->fit_hint, "%.*s/.osmos-fit-%016llx", slash ? (int)(slash - model) : 1,
           slash ? model : ".", (unsigned long long)h);
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
               "[--no-guess] [--stream-ngram] [--kv-q8] [--no-int8] [--npu <dir> [--npu-share F]] "
               "[--overlay <gguf>] [--lsa <maps>] [--profile] [--rope-stages <from:factor,...>] [--no-fit-hint] [--serve]");
  SlotFlags sf = {opt(argc, argv, "--slot-dir", NULL), 0, 0, 0, 0, 0}; /* (read first: a bad one refused before a build) */
  if (!flag_num(argc, argv, "--slot-cap", 4, 1, 1e6, &sf.cap) || !flag_num(argc, argv, "--slot-gb", 16, 0.001, 1e6, &sf.gb) ||
      !flag_num(argc, argv, "--slot-ram-gb", -1, 0, 1e6, &sf.ram_gb) ||
      !flag_num(argc, argv, "--slot-min", 1024, 1, 1e9, &sf.min) || !flag_num(argc, argv, "--slot-days", 7, 0, 1e5, &sf.days) ||
      sf.cap != (double)(uint32_t)sf.cap || sf.min != (double)(uint32_t)sf.min)
    return die("--slot-cap / --slot-min whole counts, --slot-gb / --slot-ram-gb / --slot-days numbers, each in its range");
  Server sv = {0};
  sv.max_ctx = (uint32_t)atol(opt(argc, argv, "--max-context", "1048576"));
  sv.max_tokens = (uint32_t)atol(opt(argc, argv, "--max-tokens", "2048"));
  if (atol(opt(argc, argv, "--max-tokens", "2048")) < 1) /* 0: a prompt read 0 tokens a step, forever */
    return die("--max-tokens takes a step of 1 token or more");
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
  if (lanes < 1 || lanes > OSMOS_MAX_LANES) /* (a round's lane arrays hold 64) */
    return die("--parallel takes 1 to 64 (OSMOS_MAX_LANES)");
  char *list = strdup(opt(argc, argv, "--eos", "1")), *sv_ = NULL;
  for (char *x = strtok_r(list, ",", &sv_); x && sv.n_eos < SERVE_MAX_EOS; x = strtok_r(NULL, ",", &sv_))
    sv.eos[sv.n_eos++] = (int32_t)atol(x);
  free(list);
  char err[OSMOS_ERR] = "";
  if (opt(argc, argv, "--answer-after", NULL) && !answer_of(&sv, model, opt(argc, argv, "--answer-after", ""), err))
    return die(err);
  const OsmosFiles files = {model, opt(argc, argv, "--engram", NULL), opt(argc, argv, "--overlay", NULL),
                            opt(argc, argv, "--lsa", NULL)};
  OsmosModel *m = osmos_open(&files, err);
  if (!m)
    return die(err);
  const OsmosFacts fx = osmos_facts(m);
  sv.round_tokens = fx.round_tokens;
  sv.window = fx.seed_window;
  const char *share_arg = opt(argc, argv, "--npu-share", NULL);
  const float npu_share = share_arg ? (float)atof(share_arg) : 0;
  if (share_arg && !(npu_share > 0 && npu_share < 1))
    return die("--npu-share takes a share between 0 and 1");
  if (share_arg && !opt(argc, argv, "--npu", NULL))
    return die("--npu-share is nothing without --npu (the NPU's kernels' directory)");
  const OsmosOptions o = {sv.max_ctx,     sv.max_tokens,     (uint64_t)(budget * (double)(1ull << 30)),
                          opt(argc, argv, "--picks", NULL), has(argc, argv, "--no-guess"),
                          opt(argc, argv, "--dspark", NULL), true, sv.ram_kept, has(argc, argv, "--stream-ngram"),
                          opt(argc, argv, "--npu", NULL), npu_share, has(argc, argv, "--kv-q8"),
                          has(argc, argv, "--profile"), opt(argc, argv, "--rope-stages", NULL),
                          has(argc, argv, "--no-int8")};
  if (!has(argc, argv, "--no-fit-hint"))
    fit_hint_path(&sv, model, argc, argv);
  Conv *conv = calloc((size_t)lanes, sizeof *conv);
  if (!conv)
    return die("out of memory");
  if (!serve_build(&sv, conv, lanes, m, o, err))
    return die(err);
  if (osmos_reads_apart(sv.e)) /* (a client's - a test's - what a next turn reuses: its prompt's point, not its answer) */
    out("NOTE reads apart: a turn's earlier answer is read again from its prompt's point, never continued on\n");
  if (!has(argc, argv, "--no-slots") && !open_slots(&sv, model, &sf, err))
    return die(err);
  const int rc = par ? serve_multi(&sv, conv, lanes) : serve_single(&sv, conv);
  if (sv.slots) /* the end: every conversation set aside, RAM's written to the folder */
  {
    char serr[OSMOS_ERR] = "";
    for (int i = 0; i < lanes; i++)
      conv_set_aside(&conv[i], &sv);
    if (!slots_flush(sv.slots, serr))
      out("NOTE slots: %s\n", serr);
    slots_destroy(sv.slots);
  }
  if (has(argc, argv, "--stats") && sv.e) /* the engine's counters at the end (a bench's: where its time waits) */
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
