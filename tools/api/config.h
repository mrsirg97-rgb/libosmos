#pragma once
#include <limits.h>
#include <stdbool.h>

#include "api.h"

#define CONFIG_ARGS_MAX 40 /* config_argv's every flag and its NULL (36 with all: room past it) */

/* a pack's osmos.json (the installer's: tools/install/layout.py) - what the engine is started with */
typedef struct Config
{
  char pack[64];
  char model[PATH_MAX], picks[PATH_MAX], drafter[PATH_MAX], overlay[PATH_MAX], lsa[PATH_MAX]; /* "" when none */
  double budget_gib, drafted_budget_gib; /* a fixed budget, plain and drafting; 0: sized to the GPU (auto) */
  bool fill_set;  /* fill_gib given: osmos-serve's --fill (the RAM kept free); else its own 8 GiB */
  double fill_gib;
  bool no_guess;
  bool fill_orphans; /* 0.7.3 (#47): an orphaned tool call given LOOP_ORPHAN_TEXT as its result (loop.h); else flagged */
  bool preserve_thinking; /* 0.11.3: a request's chat_template_kwargs.preserve_thinking unset, this (LFM3's: its
                             template drops an earlier turn's reasoning - a long tool loop then read again at the
                             next user turn; a pack's own choice, its training's) */
  bool stream_ngram; /* osmos-serve's --stream-ngram */
  bool kv_q8;        /* osmos-serve's --kv-q8 (Inkling's full layers' cached values Q8, not Q5) */
  bool profile;      /* osmos-serve's --profile (a measurement's build: a step's sections timed, said at its end) */
  char rope_stages[256]; /* osmos-serve's --rope-stages (osmos1.7b's YaRN stages, "from:factor,..."): "" none */
  double temperature, top_p; /* 0.14.0: the pack's sampling, a request's unset (its release's: Inkling-2-Mini's 1.0);
                                temperature 0 greedy (the default), top_p 1 */
  unsigned top_k;            /* 0.15.2: likewise (Qwen3.8's 20, osmos1.7b's 50); 0 every token */
  char npu[PATH_MAX]; /* the NPU's kernels' directory (osmos-serve's --npu): "" none */
  double npu_share;   /* its share of a layer's pairs (--npu-share): 0 osmos-serve's own */
  unsigned max_context, parallel;
  unsigned max_tokens; /* a prompt step's tokens (osmos-serve's --max-tokens); 0: osmos-serve's own */
  char num[7][32]; /* config_argv's numbers as text */
  char eng[PATH_MAX];
} Config;

bool config_read(const char *path, Config *c, char err[OSMOS_ERR]);
bool config_argv(Config *c, const char *engine, int eos, const char *answer_after, char *argv[CONFIG_ARGS_MAX],
                 char err[OSMOS_ERR]);
