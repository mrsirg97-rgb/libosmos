#pragma once
#include <limits.h>
#include <stdbool.h>

#include "api.h"

#define CONFIG_ARGS_MAX 32

/* a pack's osmos.json (the installer's: tools/install/layout.py) - what the engine is started with */
typedef struct Config
{
  char pack[64];
  char model[PATH_MAX], picks[PATH_MAX], drafter[PATH_MAX], overlay[PATH_MAX]; /* "" when none */
  double budget_gib, drafted_budget_gib; /* a fixed budget, plain and drafting; 0: sized to the GPU (auto) */
  bool no_guess;
  unsigned max_context, parallel;
  char num[4][32]; /* config_argv's numbers as text */
  char eng[PATH_MAX];
} Config;

bool config_read(const char *path, Config *c, char err[API_ERR]);
bool config_argv(Config *c, const char *engine, int eos, char *argv[CONFIG_ARGS_MAX], char err[API_ERR]);
