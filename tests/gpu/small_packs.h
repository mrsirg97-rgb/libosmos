// the small packs an engine-contract test runs on (each skipped where its files are not here): a first shard under
// $HOME, and the options that build it - a streaming family's budget past the pack (every expert resident), none else
#pragma once
#include <cstdlib>
#include <string>
#include <unistd.h>

extern "C" {
#include "osmos/run/engine.h"
}

struct SmallPack {
  const char* name;
  const char* path;  // under $HOME
};

static const SmallPack kSmallPacks[] = {
    {"osmos1.7b", "/Models/osmos/osmos1.7b/osmos1.7b-Q4_0.gguf"},
    {"inkling-2-mini", "/Models/osmos/inkling-2-mini/Inkling-2-Mini-00001-of-00002.gguf"},
};

static std::string small_pack_path(const SmallPack& p) {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : "") + p.path;
}

static bool small_pack_here(const SmallPack& p) { return access(small_pack_path(p).c_str(), R_OK) == 0; }

static OsmosOptions small_pack_options(const OsmosModel* m, uint32_t ctx) {
  OsmosOptions o = {};
  o.max_ctx = ctx, o.budget = osmos_facts(m).streams ? 1ull << 40 : 0;
  return o;
}
