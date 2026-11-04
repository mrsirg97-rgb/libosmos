// the model a GPU test reads: ds1017 (ISTA-DASLab's GSQ-RCO, ~/models/ds1017) where it is, else unsloth's UD-IQ1_M
// (~/Models/ds1017/UD-IQ1_M: its own types by layer - IQ1_S and IQ2_S among them, Q6_K attention, an F32 router;
// DESIGN.md, "UD-IQ1_M: the repack"). a test names its tensors for each
#pragma once
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

enum class TestModel { kNone, kGsq, kUd };

constexpr const char* kGsqFirst = "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf";
constexpr const char* kUdFirst = "%s/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf";

// the first shard's path into path (n bytes), and which model it is; kNone (path empty) where neither is readable.
// OSMOS_TEST_UD: another UD-IQ1_M set's first shard in its place (the MXFP8 Engram pack, say) - its tensors UD's
inline TestModel test_model(char* path, size_t n) {
  const char* home = std::getenv("HOME");
  if (const char* ud = std::getenv("OSMOS_TEST_UD")) {
    std::snprintf(path, n, "%s", ud);
    if (access(path, R_OK) == 0) {
      std::printf("  model: %s (OSMOS_TEST_UD)\n", path);
      return TestModel::kUd;
    }
    path[0] = '\0';
    return TestModel::kNone;
  }
  for (const auto& [fmt, which] : {std::pair{kGsqFirst, TestModel::kGsq}, std::pair{kUdFirst, TestModel::kUd}}) {
    std::snprintf(path, n, fmt, home ? home : "");
    if (access(path, R_OK) == 0) {
      std::printf("  model: %s\n", path);
      return which;
    }
  }
  path[0] = '\0';
  return TestModel::kNone;
}

// each expert type's gate and down: ds1017's (IQ1_M, IQ2_XXS, IQ2_XS), UD-IQ1_M's (IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S)
static const std::vector<const char*> kGsqExperts = {
    "blk.0.ffn_gate_exps.weight",  "blk.0.ffn_down_exps.weight",  "blk.10.ffn_gate_exps.weight",
    "blk.10.ffn_down_exps.weight", "blk.17.ffn_gate_exps.weight", "blk.17.ffn_down_exps.weight",
    "blk.23.ffn_down_exps.weight"};
static const std::vector<const char*> kUdExperts = {
    "blk.0.ffn_gate_exps.weight",  "blk.0.ffn_down_exps.weight",  "blk.20.ffn_gate_exps.weight",
    "blk.5.ffn_down_exps.weight",  "blk.38.ffn_gate_exps.weight", "blk.30.ffn_down_exps.weight",
    "blk.20.ffn_down_exps.weight"};
