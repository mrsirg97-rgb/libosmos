// a scope with the int8 decode off (OSMOS_Q8=0: q8_enabled, read at every step by the families that call it) - an
// f16 path's contract or a reference taken on it held there - then the run's own setting back
#pragma once
#include <cstdlib>
#include <string>

struct Q8Off {
  std::string was;
  bool had;
  Q8Off() : had(std::getenv("OSMOS_Q8") != nullptr) {
    if (had) was = std::getenv("OSMOS_Q8");
    setenv("OSMOS_Q8", "0", 1);
  }
  ~Q8Off() { had ? setenv("OSMOS_Q8", was.c_str(), 1) : unsetenv("OSMOS_Q8"); }
  Q8Off(const Q8Off&) = delete;
  Q8Off& operator=(const Q8Off&) = delete;
};
