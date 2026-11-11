// a kernel file's launch helpers, one definition for every file: an elementwise kernel's block and the grid for n
// threads' work, the stream a C caller hands over, a launch's error
#pragma once
#include <hip/hip_runtime.h>

namespace launches {

constexpr int kThreads = 256;

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

}  // namespace launches
