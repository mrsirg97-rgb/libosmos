// the attention core (osmos/gpu/attn_core.h). B1: the schema and its tests alone - every desc refused until B2
// builds the core (docs/attention-core.md)
#include <hip/hip_runtime.h>

extern "C" {
#include "osmos/gpu/attn_core.h"
}

extern "C" AttnCore* attn_core_create(const AttnCoreDesc* d, OsmosLedger* l, char err[OSMOS_ERR]) {
  (void) d, (void) l;
  osmos_fail(err, "the attention core: not built (B1: its schema and tests)");
  return nullptr;
}

extern "C" void attn_core_destroy(AttnCore* c) { (void) c; }

extern "C" bool attn_core_run(const AttnCore* c, const AttnStep* s, void* stream) {
  (void) c, (void) s, (void) stream;
  return false;
}
