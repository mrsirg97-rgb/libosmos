#ifndef OSMOS_MEM_LEDGER_HIP_H
#define OSMOS_MEM_LEDGER_HIP_H
/* the ledger's HIP allocations */
#include <hip/hip_runtime_api.h>

#include "osmos/mem/ledger.h"

#ifdef __cplusplus
extern "C" {
#endif

/* an allocation's result in the error shape (osmos/err_hip.h's osmos_hip): the ledger's refusal said as its own -
   "<what>: refused by the ledger: <why>" - a HIP failure as HIP's */
bool ledger_check(const OsmosLedger *l, hipError_t e, char err[OSMOS_ERR], const char *what);

/* hipMalloc / hipHostMalloc counted (a NULL ledger: as HIP's own); refused: hipErrorOutOfMemory, nothing taken, why
   kept. _free / _host_free: hipFree / hipHostFree, what it counted given back */
hipError_t ledger_malloc(OsmosLedger *l, void **p, size_t bytes);
hipError_t ledger_host_malloc(OsmosLedger *l, void **p, size_t bytes, unsigned int flags);
hipError_t ledger_free(OsmosLedger *l, void *p);
hipError_t ledger_host_free(OsmosLedger *l, void *p);

#ifdef __cplusplus
}
#endif

#endif
