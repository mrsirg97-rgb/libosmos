#ifndef ROUTE_H
#define ROUTE_H

#include <stdbool.h>
#include <stdint.h>

/* a decode's expert tables on the GPU, any family's (route_tok below) */
#define ROUTE_MAX_EXPERTS 1024
#define ROUTE_MAX_K 64
bool route_tok(const int32_t *idx, const float *wtok, uint32_t n_tok, uint32_t k, uint32_t n_exp, uint32_t chunk,
               const void *const *tab, int32_t *gidx, int32_t *sidx, float *wsel, int32_t *cnt, int32_t *xrow,
               const void **wptr, int32_t *fault, void *stream);

#endif

/*
route_tok
  a decode's routed picks (idx/wtok: n_tok x k, each token's k experts distinct, < n_exp) as the expert matvecs' tables,
  no host between: the picks by expert ascending, an expert's tokens ascending - row r's token gidx[r] and weight
  wsel[r], pick m's row sidx[m]; the entries (each expert's rows in chunks of `chunk`, experts ascending) cnt, xrow (its
  first row) and wptr (3 tables, n_tok x k apart: tab[i n_exp + e] - the membrane's addresses of expert e's gate, up,
  down, a device table [3][n_exp]), then count 0 to n_tok x k (mv_group_tok's skipped entries: a launch to that bound).
  a pick out of range or twice in a token: *fault = 1 (the tables then not used). false, nothing queued, for k 0 or past
  ROUTE_MAX_K, n_exp past ROUTE_MAX_EXPERTS, chunk 0, n_tok 0.
  the tables q8_experts (osmos/gpu/q8.h, exact) and mv_group_tok read as they are.
*/
