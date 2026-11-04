#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lane.h"
#include "osmos/cpu/expert.h"
#include "osmos/cpu/q8.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"

/* the type table against every host-side list of types: for each id, gguf_bytes, the CPU lane's matvec and int8
   experts and the lane contract's ids take it exactly when its row names that path; the rows' own invariants.
   test_qtype_gpu holds the kernels to the same table */

#define ID_SWEEP 256 /* past every ggml type id (GGML_TYPE_COUNT 43 at llama.cpp 3cf03257, maya's pin): any id a file names */
#define COLS 256     /* one block of a 256-value type; whole blocks of every type in the table */
#define ROWS 2

static int failures;

/* a section's line only when it found nothing wrong */
#define SAID_IF_CLEAN(before, ...) (failures == (before) ? printf(__VA_ARGS__) : 0)

static void check(bool ok, uint32_t id, const char *what)
{
  if (!ok)
  {
    const QTypeInfo *t = qtype_info(id);
    fprintf(stderr, "  type %u (%s): %s\n", id, t ? t->name : "not in the table", what);
    failures++;
  }
}

static void rows_and_ids(void)
{
  const int before = failures;
  size_t n = 0;
  for (uint32_t id = 0; id < ID_SWEEP; id++)
  {
    const QTypeInfo *t = qtype_info(id);
    if (!t)
    {
      check(qtype_paths(id) == 0, id, "no row, yet paths");
      continue;
    }
    n++;
    check(t->id == id, id, "its row names another id");
    check(t->values > 0 && t->bytes > 0, id, "a block of no values or no bytes");
    check(COLS % t->values == 0, id, "COLS not whole blocks of it");
    check(t->paths != 0, id, "a row no path reads");
    check(t->name && t->name[0], id, "no name");
    /* the lane's contract: which side computes an expert changes no bit, so each takes what the other takes */
    check(!(t->paths & QTYPE_Q8) == !(t->paths & QTYPE_CPU_Q8), id, "int8 experts on one side only");
    check(!(t->paths & QTYPE_MV_TOK) || (t->paths & QTYPE_MV), id, "an expert's tokens without the matvec");
  }
  check(qtype_info(UINT32_MAX) == NULL, UINT32_MAX, "a row for an id no file holds");
  for (uint32_t id = 0; id < ID_SWEEP; id++) /* each row found by its name, and only it */
    if (qtype_info(id))
      check(qtype_named(qtype_info(id)->name) == qtype_info(id), id, "its name finds another row");
  check(!qtype_named("Q4_K") && !qtype_named(NULL) && !qtype_named(""), 0, "a name no row has found");
  SAID_IF_CLEAN(before, "  %zu rows, ids unique and below %d\n", n, ID_SWEEP);
}

static void file_sizes(void)
{
  const int before = failures;
  for (uint32_t id = 0; id < ID_SWEEP; id++)
  {
    const QTypeInfo *t = qtype_info(id);
    const bool file = t && (t->paths & QTYPE_FILE);
    if (!file)
    {
      check(gguf_bytes(id, COLS) == 0, id, "gguf_bytes takes a type no file may hold");
      continue;
    }
    check(gguf_bytes(id, t->values) == t->bytes, id, "gguf_bytes: a block's bytes not the table's");
    check(gguf_bytes(id, 3ull * t->values) == 3ull * t->bytes, id, "gguf_bytes: three blocks not three times one");
    if (t->values > 1)
      check(gguf_bytes(id, t->values + 1) == 0, id, "gguf_bytes takes a part block");
  }
  SAID_IF_CLEAN(before, "  gguf_bytes: the table's file types, nothing else\n");
}

static void cpu_paths(void)
{
  const int before = failures;
  static uint8_t w[ROWS * COLS * 8]; /* zeros: wider than a row of any type's bytes */
  static int8_t q[COLS];
  static float d[COLS / LANE_Q8_VALUES], y[ROWS];
  const bool q8 = cpu_q8_ready();
  for (uint32_t id = 0; id < ID_SWEEP; id++)
  {
    const uint32_t paths = qtype_paths(id);
    check(cpu_mv_known(id) == !!(paths & QTYPE_CPU_MV), id, "cpu_mv_known disagrees with QTYPE_CPU_MV");
    check(!!lane_q8_type(id) == !!(paths & QTYPE_Q8), id, "lane_q8_type disagrees with QTYPE_Q8");
    if (q8)
      check(cpu_q8_rows(id, w, COLS, 0, ROWS, q, d, 1, y, ROWS) == !!(paths & QTYPE_CPU_Q8), id,
            "cpu_q8_rows disagrees with QTYPE_CPU_Q8");
  }
  SAID_IF_CLEAN(before, "  the CPU lane's matvec and int8 experts, the lane's ids: the table's%s\n",
         q8 ? "" : " (cpu_q8_rows: no AVX-512 kernels here, not run)");
}

int main(void)
{
  printf("[test_qtype] start\n");
  rows_and_ids();
  file_sizes();
  cpu_paths();
  if (failures)
  {
    fprintf(stderr, "[test_qtype] %d failures\n", failures);
    return 1;
  }
  printf("[test_qtype] ok\n");
  return 0;
}
