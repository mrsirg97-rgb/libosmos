#ifndef CPU_LANE_H
#define CPU_LANE_H

#include <stdbool.h>
#include <stdint.h>

#define CPU_LANE_THREADS_MAX 64 /* a pool's most workers */
#define CPU_LANE_FAILED 0x80000000u /* a job's posted value with this bit: it failed (its outputs not written) */

typedef struct CpuLane CpuLane;

/* an expert of a job: its three matrices in host memory (a GGUF's slices: gate and up FF rows of D, down D rows of
   FF) and its rows of the job's x, row0 .. row0 + cnt */
typedef struct CpuLaneExpert
{
  const void *gate, *up, *down;
  uint32_t row0, cnt;
} CpuLaneExpert;

typedef struct CpuLaneJob
{
  uint32_t type_gu, type_down; /* the experts' types (lane_q8_type's) */
  uint32_t dim, ff;            /* D (the FFN's input and output), FF (the experts' hidden) */
  float lim;                   /* the activation's clamp */
  const float *x;              /* n_rows rows of dim: row r a (token, expert) pair's FFN input */
  const float *scale;          /* row r's routed weight */
  uint32_t n_rows;
  const CpuLaneExpert *e;
  uint32_t n_e;
  float *y;           /* row r's output at y + r * dim */
  uint32_t *seq;      /* where `value` is posted when the job is done (a release store), | CPU_LANE_FAILED if not */
  uint32_t value;
} CpuLaneJob;

CpuLane *cpu_lane_create(int threads, uint32_t max_rows, uint32_t dim, uint32_t ff);
void cpu_lane_destroy(CpuLane *c);
bool cpu_lane_submit(CpuLane *c, const CpuLaneJob *job);
bool cpu_lane_wait(CpuLane *c);

#endif

/*
cpu lane
  the CPU's half of forward_lane: a layer's experts that are not in GPU memory computed by a pool of threads as the
  lane computes them on the GPU (q8_round, q8_gate_up, q8_round, q8_group_tok - kernels/lane.h's contract): each row
  of x to int8 (cpu_q8_round), gate and up (cpu_q8_rows) and lane_swiglu with the row's weight, the activations to
  int8, then down - the GPU's bits for each row, whichever thread takes which rows. the GPU reads the outputs where
  they are (y and seq in host memory it maps) once seq holds the job's value.

cpu_lane_create
  `threads` workers (1 .. CPU_LANE_THREADS_MAX), each job at most max_rows rows of dim and ff (multiples of 256).
  NULL on a CPU without the kernels (cpu_q8_ready), a size it cannot take, or a thread or memory refused.

cpu_lane_destroy
  waits for a job under way, then joins the workers and frees. NULL is nothing to do.

cpu_lane_submit
  queues one job and returns at once: the workers round x, then each expert's gate and up rows (the experts' rows
  split over the workers), then the activations, then the down rows; the last worker done posts seq. the job's
  memory (x, scale, e, y, seq) must stay until it is posted. false, nothing queued and seq untouched, for a job
  under way (one at a time: cpu_lane_wait first), a type not lane_q8_type's, dim or ff not the pool's, rows past
  max_rows or an expert's rows past n_rows, or no outputs.

cpu_lane_wait
  waits for the job under way (none: at once); true if it, and every job since the last wait, succeeded.
*/
