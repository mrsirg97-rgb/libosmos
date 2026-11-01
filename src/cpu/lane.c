#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "lane.h"
#include "substrate/cpu/lane.h"
#include "substrate/cpu/q8.h"

/* the pool: workers asleep between jobs (a condition), within one moving through its phases together (a spinning
   barrier: a phase is tens of microseconds) - x to int8; gate, up and the activation; the activations to int8;
   down. each phase's rows split evenly over the workers, the experts' rows one range: any split the same bits (each
   row's sums its own: test_cpu_lane) */

#define QK 256

struct CpuLane
{
  int n;
  pthread_t th[CPU_LANE_THREADS_MAX];
  uint32_t max_rows, dim, ff;
  int8_t *q, *qh;          /* the rows of x, then the activations, as int8 */
  float *qd, *qhd;         /* their blocks' scales */
  float *g, *u, *h;        /* each row's gate, up and activation */
  pthread_mutex_t mu;
  pthread_cond_t wake, idle;
  CpuLaneJob job;
  uint64_t gen;            /* jobs given out: a worker runs each once */
  bool busy, quit, all_ok; /* (mu's) */
  int left;                /* workers yet to finish the job (atomic) */
  bool failed;             /* a step refused (atomic) */
  int arrived;             /* the barrier: workers at it, and its phase (atomic) */
  unsigned phase;
};

/* every worker here before any goes on: the last one turns the phase */
static void barrier(CpuLane *c)
{
  const unsigned ph = __atomic_load_n(&c->phase, __ATOMIC_ACQUIRE);
  if (__atomic_add_fetch(&c->arrived, 1, __ATOMIC_ACQ_REL) == c->n)
  {
    __atomic_store_n(&c->arrived, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&c->phase, ph + 1, __ATOMIC_RELEASE);
    return;
  }
  while (__atomic_load_n(&c->phase, __ATOMIC_ACQUIRE) == ph)
    _mm_pause();
}

static void fail(CpuLane *c) { __atomic_store_n(&c->failed, true, __ATOMIC_RELAXED); }

/* worker id's share of `total` items */
static void share(int id, int n, uint64_t total, uint64_t *a, uint64_t *b)
{
  *a = total * (uint64_t)id / (uint64_t)n, *b = total * (uint64_t)(id + 1) / (uint64_t)n;
}

/* expert rows a .. b of the experts' rows laid end to end (each `rows` long): fn on each expert's part */
typedef void (*part_fn)(CpuLane *c, const CpuLaneExpert *e, uint32_t r0, uint32_t r1);

static void parts(CpuLane *c, uint32_t rows, uint64_t a, uint64_t b, part_fn fn)
{
  for (uint32_t k = (uint32_t)(a / rows); k < c->job.n_e && (uint64_t)k * rows < b; k++)
  {
    const uint64_t lo = (uint64_t)k * rows, r0 = a > lo ? a - lo : 0, r1 = b < lo + rows ? b - lo : rows;
    fn(c, &c->job.e[k], (uint32_t)r0, (uint32_t)r1);
  }
}

static void gate_up(CpuLane *c, const CpuLaneExpert *e, uint32_t r0, uint32_t r1)
{
  const CpuLaneJob *j = &c->job;
  const uint32_t D = j->dim, F = j->ff;
  const int8_t *q = c->q + (uint64_t)e->row0 * D;
  const float *qd = c->qd + (uint64_t)e->row0 * (D / LANE_Q8_VALUES);
  float *g = c->g + (uint64_t)e->row0 * F, *u = c->u + (uint64_t)e->row0 * F, *h = c->h + (uint64_t)e->row0 * F;
  if (!cpu_q8_rows(j->type_gu, e->gate, D, r0, r1, q, qd, e->cnt, g, F) ||
      !cpu_q8_rows(j->type_gu, e->up, D, r0, r1, q, qd, e->cnt, u, F))
    return fail(c);
  for (uint32_t t = 0; t < e->cnt; t++)
    for (uint32_t r = r0; r < r1; r++)
    {
      const uint64_t i = (uint64_t)t * F + r;
      h[i] = lane_swiglu(g[i], u[i], j->lim, j->scale[e->row0 + t]);
    }
}

static void down(CpuLane *c, const CpuLaneExpert *e, uint32_t r0, uint32_t r1)
{
  const CpuLaneJob *j = &c->job;
  const uint32_t D = j->dim, F = j->ff;
  if (!cpu_q8_rows(j->type_down, e->down, F, r0, r1, c->qh + (uint64_t)e->row0 * F,
                   c->qhd + (uint64_t)e->row0 * (F / LANE_Q8_VALUES), e->cnt, j->y + (uint64_t)e->row0 * D, D))
    fail(c);
}

static void run(CpuLane *c, int id)
{
  const CpuLaneJob *j = &c->job;
  const uint32_t D = j->dim, F = j->ff;
  uint64_t a, b;
  for (uint32_t r = (uint32_t)id; r < j->n_rows; r += (uint32_t)c->n) /* x's rows to int8 */
    if (!cpu_q8_round(j->x + (uint64_t)r * D, 1, D, D, c->q + (uint64_t)r * D,
                      c->qd + (uint64_t)r * (D / LANE_Q8_VALUES)))
      fail(c);
  barrier(c);
  share(id, c->n, (uint64_t)j->n_e * F, &a, &b);
  parts(c, F, a, b, gate_up);
  barrier(c);
  for (uint32_t r = (uint32_t)id; r < j->n_rows; r += (uint32_t)c->n) /* the activations to int8 */
    if (!cpu_q8_round(c->h + (uint64_t)r * F, 1, F, F, c->qh + (uint64_t)r * F,
                      c->qhd + (uint64_t)r * (F / LANE_Q8_VALUES)))
      fail(c);
  barrier(c);
  share(id, c->n, (uint64_t)j->n_e * D, &a, &b);
  parts(c, D, a, b, down);
}

typedef struct Arg
{
  CpuLane *c;
  int id;
} Arg;

static void *worker(void *p)
{
  Arg *arg = p;
  CpuLane *c = arg->c;
  const int id = arg->id;
  free(arg);
  uint64_t seen = 0;
  for (;;)
  {
    pthread_mutex_lock(&c->mu);
    while (c->gen == seen && !c->quit)
      pthread_cond_wait(&c->wake, &c->mu);
    if (c->quit)
    {
      pthread_mutex_unlock(&c->mu);
      return NULL;
    }
    seen = c->gen;
    pthread_mutex_unlock(&c->mu);
    run(c, id);
    if (__atomic_sub_fetch(&c->left, 1, __ATOMIC_ACQ_REL) == 0) /* the last one done: post, then the pool idle */
    {
      const bool failed = __atomic_load_n(&c->failed, __ATOMIC_RELAXED);
      __atomic_store_n(c->job.seq, c->job.value | (failed ? CPU_LANE_FAILED : 0), __ATOMIC_RELEASE);
      pthread_mutex_lock(&c->mu);
      c->all_ok = c->all_ok && !failed, c->busy = false;
      pthread_cond_broadcast(&c->idle);
      pthread_mutex_unlock(&c->mu);
    }
  }
}

CpuLane *cpu_lane_create(int threads, uint32_t max_rows, uint32_t dim, uint32_t ff)
{
  if (threads < 1 || threads > CPU_LANE_THREADS_MAX || !max_rows || !dim || dim % QK || !ff || ff % QK ||
      !cpu_q8_ready())
    return NULL;
  CpuLane *c = calloc(1, sizeof *c);
  if (!c)
    return NULL;
  c->max_rows = max_rows, c->dim = dim, c->ff = ff, c->all_ok = true;
  const uint64_t rd = (uint64_t)max_rows * dim, rf = (uint64_t)max_rows * ff;
  c->q = malloc(rd), c->qh = malloc(rf);
  c->qd = malloc(rd / LANE_Q8_VALUES * sizeof(float)), c->qhd = malloc(rf / LANE_Q8_VALUES * sizeof(float));
  c->g = malloc(rf * sizeof(float)), c->u = malloc(rf * sizeof(float)), c->h = malloc(rf * sizeof(float));
  pthread_mutex_init(&c->mu, NULL), pthread_cond_init(&c->wake, NULL), pthread_cond_init(&c->idle, NULL);
  if (!c->q || !c->qh || !c->qd || !c->qhd || !c->g || !c->u || !c->h)
    return cpu_lane_destroy(c), NULL;
  for (int i = 0; i < threads; i++)
  {
    Arg *a = malloc(sizeof *a);
    if (a)
      *a = (Arg){c, i};
    if (!a || pthread_create(&c->th[i], NULL, worker, a) != 0)
      return free(a), cpu_lane_destroy(c), NULL;
    c->n = i + 1;
  }
  return c;
}

void cpu_lane_destroy(CpuLane *c)
{
  if (!c)
    return;
  cpu_lane_wait(c);
  pthread_mutex_lock(&c->mu);
  c->quit = true;
  pthread_cond_broadcast(&c->wake);
  pthread_mutex_unlock(&c->mu);
  for (int i = 0; i < c->n; i++)
    pthread_join(c->th[i], NULL);
  pthread_mutex_destroy(&c->mu), pthread_cond_destroy(&c->wake), pthread_cond_destroy(&c->idle);
  free(c->q), free(c->qh), free(c->qd), free(c->qhd), free(c->g), free(c->u), free(c->h);
  free(c);
}

static bool valid(const CpuLane *c, const CpuLaneJob *j)
{
  if (!lane_q8_type(j->type_gu) || !lane_q8_type(j->type_down) || j->dim != c->dim || j->ff != c->ff || !j->x ||
      !j->scale || !j->e || !j->n_e || !j->y || !j->seq || !j->n_rows || j->n_rows > c->max_rows)
    return false;
  for (uint32_t k = 0; k < j->n_e; k++)
    if (!j->e[k].gate || !j->e[k].up || !j->e[k].down || !j->e[k].cnt || j->e[k].row0 + j->e[k].cnt > j->n_rows)
      return false;
  return true;
}

bool cpu_lane_submit(CpuLane *c, const CpuLaneJob *job)
{
  if (!c || !job || !valid(c, job))
    return false;
  pthread_mutex_lock(&c->mu);
  const bool free_ = !c->busy;
  if (free_)
  {
    c->job = *job, c->busy = true, c->gen++;
    __atomic_store_n(&c->failed, false, __ATOMIC_RELAXED);
    __atomic_store_n(&c->left, c->n, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&c->wake);
  }
  pthread_mutex_unlock(&c->mu);
  return free_;
}

bool cpu_lane_wait(CpuLane *c)
{
  if (!c)
    return false;
  pthread_mutex_lock(&c->mu);
  while (c->busy)
    pthread_cond_wait(&c->idle, &c->mu);
  const bool ok = c->all_ok;
  c->all_ok = true;
  pthread_mutex_unlock(&c->mu);
  return ok;
}
