/* osmos/gpu/lsa_close.h: a closer's thread buckets queued blocks while the GPU goes on (a slot a job) */
#include <hip/hip_runtime.h>
#include <pthread.h>
#include <stdlib.h>

#include "osmos/gpu/bufs.h"
#include "osmos/gpu/lsa_close.h"
#include "osmos/lsa.h"

#define F sizeof(float)

typedef struct
{
  float *ik, *cent; /* pinned: the blocks' index vectors in, their centroids out */
  int32_t *mem;     /* pinned: their members out */
  hipEvent_t copied; /* after the index vectors' copy, on the queuing stream */
  /* the job (valid while queued or running) */
  uint32_t b0, k;
  float *dev_cent;
  int32_t *dev_mem;
  bool *broken;
  bool failed; /* done, and failed: applied (and cleared) at the join */
} Slot;

struct LsaCloser
{
  pthread_mutex_t mu;
  pthread_cond_t go, done;
  pthread_t th;
  bool started, stop;
  hipStream_t st; /* the buckets' copies to the device */
  GpuBufs bufs;
  uint32_t n_slots, blocks_max;
  Slot *slot;
  uint32_t *queue, head, count; /* the slots queued, in order (a ring of n_slots) */
  uint32_t busy;                /* the slots queued or running */
  bool *free_slot;
};

/* a job: its copy waited for, each block bucketed in order, the buckets to the device; false: it failed */
static bool run(LsaCloser *c, Slot *s)
{
  if (hipEventSynchronize(s->copied) != hipSuccess)
    return false;
  for (uint32_t i = 0; i < s->k; i++)
    if (!lsa_block(s->ik + (uint64_t)i * LSA_WINDOW * LSA_DIM, s->b0 + i, s->cent + (uint64_t)i * LSA_BUCKETS * LSA_DIM,
                   s->mem + (uint64_t)i * LSA_BUCKETS * LSA_CAP))
      return false;
  return hipMemcpyAsync(s->dev_cent, s->cent, (uint64_t)s->k * LSA_BUCKETS * LSA_DIM * F, hipMemcpyHostToDevice, c->st) ==
             hipSuccess &&
         hipMemcpyAsync(s->dev_mem, s->mem, (uint64_t)s->k * LSA_BUCKETS * LSA_CAP * 4, hipMemcpyHostToDevice, c->st) ==
             hipSuccess &&
         hipStreamSynchronize(c->st) == hipSuccess;
}

static void *work(void *arg)
{
  LsaCloser *c = arg;
  pthread_mutex_lock(&c->mu);
  for (;;)
  {
    while (!c->count && !c->stop)
      pthread_cond_wait(&c->go, &c->mu);
    if (!c->count)
      break; /* (stopped, nothing left) */
    const uint32_t i = c->queue[c->head];
    c->head = (c->head + 1) % c->n_slots, c->count--;
    pthread_mutex_unlock(&c->mu);
    const bool ok = run(c, &c->slot[i]);
    pthread_mutex_lock(&c->mu);
    c->slot[i].failed = !ok, c->busy--;
    if (ok)
      c->free_slot[i] = true; /* (a failed one's freed at the join, its *broken set there) */
    pthread_cond_broadcast(&c->done);
  }
  pthread_mutex_unlock(&c->mu);
  return NULL;
}

LsaCloser *lsa_closer_create(OsmosLedger *l, uint32_t slots, uint32_t blocks_max)
{
  if (!slots || !blocks_max)
    return NULL;
  LsaCloser *c = calloc(1, sizeof *c);
  if (!c)
    return NULL;
  c->bufs.ledger = l, c->n_slots = slots, c->blocks_max = blocks_max;
  pthread_mutex_init(&c->mu, NULL);
  pthread_cond_init(&c->go, NULL);
  pthread_cond_init(&c->done, NULL);
  bool ok = (c->slot = calloc(slots, sizeof *c->slot)) && (c->queue = calloc(slots, sizeof *c->queue)) &&
            (c->free_slot = calloc(slots, sizeof *c->free_slot)) &&
            hipStreamCreateWithFlags(&c->st, hipStreamNonBlocking) == hipSuccess;
  for (uint32_t i = 0; ok && i < slots; i++)
  {
    Slot *s = &c->slot[i];
    ok = gpu_bufs_host(&c->bufs, l, &s->ik, (uint64_t)blocks_max * LSA_WINDOW * LSA_DIM * F) &&
         gpu_bufs_host(&c->bufs, l, &s->cent, (uint64_t)blocks_max * LSA_BUCKETS * LSA_DIM * F) &&
         gpu_bufs_host(&c->bufs, l, &s->mem, (uint64_t)blocks_max * LSA_BUCKETS * LSA_CAP * 4) &&
         hipEventCreateWithFlags(&s->copied, hipEventDisableTiming) == hipSuccess;
    c->free_slot[i] = ok;
  }
  ok = ok && pthread_create(&c->th, NULL, work, c) == 0;
  c->started = ok;
  if (!ok)
    return lsa_closer_destroy(c), NULL;
  return c;
}

bool lsa_closer_join(LsaCloser *c)
{
  if (!c)
    return true;
  pthread_mutex_lock(&c->mu);
  while (c->busy)
    pthread_cond_wait(&c->done, &c->mu);
  bool ok = true;
  for (uint32_t i = 0; i < c->n_slots; i++)
    if (c->slot[i].failed)
    {
      if (c->slot[i].broken)
        *c->slot[i].broken = true;
      c->slot[i].failed = false, c->free_slot[i] = true, ok = false;
    }
  pthread_mutex_unlock(&c->mu);
  return ok;
}

void lsa_closer_destroy(LsaCloser *c)
{
  if (!c)
    return;
  if (c->started)
  {
    (void)lsa_closer_join(c); /* (a pending job's buckets written while their memory lives) */
    pthread_mutex_lock(&c->mu);
    c->stop = true;
    pthread_cond_broadcast(&c->go);
    pthread_mutex_unlock(&c->mu);
    pthread_join(c->th, NULL);
  }
  for (uint32_t i = 0; c->slot && i < c->n_slots; i++)
    if (c->slot[i].copied)
      hipEventDestroy(c->slot[i].copied);
  if (c->st)
    hipStreamDestroy(c->st);
  gpu_bufs_free(&c->bufs);
  free(c->slot), free(c->queue), free(c->free_slot);
  pthread_cond_destroy(&c->go);
  pthread_cond_destroy(&c->done);
  pthread_mutex_destroy(&c->mu);
  free(c);
}

bool lsa_closer_queue(LsaCloser *c, const float *dev_ik, uint32_t b0, uint32_t k, float *dev_cent, int32_t *dev_mem,
                      bool *broken, void *stream)
{
  if (!c || !k || k > c->blocks_max)
    return false;
  pthread_mutex_lock(&c->mu);
  uint32_t i;
  for (;;)
  {
    for (i = 0; i < c->n_slots && !c->free_slot[i]; i++)
      ;
    if (i < c->n_slots)
      break;
    if (!c->busy) /* none free and none running: every one failed, held for the join - closed by the caller */
      return pthread_mutex_unlock(&c->mu), false;
    pthread_cond_wait(&c->done, &c->mu); /* every slot in flight: the next done frees one */
  }
  c->free_slot[i] = false;
  pthread_mutex_unlock(&c->mu);
  Slot *s = &c->slot[i];
  s->b0 = b0, s->k = k, s->dev_cent = dev_cent, s->dev_mem = dev_mem, s->broken = broken, s->failed = false;
  const bool copied = hipMemcpyAsync(s->ik, dev_ik, (uint64_t)k * LSA_WINDOW * LSA_DIM * F, hipMemcpyDeviceToHost,
                                     (hipStream_t)stream) == hipSuccess;
  if (!copied || hipEventRecord(s->copied, (hipStream_t)stream) != hipSuccess)
  {
    if (copied)
      (void)hipStreamSynchronize((hipStream_t)stream);
    pthread_mutex_lock(&c->mu);
    c->free_slot[i] = true;
    pthread_mutex_unlock(&c->mu);
    return false;
  }
  pthread_mutex_lock(&c->mu);
  c->queue[(c->head + c->count) % c->n_slots] = i, c->count++, c->busy++;
  pthread_cond_signal(&c->go);
  pthread_mutex_unlock(&c->mu);
  return true;
}
