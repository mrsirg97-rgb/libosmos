#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "forward_impl.h"
#include "lane.h"
#include "osmos/cpu/lane.h"

/* the lane's CPU half (forward_lane's cpu_threads): its pool, the model's files mapped read-only for its workers (the
   page cache's pages, read in place: no copy), and the buffers it hands rows through - the GPU maps them */

static bool refused(Forward *f, char *err, const char *why)
{
  fwd_cpu_destroy(f);
  return fwd_fail(err, why);
}

bool fwd_cpu_create(Forward *f, int threads, const int *cores, char *err)
{
  const Geometry *g = f->g;
  const uint64_t D = g->n_embd;
  if (!(f->cpu = cpu_lane_create(threads, cores, LANE_PAIRS, (uint32_t)D, (uint32_t)g->expert_ff)))
    return refused(f, err, "the CPU lane's pool (its threads, a core it cannot be pinned to, or a CPU without AVX-512 VNNI)");
  for (int i = 0; i < f->w->n_files; i++)
  {
    struct stat st;
    void *m = fstat(f->fds[i], &st) == 0 ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, f->fds[i], 0)
                                         : MAP_FAILED;
    if (m == MAP_FAILED)
      return refused(f, err, "a model file could not be mapped (the CPU lane)");
    f->map[i] = m, f->map_len[i] = (uint64_t)st.st_size;
  }
  const unsigned mapped = hipHostMallocMapped | hipHostMallocCoherent; /* the GPU reads what the CPU wrote, uncached */
  if (hipHostMalloc((void **)&f->cpu_xf, LANE_PAIRS * D * sizeof(float), 0) != hipSuccess ||
      hipHostMalloc((void **)&f->cpu_rw, LANE_PAIRS * sizeof(float), 0) != hipSuccess ||
      !(f->cpu_x = malloc(LANE_PAIRS * D * sizeof(float))) || !(f->cpu_w = malloc(LANE_PAIRS * sizeof(float))) ||
      hipHostMalloc((void **)&f->cpu_y, LANE_ROWS * D * sizeof(float), mapped) != hipSuccess ||
      hipHostMalloc((void **)&f->cpu_seq, sizeof(uint32_t), mapped) != hipSuccess ||
      hipHostGetDevicePointer((void **)&f->cpu_y_dev, f->cpu_y, 0) != hipSuccess ||
      hipHostGetDevicePointer((void **)&f->cpu_seq_dev, f->cpu_seq, 0) != hipSuccess ||
      !(f->on_cpu = calloc(f->n_expert_units, sizeof *f->on_cpu)))
    return refused(f, err, "memory refused (the CPU lane's buffers)");
  *f->cpu_seq = 0, f->cpu_value = 0;
  return true;
}

void fwd_cpu_destroy(Forward *f)
{
  cpu_lane_destroy(f->cpu); /* (waits for a job under way: nothing below is in use after it) */
  f->cpu = NULL;
  for (int i = 0; i < FWD_MAX_FILES; i++)
    if (f->map[i])
      munmap((void *)f->map[i], f->map_len[i]), f->map[i] = NULL;
  hipHostFree(f->cpu_xf), hipHostFree(f->cpu_rw), hipHostFree(f->cpu_y), hipHostFree(f->cpu_seq);
  free(f->cpu_x), free(f->cpu_w), free(f->on_cpu);
  f->cpu_xf = f->cpu_rw = f->cpu_x = f->cpu_w = f->cpu_y = NULL, f->cpu_seq = NULL, f->on_cpu = NULL;
}

bool fwd_cpu_takes(const Forward *f, uint32_t unit, uint32_t picks)
{
  if (!f->cpu || !f->lane || picks >= (uint32_t)f->g->n_experts || membrane_in_memory(f->mem, unit))
    return false;
  const LayerWeights *L = &f->w->layer[unit / (uint32_t)f->g->n_experts]; /* the contract's types (cpu_lane's) */
  return lane_q8_types(L->exp_gate->type, L->exp_up->type, L->exp_down->type) && L->exp_gate->type == L->exp_up->type;
}

bool fwd_cpu_parts(const Forward *f, uint32_t unit, const void *part[POOL_PARTS])
{
  const PoolUnit *u = &f->units[unit];
  for (int p = 0; p < u->n_parts; p++)
  {
    /* the part's file: its fd the model's buffered one or (the pool reads O_DIRECT since 0.3.0) its direct twin -
       one of the model's or no part at all (refused: never a map past the files') */
    int i = 0;
    while (i < f->w->n_files && f->fds[i] != u->part[p].fd && f->dfds[i] != u->part[p].fd)
      i++;
    if (i == f->w->n_files || !f->map[i])
      return false;
    part[p] = f->map[i] + u->part[p].offset;
    /* its pages asked for now, all at once: the workers' first touches find them read, or reading */
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE), at = (uintptr_t)part[p] & ~(page - 1);
    (void)madvise((void *)at, (uintptr_t)part[p] + u->part[p].bytes - at, MADV_WILLNEED);
  }
  return true;
}

uint64_t forward_cpu_picks(const Forward *f) { return f ? f->cpu_picks : 0; }

bool fwd_cpu_submit(Forward *f, int l, uint32_t picks, uint32_t nc, const int32_t *tok, const int32_t *pair,
                    char *err)
{
  const LayerWeights *L = &f->w->layer[l];
  const uint32_t D = f->g->n_embd;
  if (!cpu_lane_wait(f->cpu)) /* (the last job: long posted, the stream past its wait) */
    return fwd_fail(err, "the CPU lane's experts failed");
  for (uint32_t r = 0; r < picks; r++)
    memcpy(f->cpu_x + (uint64_t)r * D, f->cpu_xf + (uint64_t)tok[r] * D, D * sizeof(float)),
        f->cpu_w[r] = f->cpu_rw[pair[r]];
  const CpuLaneJob job = {L->exp_gate->type, L->exp_down->type, D, f->g->expert_ff, f->g->clamp_exp[l], f->cpu_x,
                          f->cpu_w, picks, f->cpu_e, nc, f->cpu_y, f->cpu_seq, ++f->cpu_value};
  if (!cpu_lane_submit(f->cpu, &job))
    return fwd_fail(err, "the CPU lane refused a layer's experts");
  for (uint32_t k = 0; k < nc; k++)
    f->cpu_picks += f->cpu_e[k].cnt;
  return true;
}
