// spike (design step): how a cold expert's pages get into the page cache fastest, many at once, with each one's end
// known - the pool's Fetcher. one run = one mode, the file evicted first.
//
//   fetch <file> <mode> [threads=8] [units=600] [unit_mb=3]
//     pread     worker threads read each range into a scratch buffer (a copy, then thrown away)
//     populate  worker threads madvise(MADV_POPULATE_READ) the range of a mapping (no copy)
//     willneed  posix_fadvise(WILLNEED) for every range, then mincore until all are in (this spike's only: it polls)
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

#define PAGE 4096u

static double now_s(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

typedef struct
{
  int fd;
  uint8_t *map;
  uint64_t *off;
  size_t unit;
  int n;
  atomic_int next;
  double *lat;
  int mode;
} Job;

static void *worker(void *arg)
{
  Job *j = arg;
  uint8_t *buf = j->mode == 0 ? malloc(j->unit) : NULL;
  for (int i; (i = atomic_fetch_add(&j->next, 1)) < j->n;)
  {
    const double t0 = now_s();
    if (j->mode == 0)
    {
      size_t got = 0;
      while (got < j->unit)
      {
        const ssize_t r = pread(j->fd, buf + got, j->unit - got, (off_t)(j->off[i] + got));
        if (r <= 0)
          break;
        got += (size_t)r;
      }
    }
    else if (madvise(j->map + j->off[i], j->unit, MADV_POPULATE_READ) != 0)
      perror("madvise");
    j->lat[i] = now_s() - t0;
  }
  free(buf);
  return NULL;
}

static int cmp(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static int resident(uint8_t *map, uint64_t off, size_t len, unsigned char *vec)
{
  if (mincore(map + off, len, vec) != 0)
    return 0;
  for (size_t i = 0; i < len / PAGE; i++)
    if (!(vec[i] & 1))
      return 0;
  return 1;
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    fprintf(stderr, "usage: %s <file> pread|populate|willneed [threads=8] [units=600] [unit_mb=3]\n", argv[0]);
    return 2;
  }
  const int mode = strcmp(argv[2], "pread") == 0 ? 0 : strcmp(argv[2], "populate") == 0 ? 1 : 2;
  const int threads = argc > 3 ? atoi(argv[3]) : 8, n = argc > 4 ? atoi(argv[4]) : 600;
  const size_t unit = (size_t)((argc > 5 ? atof(argv[5]) : 3.0) * 1048576.0) / PAGE * PAGE;
  const int fd = open(argv[1], O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0)
    return perror("open"), 1;
  const size_t fsize = (size_t)st.st_size / PAGE * PAGE;
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); /* cold: before anything maps it */
  uint8_t *map = mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED)
    return perror("mmap"), 1;
  uint64_t *off = malloc((size_t)n * sizeof *off);
  double *lat = calloc((size_t)n, sizeof *lat);
  const uint64_t stride = (fsize - unit) / (uint64_t)n / PAGE * PAGE; /* spread over the file, as experts are */
  for (int i = 0; i < n; i++)
    off[i] = (uint64_t)i * stride;
  const double t0 = now_s();
  if (mode < 2)
  {
    Job j = {fd, map, off, unit, n, 0, lat, mode};
    pthread_t th[64];
    for (int i = 0; i < threads && i < 64; i++)
      pthread_create(&th[i], NULL, worker, &j);
    for (int i = 0; i < threads && i < 64; i++)
      pthread_join(th[i], NULL);
  }
  else
  {
    for (int i = 0; i < n; i++)
      posix_fadvise(fd, (off_t)off[i], (off_t)unit, POSIX_FADV_WILLNEED);
    unsigned char *vec = malloc(unit / PAGE);
    int left = n;
    char *in = calloc((size_t)n, 1);
    while (left)
      for (int i = 0; i < n; i++)
        if (!in[i] && resident(map, off[i], unit, vec))
          in[i] = 1, lat[i] = now_s() - t0, left--;
    free(vec);
    free(in);
  }
  const double wall = now_s() - t0;
  unsigned char *vec = malloc(unit / PAGE);
  int all = 1;
  for (int i = 0; i < n; i++)
    all &= resident(map, off[i], unit, vec);
  qsort(lat, (size_t)n, sizeof *lat, cmp);
  printf("%-8s threads %2d: %d x %.1f MiB in %.2f s = %.2f GB/s; a unit p50 %.1f ms p95 %.1f ms max %.1f ms; %s\n",
         argv[2], mode == 2 ? 0 : threads, n, (double)unit / 1048576.0, wall, (double)unit * n / wall / 1e9,
         lat[n / 2] * 1e3, lat[n * 95 / 100] * 1e3, lat[n - 1] * 1e3, all ? "all resident" : "NOT ALL RESIDENT");
  free(vec);
  free(off);
  free(lat);
  munmap(map, fsize);
  close(fd);
  return 0;
}
