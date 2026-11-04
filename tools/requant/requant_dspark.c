/* requant_dspark: a DSpark drafter pack's routed experts requantized (MXFP4 -> an IQ type: IQ2_XS by default) with
   ggml's quantizers (llama.cpp, MIT: libggml-base), every other tensor copied as it is - a smaller drafter, its drafts
   checked by the target's verify as ever (a draft's precision is its acceptance, never the output's bits).

     requant_dspark <pack dir> <out dir> [IQ2_XS|IQ2_XXS|IQ2_S|IQ1_S] [threads]

   out dir/dspark.bin and dspark.json in the pack's form ([name, type, offset, bytes, [dims]]), each tensor 256-byte
   aligned. the importance weights: uniform (ggml's IQ2 quantizers need some; each value is then weighted by its own
   magnitude inside them) */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ggml.h"
#include "osmos/model/dspark.h"
#include "osmos/model/qtype.h"

#define ALIGN 256

typedef struct Job
{
  const DsparkIndex *ix;
  int in_fd, out_fd;
  enum ggml_type to;
  uint64_t *out_off, *out_bytes; /* each tensor's place in the new bin */
  uint32_t *out_type;
  atomic_uint next;
  atomic_int failed;
} Job;

static bool is_expert(const DsparkTensor *t)
{
  return t->type == QTYPE_MXFP4 && strstr(t->name, ".ffn.experts.") != NULL;
}

static void *work(void *arg)
{
  Job *j = arg;
  float *f = NULL, *imat = NULL;
  uint8_t *in = NULL, *outb = NULL;
  size_t cap_f = 0, cap_in = 0, cap_out = 0, cap_im = 0;
  for (uint32_t i; (i = atomic_fetch_add(&j->next, 1)) < j->ix->n && !atomic_load(&j->failed);)
  {
    const DsparkTensor *t = &j->ix->t[i];
    if (cap_in < t->bytes && !(in = realloc(in, cap_in = t->bytes)))
      goto fail;
    if (pread(j->in_fd, in, t->bytes, (off_t)t->offset) != (ssize_t)t->bytes)
      goto fail;
    const void *data = in;
    uint64_t bytes = t->bytes;
    if (is_expert(t))
    {
      const int64_t cols = (int64_t)t->dims[t->n_dims - 1];
      const int64_t rows = (int64_t)(t->bytes / ggml_row_size(GGML_TYPE_MXFP4, cols));
      const size_t nv = (size_t)(rows * cols);
      if (cols % 256 || (cap_f < nv && !(f = realloc(f, (cap_f = nv) * sizeof *f))) ||
          (cap_im < (size_t)cols && !(imat = realloc(imat, (cap_im = (size_t)cols) * sizeof *imat))))
        goto fail;
      for (int64_t c = 0; c < cols; c++)
        imat[c] = 1.0f;
      ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(in, f, (int64_t)nv);
      bytes = ggml_row_size(j->to, cols) * (size_t)rows;
      if (cap_out < bytes && !(outb = realloc(outb, cap_out = bytes)))
        goto fail;
      if (ggml_quantize_chunk(j->to, f, outb, 0, rows, cols, imat) != bytes)
        goto fail;
      data = outb;
      j->out_type[i] = (uint32_t)j->to;
    }
    if (bytes != j->out_bytes[i] || pwrite(j->out_fd, data, bytes, (off_t)j->out_off[i]) != (ssize_t)bytes)
      goto fail;
    if (is_expert(t) && i % 96 == 0)
      fprintf(stderr, "  %s\n", t->name);
    continue;
  fail:
    fprintf(stderr, "requant_dspark: %s failed\n", t->name);
    atomic_store(&j->failed, 1);
  }
  free(f), free(imat), free(in), free(outb);
  return NULL;
}

int main(int argc, char **argv)
{
  if (argc < 3)
    return fprintf(stderr, "usage: requant_dspark <pack dir> <out dir> [IQ2_XS|IQ2_XXS|IQ2_S|IQ1_S] [threads]\n"), 2;
  const char *tn = argc > 3 ? argv[3] : "IQ2_XS";
  const QTypeInfo *qt = qtype_named(tn);
  if (!qt || !(qt->paths & QTYPE_MV) || qt->values != 256)
    return fprintf(stderr, "requant_dspark: %s is not an expert type the drafter's matvec takes\n", tn), 2;
  const int threads = argc > 4 ? atoi(argv[4]) : 16;
  char err[DSPARK_ERR] = "";
  DsparkIndex *ix = dspark_index(argv[1], err);
  if (!ix)
    return fprintf(stderr, "requant_dspark: %s\n", err), 2;
  Job j = {ix, open(ix->bin, O_RDONLY), -1, (enum ggml_type)qt->id, calloc(ix->n, 8), calloc(ix->n, 8),
           calloc(ix->n, 4), 0, 0};
  char path[4096];
  mkdir(argv[2], 0755);
  snprintf(path, sizeof path, "%s/dspark.bin", argv[2]);
  j.out_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (j.in_fd < 0 || j.out_fd < 0 || !j.out_off || !j.out_bytes || !j.out_type)
    return fprintf(stderr, "requant_dspark: cannot open the pack or %s\n", path), 2;
  ggml_quantize_init(j.to);
  uint64_t at = 0, before = 0, after = 0;
  for (uint32_t i = 0; i < ix->n; i++)
  {
    const DsparkTensor *t = &ix->t[i];
    const int64_t cols = (int64_t)t->dims[t->n_dims - 1];
    j.out_type[i] = t->type;
    j.out_bytes[i] = is_expert(t) ? ggml_row_size(j.to, cols) * (t->bytes / ggml_row_size(GGML_TYPE_MXFP4, cols))
                                  : t->bytes;
    at = (at + ALIGN - 1) / ALIGN * ALIGN;
    j.out_off[i] = at, at += j.out_bytes[i];
    before += is_expert(t) ? t->bytes : 0, after += is_expert(t) ? j.out_bytes[i] : 0;
  }
  pthread_t th[64];
  const int nt = threads < 1 ? 1 : threads > 64 ? 64 : threads;
  for (int k = 0; k < nt; k++)
    pthread_create(&th[k], NULL, work, &j);
  for (int k = 0; k < nt; k++)
    pthread_join(th[k], NULL);
  if (atomic_load(&j.failed) || ftruncate(j.out_fd, (off_t)at) != 0 || close(j.out_fd) != 0)
    return fprintf(stderr, "requant_dspark: failed\n"), 1;
  snprintf(path, sizeof path, "%s/dspark.json", argv[2]);
  FILE *js = fopen(path, "w");
  if (!js)
    return fprintf(stderr, "requant_dspark: cannot write %s\n", path), 1;
  fprintf(js, "{\"from\": \"%s\", \"experts\": \"%s\", \"dense\": \"Q8_0\", \"tensors\": [", ix->bin, tn);
  for (uint32_t i = 0; i < ix->n; i++)
  {
    const DsparkTensor *t = &ix->t[i];
    fprintf(js, "%s[\"%s\", \"%s\", %llu, %llu, [", i ? ", " : "", t->name, qtype_info(j.out_type[i])->name,
            (unsigned long long)j.out_off[i], (unsigned long long)j.out_bytes[i]);
    for (uint32_t d = 0; d < t->n_dims; d++)
      fprintf(js, "%s%llu", d ? ", " : "", (unsigned long long)t->dims[d]);
    fprintf(js, "]]");
  }
  fprintf(js, "]}\n");
  fclose(js);
  printf("requant_dspark: %u tensors; the routed experts %.2f -> %.2f GiB (%s); the pack %.2f GiB\n", ix->n,
         before / 1073741824.0, after / 1073741824.0, tn, at / 1073741824.0);
  dspark_index_destroy(ix);
  return 0;
}
