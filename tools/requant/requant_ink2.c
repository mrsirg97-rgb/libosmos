/* requant_ink2: Inkling-2-Small's routed experts from the reference GGUF (ink2_ref_gguf.py's: the checkpoint's own
   NVFP4 / MXFP8), requantized with ggml's quantizers (llama.cpp, MIT: libggml-base) and an imatrix (unsloth's).

     requant_ink2 measure <reference first shard> <imatrix.gguf> <tensors> [threads]
     requant_ink2 build   <reference first shard> <imatrix.gguf> <plan> <out dir> [threads]

   tensors: a name a line (blk.L.ffn_{gate,up,down}_exps.weight); plan: a name and a ggml type a line. an expert's
   values as llama.cpp's inkling2 (unsloth's GGUFs) holds them: an NVFP4 layer's without the checkpoint's outer scale
   (e2m1 x ue4m3 - the scale a tensor of its own beside them, <name>.scale), an MXFP8 layer's whole (e4m3 x 2^(e -
   127)). each expert quantized alone with its own imatrix column (in_sum2 / counts; an expert never picked:
   uniform).
   measure prints a line a tensor and type (MEASURE_TYPES, default IQ1_S,IQ2_XXS,IQ2_S,IQ3_XXS,IQ4_XS): its bytes
   and the error sum_e sum_c imp_e[c] (w - q)^2 against sum_e sum_c imp_e[c] w^2 - and, with MODEL=<a GGUF of the
   same experts> (unsloth's), that model's own tensor's ("model:<type>"). build writes each planned tensor's bytes,
   its experts in order, to <out dir>/<name>.bin (written as .part, renamed when whole). */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "ggml.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"

#define MAX_T 512
#define MAX_TYPES 8

typedef struct
{
  char name[128];
  const GgufTensor *src;   /* the reference's */
  long rows, cols, experts;
  enum ggml_type plan;     /* build: its type */
  double err[MAX_TYPES + 1], energy;
  int fd_out;
} Ten;

static Ten ts[MAX_T];
static int nt;
static const Gguf *ref, *imat, *model;
static enum ggml_type types[MAX_TYPES];
static int n_types;
static bool build;
static const char *out_dir;
static atomic_long next_job;
static atomic_int failed;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static bool pread_all(int fd, void *p, size_t n, off_t at)
{
  for (size_t d = 0; d < n;)
  {
    const ssize_t k = pread(fd, (char *)p + d, n - d, at + (off_t)d);
    if (k <= 0)
      return false;
    d += (size_t)k;
  }
  return true;
}

static float ue4m3(uint8_t b) /* NVFP4's sub-block scale: E4M3 with no sign (0x7F: NaN, as zero) */
{
  const int e = b >> 3 & 15, m = b & 7;
  if (b == 0 || b == 0x7F)
    return 0.0f;
  return e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6);
}

static float e4m3(uint8_t b) /* MXFP8's value: E4M3 (0x7F / 0xFF: NaN, as zero) */
{
  const int s = b >> 7, e = b >> 3 & 15, m = b & 7;
  if ((b & 0x7F) == 0x7F)
    return 0.0f;
  const float v = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6);
  return s ? -v : v;
}

static const float E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6};

/* expert e of the reference's tensor, as F32 [rows][cols]: NVFP4 (a block 64 values: 4 UE4M3 scales, 32 bytes -
   sub-block s's value j in byte 8s + j's low nibble, j + 8 in its high one) without its outer scale; MXFP8 (a block
   32 values: an E8M0 scale, 32 E4M3) whole */
static float *decode(const Ten *t, long e)
{
  const GgufTensor *s = t->src;
  const uint64_t per = s->bytes / (uint64_t)t->experts;
  uint8_t *b = malloc(per);
  float *w = malloc((size_t)t->rows * t->cols * sizeof *w);
  const int fd = open(ref->paths[s->file], O_RDONLY);
  const bool ok = b && w && fd >= 0 && pread_all(fd, b, per, (off_t)(s->offset + (uint64_t)e * per));
  if (fd >= 0)
    close(fd);
  if (!ok)
    return free(b), free(w), NULL;
  const size_t n = (size_t)t->rows * t->cols;
  if (s->type == QTYPE_NVFP4)
    for (size_t k = 0; k < n / 64; k++)
    {
      const uint8_t *blk = b + k * 36;
      for (int sb = 0; sb < 4; sb++)
        for (int j = 0; j < 8; j++)
        {
          const uint8_t q = blk[4 + 8 * sb + j];
          const float sc = ue4m3(blk[sb]);
          w[k * 64 + sb * 16 + j] = E2M1[q & 15] * sc, w[k * 64 + sb * 16 + j + 8] = E2M1[q >> 4] * sc;
        }
    }
  else if (s->type == QTYPE_MXFP8)
    for (size_t k = 0; k < n / 32; k++)
    {
      const uint8_t *blk = b + k * 33;
      const float sc = ldexpf(1.0f, (int)blk[0] - 127);
      for (int j = 0; j < 32; j++)
        w[k * 32 + j] = e4m3(blk[1 + j]) * sc;
    }
  else
    return free(b), free(w), NULL;
  free(b);
  return w;
}

/* expert e's importance of each input column: the imatrix's in_sum2 / counts (never picked: uniform) */
static float *importance(const Ten *t, long e)
{
  char k1[160], k2[160];
  snprintf(k1, sizeof k1, "%s.in_sum2", t->name), snprintf(k2, sizeof k2, "%s.counts", t->name);
  const GgufTensor *a = gguf_tensor(imat, k1), *c = gguf_tensor(imat, k2);
  if (!a || !c || a->dims[0] != (uint64_t)t->cols || a->n_dims < 2 || a->dims[1] != (uint64_t)t->experts)
    return NULL;
  float *imp = malloc(t->cols * sizeof *imp), cnt = 0;
  const int fa = open(imat->paths[a->file], O_RDONLY), fc = open(imat->paths[c->file], O_RDONLY);
  const bool ok = imp && fa >= 0 && fc >= 0 && pread_all(fa, imp, t->cols * 4, (off_t)(a->offset + e * t->cols * 4)) &&
                  pread_all(fc, &cnt, 4, (off_t)(c->offset + e * 4));
  if (fa >= 0)
    close(fa);
  if (fc >= 0)
    close(fc);
  if (!ok)
    return free(imp), NULL;
  for (long k = 0; k < t->cols; k++)
    imp[k] = cnt > 0 ? imp[k] / cnt : 1.0f;
  return imp;
}

static double weighted_err(const float *w, const void *q, enum ggml_type ty, const Ten *t, const float *imp, float *row)
{
  const struct ggml_type_traits *tr = ggml_get_type_traits(ty);
  const size_t rb = ggml_row_size(ty, t->cols);
  double e = 0;
  for (long r = 0; r < t->rows; r++)
  {
    tr->to_float((const char *)q + r * rb, row, t->cols);
    for (long c = 0; c < t->cols; c++)
    {
      const double d = (double)w[r * t->cols + c] - row[c];
      e += imp[c] * d * d;
    }
  }
  return e;
}

static void *work(void *arg)
{
  (void)arg;
  float *row = malloc(16384 * sizeof *row);
  for (long j; !atomic_load(&failed) && (j = atomic_fetch_add(&next_job, 1)) < (long)nt * 384;)
  {
    Ten *t = &ts[j / 384];
    const long e = j % 384;
    if (e >= t->experts)
      continue;
    float *w = decode(t, e), *imp = importance(t, e);
    if (!w || !imp || t->cols > 16384)
    {
      fprintf(stderr, "%s#%ld: not decoded, or not in the imatrix\n", t->name, e);
      atomic_store(&failed, 1), free(w), free(imp);
      break;
    }
    double energy = 0, err[MAX_TYPES + 1] = {0};
    for (long k = 0; k < t->rows * t->cols; k++)
      energy += imp[k % t->cols] * (double)w[k] * w[k];
    for (int k = 0; k < (build ? 1 : n_types); k++)
    {
      const enum ggml_type ty = build ? t->plan : types[k];
      const size_t bytes = ggml_row_size(ty, t->cols) * t->rows;
      void *q = malloc(bytes);
      if (!q || ggml_quantize_chunk(ty, w, q, 0, t->rows, t->cols, imp) != bytes)
      {
        fprintf(stderr, "%s#%ld: %s refused\n", t->name, e, ggml_type_name(ty));
        atomic_store(&failed, 1), free(q);
        break;
      }
      if (build)
      {
        if (pwrite(t->fd_out, q, bytes, (off_t)(e * bytes)) != (ssize_t)bytes)
          fprintf(stderr, "%s#%ld: not written\n", t->name, e), atomic_store(&failed, 1);
      }
      else
        err[k] = weighted_err(w, q, ty, t, imp, row);
      free(q);
    }
    if (!build && model) /* the other model's own expert, its values as ours (unsloth's: no outer scale) */
    {
      const GgufTensor *mt = gguf_tensor(model, t->name);
      const uint64_t eb = mt ? mt->bytes / (uint64_t)t->experts : 0;
      void *m = mt ? malloc(eb) : NULL;
      const int fd = mt ? open(model->paths[mt->file], O_RDONLY) : -1;
      if (m && fd >= 0 && pread_all(fd, m, eb, (off_t)(mt->offset + (uint64_t)e * eb)))
        err[n_types] = weighted_err(w, m, (enum ggml_type)mt->type, t, imp, row);
      if (fd >= 0)
        close(fd);
      free(m);
    }
    pthread_mutex_lock(&mu);
    t->energy += energy;
    for (int k = 0; k <= n_types; k++)
      t->err[k] += err[k];
    pthread_mutex_unlock(&mu);
    free(w), free(imp);
  }
  free(row);
  return NULL;
}

static enum ggml_type type_of(const char *name)
{
  for (int k = 0; k < GGML_TYPE_COUNT; k++)
    if (ggml_get_type_traits((enum ggml_type)k)->type_name && !strcasecmp(ggml_type_name((enum ggml_type)k), name))
      return (enum ggml_type)k;
  return GGML_TYPE_COUNT;
}

int main(int argc, char **argv)
{
  if (argc < 5)
    return fprintf(stderr, "usage: see the file's head\n"), 2;
  build = !strcmp(argv[1], "build");
  if (!build && strcmp(argv[1], "measure"))
    return fprintf(stderr, "measure or build\n"), 2;
  char err[GGUF_ERR] = "";
  if (!(ref = gguf_create(argv[2], err)))
    return fprintf(stderr, "reference: %s\n", err), 1;
  if (!(imat = gguf_create(argv[3], err)))
    return fprintf(stderr, "imatrix: %s\n", err), 1;
  if (getenv("MODEL") && !(model = gguf_create(getenv("MODEL"), err)))
    return fprintf(stderr, "MODEL: %s\n", err), 1;
  FILE *lf = fopen(argv[4], "r");
  if (!lf)
    return fprintf(stderr, "%s: not opened\n", argv[4]), 1;
  char nm[128], ty[32];
  for (int got; nt < MAX_T && (got = fscanf(lf, "%127s", nm)) == 1; nt++)
  {
    Ten *t = &ts[nt];
    snprintf(t->name, sizeof t->name, "%s", nm);
    if (!(t->src = gguf_tensor(ref, nm)) || t->src->n_dims != 3)
      return fprintf(stderr, "%s: not an expert tensor of the reference\n", nm), 1;
    t->cols = (long)t->src->dims[0], t->rows = (long)t->src->dims[1], t->experts = (long)t->src->dims[2];
    if (t->experts > 384)
      return fprintf(stderr, "%s: more than 384 experts\n", nm), 1;
    if (build)
    {
      if (fscanf(lf, "%31s", ty) != 1 || (t->plan = type_of(ty)) == GGML_TYPE_COUNT)
        return fprintf(stderr, "plan: %s's type missing or unknown\n", nm), 1;
    }
  }
  fclose(lf);
  if (build)
  {
    if (argc < 6)
      return fprintf(stderr, "build: an out dir\n"), 2;
    out_dir = argv[5];
    for (int i = 0; i < nt; i++)
    {
      char p[4096];
      if (snprintf(p, sizeof p, "%s/%s.bin.part", out_dir, ts[i].name) >= (int)sizeof p)
        return fprintf(stderr, "%s: the out dir's path too long\n", out_dir), 1;
      if ((ts[i].fd_out = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644)) < 0)
        return fprintf(stderr, "%s: not created\n", p), 1;
    }
  }
  else
  {
    char buf[256];
    snprintf(buf, sizeof buf, "%s", getenv("MEASURE_TYPES") ? getenv("MEASURE_TYPES") : "IQ1_S,IQ2_XXS,IQ2_S,IQ3_XXS,IQ4_XS");
    for (char *tok = strtok(buf, ","); tok && n_types < MAX_TYPES; tok = strtok(NULL, ","))
      if ((types[n_types++] = type_of(tok)) == GGML_TYPE_COUNT)
        return fprintf(stderr, "MEASURE_TYPES: %s unknown\n", tok), 2;
  }
  const int threads = argc > (build ? 6 : 5) ? atoi(argv[build ? 6 : 5]) : 16;
  ggml_quantize_init(GGML_TYPE_IQ1_S), ggml_quantize_init(GGML_TYPE_IQ2_XXS), ggml_quantize_init(GGML_TYPE_IQ2_S);
  ggml_quantize_init(GGML_TYPE_IQ2_XS), ggml_quantize_init(GGML_TYPE_IQ3_XXS), ggml_quantize_init(GGML_TYPE_IQ1_M);
  pthread_t th[64];
  const int n = threads < 1 ? 1 : threads > 64 ? 64 : threads;
  for (int i = 0; i < n; i++)
    pthread_create(&th[i], NULL, work, NULL);
  for (int i = 0; i < n; i++)
    pthread_join(th[i], NULL);
  if (atomic_load(&failed))
    return 1;
  for (int i = 0; i < nt; i++)
  {
    const Ten *t = &ts[i];
    if (build)
    {
      char a[4096], b[4096];
      if (snprintf(a, sizeof a, "%s/%s.bin.part", out_dir, t->name) >= (int)sizeof a ||
          snprintf(b, sizeof b, "%s/%s.bin", out_dir, t->name) >= (int)sizeof b)
        return fprintf(stderr, "%s: the out dir's path too long\n", out_dir), 1;
      close(t->fd_out);
      if (rename(a, b))
        return fprintf(stderr, "%s: not renamed\n", a), 1;
      printf("built\t%s\t%s\t%zu\n", t->name, ggml_type_name(t->plan), ggml_row_size(t->plan, t->cols) * t->rows * t->experts);
      continue;
    }
    for (int k = 0; k < n_types; k++)
      printf("%s\t%s\t%zu\t%.6e\t%.6e\n", t->name, ggml_type_name(types[k]),
             ggml_row_size(types[k], t->cols) * t->rows * t->experts, t->err[k], t->energy);
    if (model && gguf_tensor(model, t->name))
      printf("%s\tmodel:%s\t%lu\t%.6e\t%.6e\n", t->name, ggml_type_name((enum ggml_type)gguf_tensor(model, t->name)->type),
             (unsigned long)gguf_tensor(model, t->name)->bytes, t->err[n_types], t->energy);
  }
  return 0;
}
