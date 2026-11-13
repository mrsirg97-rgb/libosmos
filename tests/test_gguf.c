#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/model/gguf.h"

/* a GGUF written by hand, byte by byte, as the format has it (version 3, little-endian) */
typedef struct
{
  uint8_t *b;
  size_t len;
  size_t cap;
} Buf;

static void put(Buf *w, const void *p, size_t n)
{
  if (w->len + n > w->cap)
  {
    w->cap = (w->len + n) * 2;
    w->b = realloc(w->b, w->cap);
    assert(w->b);
  }
  memcpy(w->b + w->len, p, n);
  w->len += n;
}
static void u32(Buf *w, uint32_t v) { put(w, &v, 4); }
static void u64(Buf *w, uint64_t v) { put(w, &v, 8); }
static void str(Buf *w, const char *s)
{
  u64(w, strlen(s));
  put(w, s, strlen(s));
}
static void pad(Buf *w, size_t align)
{
  static const uint8_t z[64];
  while (w->len % align)
    put(w, z, 1);
}

typedef struct
{
  const char *name;
  uint32_t type;
  uint32_t n_dims;
  uint64_t dims[4];
  uint64_t bytes;
} Spec;

/* a file: `kv_extra` (already encoded kv entries) after the split / alignment ones, then the tensors, then their data
   (byte i of the data section is (i * 7 + 3) & 255) */
static Buf model(const Spec *t, int n, uint32_t align, int split_no, int split_count, const Buf *kv_extra,
                 uint64_t n_kv_extra)
{
  Buf w = {0};
  u32(&w, 0x46554747u);
  u32(&w, 3);
  u64(&w, (uint64_t)n);
  uint64_t n_kv = n_kv_extra + (align ? 1 : 0) + (split_count ? 2 : 0);
  u64(&w, n_kv);
  if (align)
  {
    str(&w, "general.alignment");
    u32(&w, GGUF_UINT32);
    u32(&w, align);
  }
  if (split_count)
  {
    str(&w, "split.no");
    u32(&w, GGUF_UINT16);
    uint16_t no = (uint16_t)split_no;
    put(&w, &no, 2);
    str(&w, "split.count");
    u32(&w, GGUF_UINT16);
    uint16_t c = (uint16_t)split_count;
    put(&w, &c, 2);
  }
  if (kv_extra)
    put(&w, kv_extra->b, kv_extra->len);
  const uint32_t a = align ? align : 32;
  uint64_t off = 0;
  for (int i = 0; i < n; i++)
  {
    str(&w, t[i].name);
    u32(&w, t[i].n_dims);
    for (uint32_t d = 0; d < t[i].n_dims; d++)
      u64(&w, t[i].dims[d]);
    u32(&w, t[i].type);
    u64(&w, off);
    off += (t[i].bytes + a - 1) / a * a;
  }
  pad(&w, a);
  for (uint64_t i = 0; i < off; i++)
  {
    uint8_t v = (uint8_t)((i * 7 + 3) & 255);
    put(&w, &v, 1);
  }
  return w;
}

static char dir[] = "/tmp/test_gguf_XXXXXX";

static const char *write_file(const char *name, const Buf *w)
{
  static char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  assert(fwrite(w->b, 1, w->len, f) == w->len);
  fclose(f);
  return path;
}

/* every tensor lies in its file and its bytes are its type's for its dims: what a returned Gguf must hold */
static void check_whole(const Gguf *g)
{
  for (size_t i = 0; i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    assert(t->file >= 0 && t->file < g->n_files);
    assert(t->offset + t->bytes <= g->sizes[t->file]);
    uint64_t values = 1;
    for (uint32_t d = 0; d < t->n_dims; d++)
      values *= t->dims[d];
    assert(gguf_bytes(t->type, values) == t->bytes);
  }
}

static const Spec three[] = {
    {"a.f32", 0, 2, {4, 3}, 4 * 3 * 4},           /* F32: 48 B */
    {"b.q8_0", 8, 2, {64, 2}, 2 * 2 * 34},        /* Q8_0: 32 values a 34 B block */
    {"c.iq1_m", 29, 3, {256, 1, 2}, 2 * 56},      /* IQ1_M: 256 values a 56 B block */
};

static void test_round_trip(void)
{
  Buf kv = {0};
  str(&kv, "t.u8");
  u32(&kv, GGUF_UINT8);
  uint8_t u8 = 200;
  put(&kv, &u8, 1);
  str(&kv, "t.i32");
  u32(&kv, GGUF_INT32);
  int32_t i32 = -7;
  put(&kv, &i32, 4);
  str(&kv, "t.f32");
  u32(&kv, GGUF_FLOAT32);
  float f32 = 1.5f;
  put(&kv, &f32, 4);
  str(&kv, "t.bool");
  u32(&kv, GGUF_BOOL);
  uint8_t bt = 1;
  put(&kv, &bt, 1);
  str(&kv, "t.str");
  u32(&kv, GGUF_STRING);
  str(&kv, "deepseek");
  str(&kv, "t.strs");
  u32(&kv, GGUF_ARRAY);
  u32(&kv, GGUF_STRING);
  u64(&kv, 3);
  str(&kv, "x");
  str(&kv, "");
  str(&kv, "yz");
  str(&kv, "t.u32s");
  u32(&kv, GGUF_ARRAY);
  u32(&kv, GGUF_UINT32);
  u64(&kv, 4);
  for (uint32_t i = 0; i < 4; i++)
    u32(&kv, 10 + i);
  str(&kv, "t.u64");
  u32(&kv, GGUF_UINT64);
  u64(&kv, 1ull << 40);
  for (uint32_t align = 0; align <= 64; align += 64) /* the default 32, then general.alignment 64 */
  {
    Buf w = model(three, 3, align, 0, 0, &kv, 8);
    const char *path = write_file("one.gguf", &w);
    char err[GGUF_ERR] = "";
    Gguf *g = gguf_create(path, err);
    if (!g)
      fprintf(stderr, "  gguf_create: %s\n", err);
    assert(g);
    assert(g->n_files == 1 && g->sizes[0] == w.len && g->n_tensors == 3);
    check_whole(g);
    const GgufKv *k = gguf_kv(g, "t.u8");
    assert(k && k->type == GGUF_UINT8 && k->v.u == 200);
    k = gguf_kv(g, "t.i32");
    assert(k && k->type == GGUF_INT32 && k->v.i == -7);
    k = gguf_kv(g, "t.f32");
    assert(k && k->type == GGUF_FLOAT32 && k->v.f == 1.5);
    k = gguf_kv(g, "t.bool");
    assert(k && k->type == GGUF_BOOL && k->v.u == 1);
    k = gguf_kv(g, "t.str");
    assert(k && k->type == GGUF_STRING && strcmp(k->v.s, "deepseek") == 0);
    k = gguf_kv(g, "t.strs");
    assert(k && k->type == GGUF_ARRAY && k->item == GGUF_STRING && k->n == 3);
    char **s = (char **)k->v.items;
    assert(strcmp(s[0], "x") == 0 && strcmp(s[1], "") == 0 && strcmp(s[2], "yz") == 0);
    k = gguf_kv(g, "t.u32s");
    assert(k && k->item == GGUF_UINT32 && k->n == 4 && ((uint32_t *)k->v.items)[3] == 13);
    k = gguf_kv(g, "t.u64");
    assert(k && k->v.u == 1ull << 40);
    assert(gguf_kv(g, "absent") == NULL && gguf_tensor(g, "absent") == NULL);
    const GgufTensor *t = gguf_tensor(g, "c.iq1_m");
    assert(t && t->type == 29 && t->n_dims == 3 && t->dims[0] == 256 && t->dims[2] == 2 && t->bytes == 112);
    /* the data section starts aligned, each tensor aligned within it: the bytes there are the ones written */
    const uint32_t a = align ? align : 32;
    const GgufTensor *t0 = gguf_tensor(g, "a.f32"), *t1 = gguf_tensor(g, "b.q8_0");
    assert(t0->offset % a == 0 && t1->offset == t0->offset + (48 + a - 1) / a * a);
    const uint64_t data0 = t0->offset;
    assert(w.b[t1->offset] == (uint8_t)(((t1->offset - data0) * 7 + 3) & 255));
    gguf_destroy(g);
    free(w.b);
  }
  free(kv.b);
  gguf_destroy(NULL);
  printf("  round trip: every value type, strings and arrays, alignment 32 and 64\n");
}

static void test_split(void)
{
  Buf a = model(three, 2, 0, 0, 2, NULL, 0);
  Buf b = model(three + 2, 1, 0, 1, 2, NULL, 0);
  write_file("m-00002-of-00002.gguf", &b);
  const char *first = write_file("m-00001-of-00002.gguf", &a);
  char p1[512];
  snprintf(p1, sizeof p1, "%s", first);
  char err[GGUF_ERR] = "";
  Gguf *g = gguf_create(p1, err);
  if (!g)
    fprintf(stderr, "  gguf_create: %s\n", err);
  assert(g && g->n_files == 2 && g->n_tensors == 3);
  check_whole(g);
  const GgufTensor *t = gguf_tensor(g, "c.iq1_m");
  assert(t && t->file == 1);
  gguf_destroy(g);
  char p2[512];
  snprintf(p2, sizeof p2, "%s/m-00002-of-00002.gguf", dir);
  unlink(p2);
  err[0] = 0;
  assert(gguf_create(p1, err) == NULL && strstr(err, "00002") != NULL);
  char refused[GGUF_ERR];
  snprintf(refused, sizeof refused, "%s", err);
  /* the first shard alone (its metadata: a tokenizer's): read without its siblings, only its own tensors */
  Gguf *h = gguf_header(p1, err);
  assert(h && h->n_files == 1 && h->n_tensors == 2 && gguf_kv(h, "split.count") && !gguf_tensor(h, "c.iq1_m"));
  gguf_destroy(h);
  free(a.b);
  free(b.b);
  printf("  split: two shards as one model; a missing shard refused (\"%s\"); the first one's header alone read\n",
         refused);
}

/* a valid file cut at every length short of whole: refused - with a reason, nothing leaked - exactly when the cut takes
   some of a tensor's bytes; a cut in the last tensor's trailing padding leaves every tensor whole, and is accepted */
static void test_truncated(void)
{
  Buf w = model(three, 3, 0, 0, 0, NULL, 0);
  const size_t last_pad = (56 * 2 + 31) / 32 * 32 - 56 * 2; /* c.iq1_m's 112 bytes padded to 128 */
  int refused = 0, whole = 0;
  for (size_t n = 0; n < w.len; n++)
  {
    Buf cut = {w.b, n, n};
    const char *path = write_file("cut.gguf", &cut);
    char err[GGUF_ERR] = "";
    Gguf *g = gguf_create(path, err);
    if (n < w.len - last_pad)
    {
      assert(g == NULL && err[0] != 0);
      refused++;
    }
    else
    {
      assert(g);
      check_whole(g);
      gguf_destroy(g);
      whole++;
    }
  }
  free(w.b);
  printf("  truncated: %d cuts into a tensor refused, %d in the trailing padding whole\n", refused, whole);
}

typedef struct
{
  const char *what;
  size_t at;
  uint64_t value;
  int width;
} Patch;

static void test_corrupt(void)
{
  Buf w = model(three, 3, 0, 0, 0, NULL, 0);
  /* header: magic 0, version 4, n_tensors 8, n_kv 16; the first tensor info at 24: name length 24, name 32..37,
     n_dims 37, dims 41 / 49, type 57, offset 61 */
  const Patch patches[] = {
      {"magic", 0, 0x47475546u, 4},
      {"version 2", 4, 2, 4},
      {"n_tensors 2^60", 8, 1ull << 60, 8},
      {"n_kv 2^60", 16, 1ull << 60, 8},
      {"name length 2^62", 24, 1ull << 62, 8},
      {"n_dims 5", 37, 5, 4},
      {"dims overflow", 41, 1ull << 62, 8},
      {"unknown type 99", 57, 99, 4},
      {"a type outside the allowlist (Q4_0 2)", 57, 2, 4},
      {"offset past the file", 61, 1ull << 40, 8},
      {"offset not aligned", 61, 1, 8},
  };
  for (size_t i = 0; i < sizeof patches / sizeof patches[0]; i++)
  {
    Buf c = {malloc(w.len), w.len, w.len};
    memcpy(c.b, w.b, w.len);
    memcpy(c.b + patches[i].at, &patches[i].value, (size_t)patches[i].width);
    const char *path = write_file("bad.gguf", &c);
    char err[GGUF_ERR] = "";
    Gguf *g = gguf_create(path, err);
    if (g)
      fprintf(stderr, "  %s: accepted\n", patches[i].what);
    assert(g == NULL && err[0] != 0);
    free(c.b);
  }
  /* a duplicate tensor name */
  const Spec dup[] = {three[0], three[0]};
  Buf d = model(dup, 2, 0, 0, 0, NULL, 0);
  char err[GGUF_ERR] = "";
  assert(gguf_create(write_file("dup.gguf", &d), err) == NULL && err[0] != 0);
  free(d.b);
  free(w.b);
  printf("  corrupt: %zu patches and a duplicate name refused\n", sizeof patches / sizeof patches[0]);
}

/* type 43: unsloth's MXFP8 (33 bytes a 32 values) read; upstream's F8_E4M3 (a byte a value) refused - the bytes a
   tensor of it holds too few for MXFP8's, it runs into the next tensor or out of the file */
static void test_mxfp8(void)
{
  const uint64_t vals = 64 * 32; /* (past 1,024 values: a 32nd of them more than an alignment's padding) */
  const Spec mx[] = {{"e.mxfp8", 43, 2, {64, 32}, vals / 32 * 33}, three[0]};
  const Spec f8[] = {{"e.f8_e4m3", 43, 2, {64, 32}, vals}, three[0]};
  char err[GGUF_ERR] = "";
  Buf w = model(mx, 2, 0, 0, 0, NULL, 0);
  Gguf *g = gguf_create(write_file("mxfp8.gguf", &w), err);
  assert(g && g->tensors[0].bytes == vals / 32 * 33);
  check_whole(g);
  gguf_destroy(g);
  free(w.b);
  w = model(f8, 2, 0, 0, 0, NULL, 0);
  assert(!gguf_create(write_file("f8.gguf", &w), err) && strstr(err, "overlap"));
  free(w.b);
  w = model(f8, 1, 0, 0, 0, NULL, 0);
  assert(!gguf_create(write_file("f8one.gguf", &w), err) && strstr(err, "leave"));
  free(w.b);
  printf("  type 43: MXFP8's blocks read; a byte a value (upstream's F8_E4M3) refused, before a tensor or alone\n");
}

/* type 40: NVFP4 (36 bytes a 64 values: 4 UE4M3 scales, 32 bytes of codes) read, as Inkling-2's experts are written
   (ink2_ref_gguf.py: [experts, rows, columns], an F32 scale a expert beside it); a row not whole blocks refused */
static void test_nvfp4(void)
{
  const uint64_t vals = 256 * 16 * 4; /* (columns 256: four blocks a row; 16 rows; 4 experts) */
  const Spec nv[] = {{"blk.0.ffn_up_exps.weight", 40, 3, {256, 16, 4}, vals / 64 * 36},
                     {"blk.0.ffn_up_exps.scale", 0, 1, {4}, 4 * 4}};
  const Spec odd[] = {{"e.nvfp4", 40, 2, {96, 16}, 96 * 16 / 64 * 36}};
  char err[GGUF_ERR] = "";
  Buf w = model(nv, 2, 0, 0, 0, NULL, 0);
  Gguf *g = gguf_create(write_file("nvfp4.gguf", &w), err);
  assert(g && g->tensors[0].bytes == vals / 64 * 36 && g->tensors[1].bytes == 16);
  check_whole(g);
  gguf_destroy(g);
  free(w.b);
  w = model(odd, 1, 0, 0, 0, NULL, 0);
  assert(!gguf_create(write_file("nvfp4odd.gguf", &w), err));
  free(w.b);
  printf("  type 40: NVFP4's blocks read, an expert tensor and its scale; a row of 96 values (1.5 blocks) refused\n");
}

/* property: random byte changes never crash it; what it returns always holds (every tensor in its file) */
static void test_mutations(void)
{
  Buf w = model(three, 3, 64, 0, 0, NULL, 0);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  int accepted = 0;
  const int runs = 2000;
  for (int r = 0; r < runs; r++)
  {
    Buf c = {malloc(w.len), w.len, w.len};
    memcpy(c.b, w.b, w.len);
    const int flips = 1 + r % 4;
    for (int k = 0; k < flips; k++)
    {
      s ^= s << 13, s ^= s >> 7, s ^= s << 17;
      c.b[s % c.len] = (uint8_t)(s >> 32);
    }
    const char *path = write_file("mut.gguf", &c);
    char err[GGUF_ERR] = "";
    Gguf *g = gguf_create(path, err);
    if (g)
    {
      check_whole(g);
      accepted++;
      gguf_destroy(g);
    }
    else
      assert(err[0] != 0);
    free(c.b);
  }
  free(w.b);
  printf("  mutations: %d runs, %d accepted and whole, the rest refused with a reason\n", runs, accepted);
}

/* the real model, where this machine has it */
static void test_ds1017(void)
{
  const char *home = getenv("HOME");
  char p[512];
  snprintf(p, sizeof p, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
           home ? home : "");
  if (access(p, R_OK) != 0)
  {
    printf("  ds1017: not on this machine, skipped\n");
    return;
  }
  char err[GGUF_ERR] = "";
  Gguf *g = gguf_create(p, err);
  if (!g)
    fprintf(stderr, "  gguf_create: %s\n", err);
  assert(g && g->n_files == 4 && g->n_tensors == 1008);
  check_whole(g);
  const GgufTensor *t = gguf_tensor(g, "blk.0.ffn_gate_exps.weight");
  assert(t && t->type == 29 && t->n_dims == 3 && t->dims[0] == 5120 && t->dims[1] == 2304 && t->dims[2] == 384);
  gguf_destroy(g);
  snprintf(p, sizeof p, "%s/models/ds1017/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", home);
  g = gguf_create(p, err);
  assert(g && g->n_files == 1 && g->n_tensors == 2);
  check_whole(g);
  gguf_destroy(g);
  printf("  ds1017: 4 shards, 1008 tensors, all in their files; the Engram file, 2 tensors\n");
}

int main(void)
{
  printf("[test_gguf] start\n");
  assert(mkdtemp(dir));
  test_round_trip();
  test_split();
  test_truncated();
  test_corrupt();
  test_mxfp8();
  test_nvfp4();
  test_mutations();
  test_ds1017();
  char cmd[600];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  assert(system(cmd) == 0);
  printf("[test_gguf] ok\n");
  return 0;
}
