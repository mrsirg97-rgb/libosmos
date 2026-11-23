/* st_create (osmos/model/safetensors.h): a safetensors file read as a Gguf - its tensors' names, dims in ggml's
   order, types and bytes where data_offsets put them; refusals named. then Hy4-flash's DFly2 shard
   (~/Models/hy4-flash/model-00148-of-00148.safetensors, or OSMOS_TEST_DFLY), when on this machine: its 87 tensors */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/model/qtype.h"
#include "osmos/model/safetensors.h"

static char dir[] = "/tmp/test_st_XXXXXX";

/* a file of the header's JSON and `data` bytes of 0x5A after it */
static const char *file(const char *name, const char *header, size_t data)
{
  static char path[512];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  const uint64_t n = strlen(header);
  uint8_t len8[8];
  for (int i = 0; i < 8; i++)
    len8[i] = (uint8_t)(n >> (8 * i));
  assert(fwrite(len8, 1, 8, f) == 8 && fwrite(header, 1, n, f) == n);
  for (size_t i = 0; i < data; i++)
    fputc(0x5A, f);
  fclose(f);
  return path;
}

static void refused(const char *what, const char *header, size_t data, const char *says)
{
  char err[OSMOS_ERR] = "";
  Gguf *g = st_create(file("bad.safetensors", header, data), err);
  if (g || !strstr(err, says))
    fprintf(stderr, "  %s: %s\n", what, g ? "accepted" : err);
  assert(!g && strstr(err, says));
  printf("    %-44s -> %s\n", what, strstr(err, ": ") + 2);
}

int main(void)
{
  printf("[test_safetensors] start\n");
  assert(mkdtemp(dir));
  char err[OSMOS_ERR] = "";
  /* two tensors and the metadata: a [3, 4] BF16 (24 bytes) after a [2] F32 (8 bytes) */
  const char *good = "{\"__metadata__\":{\"format\":\"pt\"},\"b\":{\"dtype\":\"BF16\",\"shape\":[3,4],\"data_offsets\":[8,32]},"
                     "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
  Gguf *g = st_create(file("good.safetensors", good, 32), err);
  assert(g && g->n_tensors == 2 && g->n_kv == 0 && g->n_files == 1);
  const GgufTensor *b = gguf_tensor(g, "b"), *a = gguf_tensor(g, "a");
  const uint64_t data = 8 + strlen(good);
  assert(b && b->type == QTYPE_BF16 && b->n_dims == 2 && b->dims[0] == 4 && b->dims[1] == 3 && b->bytes == 24 &&
         b->offset == data + 8);
  assert(a && a->type == QTYPE_F32 && a->n_dims == 1 && a->dims[0] == 2 && a->bytes == 8 && a->offset == data);
  gguf_destroy(g);
  printf("  two tensors and the metadata: names, ggml's dims, types, offsets past the header\n");
  printf("  refused:\n");
  assert(!st_create("/nonexistent/x.safetensors", err) && strstr(err, "cannot open"));
  g = st_create(file("empty.safetensors", "{}", 0), err); /* (no tensors: a file, read) */
  assert(g && g->n_tensors == 0);
  gguf_destroy(g);
  remove(file("empty.safetensors", "{}", 0));
  refused("a tensor without its fields", "{\"a\":{}}", 0, "");
  refused("a header that is a JSON array", "[]", 0, "not a JSON object");
  refused("a dtype not read (I64)", "{\"a\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[0,8]}}", 8,
          "a dtype this build does not read");
  refused("bytes not the shape's", "{\"a\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]}}", 8,
          "not its dtype's");
  refused("bytes past the file", "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", 8,
          "past the file");
  refused("two tensors' bytes overlapping",
          "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
          "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}",
          12, "overlap");
  refused("a dim 0", "{\"a\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}", 0, "dims are 0");
  refused("no data_offsets", "{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}", 4, "without its dtype");
  remove(file("good.safetensors", "{}", 0));
  remove(file("bad.safetensors", "{}", 0));
  rmdir(dir);

  char path[512];
  if (getenv("OSMOS_TEST_DFLY"))
    snprintf(path, sizeof path, "%s", getenv("OSMOS_TEST_DFLY"));
  else
    snprintf(path, sizeof path, "%s/Models/hy4-flash/model-00148-of-00148.safetensors", getenv("HOME") ? getenv("HOME") : "");
  if (!(g = st_create(path, err)))
    printf("  Hy4-flash's DFly2 shard: not on this machine, skipped (%s)\n", err);
  else
  {
    const GgufTensor *kvb = gguf_tensor(g, "model.dfly_layers.0.self_attn.kv_b_proj.weight");
    assert(g->n_tensors == 87 && kvb && kvb->type == QTYPE_BF16 && kvb->dims[0] == 512 && kvb->dims[1] == 14336);
    printf("  Hy4-flash's DFly2 shard: 87 BF16 tensors, kv_b_proj [14336 x 512]\n");
    gguf_destroy(g);
  }
  printf("[test_safetensors] ok\n");
  return 0;
}
