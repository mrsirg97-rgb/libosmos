#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "osmos/model/fnv.h"

/* fnv_file (0.9.2): a weights file rewritten in place at the same size and name another identity - a kept
   conversation's state never taken for one read with other weights; the same file, left alone, the same */
static void write_file(const char *path, char c)
{
  FILE *f = fopen(path, "wb");
  assert(f);
  for (int i = 0; i < 4096; i++)
    fputc(c, f);
  fclose(f);
}

int main(void)
{
  printf("[test_fnv_file] start\n");
  char dir[] = "/tmp/test_fnv_file_XXXXXX", path[64];
  assert(mkdtemp(dir));
  snprintf(path, sizeof path, "%s/overlay.gguf", dir);
  write_file(path, 'a');
  const uint64_t a = fnv_file(path, FNV_SEED);
  assert(a == fnv_file(path, FNV_SEED)); /* the same file, the same */
  usleep(20 * 1000);
  write_file(path, 'b'); /* made again: its name and size the same */
  const uint64_t b = fnv_file(path, FNV_SEED);
  assert(a != b);
  printf("  rewritten in place, the same name and size: another identity (%016llx, %016llx)\n",
         (unsigned long long)a, (unsigned long long)b);
  unlink(path), rmdir(dir);
  printf("[test_fnv_file] ok\n");
  return 0;
}
