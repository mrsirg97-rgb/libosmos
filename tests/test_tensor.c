/* a tensor read from its file, one copy every family's (osmos/model/tensor.h; 0.16.0: Inkling's, Hy4's, DeepSeek's
   and the Engram's own): its bytes whole by path or descriptor, a short file refused; its values as F32 from F32, F16
   and BF16 - each the exact value (a half's subnormal, its infinity, a bfloat's), another type refused */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "osmos/model/tensor.h"

int main(void)
{
  printf("[test_tensor] start\n");
  char path[] = "/tmp/test_tensor_XXXXXX";
  const int wfd = mkstemp(path);
  assert(wfd >= 0);
  /* at 16: three F32; at 32: four F16 (1, -2, the least subnormal 2^-24, +inf); at 48: two BF16 (1.5, -0.25) */
  uint8_t buf[64] = {0};
  const float f[3] = {1.25f, -3.5f, 1e-30f};
  memcpy(buf + 16, f, sizeof f);
  const uint16_t h[4] = {0x3c00, 0xc000, 0x0001, 0x7c00}, bf[2] = {0x3fc0, 0xbe80};
  memcpy(buf + 32, h, sizeof h), memcpy(buf + 48, bf, sizeof bf);
  assert(write(wfd, buf, sizeof buf) == (ssize_t)sizeof buf);
  close(wfd);
  char *paths[1] = {path};
  Gguf g = {1, paths, NULL, NULL, 0, NULL, 0};
  const GgufTensor tf = {"f", 0, 1, {3}, 0, 16, 12}, th = {"h", 1, 1, {4}, 0, 32, 8}, tb = {"b", 30, 1, {2}, 0, 48, 4};
  const GgufTensor tq = {"q", 8, 1, {32}, 0, 0, 34}, past = {"p", 0, 1, {8}, 0, 40, 32};

  uint8_t out[32];
  assert(tensor_read(&g, &tf, out) && !memcmp(out, buf + 16, 12));
  const int fd = open(path, O_RDONLY);
  assert(fd >= 0 && tensor_read_fd(fd, &th, out) && !memcmp(out, buf + 32, 8));
  assert(!tensor_read(&g, &past, out) && !tensor_read_fd(fd, &past, out)); /* past the file's end: refused */
  close(fd);

  uint64_t n = 0;
  float *v = tensor_f32(&g, &tf, &n);
  assert(v && n == 3 && v[0] == 1.25f && v[1] == -3.5f && v[2] == 1e-30f);
  free(v);
  v = tensor_f32(&g, &th, &n);
  assert(v && n == 4 && v[0] == 1.0f && v[1] == -2.0f && v[2] == ldexpf(1.0f, -24) && isinf(v[3]) && v[3] > 0);
  free(v);
  v = tensor_f32(&g, &tb, &n);
  assert(v && n == 2 && v[0] == 1.5f && v[1] == -0.25f);
  free(v);
  assert(!tensor_f32(&g, &tq, &n)); /* Q8_0: not a float type */
  unlink(path);
  printf("  bytes whole by path and descriptor, a short file refused; F32 / F16 / BF16 exact, another type refused\n");
  printf("[test_tensor] ok\n");
  return 0;
}
