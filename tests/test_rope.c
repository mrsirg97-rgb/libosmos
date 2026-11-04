#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "osmos/model/rope.h"

/* torch's (ref.py freqs_cis, oracle venv, torch 2.15.0+cpu): the window's and the compressed rows' */
static const uint32_t kWindow[32] = {
  0x3f800000u, 0x3f3ff911u, 0x3f0ff59au, 0x3ed7e89bu, 0x3ea1e89bu, 0x3e72d423u, 0x3e361887u, 0x3e088d77u,
  0x3dcccccdu, 0x3d99940du, 0x3d6655c2u, 0x3d2cba15u, 0x3d0186e3u, 0x3cc2434fu, 0x3c91ad39u, 0x3c5a7bf2u,
  0x3c23d70au, 0x3bf5b9b0u, 0x3bb8449cu, 0x3b8a2e77u, 0x3b4f3e38u, 0x3b1b690du, 0x3ae91528u, 0x3aaec98eu,
  0x3a83126fu, 0x3a44948cu, 0x3a136a16u, 0x39dd1725u, 0x39a5cb60u, 0x3978a815u, 0x393a7753u, 0x390bd472u,
};
static const uint32_t kCompress[32] = {
  0x3f800000u, 0x3f300a3au, 0x3ef21c1fu, 0x3ea67d01u, 0x3e64f92eu, 0x3e1d7475u, 0x3dd88cb4u, 0x3d94e963u,
  0x3d4ccccdu, 0x3d0cd4fbu, 0x3cc1b019u, 0x3c8530ceu, 0x3c372dbfu, 0x3bfbed88u, 0x3bad3d5eu, 0x3b6e4237u,
  0x3b147ae1u, 0x3ab714e0u, 0x3a5ebdb6u, 0x3a0530ceu, 0x399bb3afu, 0x39305978u, 0x38be904du, 0x383e9b5fu,
  0x37a3d70cu, 0x36b443d0u, 0x3677eba6u, 0x362a7be8u, 0x35ea77ffu, 0x35a13bdcu, 0x355dbf30u, 0x35187c4du,
};

static void same(const char *what, const Rope *r, const uint32_t *want)
{
  float f[32];
  assert(rope_freqs(r, f));
  for (int i = 0; i < 32; i++)
  {
    uint32_t got;
    memcpy(&got, &f[i], 4);
    if (got != want[i])
    {
      printf("  %s: freq %d is 0x%08x, torch's 0x%08x\n", what, i, got, want[i]);
      assert(0);
    }
  }
  printf("  %s: 32 frequencies, torch's bits\n", what);
}

int main(void)
{
  printf("[test_rope] start\n");
  same("window (base 1e4)", &(Rope){64, 10000.0f, 0, 16.0f, 32.0f, 1.0f}, kWindow);
  same("compressed (base 1.6e5, YaRN 65536 x16)", &(Rope){64, 160000.0f, 65536, 16.0f, 32.0f, 1.0f}, kCompress);
  float f[ROPE_MAX_DIM];
  assert(!rope_freqs(NULL, f) && !rope_freqs(&(Rope){63, 10000.0f, 0, 0, 0, 0}, f));
  assert(!rope_freqs(&(Rope){512, 10000.0f, 0, 0, 0, 0}, f) && !rope_freqs(&(Rope){64, 1.0f, 0, 0, 0, 0}, f));
  assert(!rope_freqs(&(Rope){64, 10000.0f, 65536, 0, 32.0f, 1.0f}, f));
  printf("  refusals: no rope, an odd or too-wide dim, base 1, YaRN without a factor\n");
  printf("[test_rope] ok\n");
  return 0;
}
