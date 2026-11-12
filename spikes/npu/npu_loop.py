# the NPU kept busy reading and writing DRAM (mlir-aie's getting_started memcpy: every shim DMA pair, 64 MiB each
# way) until <stop file> exists - the GPU's loss beside it measured by the caller. prints the NPU's own rate.
#   python npu_loop.py <stop file> (in ~/npu-venv, utils/env_setup.sh sourced)
import os, sys, time
sys.path.insert(0, os.path.expanduser('~/Projects/mlir-aie/programming_examples/getting_started/00_memcpy'))
import numpy as np, aie.iron as iron
from memcpy import my_memcpy
stop, n = sys.argv[1], 16777216
x = iron.arange(n, dtype=np.int32, device="npu"); y = iron.zeros_like(x)
my_memcpy(x, y, size=n, xfr_dtype=np.int32)  # compiled (cached) and warm
runs, t0 = 0, time.time()
while not os.path.exists(stop):
    my_memcpy(x, y, size=n, xfr_dtype=np.int32); runs += 1
dt = time.time() - t0
print(f"npu_loop: {runs} copies in {dt:.1f} s, {2 * n * 4 * runs / dt / 1e9:.1f} GB/s read + write (end to end)", flush=True)
