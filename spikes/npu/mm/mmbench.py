# 0.7.0 spike: one AIE2P core's bf16 matmul rate (mmbench.cc): REPS 64x64x64 tiles, native or bfp16-emulated; the
# run's time against REPS = 1 removes the fixed cost (dispatch, the tiles' fill)
import os, sys, numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import cxx_header_path
from aie.utils.benchmark import run_iters
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.expanduser('~/Projects/mlir-aie/aie_kernels')

@iron.jit
def mmbench(inp: In, out: Out, *, reps: CompileTime[int], emu: CompileTime[int]):
    t = np.ndarray[(16,), np.dtype[np.int32]]
    a, b = ObjectFifo(t, name="mmin"), ObjectFifo(t, name="mmout")
    flags = [f"-DREPS={reps}"] + (["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"] if emu else [])
    k = ExternalFunction("mmbench", source_file=os.path.join(HERE, "mmbench.cc"), arg_types=[t, t],
                         include_dirs=[cxx_header_path(), KD], compile_flags=flags)
    def body(a, b, k):
        x, y = a.acquire(1), b.acquire(1); k(x, y); a.release(1); b.release(1)
    w = Worker(body, fn_args=[a.cons(), b.prod(), k])
    def seq(i, o, ap, bc):
        ap.fill(i); bc.drain(o, wait=True)
    return Program(iron.get_current_device(), Runtime(seq, [t, t, a.prod(), b.cons()]), workers=[w]).resolve_program()

def main():
    x = iron.zeros(16, dtype=np.int32, device="npu"); y = iron.zeros(16, dtype=np.int32, device="npu")
    for emu in (0, 1):
        us = {}
        for reps in (1, 257):
            us[reps] = run_iters(mmbench, x, y, reps=reps, emu=emu, warmup=1, iters=5).npu.avg_us
        per = (us[257] - us[1]) / 256
        print(f"{'bfp16-emulated 8x8x8' if emu else 'native bf16 4x8x8'}: a 64x64x64 tile {per:.2f} us -> "
              f"{2 * 64 ** 3 / per / 1e6:.2f} TFLOP/s one core ({us[1]:.0f} / {us[257]:.0f} us)")

if __name__ == "__main__":
    main()
