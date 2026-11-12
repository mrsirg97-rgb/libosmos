# 0.7.0 spike: Q2_K decoded on one AIE2P core (q2k_decode.cc), checked against gguf-py's dequantization of real
# evo-A bytes (blk.20.ffn_down_exps: Q2_K), timed: values a second, with REPS decodes of each buffer (the core alone)
#   python q2k_bench.py [vec 0|1] [reps]   (~/npu-venv, mlir-aie's utils/env_setup.sh sourced)
import os, sys, numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path
from aie.utils.benchmark import run_iters
from ml_dtypes import bfloat16
HERE = os.path.dirname(os.path.abspath(__file__))
NB, BLOCKS = 16, 4096  # blocks a fifo element; blocks in all (1M values)
VEC = int(sys.argv[1]) if len(sys.argv) > 1 else 1
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NOSCALE = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BFP = int(sys.argv[4]) if len(sys.argv) > 4 else 0

@iron.jit
def q2k(inp: In, out: Out, *, vec: CompileTime[int], reps: CompileTime[int], noscale: CompileTime[int] = 0, bfp: CompileTime[int] = 0):
    in_t = np.ndarray[(NB * 84,), np.dtype[np.uint8]]
    out_t = np.ndarray[(NB * 288,), np.dtype[np.uint8]] if bfp else np.ndarray[(NB * 256,), np.dtype[bfloat16]]
    of_in, of_out = ObjectFifo(in_t, name="qin"), ObjectFifo(out_t, name="vout")
    k = ExternalFunction("q2k_decode", source_file=os.path.join(HERE, "q2k_decode.cc"), arg_types=[in_t, out_t],
                         include_dirs=[cxx_header_path()], compile_flags=[f"-DNB={NB}", f"-DREPS={reps}", f"-DQ2K_VEC={vec}", f"-DQ2K_NOSCALE={noscale}", f"-DQ2K_BFP={bfp}"])
    def body(of_in, of_out, k):
        for _ in range_(BLOCKS // NB):
            a, b = of_in.acquire(1), of_out.acquire(1)
            k(a, b)
            of_in.release(1); of_out.release(1)
    w = Worker(body, fn_args=[of_in.cons(), of_out.prod(), k])
    all_in = np.ndarray[(BLOCKS * 84,), np.dtype[np.uint8]]
    all_out = np.ndarray[(BLOCKS * 288,), np.dtype[np.uint8]] if bfp else np.ndarray[(BLOCKS * 256,), np.dtype[bfloat16]]
    def seq(a, c, pin, cout):
        pin.fill(a); cout.drain(c, wait=True)
    rt = Runtime(seq, [all_in, all_out, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

def main():
    sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(os.path.expanduser('~/Models/ds1017/osmos-evo/delta/DeepSeek-V4.1-Flash-1017-osmos-evo-A-delta.gguf'))
    t = next(t for t in r.tensors if t.name == 'blk.20.ffn_down_exps.weight')
    assert t.tensor_type.name == 'Q2_K', t.tensor_type
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)[:BLOCKS * 84].copy()
    ref = dequantize(raw, t.tensor_type).astype(np.float32).reshape(-1)
    x = iron.tensor(raw, dtype=np.uint8, device="npu")
    y = iron.zeros(BLOCKS * 288, dtype=np.uint8, device="npu") if BFP else iron.zeros(BLOCKS * 256, dtype=bfloat16, device="npu")
    bench = run_iters(q2k, x, y, vec=VEC, reps=REPS, noscale=NOSCALE, bfp=BFP, warmup=1, iters=5)
    if BFP:  # the bfp16ebs8 blocks decoded on the host (mlir-aie's codec), against the reference quantized as the core
        from aie.utils import bfp as B  # converts (floor): their difference in the block's LSBs
        got = B.decode(y.numpy())
        q = B.quantize(ref.reshape(-1, 8)).reshape(-1)
        blk = np.abs(ref.reshape(-1, 8)).max(1)
        lsb = np.repeat(np.ldexp(1.0, np.floor(np.log2(np.maximum(blk, 1e-30))).astype(int) - 6), 8)
        print(f"  bfp: within the core's own rounding of the reference: {np.mean(got == q) * 100:.2f}% exact, max {np.abs(got - q).max() / lsb[np.argmax(np.abs(got - q))]:.1f} LSB")
    else:
        got = y.numpy().astype(np.float32)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    us = bench.npu.avg_us
    print(f"vec={VEC} bfp={BFP} reps={REPS} noscale={NOSCALE}: {BLOCKS * 256} values x{REPS} in {us:.0f} us on the NPU -> "
          f"{BLOCKS * 256 * REPS / us / 1e3:.2f} Gvalues/s on one core; max |err| / max |ref| {err:.2e} (bf16's ~4e-3)")

if __name__ == "__main__":
    main()
