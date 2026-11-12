# 0.7.0 spike: one core's fused expert matmul (fused_iq1s.cc) - 64 IQ1_S rows of 256 (UD-IQ1_M's own bytes)
# decoded in the core, C = A x W^T against the host's Aq x Wq^T (both operands quantized to bfp16 as the core does),
# and its rate: REPS of decode + matmul, and of the decode alone
#   python fused_bench.py   (~/npu-venv, mlir-aie's utils/env_setup.sh sourced)
import os, sys, numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.config import cxx_header_path
from aie.utils.benchmark import run_iters
from aie.utils import bfp as B
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.expanduser('~/Projects/mlir-aie/aie_kernels')
M, K, N = 64, 256, 64
V = int(sys.argv[1]) if len(sys.argv) > 1 else 2
WB, AB, CB = N * 50, M * K * 9 // 8, M * N * 9 // 8

@iron.jit
def fused1(w: In, a: In, c: Out, *, reps: CompileTime[int], decode_only: CompileTime[int] = 0, v: CompileTime[int] = 2, setup: CompileTime[int] = 0):
    wt, at, ct = (np.ndarray[(n,), np.dtype[np.uint8]] for n in (WB, AB, CB))
    ow, oa, oc = ObjectFifo(wt, name="gw", depth=1), ObjectFifo(at, name="ga", depth=1), ObjectFifo(ct, name="gc", depth=1)
    k = ExternalFunction("fused_iq1s", source_file=os.path.join(HERE, "fused_iq1s.cc"), arg_types=[wt, at, ct],
                         include_dirs=[cxx_header_path(), KD, HERE], compile_flags=[f"-DREPS={reps}", f"-DDECODE_ONLY={decode_only}"])
    def body(ow, oa, oc, k):
        x, y, z = ow.acquire(1), oa.acquire(1), oc.acquire(1); k(x, y, z); ow.release(1); oa.release(1); oc.release(1)
    wk = Worker(body, fn_args=[ow.cons(), oa.cons(), oc.prod(), k])
    def seq(w, a, c, pw, pa, cc):
        pw.fill(w); pa.fill(a); cc.drain(c, wait=True)
    return Program(iron.get_current_device(), Runtime(seq, [wt, at, ct, ow.prod(), oa.prod(), oc.cons()]), workers=[wk]).resolve_program()

def main():
    sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00002-of-00005.gguf'))
    t = next(t for t in r.tensors if t.name.endswith('ffn_gate_exps.weight') and t.tensor_type.name == 'IQ1_S')
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)[:WB].copy()
    Wf = dequantize(raw, t.tensor_type).astype(np.float32).reshape(N, K)
    rng = np.random.default_rng(1017)
    A = rng.standard_normal((M, K)).astype(np.float32)
    a_bytes = B.shuffle(B.encode(A), K, M, K, M).reshape(-1)
    Cref = B.quantize(A).astype(np.float64) @ B.quantize(Wf).astype(np.float64).T
    w = iron.tensor(raw, dtype=np.uint8, device="npu"); a = iron.tensor(a_bytes, dtype=np.uint8, device="npu")
    c = iron.zeros(CB, dtype=np.uint8, device="npu")
    run_iters(fused1, w, a, c, reps=1, v=V, warmup=0, iters=1)
    got = B.decode(B.shuffle(c.numpy(), N, M, N, M, unshuffle=True)).reshape(M, N)
    rel = np.abs(got - Cref).max() / np.abs(Cref).max()
    print(f"C against Aq x Wq^T: max |err| / max |C| {rel:.2e} (C's own bfp16 rounding ~1e-2); corr {np.corrcoef(got.ravel(), Cref.ravel())[0, 1]:.6f}")
    flop = 2 * M * K * N
    print(f'version {V}')
    us = {n: run_iters(fused1, w, a, c, reps=n, decode_only=1, v=V, setup=1, warmup=1, iters=5).npu.avg_us for n in (1, 65)}
    print(f"setup pass alone: {(us[65] - us[1]) / 64:.2f} us a tile")
    for do in (0, 1):
        us = {n: run_iters(fused1, w, a, c, reps=n, decode_only=do, v=V, warmup=1, iters=5).npu.avg_us for n in (1, 65)}
        per = (us[65] - us[1]) / 64
        what = ["decode + matmul", "decode alone", "matmul alone (+ C zeroed)"][do]
        print(f"{what}: {per:.2f} us a 64x256 weight tile" + ("" if do else f" -> {flop / per / 1e6:.3f} TFLOP/s one core at M = {M}"))

if __name__ == "__main__":
    main()
