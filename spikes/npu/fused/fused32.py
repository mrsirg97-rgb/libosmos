# 0.7.0 spike: one IQ2_XXS expert matmul on the whole array - C (64 x 2048) = A (64 x 4096) x W^T, W 2048 rows of
# 4096 in IQ2_XXS (UD-IQ1_M's bytes): core (row r, column c) owns 64 output rows, all 16 K-steps of 256 through it.
# a K-step's A tile (64 x 256 bfp16, 18 KB) from each column's shim through its memtile, broadcast to the column's
# 4 cores; a column's W stream (its 4 cores' tiles a K-step) split at its memtile; C (bfp16, accumulated in the core
# over the K-steps) joined at the memtile. checked against the host's Aq x Wq^T, timed.
#   python fused32.py [cols]   (~/npu-venv, mlir-aie's utils/env_setup.sh sourced)
import os, sys, numpy as np
import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.config import cxx_header_path
from aie.utils.benchmark import run_iters
from aie.utils import bfp as B
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.expanduser('~/Projects/mlir-aie/aie_kernels')
M, KT, NR, ROWS = 64, 256, 64, 4          # A rows, a K-step, a core's output rows, cores a column
KS = 16                                    # K-steps: K = 4096
COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = COLS * ROWS * NR
WT, AT, CT = NR * 66 // 4, M * KT * 9 // 8 // 4, M * NR * 9 // 8 // 4   # words: a core's W tile, an A tile, a C tile

def tap(total, off, n):  # n words from off, as rows of at most 1024
    inner = 1024 if n % 1024 == 0 else next(d for d in range(1024, 0, -1) if n % d == 0)
    return TensorAccessPattern((total,), off, [1, 1, n // inner, inner], [0, 0, inner, 1])

@iron.jit
def fused32(w: In, a: In, c: Out, *, cols: CompileTime[int]):
    i32 = np.dtype[np.int32]
    wt, at, ct = (np.ndarray[(n,), i32] for n in (WT, AT, CT))
    step = ExternalFunction("fused_step", source_file=os.path.join(HERE, "fused_iq2xxs.cc"),
                            arg_types=[wt, at, ct, np.int32], include_dirs=[cxx_header_path(), KD, HERE],
                            compile_flags=["-DFUSED_V=5", "-DMULTI"])
    a_l3, w_l3, c_l3, workers = [], [], [], []
    for col in range(cols):
        a3 = ObjectFifo(np.ndarray[(AT,), i32], name=f"A3_{col}", depth=1)
        a2 = a3.cons().forward(obj_type=at, name=f"A2_{col}", depth=1)
        w3 = ObjectFifo(np.ndarray[(ROWS * WT,), i32], name=f"W3_{col}", depth=2)
        w2 = w3.cons().split([WT * r for r in range(ROWS)], obj_types=[wt] * ROWS,
                             names=[f"W2_{col}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        c3 = ObjectFifo(np.ndarray[(ROWS * CT,), i32], name=f"C3_{col}", depth=1)
        c2 = c3.prod().join([CT * r for r in range(ROWS)], obj_types=[ct] * ROWS,
                            names=[f"C2_{col}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        a_l3.append(a3); w_l3.append(w3); c_l3.append(c3)
        for r in range(ROWS):
            def body(ai, wi, co, k):
                cc = co.acquire(1)
                for s in range_(KS):
                    x, y = wi.acquire(1), ai.acquire(1)
                    k(x, y, cc, s)
                    wi.release(1); ai.release(1)
                co.release(1)
            workers.append(Worker(body, fn_args=[a2.cons(), w2[r].cons(), c2[r].prod(), step], tile=Tile(col, 2 + r), stack_size=0xD00))
    TW, TA, TC = cols * KS * ROWS * WT, KS * AT, cols * ROWS * CT
    def seq(W, A, C, *hs):
        ap, wp, cc = hs[:cols], hs[cols:2 * cols], hs[2 * cols:]
        for col in range(cols):
            ap[col].fill(A, tap=tap(TA, 0, TA))
            wp[col].fill(W, tap=tap(TW, col * KS * ROWS * WT, KS * ROWS * WT))
        for col in range(cols):
            cc[col].drain(C, tap=tap(TC, col * ROWS * CT, ROWS * CT), wait=True)
    rt = Runtime(seq, [np.ndarray[(TW,), i32], np.ndarray[(TA,), i32], np.ndarray[(TC,), i32]]
                 + [f.prod() for f in a_l3] + [f.prod() for f in w_l3] + [f.cons() for f in c_l3])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

def main():
    sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00002-of-00005.gguf'))
    t = next(t for t in r.tensors if t.name.endswith('ffn_down_exps.weight') and t.tensor_type.name == 'IQ2_XXS')
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)[:N * KS * 66].copy().reshape(N, KS, 66)  # row n, block ks
    Wf = dequantize(raw.reshape(-1), t.tensor_type).astype(np.float32).reshape(N, KS * KT)
    # W stream: [col][ks][r][its 64 rows' block ks]; A stream: [ks][the K-step's tile, encoded + shuffled]
    w = np.empty((COLS, KS, ROWS, NR, 66), np.uint8)
    for col in range(COLS):
        for r_ in range(ROWS):
            band = (col * ROWS + r_) * NR
            w[col, :, r_] = raw[band:band + NR].transpose(1, 0, 2)
    rng = np.random.default_rng(1017)
    A = rng.standard_normal((M, KS * KT)).astype(np.float32)
    a = np.concatenate([B.shuffle(B.encode(A[:, s * KT:(s + 1) * KT]), KT, M, KT, M).reshape(-1) for s in range(KS)])
    Cref = B.quantize(A).astype(np.float64) @ np.concatenate(
        [B.quantize(Wf[:, s * KT:(s + 1) * KT]) for s in range(KS)], axis=1).astype(np.float64).T
    wt_ = iron.tensor(w.reshape(-1).view(np.int32), dtype=np.int32, device="npu")
    at_ = iron.tensor(a.view(np.int32), dtype=np.int32, device="npu")
    ct_ = iron.zeros(COLS * ROWS * CT, dtype=np.int32, device="npu")
    bench = run_iters(fused32, wt_, at_, ct_, cols=COLS, warmup=1, iters=5)
    cb = ct_.numpy().view(np.uint8).reshape(COLS * ROWS, M * NR * 9 // 8)
    got = np.concatenate([B.decode(B.shuffle(cb[i], NR, M, NR, M, unshuffle=True)).reshape(M, NR) for i in range(COLS * ROWS)], axis=1)
    rel = np.abs(got - Cref).max() / np.abs(Cref).max()
    us = bench.npu.avg_us
    flop = 2 * M * KS * KT * N
    print(f"{COLS} columns x {ROWS} cores: C {M} x {N}, K {KS * KT}: corr {np.corrcoef(got.ravel(), Cref.ravel())[0, 1]:.6f}, "
          f"max err / max |C| {rel:.2e}; {us:.0f} us on the NPU -> {flop / us / 1e6:.2f} TFLOP/s at M = {M} "
          f"({flop / us / 1e6 / (COLS * ROWS):.3f} a core; one core alone 0.163)")

if __name__ == "__main__":
    main()
