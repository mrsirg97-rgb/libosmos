# 0.7.0: DeepSeek's routed expert gate + up for 128 tokens on the whole array, at its real shape and from its own
# bytes - the engine's first NPU kernel. C_gate, C_up (128 x 2304, f32) = A (128 x 5120) x W^T, W an expert's
# ffn_gate_exps / ffn_up_exps (IQ2_XXS, 2304 rows of 5120: 1,320 bytes a row, GGUF's order) read as they lie in
# memory: no relayout on the host. a W tile is a K-step pair (two 66-byte blocks a row, 132 bytes: the DMA's words
# need it - one block's 66 is not a multiple of 4) for 24 rows; core i (column c, row r: i = 4 c + r) owns rows
# (32 p + i) 24 .. + 24 for passes p = 0, 1, 2 (2304 = 3 x 32 x 24). A as fused32.py's (bfp16, a quarter at a time,
# broadcast down a column), streamed once a matrix's pass; C f32 a core's (128 x 24) tile a pass, joined at the
# memtile. checked against the host's bfp16-rounded reference, timed.
#   python fused_ds.py [layer] [expert]   (~/npu-venv, mlir-aie's utils/env_setup.sh sourced)
import os, sys, glob, numpy as np
import aie.iron as iron
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.config import cxx_header_path
from aie.utils.benchmark import run_iters
from aie.utils import bfp as B

sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.expanduser('~/Projects/mlir-aie/aie_kernels')
COLS, ROWS, NR, H = 8, 4, 24, 2           # columns, cores a column, a core's rows a pass, A's 64-row halves
M, KT, BB = 64 * H, 256, 66               # tokens, a K-step, an IQ2_XXS block's bytes
K, N = 5120, 2304                         # DeepSeek's hidden and expert width (gate / up: K -> N)
KS, KP = K // KT, K // KT // 2            # K-steps, K-step pairs
PASSES = N // (COLS * ROWS * NR)          # 3
RW = K // KT * BB // 4                    # a row's words: 330
WT = NR * 2 * BB // 4                     # a core's W tile (24 rows x 132 bytes): 792 words
AT = 64 * KT * 9 // 8 // 4 // 2           # an A quarter (64 x 128 bfp16): 2304 words
CT = M * NR                               # a core's C tile a pass (f32)
MATS = 2                                  # gate, up
assert N == PASSES * COLS * ROWS * NR

def tap(total, off, sizes, strides):
    return TensorAccessPattern((total,), off, sizes, strides)

@iron.jit
def fused_ds(wg: In, wu: In, a: In, c: Out):
    i32, f32 = np.dtype[np.int32], np.dtype[np.float32]
    wt, at, ct = np.ndarray[(WT,), i32], np.ndarray[(AT,), i32], np.ndarray[(CT,), f32]
    step = ExternalFunction("fused_step_f32", source_file=os.path.join(HERE, "fused_iq2xxs.cc"),
                            arg_types=[wt, at, ct, np.int32, np.int32], include_dirs=[cxx_header_path(), KD, HERE],
                            compile_flags=["-DFUSED_V=6", "-DMULTI_F32", f"-DHALVES={H}", "-DKHALF=1", f"-DNROWS={NR}",
                                           "-DROWSTRIDE=132"])
    a_l3, w_l3, c_l3, workers = [], [], [], []
    for col in range(COLS):
        a3 = ObjectFifo(np.ndarray[(AT,), i32], name=f"A3_{col}", depth=2)
        a2 = a3.cons().forward(obj_type=at, name=f"A2_{col}", depth=2)
        w3 = ObjectFifo(np.ndarray[(ROWS * WT,), i32], name=f"W3_{col}", depth=2)
        w2 = w3.cons().split([WT * r for r in range(ROWS)], obj_types=[wt] * ROWS,
                             names=[f"W2_{col}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        c3 = ObjectFifo(np.ndarray[(ROWS * CT,), f32], name=f"C3_{col}", depth=1)
        c2 = c3.prod().join([CT * r for r in range(ROWS)], obj_types=[ct] * ROWS,
                            names=[f"C2_{col}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        a_l3.append(a3); w_l3.append(w3); c_l3.append(c3)
        for r in range(ROWS):
            def body(ai, wi, co, k):
                for _ in range_(MATS * PASSES):
                    cc = co.acquire(1)
                    for s in range_(KP):
                        x = wi.acquire(1)
                        for b in range(2):
                            for piece in range(2 * H):
                                y = ai.acquire(1)
                                k(x, y, cc, 2 * s + b, piece)
                                ai.release(1)
                        wi.release(1)
                    co.release(1)
            workers.append(Worker(body, fn_args=[a2.cons(), w2[r].cons(), c2[r].prod(), step], tile=Tile(col, 2 + r),
                                  stack_size=0x1100))
    TW, TA, TC = N * RW, KS * 2 * H * AT, MATS * N * M

    def seq(WG, WU, A, C, *hs):
        ap, wp, cp = hs[:COLS], hs[COLS:2 * COLS], hs[2 * COLS:]
        for col in range(COLS):
            # A: the whole stream once a (matrix, pass)
            ap[col].fill(A, tap=tap(TA, 0, [MATS * PASSES, TA // 1024, 1024], [0, 1024, 1]))
            # W: pass p, pair s, the column's 96 rows (its 4 cores' 24 each, in order), 33 words each
            base = col * ROWS * NR * RW
            for Wm in (WG, WU):
                wp[col].fill(Wm, tap=tap(TW, base, [PASSES, KP, ROWS * NR, 2 * BB // 4],
                                         [COLS * ROWS * NR * RW, 2 * BB // 4, RW, 1]))
        for col in range(COLS):
            # C: [matrix][pass][column][core][128 x 24 f32, sub-tile order]
            cp[col].drain(C, tap=tap(TC, col * ROWS * CT, [MATS * PASSES, ROWS * CT // 1024, 1024],
                                     [COLS * ROWS * CT, 1024, 1]), wait=True)
    rt = Runtime(seq, [np.ndarray[(TW,), i32], np.ndarray[(TW,), i32], np.ndarray[(TA,), i32], np.ndarray[(TC,), f32]]
                 + [f.prod() for f in a_l3] + [f.prod() for f in w_l3] + [f.cons() for f in c_l3])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

def expert_bytes(layer, kind, e):
    sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    for p in sorted(glob.glob(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/*.gguf'))):
        for t in GGUFReader(p).tensors:
            if t.name == f"blk.{layer}.{kind}.weight":
                assert t.tensor_type.name == 'IQ2_XXS', (t.name, t.tensor_type.name)
                raw = np.asarray(t.data).reshape(-1).view(np.uint8)
                return raw[e * N * RW * 4:(e + 1) * N * RW * 4].copy(), t.tensor_type
    raise SystemExit(f"no blk.{layer}.{kind}")

def main():
    from gguf.quants import dequantize
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if layer < 0:  # the first layer whose gate and up are both IQ2_XXS
        sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
        from gguf import GGUFReader
        ty = {}
        for p in sorted(glob.glob(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/*.gguf'))):
            for t in GGUFReader(p).tensors:
                if '_exps' in t.name: ty[t.name] = t.tensor_type.name
        layer = next(l for l in range(64) if ty.get(f"blk.{l}.ffn_gate_exps.weight") == 'IQ2_XXS'
                     and ty.get(f"blk.{l}.ffn_up_exps.weight") == 'IQ2_XXS')
    (g, tt), (u, _) = expert_bytes(layer, 'ffn_gate_exps', e), expert_bytes(layer, 'ffn_up_exps', e)
    rng = np.random.default_rng(1017)
    A = rng.standard_normal((M, K)).astype(np.float32)
    a = np.concatenate([B.shuffle(B.encode(A[h * 64:(h + 1) * 64, s * KT + q * 128:s * KT + (q + 1) * 128]), 128, 64, 128, 64).reshape(-1)
                        for s in range(KS) for h in range(H) for q in range(2)])
    Aq = B.quantize(A).astype(np.float64)
    refs = []
    for raw in (g, u):
        Wf = dequantize(raw, tt).astype(np.float32).reshape(N, K)
        refs.append(Aq @ np.concatenate([B.quantize(Wf[:, s * KT:(s + 1) * KT]) for s in range(KS)], axis=1).astype(np.float64).T)
    wg_ = iron.tensor(g.view(np.int32), dtype=np.int32, device="npu")
    wu_ = iron.tensor(u.view(np.int32), dtype=np.int32, device="npu")
    at_ = iron.tensor(a.view(np.int32), dtype=np.int32, device="npu")
    ct_ = iron.zeros(MATS * N * M, dtype=np.float32, device="npu")
    bench = run_iters(fused_ds, wg_, wu_, at_, ct_, warmup=1, iters=5)
    # C: [mat][pass][core i][half][z][j][8 m + n]; core i's rows of pass p: (32 p + i) 24 ..
    cf = ct_.numpy().reshape(MATS, PASSES, COLS * ROWS, H, 8, NR // 8, 8, 8)
    worst, corr = 0.0, 1.0
    for m in range(MATS):
        got = np.empty((M, N), np.float32)
        for p in range(PASSES):
            for i in range(COLS * ROWS):
                n0 = (32 * p + i) * NR
                got[:, n0:n0 + NR] = cf[m, p, i].transpose(0, 1, 3, 2, 4).reshape(M, NR)
        print(f'  mat {m}: |got| max {np.abs(got).max():.4g} mean {np.abs(got).mean():.4g}, |ref| max {np.abs(refs[m]).max():.4g}, max diff {np.abs(got - refs[m]).max():.3g}, nonzero {np.count_nonzero(got)}/{got.size}')
        bad = ~np.isfinite(got)
        if bad.any():
            nb = bad.reshape(M, PASSES, COLS * ROWS, NR).transpose(1, 2, 0, 3).reshape(PASSES, COLS * ROWS, -1).sum(-1)
            print(f'  mat {m}: {bad.sum()} non-finite; by pass x core:\n{nb}')
            fin = np.isfinite(got)
            print('  finite vs ref: max diff', np.abs(got[fin] - refs[m][fin]).max(), ' ref max', np.abs(refs[m]).max())
            print('  bad by token (first 16):', bad.sum(1)[:16], ' tokens with any:', (bad.sum(1) > 0).sum())
            print('  bad columns (n) with any:', (bad.sum(0) > 0).sum(), 'of', N, '; col counts max', bad.sum(0).max())
            worst, corr = np.inf, np.nan
            continue
        worst = max(worst, np.abs(got - refs[m]).max() / np.abs(refs[m]).max())
        corr = min(corr, np.corrcoef(got.ravel(), refs[m].ravel())[0, 1])
    us = bench.npu.avg_us
    flop = 2 * MATS * M * K * N
    print(f"blk.{layer} expert {e}: gate + up, {M} x {K} -> {N}: corr {corr:.6f}, max err / max |C| {worst:.2e}; "
          f"{us:.0f} us on the NPU -> {flop / us / 1e6:.2f} TFLOP/s ({flop / 1e9:.2f} GFLOP)")

if __name__ == "__main__":
    main()
