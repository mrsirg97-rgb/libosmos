# 0.7.0: DeepSeek's routed expert gate + up for 128 tokens on the whole array, tiled for the data each core receives.
# fused_ds.py's cores each held 24 rows of 128 tokens' C and made 3 passes of rows a matrix, A (all 128 tokens) streamed
# into every core again each pass: ~4.4 MB a core a part, at one stream port's rate (~5 GB/s) the kernel's whole time
# (its cores idle: the same with no decode or matmul at all). here a core holds all its rows - 72 of 2304 - for MT
# tokens at a time: MT / 128 token passes a matrix, A streamed once a matrix (1.5 MB a core a part), the core's W (95 KB)
# once a token pass, decoded again each (its decode x 128 / MT: the price). A broadcast from one shim to all 32 cores;
# a W tile a K-step pair (132 bytes a row: the DMA's words); C in rows (c_rows), each core's tile where its rows land.
#   python fused_ds2.py [layer] [expert]   (FMT=iq2xxs | iq1s | iq2xs; ~/npu-venv, mlir-aie's env_setup.sh sourced)
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
FMT = os.environ.get('FMT', 'iq2xxs')
QT, BB, SRC, FLAGS = {'iq2xxs': ('IQ2_XXS', 66, 'fused_iq2xxs.cc', ['-DFUSED_V=6']),
                      'iq1s': ('IQ1_S', 50, 'fused_iq1s.cc', ['-DIQ1_V=5']),
                      'iq2xs': ('IQ2_XS', 74, 'fused_iq2xs.cc', [])}[FMT]
COLS, ROWS = 8, 4                        # columns, cores a column
M, KT = 128, 256                         # tokens, a K-step
MT = int(os.environ.get('MT', '32'))     # a token pass's tokens
MP = M // MT                             # token passes a matrix
K, N = 5120, 2304                        # DeepSeek's hidden and expert width (gate / up: K -> N)
KS, KP = K // KT, K // KT // 2           # K-steps, K-step pairs
NR = N // (COLS * ROWS)                  # a core's rows: 72, all of them at once
MATS = 2                                 # gate, up
RB = KS * BB                             # a row's bytes (1,320 IQ2_XXS)
RW = RB // 4                             # its words
WT = NR * 2 * BB // 4                    # a core's W tile: its rows' K-step pair (72 x 132 bytes)
AT = MT * 128 * 9 // 8 // 4              # an A piece: MT tokens x a K half (128) of bfp16
CT = MT * NR                             # a core's C tile (f32)
assert MT % 8 == 0 and M % MT == 0 and NR % 8 == 0 and RB % 4 == 0

def tap(total, off, sizes, strides):
    return TensorAccessPattern((total,), off, sizes, strides)

@iron.jit
def fused_ds2(wg: In, wu: In, a: In, c: Out, c2: Out):  # c: gate's rows, c2: up's
    i32, f32 = np.dtype[np.int32], np.dtype[np.float32]
    wt, at, ct = np.ndarray[(WT,), i32], np.ndarray[(AT,), i32], np.ndarray[(CT,), f32]
    step = ExternalFunction("fused_step_ms", source_file=os.path.join(HERE, SRC),
                            arg_types=[wt, at, ct, np.int32, np.int32], include_dirs=[cxx_header_path(), KD, HERE],
                            compile_flags=FLAGS + ["-DMULTI_F32", "-DKHALF=1", f"-DNROWS={NR}", f"-DROWSTRIDE={2 * BB}",
                                                   f"-DMSPLIT={MT}", f"-DROWMAJOR={KS}", f"-DSPLIT={os.environ.get('SPLIT', '0')}"])
    a3 = ObjectFifo(np.ndarray[(AT,), i32], name="A3", depth=2)  # A: one stream, broadcast to every core
    a2 = a3.cons().forward(obj_type=at, name="A2", depth=2)
    w_l3, c_l3, workers = [], [], []
    for col in range(COLS):
        w3 = ObjectFifo(np.ndarray[(ROWS * WT,), i32], name=f"W3_{col}", depth=2)
        w2 = w3.cons().split([WT * r for r in range(ROWS)], obj_types=[wt] * ROWS,
                             names=[f"W2_{col}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        c3 = ObjectFifo(np.ndarray[(ROWS * CT,), f32], name=f"C3_{col}", depth=1)
        c2_ = c3.prod().join([CT * r for r in range(ROWS)], obj_types=[ct] * ROWS,
                             names=[f"C2_{col}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        w_l3.append(w3); c_l3.append(c3)
        for r in range(ROWS):
            def body(ai, wi, co, k):
                for _ in range_(MATS * MP):  # a matrix's token passes, gate's then up's
                    cc = co.acquire(1)
                    for s in range_(KP):
                        x = wi.acquire(1)
                        for b in range(2):
                            for kh in range(2):
                                y = ai.acquire(1)
                                k(x, y, cc, 2 * s + b, kh)
                                ai.release(1)
                        wi.release(1)
                    co.release(1)
            workers.append(Worker(body, fn_args=[a2.cons(), w2[r].cons(), c2_[r].prod(), step], tile=Tile(col, 2 + r),
                                  stack_size=0x1100))
    TW, TA, TC = N * RW, MP * KS * 2 * AT, M * N

    def seq(WG, WU, A, C, C2, *hs):
        ap, wp, cp = hs[0], hs[1:1 + COLS], hs[1 + COLS:]
        inner = next(d for d in range(1024, 0, -1) if TA % d == 0)
        ap.fill(A, tap=tap(TA, 0, [MATS, TA // inner, inner], [0, inner, 1]))  # [pass][K-step][half] a matrix
        for col in range(COLS):
            base = col * ROWS * NR * RW  # the column's 4 x 72 rows, contiguous
            for Wm in (WG, WU):  # token pass (again), pair s, the column's rows, a pair's 33 words each
                wp[col].fill(Wm, tap=tap(TW, base, [MP, KP, ROWS * NR, 2 * BB // 4], [0, 2 * BB // 4, RW, 1]))
        for col in range(COLS):  # token pass p, core r, its MT rows of NR where they land
            for Cm in (C, C2):
                cp[col].drain(Cm, tap=tap(TC, col * ROWS * NR, [MP, ROWS, MT, NR], [MT * N, NR, N, 1]), wait=True)
    rt = Runtime(seq, [np.ndarray[(TW,), i32], np.ndarray[(TW,), i32], np.ndarray[(TA,), i32], np.ndarray[(TC,), f32],
                       np.ndarray[(TC,), f32]] + [a3.prod()] + [f.prod() for f in w_l3] + [f.cons() for f in c_l3])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

def expert_bytes(layer, kind, e):
    from gguf import GGUFReader
    for p in sorted(glob.glob(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/*.gguf'))):
        for t in GGUFReader(p).tensors:
            if t.name == f"blk.{layer}.{kind}.weight":
                assert t.tensor_type.name == QT, (t.name, t.tensor_type.name)
                raw = np.asarray(t.data).reshape(-1).view(np.uint8)
                return raw[e * N * RB:(e + 1) * N * RB].copy(), t.tensor_type
    raise SystemExit(f"no blk.{layer}.{kind}")

def main():
    from gguf import GGUFReader
    from gguf.quants import dequantize
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if layer < 0:
        ty = {t.name: t.tensor_type.name for p in sorted(glob.glob(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/*.gguf')))
              for t in GGUFReader(p).tensors if '_exps' in t.name}
        layer = next(l for l in range(64) if ty.get(f"blk.{l}.ffn_gate_exps.weight") == QT
                     and ty.get(f"blk.{l}.ffn_up_exps.weight") == QT)
    (g, tt), (u, _) = expert_bytes(layer, 'ffn_gate_exps', e), expert_bytes(layer, 'ffn_up_exps', e)
    A = np.random.default_rng(1017).standard_normal((M, K)).astype(np.float32)
    # A's stream: [token pass][K-step][K half], each piece MT x 128, encoded and in the mmul's sub-tile order
    a = np.concatenate([B.shuffle(B.encode(A[p * MT:(p + 1) * MT, s * KT + q * 128:s * KT + (q + 1) * 128]), 128, MT, 128, MT).reshape(-1)
                        for p in range(MP) for s in range(KS) for q in range(2)])
    Aq = B.quantize(A).astype(np.float64)
    refs = []
    for raw in (g, u):
        Wf = dequantize(raw, tt).astype(np.float32).reshape(N, K)
        refs.append(Aq @ np.concatenate([B.quantize(Wf[:, s * KT:(s + 1) * KT]) for s in range(KS)], axis=1).astype(np.float64).T)
    wg_ = iron.tensor(g.view(np.int32), dtype=np.int32, device="npu")
    wu_ = iron.tensor(u.view(np.int32), dtype=np.int32, device="npu")
    at_ = iron.tensor(a.view(np.int32), dtype=np.int32, device="npu")
    ct_, ct2_ = iron.zeros(M * N, dtype=np.float32, device="npu"), iron.zeros(M * N, dtype=np.float32, device="npu")
    bench = run_iters(fused_ds2, wg_, wu_, at_, ct_, ct2_, warmup=1, iters=5)
    worst, corr = 0.0, 1.0
    for m, out in enumerate((ct_, ct2_)):
        got = out.numpy().reshape(M, N).astype(np.float64)
        if not np.isfinite(got).all():
            bad = ~np.isfinite(got)
            print(f"  mat {m}: {bad.sum()} non-finite; tokens with any {(bad.sum(1) > 0).sum()}, rows with any {(bad.sum(0) > 0).sum()}")
            worst, corr = np.inf, np.nan
            continue
        worst = max(worst, np.abs(got - refs[m]).max() / np.abs(refs[m]).max())
        corr = min(corr, np.corrcoef(got.ravel(), refs[m].ravel())[0, 1])
    us = bench.npu.avg_us
    flop = 2 * MATS * M * K * N
    print(f"ds2 {FMT} blk.{layer} expert {e}: gate + up, {M} x {K} -> {N}, {MT}-token passes: corr {corr:.6f}, "
          f"max err / max |C| {worst:.2e}; {us:.0f} us on the NPU -> {flop / us / 1e6:.2f} TFLOP/s")

if __name__ == "__main__":
    main()
