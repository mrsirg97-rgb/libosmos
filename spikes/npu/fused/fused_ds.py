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
COLS, ROWS, H = 8, 4, 2                   # columns, cores a column, A's 64-row halves
FMT = os.environ.get('FMT', 'iq2xxs')     # the experts' format: IQ2_XXS (66-byte blocks) or IQ1_S (50)
QT, BB, SRC, FLAGS = {'iq2xxs': ('IQ2_XXS', 66, 'fused_iq2xxs.cc', ['-DFUSED_V=6']),
                      'iq1s': ('IQ1_S', 50, 'fused_iq1s.cc', ['-DIQ1_V=5']),
                      'iq2xs': ('IQ2_XS', 74, 'fused_iq2xs.cc', []),
                      'iq2s': ('IQ2_S', 82, 'fused_qwen.cc', ['-DQFMT=3'])}[FMT]  # its grid packed (2 KB)
# gateup: gate and up (K 5120 -> N 2304), a W tile a K-step pair (two blocks a row: 4-byte words) for 24 rows, 3
# passes. down: K 2304 -> N 5120 - a row's 9 blocks are no whole words (594 bytes), a core's 16 rows are: its W tile
# those rows whole, once a pass, a block decoded a K-step; 10 passes
KIND = os.environ.get('KIND', 'gateup')
ROWMAJ = int(os.environ.get('ROWMAJOR', '1'))  # C to rows where the GPU reads (else sub-tile order, contiguous: timing only)
DOWN = KIND == 'down'
M, KT = 64 * H, 256                       # tokens, a K-step
K, N = (2304, 5120) if DOWN else (5120, 2304)
NR = 16 if DOWN else 24                   # a core's rows a pass
MATS = 1 if DOWN else 2                   # down; gate, up
KS, KP = K // KT, K // KT // 2            # K-steps, K-step pairs
PASSES = N // (COLS * ROWS * NR)          # 10; 3
RB = K // KT * BB                         # a row's bytes (594 IQ2_XXS down: not a whole word count)
WT = (NR * RB if DOWN else NR * 2 * BB) // 4  # a core's W tile, words
AT = 64 * KT * 9 // 8 // 4 // 2           # an A quarter (64 x 128 bfp16): 2304 words
CT = M * NR                               # a core's C tile a pass (f32)
assert N == PASSES * COLS * ROWS * NR and (NR * RB) % 4 == 0

def tap(total, off, sizes, strides):
    return TensorAccessPattern((total,), off, sizes, strides)

@iron.jit
def fused_ds(wg: In, wu: In, a: In, c: Out, c2: Out):  # c: gate's (or the down's) rows, c2: up's (the down: unwritten)
    i32, f32 = np.dtype[np.int32], np.dtype[np.float32]
    wt, at, ct = np.ndarray[(WT,), i32], np.ndarray[(AT,), i32], np.ndarray[(CT,), f32]
    step = ExternalFunction("fused_step_f32", source_file=os.path.join(HERE, SRC),
                            arg_types=[wt, at, ct, np.int32, np.int32], include_dirs=[cxx_header_path(), KD, HERE],
                            compile_flags=FLAGS + ["-DMULTI_F32", f"-DHALVES={H}", "-DKHALF=1", f"-DNROWS={NR}",
                                                   f"-DROWSTRIDE={RB if DOWN else 2 * BB}", f"-DSPLIT={os.environ.get('SPLIT', '0')}"] + ([f"-DROWMAJOR={KS}"] if ROWMAJ else []) + (["-DWHOLEK"] if DOWN else []))
    a_l3, w_l3, c_l3, workers = [], [], [], []
    # A: one stream from one shim, through one memtile, to all 32 cores (ABCAST, the default): every core multiplies
    # the same A tiles, so a column's own copy was 8 x the DRAM reads; ABCAST=0 a stream a column, as before
    ABC = int(os.environ.get('ABCAST', '1'))
    if ABC:
        a3b = ObjectFifo(np.ndarray[(AT,), i32], name="A3", depth=2)
        a2b = a3b.cons().forward(obj_type=at, name="A2", depth=2)
        a_l3.append(a3b)
    for col in range(COLS):
        if ABC:
            a2 = a2b
        else:
            a3 = ObjectFifo(np.ndarray[(AT,), i32], name=f"A3_{col}", depth=2)
            a2 = a3.cons().forward(obj_type=at, name=f"A2_{col}", depth=2)
        w3 = ObjectFifo(np.ndarray[(ROWS * WT,), i32], name=f"W3_{col}", depth=2)
        w2 = w3.cons().split([WT * r for r in range(ROWS)], obj_types=[wt] * ROWS,
                             names=[f"W2_{col}_{r}" for r in range(ROWS)], depths=[2] * ROWS)
        c3 = ObjectFifo(np.ndarray[(ROWS * CT,), f32], name=f"C3_{col}", depth=1)
        c2 = c3.prod().join([CT * r for r in range(ROWS)], obj_types=[ct] * ROWS,
                            names=[f"C2_{col}_{r}" for r in range(ROWS)], depths=[1] * ROWS)
        if not ABC:
            a_l3.append(a3)
        w_l3.append(w3); c_l3.append(c3)
        for r in range(ROWS):
            def body(ai, wi, co, k):
                for _ in range_(MATS * PASSES):
                    cc = co.acquire(1)
                    if DOWN:  # the rows whole: one tile a pass, a block a K-step
                        x = wi.acquire(1)
                        for s in range_(KS):
                            for piece in range(2 * H):
                                y = ai.acquire(1)
                                k(x, y, cc, s, piece)
                                ai.release(1)
                        wi.release(1)
                    else:
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
    TW, TA, TC = N * RB // 4, KS * 2 * H * AT, M * N  # C: a matrix's rows, M x N f32, row-major

    def seq(WG, WU, A, C, C2, *hs):  # (down: WG its matrix, WU unread, C2 unwritten)
        na = len(a_l3)
        ap, wp, cp = hs[:na], hs[na:na + COLS], hs[na + COLS:]
        for col in range(COLS):
            # A: the whole stream once a (matrix, pass)
            if col < na:
                ap[col].fill(A, tap=tap(TA, 0, [MATS * PASSES, TA // 1024, 1024], [0, 1024, 1]))
            if DOWN:  # W: pass p, the column's 64 rows whole (its 4 cores' 16, in order): contiguous
                base, n = col * ROWS * WT, ROWS * WT
                inner = next(d for d in range(1024, 0, -1) if n % d == 0)
                wp[col].fill(WG, tap=tap(TW, base, [PASSES, n // inner, inner], [COLS * ROWS * WT, inner, 1]))
            else:  # W: pass p, pair s, the column's 96 rows (its 4 cores' 24 each, in order), 33 words each
                RW = RB // 4
                base = col * ROWS * NR * RW
                for Wm in (WG, WU):
                    wp[col].fill(Wm, tap=tap(TW, base, [PASSES, KP, ROWS * NR, 2 * BB // 4],
                                             [COLS * ROWS * NR * RW, 2 * BB // 4, RW, 1]))
        for col in range(COLS):
            # C: a pass's core tiles (in rows: the kernels' c_rows) where their rows go - pass p, the column's core r,
            # row m, its NR columns at (32 p + 4 col + r) NR; gate to C, up to C2 (the down: C, its passes alone)
            for Cm in ((C,) if DOWN else (C, C2)):
                # (a core's NR-word rows a burst each: the NPU's time, off the GPU's path - its rows outlast these)
                if ROWMAJ:
                    cp[col].drain(Cm, tap=tap(TC, col * ROWS * NR, [PASSES, ROWS, M, NR], [COLS * ROWS * NR, NR, N, 1]),
                                  wait=True)
                else:  # timing only: each core's tile contiguous ([pass][column][core][tile]), the values unchecked
                    cp[col].drain(Cm, tap=tap(TC, col * ROWS * CT, [PASSES, ROWS * CT // 1024, 1024],
                                              [COLS * ROWS * CT, 1024, 1]), wait=True)
    rt = Runtime(seq, [np.ndarray[(TW,), i32], np.ndarray[(TW,), i32], np.ndarray[(TA,), i32], np.ndarray[(TC,), f32], np.ndarray[(TC,), f32]]
                 + [f.prod() for f in a_l3] + [f.prod() for f in w_l3] + [f.cons() for f in c_l3])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

def expert_bytes(layer, kind, e):
    sys.path.insert(0, os.path.expanduser('~/Projects/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    for p in sorted(glob.glob(os.path.expanduser('~/Models/ds1017/UD-IQ1_M/*.gguf'))):
        for t in GGUFReader(p).tensors:
            if t.name == f"blk.{layer}.{kind}.weight":
                assert t.tensor_type.name == QT, (t.name, t.tensor_type.name)
                raw = np.asarray(t.data).reshape(-1).view(np.uint8)
                return raw[e * N * RB:(e + 1) * N * RB].copy(), t.tensor_type
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
        layer = next(l for l in range(64) if (ty.get(f"blk.{l}.ffn_down_exps.weight") == QT if DOWN else
                     ty.get(f"blk.{l}.ffn_gate_exps.weight") == QT and ty.get(f"blk.{l}.ffn_up_exps.weight") == QT))
    if DOWN:
        (g, tt) = expert_bytes(layer, 'ffn_down_exps', e)
        u = g
    else:
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
    ct_ = iron.zeros(M * N, dtype=np.float32, device="npu")
    ct2_ = iron.zeros(M * N, dtype=np.float32, device="npu")
    bench = run_iters(fused_ds, wg_, wu_, at_, ct_, ct2_, warmup=1, iters=5)
    if os.environ.get('DUMP'):  # the host harness's inputs and this run's output (host/ds_npu.hip: bit-identical)
        d = os.environ['DUMP']; os.makedirs(d, exist_ok=True)
        for name, arr in (('wg', g), ('wu', u), ('a', a), ('c', np.concatenate([ct_.numpy(), ct2_.numpy()])), ('x', A)):
            np.ascontiguousarray(arr).tofile(os.path.join(d, name + '.bin'))
        # the engine's encoding (kernels/npu_pack.hip): nearest-even, the exponent raised on a carry - its bytes,
        # and the reference it should reach
        def enc_even(X):
            bl = X.reshape(X.shape[0], -1, 8)
            e = ((bl.view(np.uint32) >> 23) & 0xFF).max(-1, keepdims=True).astype(np.int64)
            q = np.rint(np.ldexp(bl.astype(np.float64), (133 - e)))
            cy = (q > 127).any(-1, keepdims=True)
            e = e + cy
            q = np.clip(np.rint(np.ldexp(bl.astype(np.float64), (133 - e))), -128, 127).astype(np.int8)
            out = np.empty(bl.shape[:2] + (9,), np.uint8)
            out[..., 0] = e[..., 0].astype(np.uint8); out[..., 1:] = q.view(np.uint8)
            return out.reshape(X.shape[0], -1)
        ae = np.concatenate([B.shuffle(enc_even(A[h * 64:(h + 1) * 64, s * KT + q * 128:s * KT + (q + 1) * 128]), 128, 64, 128, 64).reshape(-1)
                             for s in range(KS) for h in range(H) for q in range(2)])
        ae.tofile(os.path.join(d, 'a_even.bin'))
        Ae = B.quantize(A, rounding='conv_even').astype(np.float64)
        re = np.stack([Ae @ np.concatenate([B.quantize(dequantize(raw, tt).astype(np.float32).reshape(N, K)[:, s * KT:(s + 1) * KT])
                                            for s in range(KS)], axis=1).astype(np.float64).T for raw in (g, u)]).astype(np.float32)
        re.tofile(os.path.join(d, 'ref_even.bin'))
        GOT = []
    # C: [mat][pass][core i][half][z][j][8 m + n]; core i's rows of pass p: (32 p + i) 24 ..
    outs = [ct_.numpy().reshape(M, N), ct2_.numpy().reshape(M, N)]  # in rows, where the GPU reads them
    worst, corr = 0.0, 1.0
    for m in range(MATS):
        got = outs[m].astype(np.float32)
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
        if os.environ.get('DUMP'): GOT.append(got.copy())
        worst = max(worst, np.abs(got - refs[m]).max() / np.abs(refs[m]).max())
        corr = min(corr, np.corrcoef(got.ravel(), refs[m].ravel())[0, 1])
    if os.environ.get('DUMP') and len(GOT) == 2: np.stack(GOT).astype(np.float32).tofile(os.path.join(os.environ['DUMP'], 'got.bin'))
    us = bench.npu.avg_us
    flop = 2 * MATS * M * K * N
    print(f"{FMT} blk.{layer} expert {e}: {'down' if DOWN else 'gate + up'}, {M} x {K} -> {N}: corr {corr:.6f}, max err / max |C| {worst:.2e}; "
          f"{us:.0f} us on the NPU -> {flop / us / 1e6:.2f} TFLOP/s ({flop / 1e9:.2f} GFLOP)")

if __name__ == "__main__":
    main()
