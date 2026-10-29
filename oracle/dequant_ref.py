"""reference dequantization for tests/gpu/test_quant_gpu.hip: per type, real blocks of a ds1017 tensor (its start, a
seeded middle, its end) and random blocks (any bytes are a block; those whose values are not all finite are drawn
again), each with gguf-py's values. one file a type: b"SUBQ", u32 version, u32 type, u64 values, u64 bytes, the bytes,
the values (f32). run: PYTHONPATH=<llama.cpp>/gguf-py ../.oracle-venv/bin/python dequant_ref.py --out DIR"""
import argparse
import glob
import os
import struct

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T, quants as Q
from gguf.constants import GGML_QUANT_SIZES

MODEL = os.path.expanduser("~/models/ds1017")
TENSORS = {  # a tensor of each type the model has, and the file it is in
    T.Q8_0: ("blk.0.attn_q_a.weight", "model"),
    T.Q5_K: ("blk.0.attn_q_b.weight", "model"),
    T.Q6_K: ("blk.0.attn_output_b.weight", "model"),
    T.IQ2_XXS: ("blk.10.ffn_gate_exps.weight", "model"),
    T.IQ2_XS: ("blk.17.ffn_gate_exps.weight", "model"),
    T.IQ1_M: ("blk.0.ffn_gate_exps.weight", "model"),
    T.IQ4_NL: ("blk.1.engram_embd.weight", "engram"),
    T.BF16: ("blk.0.ffn_gate_inp.weight", "model"),
}
RANGE_BLOCKS = 2048
RANDOM_BLOCKS = 8192
SEED = 1017


def readers():
    files = {"model": sorted(glob.glob(MODEL + "/*-of-*.gguf")), "engram": glob.glob(MODEL + "/*Engram*.gguf")}
    return {k: [GGUFReader(f) for f in v] for k, v in files.items()}


def raw(rs, name):
    for r in rs:
        for t in r.tensors:
            if t.name == name:
                return t.tensor_type, np.asarray(t.data).reshape(-1).view(np.uint8)
    raise SystemExit("no tensor " + name)


def values(qtype, b):
    with np.errstate(invalid="ignore", over="ignore"):  # a random draw may decode to NaN: it is drawn again
            return Q.dequantize(b.reshape(1, -1), qtype).reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rs = readers()
    rng = np.random.default_rng(SEED)
    for qtype, (name, where) in TENSORS.items():
        got, data = raw(rs[where], name)
        assert got == qtype, (name, got)
        bv, bb = GGML_QUANT_SIZES[qtype]
        n = data.size // bb
        mid = int(rng.integers(RANGE_BLOCKS, n - 2 * RANGE_BLOCKS))
        parts = [data[s * bb:(s + RANGE_BLOCKS) * bb] for s in (0, mid, n - RANGE_BLOCKS)]
        rand = []
        while len(rand) < RANDOM_BLOCKS:
            blk = rng.integers(0, 256, bb, dtype=np.uint8)
            if np.isfinite(values(qtype, blk)).all():
                rand.append(blk)
        b = np.concatenate(parts + rand)
        v = values(qtype, b)
        assert v.size == b.size // bb * bv
        path = os.path.join(a.out, qtype.name + ".bin")
        with open(path, "wb") as f:
            f.write(b"SUBQ" + struct.pack("<IIQQ", 1, int(qtype), v.size, b.size))
            f.write(b.tobytes())
            f.write(v.tobytes())
        print("%-8s %s: blocks 0, %d, %d (+%d each) and %d random: %d values" %
              (qtype.name, name, mid, n - RANGE_BLOCKS, RANGE_BLOCKS, RANDOM_BLOCKS, v.size))


main()
