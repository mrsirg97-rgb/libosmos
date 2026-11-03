"""reference dequantization for tests/gpu/test_quant_gpu.hip: per type, real blocks of a model's tensor (the first of
that type: its start, a seeded middle, its end) and random blocks (any bytes are a block; those whose values are not all finite are drawn
again), each with gguf-py's values. one file a type: b"SUBQ", u32 version, u32 type, u64 values, u64 bytes, the bytes,
the values (f32). run: PYTHONPATH=<llama.cpp>/gguf-py ../.oracle-venv/bin/python dequant_ref.py --out DIR [--model DIR] [--dspark DIR]"""
import argparse
import glob
import os
import struct

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T, quants as Q
from gguf.constants import GGML_QUANT_SIZES

DSPARK_DEFAULT = "~/models/dspark"
RANGE_BLOCKS = 2048
RANDOM_BLOCKS = 8192
SEED = 1017
MXFP4_TENSOR = "mtp.0.ffn.experts.0.w1.weight"  # DSpark's experts: dspark.bin, as dspark.json says


def tensors(model_dir):
    """the first tensor of each quantized type in the model's files (its shards, then any other GGUF beside them: the
    Engram's), long enough for three ranges of RANGE_BLOCKS blocks: (type -> (name, its bytes))"""
    files = sorted(glob.glob(model_dir + "/*-of-*.gguf"))
    files += sorted(f for f in glob.glob(model_dir + "/*.gguf") if f not in files and "mmproj" not in f)
    if not files:
        raise SystemExit("no GGUF files in " + model_dir)
    picked = {}
    for f in files:
        for t in GGUFReader(f).tensors:
            q = t.tensor_type
            if q in picked or q in (T.F32, T.F16) or t.n_bytes < 3 * RANGE_BLOCKS * GGML_QUANT_SIZES[q][1]:
                continue
            picked[q] = (t.name, np.asarray(t.data).reshape(-1).view(np.uint8))
    return picked


def dspark_mxfp4(dspark_dir):
    """DSpark's MXFP4 experts (not a GGUF: dspark.json's [name, type, offset, bytes, shape] into dspark.bin), or None"""
    import json
    meta = os.path.join(dspark_dir, "dspark.json")
    if not os.path.exists(meta):
        return None
    for n, dt, off, nb, _ in json.load(open(meta))["tensors"]:
        if n == MXFP4_TENSOR:
            assert getattr(T, dt) == T.MXFP4, (n, dt)
            return n, np.fromfile(os.path.join(dspark_dir, "dspark.bin"), np.uint8, nb, offset=off)
    raise SystemExit("no tensor " + MXFP4_TENSOR + " in " + meta)


def values(qtype, b):
    with np.errstate(invalid="ignore", over="ignore"):  # a random draw may decode to NaN: it is drawn again
            return Q.dequantize(b.reshape(1, -1), qtype).reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="~/models/ds1017", help="a folder of a model's GGUF files (ds1017; UD-IQ1_M)")
    ap.add_argument("--dspark", default=DSPARK_DEFAULT, help="maya's dspark.bin / dspark.json (MXFP4); absent: none")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    picked = tensors(os.path.expanduser(a.model))
    mx = dspark_mxfp4(os.path.expanduser(a.dspark))
    if mx:
        picked[T.MXFP4] = mx
    rng = np.random.default_rng(SEED)
    for qtype, (name, data) in sorted(picked.items(), key=lambda kv: int(kv[0])):
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
