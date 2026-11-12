#!/usr/bin/env python3
"""tools/convert/ds4_experts_mxfp4.py - DeepSeek V4.1 Flash's routed experts as they are stored (deepseek-ai's
safetensors: E2M1 codes two to a byte, an E8M0 scale every 32 inputs) as an overlay of MXFP4 GGUF shards: no value
changed, the codes and scales repacked. osmos's reference model (DESIGN.md, "0.6.6: the reference") is a model's GGUF
with this overlay, so a quant's distance from it is its experts' and dense weights' alone.

    python3 tools/convert/ds4_experts_mxfp4.py <headers.json> <out dir> [--local dir] [--shards 8] [--only k,...]

headers.json: every shard's safetensors header, {shard: {tensor: {dtype, shape, data_offsets}, "__base__": n}}
(the header's 8-byte length + the JSON: the data's first byte). the bytes come from the repo by HTTP range requests
(HF_TOKEN, or `hf auth token`), or from a shard already in --local. out: <out>/ds41-experts-mxfp4-0000k-of-0000N.gguf,
each written as .part and renamed when whole (a shard there is skipped: a stopped run goes on where it was); its last
shard (N = shards + 1) is the dense one, written by another tool - the count is fixed here, for every header.

the repack, value for value: deepseek's byte i of a row holds codes 2i (low nibble) and 2i + 1 (high), worth E2M1
(0, 0.5, 1, 1.5, 2, 3, 4, 6, then the negatives) times 2^(e - 127); ggml's MXFP4 block is e, then 16 bytes, byte j
code j (low) and code j + 16 (high), worth E2M1 doubled times 2^(e - 128) - the same values, the same codes and e.
w1 is the gate, w3 the up, w2 the down (model.py's Expert: w2(silu(w1(x)) * w3(x))).
"""
import concurrent.futures
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request

import numpy as np

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
LAYERS, EXPERTS, DIM, FF = 40, 384, 5120, 2304
MXFP4, BLOCK, BLOCK_BYTES, ALIGN = 39, 32, 17, 32
KINDS = (("gate", "w1", DIM, FF), ("up", "w3", DIM, FF), ("down", "w2", FF, DIM))  # name, theirs, cols (in), rows (out)
CHUNK, STREAMS = 64 << 20, 16


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    return subprocess.run(["hf", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]


def fetch(shard, a, b, tok, local):
    """bytes a .. b - 1 of a shard: the local copy's, or the repo's (retried)"""
    path = os.path.join(local, shard) if local else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            f.seek(a)
            return f.read(b - a)
    url = f"https://huggingface.co/{REPO}/resolve/main/{shard}"
    for attempt in range(8):
        try:
            rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", "Authorization": f"Bearer {tok}",
                                                      "User-Agent": "osmos-ref"})
            with urllib.request.urlopen(rq, timeout=120) as r:
                data = r.read()
            if len(data) == b - a:
                return data
            raise IOError(f"{len(data)} bytes of {b - a}")
        except Exception as e:  # noqa: BLE001 - every failure is retried, then raised
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry {shard} {a}: {e}", file=sys.stderr)


def layer_bytes(hs, layer, tok, local):
    """every tensor of a layer's experts: {name: bytes}, read as few large ranges"""
    want = []
    for shard, h in hs.items():
        for k, v in h.items():
            if k.startswith(f"layers.{layer}.ffn.experts."):
                want.append((shard, h["__base__"] + v["data_offsets"][0], h["__base__"] + v["data_offsets"][1], k))
    want.sort()
    if len(want) != EXPERTS * 6:
        raise SystemExit(f"layer {layer}: {len(want)} expert tensors, not {EXPERTS * 6}")
    spans = []  # contiguous runs, cut into CHUNK pieces
    for shard, a, b, _ in want:
        if spans and spans[-1][0] == shard and spans[-1][2] == a:
            spans[-1][2] = b
        else:
            spans.append([shard, a, b])
    jobs = [(s, x, min(x + CHUNK, b)) for s, a, b in spans for x in range(a, b, CHUNK)]
    with concurrent.futures.ThreadPoolExecutor(STREAMS) as ex:
        parts = dict(zip(jobs, ex.map(lambda j: fetch(j[0], j[1], j[2], tok, local), jobs)))
    out = {}
    for shard, a, b, k in want:
        buf = bytearray()
        for (s, x, y), data in ((j, parts[j]) for j in jobs if j[0] == shard and j[1] < b and j[2] > a):
            buf += data[max(a, x) - x:min(b, y) - x]
        out[k] = bytes(buf)
    return out


def repack(weight, scale, rows, cols):
    """one expert's matrix: deepseek's [rows][cols / 2] codes and [rows][cols / 32] scales -> MXFP4 rows"""
    w = np.frombuffer(weight, np.uint8).reshape(rows, cols // 2)
    e = np.frombuffer(scale, np.uint8).reshape(rows, cols // BLOCK)
    codes = np.empty((rows, cols), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = w & 0x0F, w >> 4
    codes = codes.reshape(rows, cols // BLOCK, BLOCK)
    qs = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return np.concatenate([e[:, :, None], qs], axis=2).tobytes()


def tensors_of(layers):
    out = []
    for layer in layers:
        for name, _, cols, rows in KINDS:
            out.append((f"blk.{layer}.ffn_{name}_exps.weight", (cols, rows, EXPERTS),
                        rows * cols // BLOCK * BLOCK_BYTES * EXPERTS))
    return out


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def header(tensors, no, count):
    kv = [s("general.alignment") + struct.pack("<II", 4, ALIGN), s("general.name") + struct.pack("<I", 8) + s("ds41-experts-mxfp4"),
          s("split.no") + struct.pack("<IH", 2, no), s("split.count") + struct.pack("<IH", 2, count),
          s("split.tensors.count") + struct.pack("<Ii", 5, LAYERS * len(KINDS))]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv)
    off = 0
    for name, dims, nbytes in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", MXFP4, off)
        off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    return out + b"\0" * ((-len(out)) % ALIGN)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    hs = json.load(open(args[0]))
    out_dir, local, shards, only = args[1], None, 8, None
    for i, a in enumerate(args):
        if a == "--local":
            local = args[i + 1]
        if a == "--shards":
            shards = int(args[i + 1])
        if a == "--only":
            only = {int(x) for x in args[i + 1].split(",")}
    os.makedirs(out_dir, exist_ok=True)
    tok, per, count = token(), LAYERS // shards, shards + 1
    for k in range(shards):
        path = os.path.join(out_dir, f"ds41-experts-mxfp4-{k + 1:05d}-of-{count:05d}.gguf")
        if os.path.exists(path) or (only and k + 1 not in only):
            continue
        layers = range(k * per, (k + 1) * per)
        tensors = tensors_of(layers)
        t0 = time.time()
        with open(path + ".part", "wb") as f:
            f.write(header(tensors, k, count))
            for layer in layers:
                t1 = time.time()
                raw = layer_bytes(hs, layer, tok, local)
                for name, theirs, cols, rows in KINDS:
                    n = 0
                    for e in range(EXPERTS):
                        p = f"layers.{layer}.ffn.experts.{e}.{theirs}"
                        n += f.write(repack(raw[p + ".weight"], raw[p + ".scale"], rows, cols))
                    f.write(b"\0" * ((-n) % ALIGN))
                del raw
                print(f"layer {layer}: {time.time() - t1:.0f} s", flush=True)
        os.rename(path + ".part", path)
        print(f"shard {k + 1} of {count}: layers {layers.start}-{layers.stop - 1} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
