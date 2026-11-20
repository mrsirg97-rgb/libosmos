#!/usr/bin/env python3
"""requant_hy4.py <first shard> <out.gguf> <ggml type: q4_k|q5_k|q6_k> [kinds: attn_gate,attn_output] - Hy4-flash's
dense matrices of those kinds (blk.L.<kind>.weight, every layer but the MTP's) requantized from the file's own values
(decoded: oracle/hy4_gguf.py) with ggml's quantizers (llama.cpp's libggml-base, MIT; no imatrix) into a replacements
GGUF for tools/convert/gguf_assemble.py (the same names and dims, the new type). a measurement's first pass: the
values twice rounded (Q8_0, then this); the release's BF16 and an imatrix would do better.
env GGML_LIB: libggml-base.so (default ~/Projects/llama.cpp/build-hip/bin/libggml-base.so)"""
import ctypes
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "oracle"))
from hy4_gguf import Gguf  # noqa: E402

TYPES = {"q4_k": 12, "q5_k": 13, "q6_k": 14}
src, out, tname = sys.argv[1:4]
kinds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["attn_gate", "attn_output"]
ty = TYPES[tname]
lib = ctypes.CDLL(os.environ.get("GGML_LIB", os.path.expanduser("~/Projects/llama.cpp/build-hip/bin/libggml-base.so")))
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_void_p]
lib.ggml_row_size.restype = ctypes.c_size_t
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
lib.ggml_quantize_init.argtypes = [ctypes.c_int]
lib.ggml_quantize_init(ty)
g = Gguf(src)
names = sorted((n for n in g.tensors if (m := re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", n)) and m[2] in kinds
                and int(m[1]) < 42), key=lambda n: (int(n.split(".")[1]), n))
blobs = []
for n in names:
    dims = g.tensors[n][0]
    x = np.ascontiguousarray(g.tensor(n), dtype=np.float32)
    rows, cols = int(np.prod(dims)) // dims[0], dims[0]
    size = lib.ggml_row_size(ty, cols) * rows
    buf = ctypes.create_string_buffer(size)
    assert lib.ggml_quantize_chunk(ty, x.ctypes.data, buf, 0, rows, cols, None) == size
    blobs.append((n, dims, buf.raw))
    print(f"  {n} {dims} -> {tname} {size / 1e6:.1f} MB", flush=True)


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


kv = [s("general.alignment") + struct.pack("<II", 4, 32)]
hd = b"GGUF" + struct.pack("<IQQ", 3, len(blobs), len(kv)) + b"".join(kv)
info, off = b"", 0
for n, dims, b in blobs:
    info += s(n) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", ty, off)
    off = (off + len(b) + 31) // 32 * 32
with open(out + ".part", "wb") as f:
    f.write(hd + info)
    f.write(b"\0" * ((-f.tell()) % 32))
    for n, dims, b in blobs:
        f.write(b)
        f.write(b"\0" * ((-len(b)) % 32))
os.rename(out + ".part", out)
print(f"{out}: {len(blobs)} tensors")
