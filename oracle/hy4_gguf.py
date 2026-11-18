"""a GGUF read without llama.cpp's gguf-py: the metadata, and a tensor's values (F32 as stored, Q8_0 dequantised as
ggml does: an f16 scale times 32 int8 codes) or some of its rows - the oracle's own reading, sharing none of osmos's
code. Hy4-flash's oracles (hy4_pieces.py) read its weights through it."""
import struct

import numpy as np

TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 30: ("BF16", 1, 2), 40: ("NVFP4", 64, 36)}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        if self.f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        self._u("<I")
        n_tensors, n_kv = self._u("<Q"), self._u("<Q")
        self.kv = {}
        for _ in range(n_kv):
            key = self._s()
            self.kv[key] = self._value(self._u("<I"))
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._s()
            dims = [self._u("<Q") for _ in range(self._u("<I"))]
            self.tensors[name] = (dims, self._u("<I"), self._u("<Q"))
        align = self.kv.get("general.alignment", 32)
        self.base = (self.f.tell() + align - 1) // align * align

    def _u(self, fmt):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def _s(self):
        return self.f.read(self._u("<Q")).decode("utf-8", "replace")

    def _value(self, t):
        if t == 8:
            return self._s()
        if t == 9:
            item, n = self._u("<I"), self._u("<Q")
            return [self._value(item) for _ in range(n)]
        return self._u(SCALAR[t])

    def _row_bytes(self, ty, ne0):
        name, block, size = TYPES[ty]
        if ne0 % block:
            raise SystemExit(f"a row of {ne0} not whole {name} blocks")
        return ne0 // block * size

    def _decode(self, ty, raw, count):
        name = TYPES[ty][0]
        if name == "F32":
            return np.frombuffer(raw, dtype=np.float32, count=count).copy()
        if name == "Q8_0":
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
            scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            codes = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
            return (codes * scale).reshape(-1)[:count]
        raise SystemExit(f"type {name}: not read by this oracle")

    def tensor(self, name):
        """the whole tensor as F32, numpy's shape the GGUF dims reversed (rows last-dim-major: [ne1][ne0])"""
        dims, ty, off = self.tensors[name]
        count = int(np.prod(dims))
        rows = count // dims[0]
        self.f.seek(self.base + off)
        raw = self.f.read(rows * self._row_bytes(ty, dims[0]))
        return self._decode(ty, raw, count).reshape(list(reversed(dims)))

    def rows(self, name, ids):
        """rows ids of a 2-D tensor ([len(ids)][ne0]), each read alone (a table too large to read whole)"""
        dims, ty, off = self.tensors[name]
        rb = self._row_bytes(ty, dims[0])
        out = np.empty((len(ids), dims[0]), dtype=np.float32)
        for i, r in enumerate(ids):
            self.f.seek(self.base + off + int(r) * rb)
            out[i] = self._decode(ty, self.f.read(rb), dims[0])
        return out
