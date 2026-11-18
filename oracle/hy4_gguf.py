"""a GGUF read without llama.cpp's gguf-py: the metadata, and a tensor's values (F32 as stored, Q8_0 dequantised as
ggml does: an f16 scale times 32 int8 codes; NVFP4 as ggml's dequantize_row_nvfp4: 4 UE4M3 scales a 64-value block,
one a 16, then 32 bytes - sub-block s's value j byte 4 + 8s + j's low nibble, value j + 8 its high - each E2M1 code
doubled times half its scale) or some of its rows - the oracle's own reading, sharing none of osmos's
code. Hy4-flash's oracles (hy4_pieces.py) read its weights through it."""
import struct

import numpy as np

E2M1_DOUBLED = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def ue4m3(codes):
    """an unsigned E4M3 scale's value as ggml reads one: the sign bit not read, 0x7F zero"""
    c = codes.astype(np.int32) & 0x7F
    e, m = c >> 3, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8.0) * 2.0 ** (e - 7)).astype(np.float32)
    return np.where(c == 0x7F, np.float32(0), v)

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
        if name == "NVFP4":
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36)
            half = ue4m3(blocks[:, :4]) * np.float32(0.5)                       # [blocks][4]
            qs = blocks[:, 4:].reshape(-1, 4, 8)                                 # [blocks][sub-block][8]
            codes = np.concatenate([qs & 0x0F, qs >> 4], axis=2)                 # [blocks][4][16]: j, then j + 8
            return (E2M1_DOUBLED[codes] * half[:, :, None]).reshape(-1)[:count]
        raise SystemExit(f"type {name}: not read by this oracle")

    def tensor(self, name):
        """the whole tensor as F32, numpy's shape the GGUF dims reversed (rows last-dim-major: [ne1][ne0])"""
        dims, ty, off = self.tensors[name]
        count = int(np.prod(dims))
        rows = count // dims[0]
        self.f.seek(self.base + off)
        raw = self.f.read(rows * self._row_bytes(ty, dims[0]))
        return self._decode(ty, raw, count).reshape(list(reversed(dims)))

    def expert(self, name, e):
        """expert e of a 3-D experts tensor ([ne2][ne1][ne0]): its [ne1][ne0] values, read alone"""
        dims, ty, off = self.tensors[name]
        rb = self._row_bytes(ty, dims[0])
        size = rb * dims[1]
        self.f.seek(self.base + off + e * size)
        return self._decode(ty, self.f.read(size), dims[0] * dims[1]).reshape(dims[1], dims[0])

    def expert_raw(self, name, e):
        """expert e's bytes as stored"""
        dims, ty, off = self.tensors[name]
        size = self._row_bytes(ty, dims[0]) * dims[1]
        self.f.seek(self.base + off + e * size)
        return np.frombuffer(self.f.read(size), dtype=np.uint8).copy()

    def raw(self, name):
        """the tensor's bytes as stored (its blocks), a uint8 array"""
        dims, ty, off = self.tensors[name]
        rows = int(np.prod(dims)) // dims[0]
        self.f.seek(self.base + off)
        return np.frombuffer(self.f.read(rows * self._row_bytes(ty, dims[0])), dtype=np.uint8).copy()

    def rows(self, name, ids):
        """rows ids of a 2-D tensor ([len(ids)][ne0]), each read alone (a table too large to read whole)"""
        dims, ty, off = self.tensors[name]
        rb = self._row_bytes(ty, dims[0])
        out = np.empty((len(ids), dims[0]), dtype=np.float32)
        for i, r in enumerate(ids):
            self.f.seek(self.base + off + int(r) * rb)
            out[i] = self._decode(ty, self.f.read(rb), dims[0])
        return out
