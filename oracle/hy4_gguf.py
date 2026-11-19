"""a GGUF read without llama.cpp's gguf-py: the metadata, and a tensor's values (F32 as stored, Q8_0 dequantised as
ggml does: an f16 scale times 32 int8 codes; NVFP4 as ggml's dequantize_row_nvfp4: 4 UE4M3 scales a 64-value block,
one a 16, then 32 bytes - sub-block s's value j byte 4 + 8s + j's low nibble, value j + 8 its high - each E2M1 code
doubled times half its scale) or some of its rows - the oracle's own reading, sharing none of osmos's
code. Q4_K and Q6_K as ggml's dequantize_row_q4_K / _q6_K (F32 products in its order); BF16 an F32's top half;
upstream's F8_E4M3 (ggml type 43, a byte a value, as torch's float8_e4m3fn - AngelSlim's Engram table; told from
unsloth's MXFP8 43, not read here, by its bytes filling the tensor's room). a split model (gguf-split's
-00001-of-NNNNN) is opened by its first shard, every shard's tensors one table. Hy4-flash's oracles (hy4_pieces.py)
read its weights through it."""
import struct

import numpy as np

E2M1_DOUBLED = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def ue4m3(codes):
    """an unsigned E4M3 scale's value as ggml reads one: the sign bit not read, 0x7F zero"""
    c = codes.astype(np.int32) & 0x7F
    e, m = c >> 3, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8.0) * 2.0 ** (e - 7)).astype(np.float32)
    return np.where(c == 0x7F, np.float32(0), v)

def _f16(b, at):
    return b[:, at:at + 2].copy().view(np.float16).astype(np.float32)


def q4_k(b):
    """ggml's dequantize_row_q4_K, blocks [n][144]: d, dmin (f16), 12 bytes of 6-bit scales and mins, 128 of codes;
    each 64 values two sub-blocks - the low nibbles d * sc - dmin * m, the high the next pair's"""
    d, dmin = _f16(b, 0), _f16(b, 2)
    q = b[:, 4:16].astype(np.int32)
    sc, mn = np.empty((len(b), 8), np.int32), np.empty((len(b), 8), np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j], mn[:, j] = q[:, j] & 63, q[:, j + 4] & 63
        else:
            sc[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
            mn[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    qs = b[:, 16:].reshape(-1, 4, 32).astype(np.float32)
    lo, hi = np.floor_divide(qs, 1) % 16, np.floor_divide(qs, 16)
    d1, m1 = d * sc[:, 0::2].astype(np.float32), dmin * mn[:, 0::2].astype(np.float32)   # [n][4]
    d2, m2 = d * sc[:, 1::2].astype(np.float32), dmin * mn[:, 1::2].astype(np.float32)
    y = np.concatenate([d1[:, :, None] * lo - m1[:, :, None], d2[:, :, None] * hi - m2[:, :, None]], axis=2)
    return y.astype(np.float32)


def q6_k(b):
    """ggml's dequantize_row_q6_K, blocks [n][210]: 128 bytes of low nibbles, 64 of high pairs, 16 int8 scales, d (f16);
    each half 128 values, value l + 32k: (ql's nibble | qh's pair << 4) - 32, times d * its scale"""
    ql, qh = b[:, :128].astype(np.int32), b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = _f16(b, 208)
    y = np.empty((len(b), 256), np.float32)
    for h in range(2):
        L, H, S = ql[:, 64 * h:64 * h + 64], qh[:, 32 * h:32 * h + 32], sc[:, 8 * h:8 * h + 8]
        qv = [(L[:, :32] & 0xF) | ((H & 3) << 4), (L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4),
              (L[:, :32] >> 4) | (((H >> 4) & 3) << 4), (L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)]
        for k in range(4):
            s = np.repeat(S[:, 2 * k:2 * k + 2], 16, axis=1)  # value l's scale: is = l / 16, then + 2k
            y[:, 128 * h + 32 * k:128 * h + 32 * k + 32] = (d * s) * (qv[k] - 32).astype(np.float32)
    return y


def e4m3fn_table():
    """float8_e4m3fn's 256 codes: bias 7, subnormals at e 0, 0x7F / 0xFF NaN, no infinity"""
    c = np.arange(256)
    e, m = (c >> 3) & 15, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v = np.where((c & 0x7F) == 0x7F, np.nan, v)
    return np.where(c & 0x80, -v, v).astype(np.float32)


E4M3FN = e4m3fn_table()
F8_E4M3 = 0x100 | 43  # (osmos's own id for upstream's 43, here too)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 14: ("Q6_K", 256, 210),
         30: ("BF16", 1, 2), 40: ("NVFP4", 64, 36), F8_E4M3: ("F8_E4M3", 1, 1)}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Gguf:
    def __init__(self, path):
        self.tensors, self.files = {}, []
        self._read(path, first=True)
        count = self.kv.get("split.count", 1)
        for no in range(1, count):
            tag = "-00001-of-%05d" % count
            if tag not in path:
                raise SystemExit(f"{path}: a split model's first shard is named {tag}")
            self._read(path.replace(tag, "-%05d-of-%05d" % (no + 1, count)), first=False)

    def _read(self, path, first):
        self.f = open(path, "rb")
        if self.f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        self._u("<I")
        n_tensors, n_kv = self._u("<Q"), self._u("<Q")
        kv = {}
        for _ in range(n_kv):
            key = self._s()
            kv[key] = self._value(self._u("<I"))
        if first:
            self.kv = kv
        mine = []
        for _ in range(n_tensors):
            name = self._s()
            dims = [self._u("<Q") for _ in range(self._u("<I"))]
            mine.append([name, dims, self._u("<I"), self._u("<Q")])
        align = kv.get("general.alignment", self.kv.get("general.alignment", 32))
        base = (self.f.tell() + align - 1) // align * align
        size = self.f.seek(0, 2)
        for t in mine:  # a 43 whose MXFP8 bytes leave its room but whose values fill it a byte each: F8_E4M3
            if t[2] == 43:
                room = min([u[3] for u in mine if u[3] > t[3]] + [size - base]) - t[3]
                count = int(np.prod(t[1]))
                if count // 32 * 33 > room >= count:
                    t[2] = F8_E4M3
        for name, dims, ty, off in mine:
            self.tensors[name] = (dims, ty, base + off, len(self.files))
        self.files.append(self.f)

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
        if name == "BF16":
            return (np.frombuffer(raw, dtype=np.uint16, count=count).astype(np.uint32) << 16).view(np.float32)
        if name == "F8_E4M3":
            return E4M3FN[np.frombuffer(raw, dtype=np.uint8, count=count)]
        if name == "Q4_K":
            return q4_k(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144)).reshape(-1)[:count]
        if name == "Q6_K":
            return q6_k(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)).reshape(-1)[:count]
        if name == "NVFP4":
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36)
            half = ue4m3(blocks[:, :4]) * np.float32(0.5)                       # [blocks][4]
            qs = blocks[:, 4:].reshape(-1, 4, 8)                                 # [blocks][sub-block][8]
            codes = np.concatenate([qs & 0x0F, qs >> 4], axis=2)                 # [blocks][4][16]: j, then j + 8
            return (E2M1_DOUBLED[codes] * half[:, :, None]).reshape(-1)[:count]
        raise SystemExit(f"type {name}: not read by this oracle")

    def tensor(self, name):
        """the whole tensor as F32, numpy's shape the GGUF dims reversed (rows last-dim-major: [ne1][ne0])"""
        dims, ty, off, fi = self.tensors[name]
        f = self.files[fi]
        count = int(np.prod(dims))
        rows = count // dims[0]
        f.seek(off)
        raw = f.read(rows * self._row_bytes(ty, dims[0]))
        return self._decode(ty, raw, count).reshape(list(reversed(dims)))

    def expert(self, name, e):
        """expert e of a 3-D experts tensor ([ne2][ne1][ne0]): its [ne1][ne0] values, read alone"""
        dims, ty, off, fi = self.tensors[name]
        f = self.files[fi]
        rb = self._row_bytes(ty, dims[0])
        size = rb * dims[1]
        f.seek(off + e * size)
        return self._decode(ty, f.read(size), dims[0] * dims[1]).reshape(dims[1], dims[0])

    def expert_raw(self, name, e):
        """expert e's bytes as stored"""
        dims, ty, off, fi = self.tensors[name]
        f = self.files[fi]
        size = self._row_bytes(ty, dims[0]) * dims[1]
        f.seek(off + e * size)
        return np.frombuffer(f.read(size), dtype=np.uint8).copy()

    def raw(self, name):
        """the tensor's bytes as stored (its blocks), a uint8 array"""
        dims, ty, off, fi = self.tensors[name]
        f = self.files[fi]
        rows = int(np.prod(dims)) // dims[0]
        f.seek(off)
        return np.frombuffer(f.read(rows * self._row_bytes(ty, dims[0])), dtype=np.uint8).copy()

    def rows(self, name, ids):
        """rows ids of a 2-D tensor ([len(ids)][ne0]), each read alone (a table too large to read whole)"""
        dims, ty, off, fi = self.tensors[name]
        f = self.files[fi]
        rb = self._row_bytes(ty, dims[0])
        out = np.empty((len(ids), dims[0]), dtype=np.float32)
        for i, r in enumerate(ids):
            f.seek(off + int(r) * rb)
            out[i] = self._decode(ty, f.read(rb), dims[0])
        return out
