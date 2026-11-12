#!/usr/bin/env python3
"""tools/convert/ds4_ref_dense.py - the reference overlay's last shard (DESIGN.md, "0.6.6: the reference, and how a quant
is scored"): DeepSeek V4.1 Flash's dense weights as deepseek-ai stores them, for a GGUF model's names and dims.

    python3 tools/convert/ds4_ref_dense.py <headers.json> <base first shard> <out shard> [--local a.safetensors ...]

each FP8 original (E4M3 codes, a UE8M0 scale every 32 x 32 block) becomes BF16 exactly (4 significant bits in 8, the
scale a power of two: checked, a value BF16 would round refused); the head and the embedding are BF16 as stored. the
attention's wo_a (attn_output_a) takes mm_group alone, which has no BF16 path: Q8_0 there (gguf-py's quantizer), the
reference's one approximation. the base model (any DeepSeek V4.1 GGUF: evo's, UD's) gives every tensor's dims, checked;
its F32 and BF16 tensors are the originals already (the norms, the router, the compressor and indexer) and stay its.
the bytes: from --local safetensors holding the original names (a tensor found there is read there), else from the
repo by range (ds4_experts_mxfp4.py's headers and fetch). the shard's split.no / split.count are 8 / 9: the expert
shards' (ds4_experts_mxfp4.py) last.
"""
import glob
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds4_experts_mxfp4 as X  # noqa: E402

sys.path.insert(0, os.environ.get("OSMOS_GGUF_PY", os.path.expanduser("~/Projects/llama.cpp/gguf-py")))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

NAMES = {"attn.wkv": "attn_kv", "attn.wq_a": "attn_q_a", "attn.wq_b": "attn_q_b", "attn.wo_a": "attn_output_a",
         "attn.wo_b": "attn_output_b", "ffn.shared_experts.w1": "ffn_gate_shexp",
         "ffn.shared_experts.w2": "ffn_down_shexp", "ffn.shared_experts.w3": "ffn_up_shexp", "engram.wkv": "engram_kv"}
Q8, BF16 = 8, 30


def gguf_name(k):
    if k == "head.weight":
        return "output.weight"
    if k == "embed.weight":
        return "token_embd.weight"
    p = k.split(".")
    if p[0] == "layers" and p[-1] == "weight" and ".".join(p[2:-1]) in NAMES:
        return f"blk.{p[1]}.{NAMES['.'.join(p[2:-1])]}.weight"
    return None


E4 = np.zeros(256, np.float32)  # float8_e4m3fn: sign, 4 exponent bits (bias 7), 3 mantissa; 0x7F / 0xFF NaN
for b in range(256):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    E4[b] = np.nan if (b & 0x7F) == 0x7F else (-v if s else v)


class Source:
    """a tensor's bytes by its original name: a local safetensors that has it, else the repo by range"""

    def __init__(self, hs, local, tok):
        self.hs, self.tok, self.local = hs, tok, {}
        for p in local:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            for k, v in h.items():
                if not k.startswith("__"):
                    self.local[k] = (p, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v)
        self.where = {}
        for shard, h in hs.items():
            for k, v in h.items():
                if not k.startswith("__"):
                    self.where[k] = (shard, h["__base__"] + v["data_offsets"][0], h["__base__"] + v["data_offsets"][1], v)

    def meta(self, k):
        return (self.local[k][3] if k in self.local else self.where[k][3])

    def read(self, k):
        if k in self.local:
            p, a, nb, _ = self.local[k]
            with open(p, "rb") as f:
                f.seek(a)
                return f.read(nb)
        shard, a, b, _ = self.where[k]
        return X.fetch(shard, a, b, self.tok, None)


def tensor_bytes(src, k, ty, rows, cols):
    v = src.meta(k)
    raw = src.read(k)
    if v["dtype"] == "BF16":
        assert ty == BF16, k
        return raw
    assert v["dtype"] == "F8_E4M3", (k, v["dtype"])
    sm = src.meta(k[: -len("weight")] + "scale")
    assert sm["dtype"] == "F8_E8M0" and sm["shape"] == [(rows + 31) // 32, (cols + 31) // 32], (k, sm)
    sc = np.frombuffer(src.read(k[: -len("weight")] + "scale"), np.uint8).reshape(sm["shape"]).astype(np.int32)
    w = E4[np.frombuffer(raw, np.uint8).reshape(rows, cols)] * np.ldexp(
        np.float32(1), np.repeat(np.repeat(sc - 127, 32, 0), 32, 1)[:rows, :cols])
    if ty == Q8:
        return quants.quantize(w.astype(np.float32), gguf.GGMLQuantizationType.Q8_0).tobytes()
    u = w.astype(np.float32).view(np.uint32)
    assert np.all((u & 0xFFFF) == 0), f"{k}: a value BF16 does not hold exactly"
    return (u >> 16).astype(np.uint16).tobytes()


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)
    local = a[a.index("--local") + 1:] if "--local" in a else []
    hp, base_first, out = a[:3]
    tok = X.token()
    src = Source(X.headers(hp, tok), local, tok)
    dims = {}
    for p in sorted(glob.glob(base_first.replace("-00001-of-", "-0000?-of-"))):
        for t in gguf.GGUFReader(p).tensors:
            dims[t.name] = tuple(int(x) for x in t.shape)
    tens = []
    for k in sorted(set(src.where) | set(src.local)):
        g = gguf_name(k)
        if not g or g not in dims:
            continue
        rows, cols = src.meta(k)["shape"]
        assert dims[g] == (cols, rows), (g, dims[g], (cols, rows))
        ty = Q8 if g.endswith("attn_output_a.weight") else BF16
        tens.append((g, k, ty, rows, cols, rows * cols // 32 * 34 if ty == Q8 else rows * cols * 2))
    s = X.s
    kv = [s("general.alignment") + struct.pack("<II", 4, 32),
          s("general.name") + struct.pack("<I", 8) + s("ds41-reference-dense"),
          s("split.no") + struct.pack("<IH", 2, 8), s("split.count") + struct.pack("<IH", 2, 9),
          s("split.tensors.count") + struct.pack("<Ii", 5, 120 + len(tens))]
    hd = b"GGUF" + struct.pack("<IQQ", 3, len(tens), len(kv)) + b"".join(kv)
    o = 0
    for g, k, ty, rows, cols, nb in tens:
        hd += s(g) + struct.pack("<I", 2) + struct.pack("<2Q", cols, rows) + struct.pack("<IQ", ty, o)
        o += (nb + 31) // 32 * 32
    hd += b"\0" * ((-len(hd)) % 32)
    with open(out + ".part", "wb") as w:
        w.write(hd)
        for g, k, ty, rows, cols, nb in tens:
            data = tensor_bytes(src, k, ty, rows, cols)
            assert len(data) == nb, g
            w.write(data)
            w.write(b"\0" * ((-nb) % 32))
    os.rename(out + ".part", out)
    print(f"{len(tens)} tensors ({sum(t[2] == BF16 for t in tens)} BF16, {sum(t[2] == Q8 for t in tens)} Q8_0), "
          f"{o / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
