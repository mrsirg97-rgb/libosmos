"""lsa_maps.py - a D-LSA map set (incarnate qwen/lsa.py's maps.*.pt: "<layer>.W_Q_idx" and "<layer>.W_K_idx", F32
[768, 64] a full-attention layer) as src/q35's file: little-endian u32 magic "LSAM", u32 layers, u32 in (768), u32 dim
(64); then for each layer ascending: u32 its index, f32 W_Q [in x dim], f32 W_K [in x dim] (row major).

    python lsa_maps.py MAPS.pt OUT.bin"""
import re
import struct
import sys

import torch

d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
layers = sorted({int(m.group(1)) for k in d for m in [re.fullmatch(r"(\d+)\.W_[QK]_idx", k)] if m})
W = {L: (d[f"{L}.W_Q_idx"].float().contiguous(), d[f"{L}.W_K_idx"].float().contiguous()) for L in layers}
i, o = W[layers[0]][0].shape
with open(sys.argv[2], "wb") as f:
    f.write(b"LSAM" + struct.pack("<III", len(layers), i, o))
    for L in layers:
        q, k = W[L]
        assert q.shape == (i, o) and k.shape == (i, o)
        f.write(struct.pack("<I", L) + q.numpy().astype("<f4").tobytes() + k.numpy().astype("<f4").tobytes())
print(f"{sys.argv[2]}: {len(layers)} layers {layers}, {i} x {o}", file=sys.stderr)
