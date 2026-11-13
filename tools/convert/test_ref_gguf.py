"""test_ref_gguf.py - a reference GGUF's experts read back (gguf-py's NVFP4 dequantize x the .scale tensor; MXFP8
from the file's bytes) against the checkpoint's values (ink2_oracle.StreamExperts' decode): every bit.

    PYTHONPATH=<llama.cpp>/gguf-py:<transformers main> python tools/convert/test_ref_gguf.py CKPT_DIR GGUF_PREFIX LAYER ...

(test_oracle.py --out DIR writes a small checkpoint to try it on: ink2_ref_gguf.py --src DIR --out DIR/ref, then
test_ref_gguf.py DIR DIR/ref 0 9 10 11)
"""
import glob
import sys

import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType as T, quants as Q
from safetensors import safe_open

ckpt, prefix, layers = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
sys.argv = ["x", "--src", "x", "--tokens", "x", "--out", "x"]
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ink2_oracle as O  # noqa: E402

src = O.Src(ckpt)
G = {}
for fn in sorted(glob.glob(prefix + "-*.gguf")):
    for t in GGUFReader(fn).tensors:
        G[t.name] = (t, fn)
bad = 0
for L in layers:
    p = f"model.llm.layers.{L}.mlp.experts."
    se = O.StreamExperts(src, L, "cpu", torch.float32, None)
    E = se.num_experts
    for nm, proj, sl in (("gate", "w13", slice(0, None, 2)), ("up", "w13", slice(1, None, 2)), ("down", "w2", slice(None))):
        t, fn = G[f"blk.{L}.ffn_{nm}_exps.weight"]
        for e in sorted({0, E // 2, E - 1}):
            want = se.weight(proj, e)[sl]
            if se.nv:
                raw = np.asarray(t.data).reshape(-1).view(np.uint8)
                per = raw.size // E
                got = Q.dequantize(raw[e * per:(e + 1) * per].reshape(1, -1), T.NVFP4).reshape(want.shape)
                sc = np.asarray(G[f"blk.{L}.ffn_{nm}_exps.scale"][0].data).reshape(-1)[e]
                got = torch.from_numpy(got.astype(np.float32)) * torch.tensor(sc, dtype=torch.float32)
            else:   # (gguf-py sizes 43 as a byte a value: the bytes from the file at the tensor's offset)
                n = want.numel() // 32 * 33
                raw = np.memmap(fn, np.uint8, "r", offset=int(t.data_offset) + e * n, shape=(n,))
                b = torch.from_numpy(np.array(raw)).reshape(-1, 33)
                got = (b[:, 1:].contiguous().view(torch.float8_e4m3fn).float() *
                       torch.exp2(b[:, :1].float() - 127)).reshape(want.shape)
            ok = torch.equal(want.contiguous().view(torch.int32), got.contiguous().view(torch.int32))
            bad += not ok
            print(f"L{L:2d} {nm:4s} e{e:3d} {'nvfp4' if se.nv else 'mxfp8'} {tuple(want.shape)}: "
                  f"{'bit-equal' if ok else 'DIFFERS max %.3g' % (want - got).abs().max()}")
print(f"{len(G)} tensors; " + ("all bit-equal" if not bad else f"{bad} DIFFER"))
sys.exit(1 if bad else 0)
