"""lsa_cases.py - tests/test_lsa's cases: one extracted document's layer (qwen/lsa.py extract), its keys' and queries'
index vectors under a map (lsa.py's index, the keys BF16 as lsa.py keeps them), and lsa.py's buckets and served reads
over them - the sets src/lsa/lsa.c is measured against.

    python lsa_cases.py DOC.safetensors MAPS.pt LAYER OUT.bin [--queries 64]

OUT.bin, little-endian: u32 magic "LSA1", u32 T, u32 Q, u32 blocks; f32 ik [T x 64]; f32 the maps' first key row
input [768] and its index [64] (lsa_index's check); f32 cent [blocks*64 x 64]; i32 members [blocks*64 x 16]; then Q
queries: u32 p, f32 iq [64], u32 n, u32 read [n]."""
import argparse
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/mnt/data/incarnate/qwen")
import lsa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc")
    ap.add_argument("maps")
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    ap.add_argument("--queries", type=int, default=64)
    a = ap.parse_args()
    f = safe_open(a.doc, "pt")
    L = a.layer
    W = lsa.load_maps(a.maps, [L])[L]
    k = f.get_tensor(f"{L}.k").float()
    q = f.get_tensor(f"{L}.q").float()
    rows = f.get_tensor("rows")
    ik = lsa.index(k, W["Wk"]).to(torch.bfloat16).float()
    iq = lsa.index(q, W["Wq"])
    cent, members = lsa.buckets(ik)
    T, G = ik.shape[0], ik.shape[0] // lsa.WINDOW
    sel = torch.linspace(0, len(rows) - 1, a.queries).round().long().unique()
    with open(a.out, "wb") as o:
        o.write(b"LSA1" + struct.pack("<III", T, len(sel), G))
        o.write(ik.numpy().astype("<f4").tobytes())
        o.write(k[0].numpy().astype("<f4").tobytes() + ik[0].numpy().astype("<f4").tobytes())
        o.write(cent.numpy().astype("<f4").tobytes())
        o.write(members.numpy().astype("<i4").tobytes())
        for j in sel.tolist():
            p = int(rows[j])
            r = lsa.served_read(iq[j], p, cent, members)
            o.write(struct.pack("<I", p) + iq[j].numpy().astype("<f4").tobytes() + struct.pack("<I", len(r)))
            o.write(r.numpy().astype("<u4").tobytes())
    with open(a.out + ".W", "wb") as o:  # the key map, for lsa_index's check
        o.write(W["Wk"].numpy().astype("<f4").tobytes())
    print(f"{a.out}: T {T}, {G} blocks, {len(sel)} queries", file=sys.stderr)


if __name__ == "__main__":
    main()
