#!/usr/bin/env python3
"""overlay.py <source first shard> <out.gguf> <layers: 0-9,20> [kinds: gate,up,down] - an OVERLAY of the source's
routed experts for those layers (blk.L.ffn_{kind}_exps.weight), cloned (btrfs: no space) - kld's OVERLAY tries
them in another pack (evo-A with UD-IQ2_XXS's experts in some layers)"""
import os, re, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import gguf_assemble as A  # noqa: E402
import gguf_delta as D  # noqa: E402
src, out, spec = sys.argv[1:4]
kinds = sys.argv[4].split(',') if len(sys.argv) > 4 else ['gate', 'up', 'down']
layers = set()
for part in spec.split(','):
    a, _, b = part.partition('-')
    layers |= set(range(int(a), int(b or a) + 1))
T = D.tensors(src)
keep = []
for name, (p, dims, ty, off, nb) in T.items():
    m = re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', name)
    if m and int(m[1]) in layers and m[2] in kinds:
        keep.append((name, p, dims, ty, off, nb))
s = A.s
kv = [s("general.alignment") + struct.pack("<II", 4, 32)]
hd = b"GGUF" + struct.pack("<IQQ", 3, len(keep), len(kv)) + b"".join(kv)
info_len = sum(len(s(n)) + 4 + 8 * len(d) + 12 for n, _, d, _, _, _ in keep)
data0 = (len(hd) + info_len + 31) // 32 * 32
rel, cur = [], 0
for n, p, d, t, off, nb in keep:
    r = (cur + 31) // 32 * 32
    r += ((off - data0) % A.BLOCK - r) % A.BLOCK
    rel.append(r)
    cur = r + nb
info = b"".join(s(n) + struct.pack("<I", len(d)) + struct.pack(f"<{len(d)}Q", *d) + struct.pack("<IQ", t, r)
                for (n, _, d, t, _, _), r in zip(keep, rel))
head = hd + info + b"\0" * (data0 - len(hd) - len(info))
cloned = 0
with open(out + ".part", "wb") as w:
    w.write(head)
    w.truncate(data0 + cur)
    w.flush()
    fds = {}
    for (n, p, d, t, off, nb), r in zip(keep, rel):
        fds.setdefault(p, os.open(p, os.O_RDONLY))
        cloned += A.clone_or_copy(fds[p], off, w.fileno(), data0 + r, nb)
os.rename(out + ".part", out)
print(f"{out}: {len(keep)} tensors, layers {spec}, {sum(x[5] for x in keep) / 1e9:.1f} GB ({cloned / 1e9:.1f} cloned)")
