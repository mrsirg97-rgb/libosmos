#!/usr/bin/env python3
"""tools/convert/gguf_delta.py - what a GGUF set has that another does not: the delta gguf_assemble.py turns the other
back into it.

    python3 tools/convert/gguf_delta.py <target first shard> <base first shard> <out.gguf> [--note TEXT] [--skip NAME ...]

out holds every tensor of the target whose bytes (or type) are not the base's tensor of that name, cloned from the
target where the filesystem has clones, and the target's first shard's metadata keys that the base's lacks or holds
otherwise - not the layout's (general.alignment, split.*) nor the architecture - and osmos.delta (the note: what it is
a delta on). --skip: tensors left to the base's set as it is installed (a pack's Engram table, per_layer_token_embd:
its table slot, the installer's). gguf_assemble.py <base> <dir> out.gguf then writes the target's tensors and metadata in the base's
layout: the same model, tensor for tensor. evo-A (DESIGN.md, "0.6.8") ships so: 34 GB of its own on UD-IQ1_M's 107.
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_assemble as A  # noqa: E402


def tensors(first):
    out = {}
    for p in sorted(glob.glob(first.replace("-00001-of-", "-0000?-of-"))):
        for name, dims, ty, off, nb in A.read_shard(p)[4]:
            out[name] = (p, tuple(dims), ty, off, nb)
    return out


def same_bytes(a, b, nb):
    (pa, oa), (pb, ob) = a, b
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        for k in range(0, nb, 1 << 26):
            m = min(1 << 26, nb - k)
            fa.seek(oa + k)
            fb.seek(ob + k)
            if fa.read(m) != fb.read(m):
                return False
    return True


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)
    target, base, out = a[:3]
    note = a[a.index("--note") + 1] if "--note" in a else f"a delta on {os.path.basename(base)}"
    skip = set()
    if "--skip" in a:
        for x in a[a.index("--skip") + 1:]:
            if x.startswith("--"):
                break
            skip.add(x)
    T, B = tensors(target), tensors(base)
    keep = []
    for name, (p, dims, ty, off, nb) in T.items():
        if name in skip:
            continue
        b = B.get(name)
        if b and b[1] == dims and b[2] == ty and b[4] == nb and same_bytes((p, off), (b[0], b[3]), nb):
            continue
        if b and b[1] != dims:
            raise SystemExit(f"{name}: dims {dims} in the target, {b[1]} in the base - not the same model")
        keep.append((name, p, dims, ty, off, nb))
    missing = set(B) - set(T)
    if missing:
        raise SystemExit(f"tensors the base has and the target lacks: {sorted(missing)[:5]}")
    tkv, bkv = dict(A.read_kv(target)), dict(A.read_kv(base))
    meta = [(k, raw) for k, raw in A.read_kv(target) if not k.startswith(A.LAYOUT_KEYS)
            and k != "general.architecture" and bkv.get(k) != raw]
    s = A.s
    kv = [s("general.alignment") + struct.pack("<II", 4, 32), s(A.DELTA_KEY) + struct.pack("<I", 8) + s(note)]
    kv += [raw for _, raw in meta]
    hd = b"GGUF" + struct.pack("<IQQ", 3, len(keep), len(kv)) + b"".join(kv)
    info_len = sum(len(s(n)) + 4 + 8 * len(d) + 12 for n, _, d, _, _, _ in keep)
    data0 = (len(hd) + info_len + 31) // 32 * 32
    rel, cur = [], 0
    for n, p, d, t, off, nb in keep:  # each at its offset in the target modulo a block: cloned from it
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
            if p not in fds:
                fds[p] = os.open(p, os.O_RDONLY)
            cloned += A.clone_or_copy(fds[p], off, w.fileno(), data0 + r, nb)
        for fd in fds.values():
            os.close(fd)
    os.rename(out + ".part", out)
    total = sum(x[5] for x in keep)
    print(f"{out}: {len(keep)} of {len(T)} tensors ({total / 1e9:.2f} GB, {cloned / 1e9:.2f} GB cloned), "
          f"{len(meta)} metadata keys: {[k for k, _ in meta]}")


if __name__ == "__main__":
    main()
