#!/usr/bin/env python3
"""tools/convert/gguf_assemble.py - a GGUF set with some of its tensors replaced, its other bytes shared with the base's.

    python3 tools/convert/gguf_assemble.py <base first shard> <out dir> <replacements.gguf> [<more.gguf> ...]

every shard of the base that holds a replaced tensor is written again under <out dir> with the same name, metadata
(verbatim, but the first shard's: a delta's own keys - gguf_delta.py's, marked osmos.delta; not its layout's,
general.alignment and split.*, nor the architecture - replace the base's of those names or are added) and tensors in the same order (a shard with nothing changed is linked); a tensor a replacement GGUF holds (the same name and dims; any type) takes its bytes from there, every
other keeps the base's. every tensor is placed at the same offset modulo 4096 as in its source (the base's file or
the replacement's), so on a filesystem with clones (btrfs, XFS) its whole blocks are cloned (FICLONERANGE: no bytes
copied, no space taken) and only the blocks' edges are written; elsewhere it is copied. each shard written as .part,
checked tensor by tensor against its source, then renamed. evo-A (DESIGN.md, "0.6.8") is evo's gguf-a with the GPTQ
dense overlay baked in this way: its abliteration then takes osmos's one overlay.
"""
import fcntl
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtypes  # noqa: E402

BLOCK = 4096
FICLONERANGE = 0x4020940D
# a type's values and bytes a block: qtype.h's table (qtypes.py reads it), and F16 - carried as bytes, not read by osmos
TYPES = {**qtypes.SIZES, 1: (1, 2)}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def skip_value(f, t):
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        (n,) = rd(f, "Q")
        f.seek(n, 1)
    elif t == 9:
        et, n = rd(f, "IQ")
        if et == 8:
            for _ in range(n):
                (m,) = rd(f, "Q")
                f.seek(m, 1)
        else:
            f.seek(size[et] * n, 1)
    else:
        f.seek(size[t], 1)


def read_kv(path):
    """a GGUF's metadata in order: [(key, its raw bytes - the key, the type, the value)]"""
    out = []
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        ver, nt, nkv = rd(f, "IQQ")
        for _ in range(nkv):
            at = f.tell()
            (n,) = rd(f, "Q")
            key = f.read(n).decode()
            (t,) = rd(f, "I")
            skip_value(f, t)
            end = f.tell()
            f.seek(at)
            out.append((key, f.read(end - at)))
    return out


LAYOUT_KEYS = ("general.alignment", "split.")  # a set's layout, never carried from a replacement
DELTA_KEY = "osmos.delta"  # a replacement that is a delta (gguf_delta.py): its other metadata is the set's own


def read_shard(path):
    """the shard's header: its KV section's raw bytes, alignment, tensors (name, dims, type, absolute offset, bytes)"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        ver, nt, nkv = rd(f, "IQQ")
        kv0, align = f.tell(), 32
        for _ in range(nkv):
            (n,) = rd(f, "Q")
            key = f.read(n).decode()
            (t,) = rd(f, "I")
            if key == "general.alignment":
                (align,) = rd(f, "I")
            else:
                skip_value(f, t)
        kv1 = f.tell()
        f.seek(kv0)
        kv = f.read(kv1 - kv0)
        tens = []
        for _ in range(nt):
            (n,) = rd(f, "Q")
            name = f.read(n).decode()
            (nd,) = rd(f, "I")
            dims = rd(f, "Q" * nd)
            ty, off = rd(f, "IQ")
            tens.append([name, dims, ty, off])
        base = (f.tell() + align - 1) // align * align
    for t in tens:
        t[3] += base
    for t in tens:
        values = 1
        for d in t[1]:
            values *= d
        per, nb = TYPES[t[2]]
        t.append(values // per * nb)
    return ver, nkv, kv, align, tens


def clone_or_copy(src, so, dst, do, n):
    """n bytes of src at so to dst at do: the blocks both share an offset in, cloned; the edges (and all of it where
    clones are refused) copied"""
    cloned = 0
    if so % BLOCK == do % BLOCK:
        a = (-so) % BLOCK
        mid = (n - a) // BLOCK * BLOCK
        if mid > 0:
            try:
                fcntl.ioctl(dst, FICLONERANGE, struct.pack("=qQQQ", src, so + a, mid, do + a))
                cloned = mid
            except OSError:
                mid = 0
        if cloned:
            copy(src, so, dst, do, a)
            copy(src, so + a + mid, dst, do + a + mid, n - a - mid)
            return cloned
    copy(src, so, dst, do, n)
    return 0


def copy(src, so, dst, do, n):
    while n > 0:
        k = os.copy_file_range(src, dst, min(n, 1 << 30), so, do)
        if k <= 0:
            raise OSError(f"copy_file_range returned {k}")
        so, do, n = so + k, do + k, n - k


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)
    first, out_dir, repl_paths = a[0], a[1], a[2:]
    shards = sorted(glob.glob(first.replace("-00001-of-", "-0000?-of-")))
    os.makedirs(out_dir, exist_ok=True)
    assemble(shards, [os.path.join(out_dir, os.path.basename(sp)) for sp in shards], repl_paths)


def assemble(shards, outs, repl_paths, log=print):
    """the base's shards (paths, in order: the first holds the set's metadata) into outs (a path each, the same order)
    with repl_paths' tensors (and a delta's metadata) in; a shard nothing of which changes linked to the base's file.
    SystemExit for a replacement the base has no tensor for, or of other dims"""
    repl, meta = {}, {}
    for p in repl_paths:
        for name, dims, ty, off, nb in read_shard(p)[4]:
            repl[name] = (p, dims, ty, off, nb)
        kvs = read_kv(p)
        if any(k == DELTA_KEY for k, _ in kvs):  # a delta's metadata (gguf_delta.py's): the first shard's to carry -
            for key, raw in kvs:                  # never its layout, its own marker or the architecture
                if not key.startswith(LAYOUT_KEYS) and key not in (DELTA_KEY, "general.architecture"):
                    meta[key] = raw
    used, total_clone, total = set(), 0, 0
    for sp, out in zip(shards, outs):
        ver, nkv, kv, align, tens = read_shard(sp)
        if sp == shards[0] and meta:  # the replacements' metadata: the base's keys of those names replaced, the others added
            entries = read_kv(sp)
            have = {k for k, _ in entries}
            entries = [(k, meta.get(k, raw)) for k, raw in entries] + [(k, raw) for k, raw in meta.items() if k not in have]
            kv, nkv = b"".join(raw for _, raw in entries), len(entries)
        if not any(t[0] in repl for t in tens) and not (sp == shards[0] and meta):  # nothing of it changes: linked
            if os.path.lexists(out):
                os.remove(out)
            os.symlink(os.path.realpath(sp), out)
            log(f"{os.path.basename(out)}: no tensor replaced - a link to the base's")
            continue
        new = []  # name, dims, type, source path, source offset, bytes
        for name, dims, ty, off, nb in tens:
            if name in repl:
                rp, rdims, rty, roff, rnb = repl[name]
                assert tuple(rdims) == tuple(dims), (name, rdims, dims)
                new.append((name, dims, rty, rp, roff, rnb)); used.add(name)
            else:
                new.append((name, dims, ty, sp, off, nb))
        hd = b"GGUF" + struct.pack("<IQQ", ver, len(new), nkv) + kv
        info = b"".join(s(n) + struct.pack("<I", len(d)) + struct.pack(f"<{len(d)}Q", *d) + struct.pack("<IQ", t, 0)
                        for n, d, t, _, _, _ in new)
        data0 = (len(hd) + len(info) + align - 1) // align * align
        rel, cur = [], 0
        for n, d, t, p, off, nb in new:
            r = (cur + align - 1) // align * align
            r += ((off - data0) % BLOCK - r) % BLOCK  # its source's offset modulo a block (the base's or the
            # replacement's), so its whole blocks are cloned from there
            rel.append(r); cur = r + nb
        info = b"".join(s(n) + struct.pack("<I", len(d)) + struct.pack(f"<{len(d)}Q", *d) + struct.pack("<IQ", t, r)
                        for (n, d, t, _, _, _), r in zip(new, rel))
        head = hd + info + b"\0" * (data0 - len(hd) - len(info))
        fds = {}
        with open(out + ".part", "wb") as w:
            w.write(head)
            w.truncate(data0 + cur)
            w.flush()
            dst = w.fileno()
            for (n, d, t, p, off, nb), r in zip(new, rel):
                if p not in fds:
                    fds[p] = os.open(p, os.O_RDONLY)
                total_clone += clone_or_copy(fds[p], off, dst, data0 + r, nb)
                total += nb
        for fd in fds.values():
            os.close(fd)
        # checked: every tensor's bytes its source's
        _, _, _, _, got = read_shard(out + ".part")
        with open(out + ".part", "rb") as g:
            for (n, d, t, p, off, nb), (gn, gd, gt, goff, gnb) in zip(new, got):
                assert (n, tuple(d), t, nb) == (gn, tuple(gd), gt, gnb), n
                with open(p, "rb") as src:
                    for k in range(0, nb, 1 << 26):
                        m = min(1 << 26, nb - k)
                        src.seek(off + k); g.seek(goff + k)
                        if src.read(m) != g.read(m):
                            raise SystemExit(f"{n}: its bytes differ from the source's")
        os.rename(out + ".part", out)
        log(f"{os.path.basename(out)}: {len(new)} tensors, {sum(1 for x in new if x[3] != sp)} replaced")
    missing = set(repl) - used
    if missing:
        raise SystemExit(f"replacement tensors the base does not have: {sorted(missing)[:5]}")
    log(f"{total / 1e9:.1f} GB of tensors, {total_clone / 1e9:.1f} GB of it cloned (shared with the base)")
    return outs


if __name__ == "__main__":
    main()
