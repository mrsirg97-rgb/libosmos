"""huihui-ai's Qwen abliteration patch turned into an overlay the engine reads (osmos-api's --overlay: tensors standing
in for the model's of their names).

the patch is one GGUF of every variant: a target tensor's copy in each type unsloth's quants use, named
"<the model's name>_<ggml type id>", its bytes XORed with the key file's text (huihui's dec.py: the key repeated over
each tensor's bytes from its first). a mode picks the targets - 0: ssm_out and ffn_down_shexp, 1: ssm_out and
attn_output, 2: all three (the patch's README: "no warnings"). each target's variant is the one of the model's type,
of the model's shape; the overlay holds those alone, decoded, under the model's names - one file, its bytes the same
for the same inputs (so its SHA-256 can be pinned like a download's)."""
import os
import re
import struct

MODES = {0: ("ssm_out", "ffn_down_shexp"), 1: ("ssm_out", "attn_output"), 2: ("ssm_out", "attn_output", "ffn_down_shexp")}
ARCH = "qwen4exp"
_ALIGN = 32           # GGUF's default alignment (general.alignment absent)
_HEAD = 1 << 26       # a header read at most: the patch's 276 tensor infos, a shard's tokenizer (~6 MB)
_CHUNK = 64 << 20     # a tensor's bytes decoded this many at a time
_STRING, _ARRAY, _UINT32 = 8, 9, 4
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class OverlayError(Exception):
    pass


def _header(path):
    """({key: text} of its string values, alignment, data start, {name: (dims, type, offset)}) of a GGUF v3""" 
    with open(path, "rb") as f:
        b = f.read(_HEAD)
    o = 0

    def take(fmt):
        nonlocal o
        if o + struct.calcsize(fmt) > len(b):
            raise OverlayError(f"{path}: a header past what this reads, or truncated")
        v = struct.unpack_from(fmt, b, o)
        o += struct.calcsize(fmt)
        return v[0]

    def text():
        nonlocal o
        n = take("<Q")
        if o + n > len(b):
            raise OverlayError(f"{path}: a string past the header")
        t = b[o:o + n].decode("utf-8", "replace")
        o += n
        return t

    def skip(ty):
        nonlocal o
        if ty == _STRING:
            text()
        elif ty == _ARRAY:
            sub, n = take("<I"), take("<Q")
            if sub in _FIXED:
                o += _FIXED[sub] * n
            else:
                for _ in range(n):
                    skip(sub)
        elif ty in _FIXED:
            o += _FIXED[ty]
        else:
            raise OverlayError(f"{path}: a value of type {ty}")

    if b[:4] != b"GGUF" or struct.unpack_from("<I", b, 4)[0] != 3:
        raise OverlayError(f"{path}: not a GGUF v3")
    o = 8
    n_t, n_kv = take("<Q"), take("<Q")
    texts, align = {}, _ALIGN
    for _ in range(n_kv):
        k, ty = text(), take("<I")
        if ty == _STRING:
            texts[k] = text()
        elif k == "general.alignment" and ty == _UINT32:
            align = take("<I")
        else:
            skip(ty)
    tensors = {}
    for _ in range(n_t):
        name, nd = text(), take("<I")
        dims = [take("<Q") for _ in range(nd)]
        tensors[name] = (dims, take("<I"), take("<Q"))
    return texts, align, o + (-o % align), tensors


def _sizes(tensors, start, path):
    """each tensor's bytes: to the next tensor's offset, the last to the file's end (the data laid out in order)"""
    ends = sorted(off for _, _, off in tensors.values()) + [os.path.getsize(path) - start]
    return {n: min(e for e in ends if e > off) - off for n, (_, _, off) in tensors.items()}


def _xor(data, key):
    """data XOR the key repeated from its first byte"""
    reps = -(-len(data) // len(key))
    k = int.from_bytes((key * reps)[:len(data)], "little")
    return (int.from_bytes(data, "little") ^ k).to_bytes(len(data), "little")


def convert(model_shards, patch, key, mode, out):
    """the overlay of `mode`'s targets, written to `out` (its temp first, renamed: a refusal leaves nothing). raises
    OverlayError naming the cause; returns the names written"""
    if mode not in MODES:
        raise OverlayError(f"mode {mode}: one of {sorted(MODES)}")
    with open(key, "rb") as f:
        k = f.read().strip()
    if not k:
        raise OverlayError(f"{key}: an empty key")
    want = re.compile(r"blk\.\d+\.(" + "|".join(MODES[mode]) + r")\.weight$")
    targets = []  # (name, dims, type) in the model's order
    for i, s in enumerate(model_shards):  # the first shard's metadata names it; a split's later ones carry none
        texts, _, _, ts = _header(s)
        arch = texts.get("general.architecture")
        if arch != ARCH and (i == 0 or arch is not None):
            raise OverlayError(f"{s}: architecture {arch}, not {ARCH}")
        targets += [(n, d, t) for n, (d, t, _) in ts.items() if want.match(n)]
    if not targets:
        raise OverlayError("the model has none of the mode's tensors")
    ptexts, _, pstart, pts = _header(patch)
    if ptexts.get("general.encrypt_type") != "xor":  # the one scheme this reads (huihui's dec.py)
        raise OverlayError(f"{patch}: general.encrypt_type {ptexts.get('general.encrypt_type')}, not xor")
    psize = _sizes(pts, pstart, patch)
    picked = []
    for name, dims, ty in targets:
        v = f"{name}_{ty}"
        if v not in pts:
            raise OverlayError(f"{patch}: no {v} (the model's {name} is type {ty})")
        vdims, vty, voff = pts[v]
        if vdims != dims or vty != ty:
            raise OverlayError(f"{patch}: {v} of dims {vdims}, type {vty}; the model's {dims}, {ty}")
        picked.append((name, dims, ty, voff, psize[v]))

    def s(t):
        b = t.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(picked), 1) + s("general.architecture") + struct.pack("<I", _STRING) + s(ARCH)
    info, at = b"", 0
    for name, dims, ty, _, n in picked:
        at += -at % _ALIGN
        info += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", ty, at)
        at += n
    tmp = out + ".part"
    try:
        with open(patch, "rb") as src, open(tmp, "wb") as dst:
            body = head + info
            dst.write(body + b"\0" * (-len(body) % _ALIGN))
            written = 0
            for name, dims, ty, voff, n in picked:
                dst.write(b"\0" * (-written % _ALIGN))
                written += -written % _ALIGN
                src.seek(pstart + voff)
                for i in range(0, n, _CHUNK):  # the key restarts at each tensor's first byte, not each chunk's
                    data = src.read(min(_CHUNK, n - i))
                    if len(data) != min(_CHUNK, n - i):
                        raise OverlayError(f"{patch}: {name}'s variant past the file's end")
                    shift = i % len(k)
                    dst.write(_xor(data, k[shift:] + k[:shift]))
                written += n
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return [p[0] for p in picked]
