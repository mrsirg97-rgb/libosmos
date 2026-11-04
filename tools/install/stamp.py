"""a GGUF's split stamp (split.no, split.count): where in the header its two values are, read, written in place,
and the file's SHA-256 as published (those two values read as another stamp) - the rest of the bytes as they are.

the header walked as GGUF v3 lays it out (the key-values before the tensor infos), not searched for: a value's place
is the parse's, and an integer's width is its type's."""
import hashlib
import struct

_INT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9
_HEAD = 1 << 24          # the most of a header read: unsloth's are ~6 MB with the tokenizer
_CHUNK = 1 << 24         # a hash's read


class StampError(Exception):
    pass


def _fields(path):
    """{'split.no': (offset, fmt), 'split.count': (offset, fmt)} - each value's place and width"""
    with open(path, "rb") as f:
        b = f.read(_HEAD)
    o = 0

    def take(fmt):
        nonlocal o
        if o + struct.calcsize(fmt) > len(b):
            raise StampError(f"{path}: a header longer than this reads, or truncated")
        v = struct.unpack_from(fmt, b, o)[0]
        o += struct.calcsize(fmt)
        return v

    def skip_string():
        nonlocal o
        o += take("<Q")

    if b[:4] != b"GGUF":
        raise StampError(f"{path}: not a GGUF")
    o = 4
    if take("<I") != 3:
        raise StampError(f"{path}: not GGUF version 3")
    take("<Q")
    found = {}
    for _ in range(take("<Q")):
        n = take("<Q")
        key = b[o:o + n].decode("utf-8", "replace")
        o += n
        t = take("<I")
        if key in ("split.no", "split.count"):
            if t not in _INT:
                raise StampError(f"{path}: {key} is not an integer")
            found[key] = (o, _INT[t])
        if t == _STRING:
            skip_string()
        elif t == _ARRAY:
            it, cnt = take("<I"), take("<Q")
            if it == _STRING:
                for _ in range(cnt):
                    skip_string()
            elif it in _FIXED:
                o += _FIXED[it] * cnt
            else:
                raise StampError(f"{path}: an array of a type this does not walk")
        elif t in _FIXED:
            o += _FIXED[t]
        else:
            raise StampError(f"{path}: a value of a type this does not walk")
    if set(found) != {"split.no", "split.count"}:
        raise StampError(f"{path}: no split stamp")
    return found


def read(path):
    """(no, count): no 1-based"""
    fields = _fields(path)
    out = []
    with open(path, "rb") as f:
        for key in ("split.no", "split.count"):
            off, fmt = fields[key]
            f.seek(off)
            out.append(struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0])
    return out[0] + 1, out[1]


def _values(path, no, count):
    fields = _fields(path)
    return {fields["split.no"][0]: struct.pack(fields["split.no"][1], no - 1),
            fields["split.count"][0]: struct.pack(fields["split.count"][1], count)}


def write(path, no, count):
    """the stamp set to (no, count) in place: two values of the header, nothing else"""
    if not 1 <= no <= count:
        raise StampError(f"{path}: stamp {no} of {count}")
    with open(path, "r+b") as f:
        for off, v in _values(path, no, count).items():
            f.seek(off)
            f.write(v)


def canonical_sha256(path, published):
    """the SHA-256 of the file with its stamp read as `published` (no, count)"""
    patch = _values(path, *published)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        at = 0
        while True:
            chunk = bytearray(f.read(_CHUNK))
            if not chunk:
                break
            for off, v in patch.items():
                if at <= off < at + len(chunk):
                    chunk[off - at:off - at + len(v)] = v  # (a header value never straddles: the head is one read)
            h.update(chunk)
            at += len(chunk)
    return h.hexdigest()
