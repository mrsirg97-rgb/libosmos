"""a manifest file among the user's own: any file under the directories given of its bytes (names are free - a
renamed shard is found), then of its SHA-256 (a stamped table's as published: stamp.canonical_sha256). the first
that is; None if none."""
import hashlib
import os

import stamp

_CHUNK = 1 << 24
# --trust-size skips the hash of a file this large alone: a model shard's minutes of hashing. a smaller one is hashed
# whatever the flag - well under a second, and files of one size are common there (two packs' picks profiles)
TRUST_FROM = 1 << 30


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates(dirs, size):
    """every regular file of `size` bytes under dirs, each once (its real path)"""
    seen = set()
    for d in dirs:
        for root, _, names in os.walk(d, followlinks=True):
            for n in names:
                p = os.path.realpath(os.path.join(root, n))
                if p in seen or not os.path.isfile(p):
                    continue
                seen.add(p)
                try:
                    if os.path.getsize(p) == size:
                        yield p
                except OSError:
                    continue


def matches(path, entry, trust_size=False):
    """the file is the entry's: its bytes, and its SHA-256 (a stamped table's read as published) - unless trust_size
    and the file is large (TRUST_FROM)"""
    if os.path.getsize(path) != entry["bytes"]:
        return False
    if trust_size and entry["bytes"] >= TRUST_FROM:
        return True
    if "stamp" in entry:
        try:
            return stamp.canonical_sha256(path, tuple(entry["stamp"])) == entry["sha256"]
        except stamp.StampError:
            return False
    return _sha256(path) == entry["sha256"]


def find(entry, dirs, trust_size=False):
    for p in candidates(dirs, entry["bytes"]):
        if matches(p, entry, trust_size):
            return p
    return None
