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


_SEEN = {}  # (real path, bytes, mtime_ns): its SHA-256 - a file hashed once a run (the plan's, then the layout's)


def _sha256(path):
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _SEEN:
        _SEEN[key] = _sha256_read(path)
    return _SEEN[key]


def _sha256_read(path):
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


MADE_STAMP = ".sha256"  # a made file's stamp beside it: the pinned SHA-256 it was checked against when made


def stamp_made(path, entry):
    """a made file, checked against the entry's SHA-256: that SHA-256 written beside it (MADE_STAMP)"""
    with open(path + MADE_STAMP, "w") as f:
        f.write(entry["sha256"] + "\n")


def made_matches(path, entry, trust_size=False):
    """a file this installer made is the entry's: under trust_size its bytes alone when its stamp is the entry's pinned
    SHA-256 (a stale recipe's, or none: hashed); otherwise, and on any doubt, matches() - stamped again when it does"""
    if os.path.getsize(path) != entry["bytes"]:
        return False
    if trust_size and entry["bytes"] >= TRUST_FROM:
        try:
            with open(path + MADE_STAMP) as f:
                if f.read().strip() == entry["sha256"]:
                    return True
        except OSError:
            pass
    if not matches(path, entry):
        return False
    stamp_made(path, entry)
    return True


def find(entry, dirs, trust_size=False):
    """the first of the user's files that is the entry's. under trust_size a large file's bytes alone - unless more than
    one file has them (0.9.2: the first taken, another pack's shard of the same size perhaps): then each hashed"""
    found = list(candidates(dirs, entry["bytes"]))
    alone = trust_size and len(found) == 1
    for p in found:
        if matches(p, entry, alone):
            return p
    return None
