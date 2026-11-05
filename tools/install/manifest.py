"""packs.json: the files a pack reads and how it runs - checked whole before anything is fetched or written.

a file: repo, revision (a commit, 40 hex - or null: ours, not yet published), path, bytes, sha256 (64 hex), stamp
(optional: [no, count], the split stamp it is published with). a pack: title, shards (file ids, in order), table
(null, or {file, slot: [no, count]}), picks, drafter (file ids), overlay (a file id or null), settings."""
import json
import re

VERSION = 1
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(Exception):
    pass


def _file(fid, f):
    for k in ("repo", "path"):
        if not isinstance(f.get(k), str) or not f[k]:
            raise ManifestError(f"file {fid}: no {k}")
    rev = f.get("revision", "")
    if rev is not None and not (isinstance(rev, str) and _HEX40.match(rev)):
        raise ManifestError(f"file {fid}: revision {rev!r} is not a pinned commit (40 hex) nor null (unpublished)")
    if not isinstance(f.get("bytes"), int) or f["bytes"] < 0:
        raise ManifestError(f"file {fid}: no bytes")
    if not isinstance(f.get("sha256"), str) or not _HEX64.match(f["sha256"]):
        raise ManifestError(f"file {fid}: sha256 is not 64 hex")
    if "stamp" in f and not _slot(f["stamp"]):
        raise ManifestError(f"file {fid}: stamp {f['stamp']!r} is not [no, count]")


def _slot(s):
    return (isinstance(s, list) and len(s) == 2 and all(isinstance(x, int) for x in s) and 1 <= s[0] <= s[1])


def _ref(m, pid, what, fid):
    if fid not in m["files"]:
        raise ManifestError(f"pack {pid}: {what} {fid!r} is not a file of the manifest")


def check(m):
    """m as a manifest, or ManifestError naming what is not"""
    if not isinstance(m, dict) or m.get("version") != VERSION:
        raise ManifestError(f"version {m.get('version') if isinstance(m, dict) else None!r}: this installer reads "
                            f"version {VERSION}")
    if not isinstance(m.get("files"), dict) or not isinstance(m.get("packs"), dict):
        raise ManifestError("no files or no packs")
    r = m.get("rig")
    if not (isinstance(r, dict) and isinstance(r.get("repo"), str) and re.fullmatch(r"[\w.-]+/[\w.-]+", r["repo"])
            and isinstance(r.get("version"), str) and re.fullmatch(r"v\d+\.\d+\.\d+", r["version"])
            and isinstance(r.get("install_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", r["install_sha256"])):
        raise ManifestError("rig: not {repo, version (a release tag vX.Y.Z), install_sha256 (its install.sh's)}")
    for fid, f in m["files"].items():
        _file(fid, f)
    for pid, p in m["packs"].items():
        if not isinstance(p.get("shards"), list) or not p["shards"]:
            raise ManifestError(f"pack {pid}: no shards")
        for fid in p["shards"]:
            _ref(m, pid, "shard", fid)
        t = p.get("table")
        if t is not None:
            _ref(m, pid, "table", t.get("file"))
            if not _slot(t.get("slot")):
                raise ManifestError(f"pack {pid}: table slot {t.get('slot')!r} is not [no, count] with no <= count")
            if "stamp" not in m["files"][t["file"]]:
                raise ManifestError(f"pack {pid}: table {t['file']} has no published stamp")
        for k in ("picks", "overlay"):
            if p.get(k) is not None:
                _ref(m, pid, k, p[k])
        for fid in p.get("drafter") or []:
            _ref(m, pid, "drafter", fid)
        if not isinstance(p.get("settings"), dict):
            raise ManifestError(f"pack {pid}: no settings")
    return m


def load(path):
    with open(path, encoding="utf-8") as f:
        return check(json.load(f))
