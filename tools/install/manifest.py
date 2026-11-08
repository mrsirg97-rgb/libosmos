"""packs.json: the files a pack reads and how it runs - checked whole before anything is fetched or written.

a file: repo, revision (a commit, 40 hex - or null: ours, not yet published), path, bytes, sha256 (64 hex), stamp
(optional: [no, count], the split stamp it is published with). a pack: title, shards (file ids, in order), table
(null, or {file, slot: [no, count]}), picks, drafter (file ids), drafter_kind (dspark: a directory of its files - the
default; mtp: one GGUF, the model's MTP head), overlay (a file id or null), rig (its row's: max_tokens, reserve,
efforts, effort - one of them - and aliases), settings.

a derived file: made, not downloaded - derive {with: the converter, its inputs (file ids of published files), mode},
path, bytes and sha256 (its result's, pinned as a download's is), no repo nor revision. overlay_q4x's: patch, key
(huihui-ai's abliteration and its key), model (the shards it stands in for), mode (overlay_q4x.MODES)."""
import json
import re

import overlay_q4x

VERSION = 1
DRAFTER_KINDS = ("dspark", "mtp")
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


def _derived(m, fid, f):
    d = f.get("derive")
    if not isinstance(d, dict) or d.get("with") != "overlay_q4x":
        raise ManifestError(f"file {fid}: derive.with {d.get('with') if isinstance(d, dict) else None!r} is not a "
                            "converter this installer runs (overlay_q4x)")
    if "repo" in f or "revision" in f:
        raise ManifestError(f"file {fid}: derived, not downloaded - no repo nor revision")
    ins = [d.get("patch"), d.get("key")] + (d.get("model") if isinstance(d.get("model"), list) else [None])
    for i in ins:
        if i not in m["files"] or "derive" in m["files"][i]:
            raise ManifestError(f"file {fid}: input {i!r} is not a published file of the manifest")
    if d.get("mode") not in overlay_q4x.MODES:
        raise ManifestError(f"file {fid}: mode {d.get('mode')!r} is not one of {sorted(overlay_q4x.MODES)}")
    if not isinstance(f.get("path"), str) or not f["path"] or "/" in f["path"]:
        raise ManifestError(f"file {fid}: no path (a file name)")
    if not isinstance(f.get("bytes"), int) or f["bytes"] < 0:
        raise ManifestError(f"file {fid}: no bytes")
    if not isinstance(f.get("sha256"), str) or not _HEX64.match(f["sha256"]):
        raise ManifestError(f"file {fid}: sha256 is not 64 hex")


def _slot(s):
    return (isinstance(s, list) and len(s) == 2 and all(isinstance(x, int) for x in s) and 1 <= s[0] <= s[1])


def _rig(pid, r):
    if not isinstance(r, dict):
        raise ManifestError(f"pack {pid}: no rig (its row's max_tokens, reserve, efforts, effort, aliases)")
    for k in ("max_tokens", "reserve"):
        if not isinstance(r.get(k), int) or r[k] < 1:
            raise ManifestError(f"pack {pid}: rig {k} {r.get(k)!r} is not a positive count of tokens")
    for k in ("efforts", "aliases"):
        if not (isinstance(r.get(k), list) and r[k] and all(isinstance(x, str) and x for x in r[k])):
            raise ManifestError(f"pack {pid}: rig {k} is not a list of names")
    if r.get("effort") not in r["efforts"]:
        raise ManifestError(f"pack {pid}: rig effort {r.get('effort')!r} is not among its efforts")


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
        _derived(m, fid, f) if "derive" in f else _file(fid, f)
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
        kind = p.get("drafter_kind", "dspark")
        if kind not in DRAFTER_KINDS:
            raise ManifestError(f"pack {pid}: drafter_kind {kind!r} is not one of {DRAFTER_KINDS}")
        if kind == "mtp" and len(p.get("drafter") or []) != 1:
            raise ManifestError(f"pack {pid}: an mtp drafter is one file (the head's GGUF)")
        _rig(pid, p.get("rig"))
        if not isinstance(p.get("settings"), dict):
            raise ManifestError(f"pack {pid}: no settings")
    return m


def load(path):
    with open(path, encoding="utf-8") as f:
        return check(json.load(f))
