"""a pack laid out: <models>/<pack>/ - its shards linked by their names, the Engram table as its shard slot, the
picks profile, the drafter (dspark/; an MTP head its one GGUF beside the shards), the overlay if asked, osmos.json. each file the user's (found by userfiles) or
fetched at its pinned commit (`fetch(entry) -> path`); ours not yet published and not found: refused.

a table's stamp: one the installer fetched is stamped in place; a user's of another stamp than the slot is copied
(a reflink where the filesystem has them) under <models>/engram/ and the copy stamped - never a user's file written."""
import json
import os
import re
import subprocess

import overlay_q4x
import stamp
import userfiles

_SHARD = re.compile(r"-(\d{5})-of-(\d{5})")


class LayoutError(Exception):
    pass


def _resolve(m, fid, have, trust_size, fetch, found=None):
    """(path, ours): ours - fetched by the installer (its to write). `found`: the user's files already matched (the
    plan's), by id"""
    e = m["files"][fid]
    p = (found or {}).get(fid) or (userfiles.find(e, have, trust_size) if have else None)
    if p:
        return p, False
    if e["revision"] is None:
        raise LayoutError(f"{fid} ({e['path']}): not published yet and not among your files")
    if fetch is None:
        raise LayoutError(f"{fid} ({e['path']}): not among your files, and fetching is off")
    return fetch(e), True


def _derive(m, fid, models, have, trust_size, fetch, found):
    """a derived file's path: the user's if they have it, else made from its inputs under <models>/derived/ - its
    bytes and SHA-256 the pinned ones, or refused with nothing of it left"""
    e = m["files"][fid]
    p = (found or {}).get(fid) or (userfiles.find(e, have, False) if have else None)
    if p:
        return p
    d = e["derive"]
    out = os.path.join(models, "derived", e["path"])
    if os.path.exists(out) and userfiles.matches(out, e):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    patch, _ = _resolve(m, d["patch"], have, trust_size, fetch, found)
    key, _ = _resolve(m, d["key"], have, trust_size, fetch, found)
    shards = [_resolve(m, s, have, trust_size, fetch, found)[0] for s in d["model"]]
    try:
        overlay_q4x.convert(shards, patch, key, d["mode"], out)
    except overlay_q4x.OverlayError as x:
        raise LayoutError(f"{fid}: {x}")
    if not userfiles.matches(out, e):
        os.remove(out)
        raise LayoutError(f"{fid} ({e['path']}): made, but not the pinned bytes and SHA-256 - refused")
    return out


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def _table(m, pack, models, have, trust_size, fetch, first, found):
    t = pack["table"]
    no, count = t["slot"]
    names = _SHARD.search(first)
    if not names or int(names.group(2)) != count:
        raise LayoutError(f"the table's slot {no} of {count} is not the shards' ({first})")
    src, ours = _resolve(m, t["file"], have, trust_size, fetch, found)
    if stamp.read(src) != (no, count):
        if ours:
            stamp.write(src, no, count)
        else:
            os.makedirs(os.path.join(models, "engram"), exist_ok=True)
            base = os.path.basename(m["files"][t["file"]]["path"]).split(".gguf")[0]
            copy = os.path.join(models, "engram", f"{base}-{no}of{count}.gguf")
            if not (os.path.exists(copy) and stamp.read(copy) == (no, count) and
                    userfiles.matches(copy, m["files"][t["file"]], trust_size)):
                subprocess.run(["cp", "--reflink=auto", src, copy], check=True)
                stamp.write(copy, no, count)
            src = copy
    return src, _SHARD.sub(f"-{no:05d}-of-{count:05d}", first)


def lay(m, pid, models, have, trust_size, abliterated, fetch, found=None):
    """the pack's folder made; the path of its osmos.json"""
    pack = m["packs"][pid]
    out = os.path.join(models, pid)
    os.makedirs(out, exist_ok=True)
    names = []
    for fid in pack["shards"]:
        src, _ = _resolve(m, fid, have, trust_size, fetch, found)
        names.append(os.path.basename(m["files"][fid]["path"]))
        _link(src, os.path.join(out, names[-1]))
    cfg = {"pack": pid, "title": pack.get("title", pid), "model": os.path.join(out, names[0])}
    if pack.get("table"):
        src, name = _table(m, pack, models, have, trust_size, fetch, names[0], found)
        _link(src, os.path.join(out, name))
    if pack.get("picks"):
        src, _ = _resolve(m, pack["picks"], have, trust_size, fetch, found)
        cfg["picks"] = os.path.join(out, "osmos-picks.bin")
        _link(src, cfg["picks"])
    if pack.get("drafter") and pack.get("drafter_kind", "dspark") == "mtp":  # the head's GGUF: osmos.json names it
        fid = pack["drafter"][0]
        src, _ = _resolve(m, fid, have, trust_size, fetch, found)
        cfg["drafter"] = os.path.join(out, os.path.basename(m["files"][fid]["path"]))
        _link(src, cfg["drafter"])
    elif pack.get("drafter"):
        d = os.path.join(out, "dspark")
        os.makedirs(d, exist_ok=True)
        for fid in pack["drafter"]:
            src, _ = _resolve(m, fid, have, trust_size, fetch, found)
            _link(src, os.path.join(d, os.path.basename(m["files"][fid]["path"])))
        cfg["drafter"] = d
    if abliterated:
        if not pack.get("overlay"):
            raise LayoutError(f"pack {pid} has no abliteration overlay")
        oid = pack["overlay"]
        src = (_derive(m, oid, models, have, trust_size, fetch, found) if "derive" in m["files"][oid]
               else _resolve(m, oid, have, trust_size, fetch, found)[0])
        cfg["overlay"] = os.path.join(out, os.path.basename(m["files"][oid]["path"]))
        _link(src, cfg["overlay"])
    cfg.update(pack["settings"])
    path = os.path.join(out, "osmos.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return path


def read_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
