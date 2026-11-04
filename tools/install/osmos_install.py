#!/usr/bin/env python3
"""osmos-install: from a stock Strix Halo to a pack one can talk to.

    tools/install/osmos_install.py --list
    tools/install/osmos_install.py ud-iq1_s [ud-iq1_m] [--models DIR] [--have DIR ...] [--trust-size]
                                   [--abliterated] [--rig [--rig-default yes|no|ask]] [--serve] [--no-build] [--dry-run]

the packs and their files: packs.json beside this (every file pinned to a commit and its SHA-256). each file taken
from the user's own (--have: any file of its bytes and hash under those directories, renamed or not) or downloaded
at its commit and checked; the engine built from this checkout; each pack laid out under --models (its shards, the
Engram table as its slot, the picks, the drafter, the overlay if --abliterated, osmos.json). then --rig registers it
in rig (~/.rig: a row rig launches, the default model only when asked), --serve starts osmos-api. design: DESIGN.md,
"the installer"."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch  # noqa: E402
import host  # noqa: E402
import layout  # noqa: E402
import manifest  # noqa: E402
import rig  # noqa: E402
import userfiles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
API = os.path.join(ROOT, "bin", "osmos-api")
BUILD = [["git", "submodule", "update", "--init"],  # libevt (osmos-serve's loop)
         ["make", "-j"], ["make", "-j", "serve"], ["make", "-j", "api"]]  # the engine, osmos-serve, osmos-api


def needed(m, pids, abliterated):
    ids = []
    for pid in pids:
        p = m["packs"][pid]
        ids += p["shards"] + ([p["table"]["file"]] if p.get("table") else [])
        ids += [p["picks"]] if p.get("picks") else []
        ids += p.get("drafter") or []
        if abliterated and p.get("overlay"):
            ids.append(p["overlay"])
    return list(dict.fromkeys(ids))


def plan(m, ids, have, trust_size):
    """({id: the user's path}, [ids to fetch]) - or refused: one of ours not published and not found"""
    found, fetch_ids = {}, []
    for fid in ids:
        e = m["files"][fid]
        p = userfiles.find(e, have, trust_size) if have else None
        if p:
            found[fid] = p
            print(f"  have   {fid:16s} {p}")
        elif e["revision"] is None:
            raise SystemExit(f"{fid} ({e['path']}): not published yet and not among your files (--have)")
        else:
            fetch_ids.append(fid)
            print(f"  fetch  {fid:16s} {e['repo']}/{e['path']} ({e['bytes'] / 1e9:.1f} GB)")
    return found, fetch_ids


def check(ok_why):
    ok, why = ok_why
    print(f"  {'ok  ' if ok else 'FAIL'} {why}")
    return ok


def ask_default(choice, rid):
    if choice != "ask":
        return choice == "yes"
    if not sys.stdin.isatty():
        print(f"  rig's default model left as it is (not a terminal: --rig-default yes sets {rid})")
        return False
    return input(f"  make {rid} rig's default model? [y/N] ").strip().lower() in ("y", "yes")


def main():
    ap = argparse.ArgumentParser(description="osmos-install: a pack from a stock Strix Halo, or from files you have")
    ap.add_argument("packs", nargs="*")
    ap.add_argument("--list", action="store_true", help="the packs and what each is")
    ap.add_argument("--manifest", default=os.path.join(HERE, "packs.json"))
    ap.add_argument("--models", default=os.path.expanduser("~/Models/osmos"), help="where packs are laid out")
    ap.add_argument("--have", nargs="*", default=[], help="directories of files you already have")
    ap.add_argument("--trust-size", action="store_true",
                    help="take your large files by bytes alone (small ones are hashed still)")
    ap.add_argument("--abliterated", action="store_true", help="the abliteration overlay (huihui-ai), opt-in")
    ap.add_argument("--rig", action="store_true", help="register each pack in rig (~/.rig/models.json)")
    ap.add_argument("--rig-home", default=os.environ.get("RIG_HOME", os.path.expanduser("~/.rig")))
    ap.add_argument("--rig-default", choices=("yes", "no", "ask"), default="ask")
    ap.add_argument("--serve", action="store_true", help="start osmos-api on the (first) pack")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="the checks and the plan, nothing written")
    a = ap.parse_args()
    m = manifest.load(a.manifest)
    if a.list or not a.packs:
        for pid, p in m["packs"].items():
            print(f"{pid:12s} {p['title']}")
        return 0
    for pid in a.packs:
        if pid not in m["packs"]:
            raise SystemExit(f"no pack {pid!r}: --list names them")
    print("the machine:")
    if not (check(host.gpu()) & check(host.gpu_memory())):
        raise SystemExit("not a machine osmos runs on as it is: the lines above say why")
    print("the files:")
    ids = needed(m, a.packs, a.abliterated)
    found, fetch_ids = plan(m, ids, a.have, a.trust_size)
    if not check(host.disk(a.models, sum(m["files"][f]["bytes"] for f in fetch_ids))):
        raise SystemExit("not enough disk")
    if not check(host.tools(bool(fetch_ids), not a.no_build)):
        raise SystemExit("a tool missing: the line above says which")
    if a.dry_run:
        return 0
    if not a.no_build:
        print("the engine:")
        for cmd in BUILD:
            subprocess.run(cmd, cwd=ROOT, check=True)
    get = fetch.fetcher(a.models)
    cfgs = []
    for pid in a.packs:
        print(f"the pack {pid}:")
        cfgs.append(layout.lay(m, pid, a.models, a.have, a.trust_size, a.abliterated, get, found))
        print(f"  {cfgs[-1]}")
    if a.rig:
        print("rig:")
        for c in cfgs:
            cfg = layout.read_config(c)
            rid = f"osmos-{cfg['pack']}"
            row = rig.register(a.rig_home, cfg, API, c, ask_default(a.rig_default, rid))
            print(f"  {row['id']} on port {row['port']} in {a.rig_home}/models.json")
    if a.serve:
        if not os.path.exists(API):
            raise SystemExit(f"{API} is not built (make api)")
        os.execv(API, [API, "--config", cfgs[0], "--port", str(rig.PORT_BASE)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
