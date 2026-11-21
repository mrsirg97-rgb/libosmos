#!/usr/bin/env python3
"""osmos-install: from a stock Strix Halo to a pack one can talk to.

    tools/install/osmos_install.py --list
    tools/install/osmos_install.py ud-iq1_s [ud-iq1_m] [--models DIR] [--have DIR ...] [--trust-size]
                                   [--abliterated] [--rig [--rig-default yes|no|ask]] [--serve] [--no-build] [--dry-run]
    tools/install/osmos_install.py --doctor [--models DIR]

the packs and their files: packs.json beside this (every file pinned to a commit and its SHA-256). each file taken
from the user's own (--have: any file of its bytes and hash under those directories, renamed or not) or downloaded
at its commit and checked; the engine built from this checkout; each pack laid out under --models (its shards, the
Engram table as its slot, the picks, the drafter - DSpark's directory, or Qwen's MTP head - the overlay if
--abliterated, osmos.json). then --rig registers it
in rig (~/.rig: a row rig launches, the default model only when asked; rig itself installed first when it is
not here - its install.sh at the manifest's pinned release, checked by SHA-256), --serve starts osmos-api. design: DESIGN.md,
"the installer"."""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import doctor  # noqa: E402
import fetch  # noqa: E402
import host  # noqa: E402
import layout  # noqa: E402
import manifest  # noqa: E402
import rig  # noqa: E402
import userfiles  # noqa: E402
import versions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
API = os.path.join(ROOT, "bin", "osmos-api")


def own_version(path=os.path.join(ROOT, "include", "osmos", "version.h")):
    """this checkout's osmos version (OSMOS_VERSION)"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#define OSMOS_VERSION "):
                return line.split('"')[1]
    raise SystemExit(f"{path}: no OSMOS_VERSION")


def rig_target(cfg, ver, h, dev, build=lambda h, v: versions.build(h, v, versions.REPO)):
    """the osmos-api and config a new rig row names: <h>/<ver> (its release built there, or kept as it is) and the
    pack's config frozen for it (osmos.<ver>.json) - the shape --upgrade moves; dev (--rig-dev): this checkout's own
    osmos-api and osmos.json, a row no upgrade moves (a branch's, to test)"""
    if dev:
        return API, cfg
    try:
        build(h, ver)
    except versions.VersionError as x:
        raise SystemExit(f"--rig: osmos {ver} has no release to build ({x}); --rig-dev registers this checkout's own")
    return os.path.join(h, ver, "bin", "osmos-api"), versions.freeze(cfg, ver, own=True)
BUILD = [["git", "submodule", "update", "--init"],  # libevt (osmos-serve's loop)
         ["make", "-j"], ["make", "-j", "serve"], ["make", "-j", "api"]]  # the engine, osmos-serve, osmos-api


def find_rig():
    """rig on PATH, or where its install.sh lands it (RIG_BIN, else ~/.local/bin)"""
    at = os.path.join(os.environ.get("RIG_BIN") or os.path.expanduser("~/.local/bin"), "rig")
    return shutil.which("rig") or (at if os.access(at, os.X_OK) else None)


def download(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def run_sh(script, env):
    print("  rig not found: installing it with its own install.sh", flush=True)
    subprocess.run(["sh", "-s"], input=script, env=env, check=True)


def inputs(m, fid, made=lambda fid: False):
    """the published files a file takes: itself, or a derived one's inputs (the layout makes it from them) - a derived
    base's own in turn (0.8.0: evo-A's patch on evo-A's shards), none for one already made (made(fid): an install
    that has evo-A fetches the patch alone)"""
    d = m["files"][fid].get("derive")
    if not d:
        return [fid]
    if made(fid):
        return []
    if d["with"] == "gguf_assemble":
        return [i for b in d["base"] for i in inputs(m, b, made)] + manifest.deltas(d)
    return [d["patch"], d["key"]] + d["model"]


def needed(m, pids, abliterated, made=lambda fid: False):
    ids = []
    for pid in pids:
        p = m["packs"][pid]
        for fid in p["shards"]:
            ids += inputs(m, fid, made)
        ids += [p["table"]["file"]] if p.get("table") else []
        ids += [p["picks"]] if p.get("picks") else []
        ids += p.get("drafter") or []
        ids += [p["lsa"]] if p.get("lsa") else []
        if abliterated and p.get("overlay"):
            ids += inputs(m, p["overlay"])
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
        elif e.get("revision", "") is None:
            raise SystemExit(f"{fid} ({e['path']}): not published yet and not among your files (--have)")
        else:
            fetch_ids.append(fid)
            where = f"release {e['release']}" if "release" in e else e["repo"]
            print(f"  fetch  {fid:16s} {where}/{e['path']} ({e['bytes'] / 1e9:.1f} GB)")
    return found, fetch_ids


def fetch_bytes(m, fetch_ids, models):
    """the bytes the files to fetch still need on disk: each one's past what the cache holds (fetch.on_disk)"""
    return sum(m["files"][f]["bytes"] - fetch.on_disk(models, m["files"][f]) for f in fetch_ids)


def gpu_need(m, packs):
    """the GPU memory the packs need to start (GiB): the largest of their gpu_gib, one never measured the measured
    machine's (host.GPU_GIB_MEASURED)"""
    return max(m["packs"][p].get("gpu_gib", host.GPU_GIB_MEASURED) for p in packs)


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


CTX_MIN, CTX_MAX = 4096, 1 << 20  # osmos-api's --ctx bounds (tools/api/main.c)


def ctx_length(v):
    """--ctx's length: N, Nk (x 1,024) or Nm (x 1,048,576), within CTX_MIN .. CTX_MAX"""
    s = v.strip().lower()
    mul = 1024 if s.endswith("k") else 1 << 20 if s.endswith("m") else 1
    try:
        n = int(s[:-1] if mul > 1 else s) * mul
    except ValueError:
        n = 0
    if not CTX_MIN <= n <= CTX_MAX:
        raise argparse.ArgumentTypeError(f"{v}: a length from {CTX_MIN} to {CTX_MAX} (N, Nk or Nm: 524288, 512k, 1m)")
    return n


def parser():
    ap = argparse.ArgumentParser(description="osmos-install: a pack from a stock Strix Halo, or from files you have")
    ap.add_argument("packs", nargs="*")
    ap.add_argument("--list", action="store_true", help="the packs and what each is")
    ap.add_argument("--manifest", default=os.path.join(HERE, "packs.json"))
    ap.add_argument("--models", default=os.path.expanduser("~/Models/osmos"), help="where packs are laid out")
    ap.add_argument("--have", nargs="+", action="extend", default=[], metavar="DIR",
                    help="directories of files you already have (more than one, or the flag again: each added)")
    ap.add_argument("--trust-size", action="store_true",
                    help="take your large files by bytes alone (small ones are hashed still)")
    ap.add_argument("--abliterated", action="store_true", help="the abliteration overlay (huihui-ai), opt-in")
    ap.add_argument("--npu", action="store_true",
                    help="the NPU beside the GPU for prompts (0.7.0, opt-in): its kernels fetched, osmos built with it")
    ap.add_argument("--ctx", type=ctx_length, default=None,
                    help="the context length, 4096 to 1M: N, Nk or Nm (the pack's own by default: DeepSeek's 1M, Qwen's 512K)")
    ap.add_argument("--rig", action="store_true", help="register each pack in rig (~/.rig/models.json), on this "
                    "version's release (built under ~/.local/share/osmos)")
    ap.add_argument("--rig-dev", action="store_true", help="with --rig: the row on this checkout's own osmos-api "
                    "(a branch's, to test; --upgrade does not move it)")
    ap.add_argument("--rig-home", default=os.environ.get("RIG_HOME", os.path.expanduser("~/.rig")))
    ap.add_argument("--rig-default", choices=("yes", "no", "ask"), default="ask")
    ap.add_argument("--serve", action="store_true", help="start osmos-api on the (first) pack")
    ap.add_argument("--port", type=int, default=None,
                    help="--serve's port (the first free one from 8091 by default: rig's rows' own may be serving)")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--upgrade", nargs="?", const="latest", metavar="VERSION",
                    help="build osmos VERSION (the newest release by default) beside the others and move rig's rows to "
                         "it - the rows on the newest version built, or --rows (versions.py)")
    ap.add_argument("--rows", default=None, help="--upgrade's rig rows: ids, comma-separated, or all")
    ap.add_argument("--rollback", action="store_true", help="the last --upgrade's rows back where they were")
    ap.add_argument("--versions", action="store_true", help="the osmos versions built here and the rig rows on each")
    ap.add_argument("--prune", action="store_true", help="the versions no rig row runs (removed with --yes)")
    ap.add_argument("--yes", action="store_true", help="--prune: remove them")
    ap.add_argument("--dry-run", action="store_true", help="the checks and the plan, nothing written")
    ap.add_argument("--doctor", action="store_true",
                    help="the machine as osmos will run on it: every check, what to change (doctor.py)")
    return ap


def free_port(start, host="127.0.0.1"):
    """the first port from start nothing listens on (0.9.2: --serve took 8091 whatever held it - rig's first row)"""
    import socket
    for p in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise SystemExit(f"--serve: no free port from {start} (--port one)")


def versions_mode(a):
    """--upgrade, --rollback, --versions, --prune: osmos's versions side by side, rig's rows moved between them"""
    h = versions.home()
    try:
        if a.versions:
            print(f"osmos versions in {h} (rig's rows on each):")
            versions.report(a.rig_home, h)
        if a.upgrade:
            repo = subprocess.run(["git", "-C", ROOT, "remote", "get-url", "origin"], capture_output=True,
                                  text=True).stdout.strip() or versions.REPO
            ver = versions.latest(repo) if a.upgrade == "latest" else a.upgrade.lstrip("v")
            print(f"osmos {ver}:")
            versions.build(h, ver, repo, dry=a.dry_run)
            print("rig:")
            moved = versions.upgrade(a.rig_home, h, ver, a.rows, dry=a.dry_run)
            if a.dry_run:  # (0.12.2: --upgrade --dry-run had built the version and moved the rows)
                print(f"  {len(moved)} row(s) would move to {ver} - a dry run, nothing built or written")
            else:
                print(f"  {len(moved)} row(s) moved; rig starts them on {ver} at their next launch (a running one: "
                      f"restart it)" if moved else "  no row moved (--rows names them, or all)")
        if a.rollback:
            print("rig:")
            versions.rollback(a.rig_home, h, dry=a.dry_run)
        if a.prune:
            print(f"versions no rig row runs ({h}):")
            versions.prune(a.rig_home, h, a.yes and not a.dry_run)
    except (versions.VersionError, rig.RigError) as x:
        raise SystemExit(str(x))
    return 0


def rig_rooms(m, packs, ctx):
    """--rig: each pack's row refused before anything is fetched, where rig would refuse it after (0.11.2: a --ctx too
    small for a pack's reply reserve found once its files were down)"""
    for pid in packs:
        why = rig.room(ctx or m["packs"][pid]["settings"]["max_context"], m["packs"][pid]["rig"]["reserve"])
        if why:
            raise SystemExit(f"--rig {pid}: {why} - a larger --ctx, or none (the pack's own)")


def main():
    """the installer: a failure of a step (a download, the layout, rig, the build) one line, not a traceback (0.11.2)"""
    try:
        return run()
    except (fetch.FetchError, layout.LayoutError, rig.RigError, versions.VersionError) as x:
        raise SystemExit(f"osmos_install: {x}")
    except subprocess.CalledProcessError as x:
        raise SystemExit(f"osmos_install: the build failed ({' '.join(map(str, x.cmd))}, exit {x.returncode}) - "
                         "its output above says why; --no-build skips it")


def run():
    a = parser().parse_args()
    missing = [d for d in a.have if not os.path.isdir(d)]
    if missing:  # 0.9.2: passed over in silence - its files then fetched
        raise SystemExit("--have: no such directory: " + ", ".join(missing))
    if a.upgrade or a.rollback or a.versions or a.prune:
        return versions_mode(a)
    if a.doctor:
        checks = doctor.run(a.models)
        for level, why in checks:
            print(f"  {level:4s} {why}")
        fails = sum(level == doctor.FAIL for level, _ in checks)
        notes = sum(level == doctor.NOTE for level, _ in checks)
        print(f"{fails} to fix, {notes} to look at" if fails or notes else "as osmos was measured")
        return 1 if fails else 0
    m = manifest.load(a.manifest)
    for old in [p for p in a.packs if manifest.pack_id(m, p) != p]:
        print(f"  {old}: now {manifest.pack_id(m, old)}")
    a.packs = list(dict.fromkeys(manifest.pack_id(m, p) for p in a.packs))
    if a.list or not a.packs:
        for pid, p in m["packs"].items():
            print(f"{pid:12s} {p['title']}")
            if p.get("quality"):  # 0.6.6's reference (DESIGN.md): the distance from the model's own weights
                qd = p["quality"]
                print(f"{'':12s} quality: KL {qd.get('wiki_kld', '-')} on Wiki text, {qd.get('agent_kld', '-')} on agent "
                      f"traces (0 the model's own weights; lower is closer)")
        return 0
    for pid in a.packs:
        if pid not in m["packs"]:
            raise SystemExit(f"no pack {pid!r}: --list names them")
    if a.rig:
        rig_rooms(m, a.packs, a.ctx)
    print("the machine:")
    if not (check(host.gpu()) & check(host.gpu_memory(gpu_need(m, a.packs)))):
        raise SystemExit("not a machine osmos runs on as it is: the lines above say why")
    if a.npu:
        if not check(host.npu()):
            raise SystemExit("--npu: no NPU osmos can use here - the line above says why (without --npu the GPU runs it)")
        for pid in a.packs:
            if not m["packs"][pid].get("npu"):
                raise SystemExit(f"--npu: pack {pid} has no NPU kernels (the GPU alone runs it)")
    print("the files:")
    made = lambda fid: layout.made(m, fid, a.models, a.have, a.trust_size)  # noqa: E731
    ids = list(dict.fromkeys(needed(m, a.packs, a.abliterated, made) +
                             ([m["packs"][p]["npu"] for p in a.packs] if a.npu else [])))
    found, fetch_ids = plan(m, ids, a.have, a.trust_size)
    if not check(host.disk(a.models, fetch_bytes(m, fetch_ids, a.models))):
        raise SystemExit("not enough disk")
    if not check(host.tools(bool(fetch_ids), not a.no_build)):
        raise SystemExit("a tool missing: the line above says which")
    if a.dry_run:
        return 0
    if not a.no_build:
        print("the engine:")
        for cmd in BUILD:  # --npu: the HIP library and its binaries with XRT (NPU=1)
            subprocess.run(cmd + (["NPU=1"] if a.npu and cmd[0] == "make" else []), cwd=ROOT, check=True)
    get = fetch.fetcher(a.models)
    cfgs = []
    for pid in a.packs:
        print(f"the pack {pid}:")
        cfgs.append(layout.lay(m, pid, a.models, a.have, a.trust_size, a.abliterated, get, found, npu=a.npu, ctx=a.ctx))
        print(f"  {cfgs[-1]}")
    if a.rig:
        print("rig:")
        exe = rig.ensure(m["rig"], find_rig, download, run_sh)
        print(f"  {exe}")
        ver, h = own_version(), versions.home()
        for c in cfgs:
            cfg = layout.read_config(c)
            rid = m["packs"][cfg["pack"]]["rig"]["id"]  # (0.11.2: the prompt named 0.5.8's osmos-<pack>)
            api, cfg_path = rig_target(c, ver, h, a.rig_dev)
            row = rig.register(a.rig_home, cfg, api, cfg_path, ask_default(a.rig_default, rid), m["packs"][cfg["pack"]]["rig"])
            print(f"  {row['id']} on port {row['port']} in {a.rig_home}/models.json")
    if a.serve:
        if not os.path.exists(API):
            raise SystemExit(f"{API} is not built (make api)")
        port = a.port or free_port(rig.PORT_BASE)
        print(f"osmos-api on 127.0.0.1:{port}", flush=True)
        os.execv(API, [API, "--config", cfgs[0], "--port", str(port)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
