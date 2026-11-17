"""osmos's versions side by side, and rig's rows moved between them (0.9.2): osmos-install's --upgrade, --rollback,
--versions and --prune.

each version a checkout of its tag built in place, <home>/<version> (home: ~/.local/share/osmos, or $OSMOS_VERSIONS)
- a rig row runs <home>/<version>/bin/osmos-api --config <pack>/osmos.<version>.json --port ${PORT}. a row moved to
another version changes its command alone (its port, window, efforts and the rest are the user's to keep); its
config frozen per version - osmos-api refuses a key it does not know, so one osmos.json a newer installer writes a
new key into would break every older row (a rollback past it). each move journaled (<home>/upgrades.json): a
rollback restores the last one's commands, on the rows still as it left them."""
import json
import os
import re
import shutil
import subprocess
import time

import rig

REPO = "https://github.com/mrsirg97-rgb/libosmos.git"  # a checkout without an origin: osmos's own
BUILD = [["git", "submodule", "update", "--init"],  # libevt (osmos-serve's loop)
         ["make", "-j"], ["make", "-j", "serve"], ["make", "-j", "api"]]  # the engine, osmos-serve, osmos-api
_CMD = re.compile(r"^(?P<dir>\S+)/bin/osmos-api --config (?P<cfg>\S+)(?P<rest>.*)$")
_VER = re.compile(r"^\d+\.\d+\.\d+$")


class VersionError(Exception):
    pass


def home():
    return os.environ.get("OSMOS_VERSIONS") or os.path.expanduser("~/.local/share/osmos")


def key(v):
    return tuple(int(x) for x in v.split("."))


def command(cmd):
    """a row's command as (version dir, config, the rest) - None for one that is not a versioned osmos-api's"""
    m = _CMD.match(cmd or "")
    return (m["dir"], m["cfg"], m["rest"]) if m and _VER.match(os.path.basename(m["dir"])) else None


def installed(h):
    """the versions built under h, oldest first"""
    if not os.path.isdir(h):
        return []
    vs = [v for v in os.listdir(h) if _VER.match(v) and os.access(os.path.join(h, v, "bin", "osmos-api"), os.X_OK)]
    return sorted(vs, key=key)


def rows_on(rows):
    """{version: [row ids]} of the rows that run a versioned osmos-api"""
    out = {}
    for r in rows:
        c = command(r.get("command"))
        if c:
            out.setdefault(os.path.basename(c[0]), []).append(r.get("id"))
    return out


def latest(repo, run=subprocess.run):
    """the repo's newest release tag (vX.Y.Z), its version"""
    p = run(["git", "ls-remote", "--tags", "--refs", repo, "v*"], capture_output=True, text=True)
    if p.returncode:
        raise VersionError(f"git ls-remote {repo}: {p.stderr.strip() or 'failed'}")
    tags = [line.rsplit("refs/tags/v", 1)[-1] for line in p.stdout.splitlines() if "refs/tags/v" in line]
    tags = [t for t in tags if _VER.match(t)]
    if not tags:
        raise VersionError(f"{repo}: no release tags")
    return max(tags, key=key)


def build(h, ver, repo, run=subprocess.run, log=print):
    """<h>/<ver> built from its tag: kept as it is when its osmos-api already says it is that version, else cloned
    beside it and built, its osmos-api's --version checked, then moved into place (nothing half-built left there)"""
    at = os.path.join(h, ver)
    api = os.path.join(at, "bin", "osmos-api")

    def says(path):
        p = run([path, "--version"], capture_output=True, text=True)
        return p.returncode == 0 and p.stdout.strip() == f"osmos {ver}"
    if os.access(api, os.X_OK) and says(api):
        log(f"  {ver}: built already ({at})")
        return at
    if os.path.exists(at):
        raise VersionError(f"{at}: there, but not osmos {ver}'s build - moved aside by hand first")
    tmp = at + ".osmos-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(h, exist_ok=True)
    log(f"  {ver}: cloning v{ver} and building it ({tmp})")
    steps = [["git", "clone", "--depth", "1", "--branch", f"v{ver}", repo, tmp]] + BUILD
    for i, s in enumerate(steps):
        p = run(s, cwd=None if i == 0 else tmp, capture_output=True, text=True)
        if p.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            raise VersionError(f"{' '.join(s)}: failed - {(p.stderr or p.stdout).strip()[-400:]}")
    if not says(os.path.join(tmp, "bin", "osmos-api")):
        shutil.rmtree(tmp, ignore_errors=True)
        raise VersionError(f"v{ver}'s build does not say osmos {ver} (osmos-api --version)")
    os.replace(tmp, at)
    return at


def freeze(cfg, ver):
    """the pack's config for version ver: osmos.<ver>.json beside it - written once, from the shared osmos.json (or
    the frozen file given), never rewritten: a later installer's keys never reach a version that does not know them"""
    d = os.path.dirname(cfg)
    out = os.path.join(d, f"osmos.{ver}.json")
    if not os.path.exists(out):
        with open(cfg, encoding="utf-8") as f:
            body = json.load(f)
        tmp = out + ".osmos-tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
            f.write("\n")
        os.replace(tmp, out)
    return out


def _journal(h):
    return os.path.join(h, "upgrades.json")


def _read_journal(h):
    p = _journal(h)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        v = json.load(f)
    if not isinstance(v, list):
        raise VersionError(f"{p}: not a list - left as it is")
    return v


def _write_json(path, v):
    tmp = path + ".osmos-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def pick(rows, which, vs):
    """the rows an upgrade moves: `which` ids (or "all": every versioned row); by default the rows on the newest
    version built before it - the ones following releases (a row kept on an older one stays there)"""
    on = rows_on(rows)
    if which == "all":
        return [i for ids in on.values() for i in ids]
    if which:
        ids = [w for w in which.split(",") if w]
        unknown = [i for i in ids if not any(r.get("id") == i for r in rows)]
        if unknown:
            raise VersionError(f"no rig row {', '.join(unknown)}")
        bad = [i for i in ids if not any(r.get("id") == i and command(r.get("command")) for r in rows)]
        if bad:
            raise VersionError(f"{', '.join(bad)}: not a versioned osmos-api row ({home()}/<version>/bin/osmos-api)")
        return ids
    newest = max((v for v in on if v in vs), key=key, default=None)
    return list(on.get(newest, []))


def upgrade(rig_home, h, ver, which=None, log=print):
    """rig's chosen rows moved to <h>/<ver> (built already: build()): each row's command alone changed, its config
    frozen for both versions (the one it leaves, the one it takes); the move journaled. the rows moved"""
    mpath = os.path.join(rig_home, "models.json")
    rows = rig._read(mpath, [])
    ids = pick(rows, which, installed(h))
    api = os.path.join(h, ver, "bin", "osmos-api")
    moves = []
    for r in rows:
        if r.get("id") not in ids:
            continue
        d, cfg, rest = command(r["command"])
        old = os.path.basename(d)
        if old == ver:
            log(f"  {r['id']}: on {ver} already")
            continue
        shared = os.path.join(os.path.dirname(cfg), "osmos.json")
        was = f"{d}/bin/osmos-api --config {freeze(cfg, old)}{rest}"  # the row as it was, its config frozen
        now = f"{api} --config {freeze(shared if os.path.exists(shared) else cfg, ver)}{rest}"
        moves.append({"id": r["id"], "from": was, "to": now, "from_version": old})
        r["command"] = now
        log(f"  {r['id']}: {old} -> {ver}")
    if moves:
        rig._write(mpath, rows)
        j = _read_journal(h)
        j.append({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": ver, "rows": moves})
        _write_json(_journal(h), j)
    return moves


def rollback(rig_home, h, log=print):
    """the last upgrade undone: each row it moved, still on the command it left, back to the one it had (its config
    the frozen one) - a row changed since left as it is, said so. the journal's last entry dropped"""
    j = _read_journal(h)
    if not j:
        raise VersionError(f"no upgrade to roll back ({_journal(h)})")
    last = j[-1]
    mpath = os.path.join(rig_home, "models.json")
    rows = rig._read(mpath, [])
    back = []
    for mv in last["rows"]:
        r = next((x for x in rows if x.get("id") == mv["id"]), None)
        if r is None or r.get("command") != mv["to"]:
            log(f"  {mv['id']}: changed since the upgrade - left as it is")
            continue
        r["command"] = mv["from"]
        back.append(mv["id"])
        log(f"  {mv['id']}: {last['version']} -> {mv['from_version']}")
    rig._write(mpath, rows)
    _write_json(_journal(h), j[:-1])
    return back


def unused(rig_home, h):
    """the versions built under h no rig row runs - the newest kept whatever (the next upgrade's start)"""
    vs = installed(h)
    on = rows_on(rig._read(os.path.join(rig_home, "models.json"), []))
    return [v for v in vs[:-1] if v not in on]


def prune(rig_home, h, yes, log=print):
    """the unused versions' folders removed (yes), or named (not)"""
    gone = unused(rig_home, h)
    for v in gone:
        p = os.path.join(h, v)
        if yes:
            shutil.rmtree(p)
        log(f"  {'removed' if yes else 'unused '} {v} ({p})")
    if gone and not yes:
        log("  (--prune --yes removes them)")
    return gone


def report(rig_home, h, log=print):
    """each version built, the rows on it; a row on a version not built here, said"""
    vs = installed(h)
    on = rows_on(rig._read(os.path.join(rig_home, "models.json"), []))
    for v in vs:
        log(f"  {v:10s} {', '.join(on.get(v, [])) or '-'}")
    for v in sorted(set(on) - set(vs), key=key):
        log(f"  {v:10s} {', '.join(on[v])} (not built here: {os.path.join(h, v)})")
