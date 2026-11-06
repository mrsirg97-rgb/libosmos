"""a pack registered in rig (the user's agent runtime: it launches a local model as a child and routes to it): a row
merged by id into <rig home>/models.json - the provider half and the child half rig starts (`command`, `${PORT}`
substituted by rig; `port`; `aliases`; `stopGrace`; `places`) - and, asked, the default model in settings.json.
no other row or key touched; a file replaced whole (written aside, then renamed); a file rig would refuse refused
here first, unwritten."""
import hashlib
import json
import os

PORT_BASE = 8091        # the first port tried for an osmos row (rig's default endpoint is 8090)
STOP_GRACE = 30         # seconds rig's SIGTERM waits: a conversation's state written out (osmos-serve's slots)
# the provider half: the context the pack runs, and what a turn may take of it (rig's row: all four required) - the
# pack's own (packs.json's rig: a reply's tokens, its thinking and its answer - rig's thinkingBudget is Anthropic's
# alone - as much kept free, the reasoning efforts its family's osmos-api dialect maps, the default among them)
KEEP_RECENT_SHARE = 4   # keepRecent: a quarter of the window


class RigError(Exception):
    pass


def _read(path, empty):
    if not os.path.exists(path):
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
    except (OSError, ValueError) as e:
        raise RigError(f"{path}: not JSON rig reads ({e}) - left as it is")
    if type(v) is not type(empty):
        raise RigError(f"{path}: not a {type(empty).__name__} - left as it is")
    return v


def _write(path, v):
    tmp = path + ".osmos-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def ensure(pin, find, download, run):
    """rig's executable: the one find() sees, or else rig's own install.sh at the pinned release - its bytes checked
    against the pin's SHA-256 before it runs (it verifies the binary it fetches against the release's checksums) -
    then the one find() sees after it. seams: find() -> path or None; download(url) -> bytes; run(script, env)"""
    exe = find()
    if exe:
        return exe
    url = f"https://raw.githubusercontent.com/{pin['repo']}/{pin['version']}/install.sh"
    script = download(url)
    got = hashlib.sha256(script).hexdigest()
    if got != pin["install_sha256"]:
        raise RigError(f"{url}: its SHA-256 {got} is not the manifest's {pin['install_sha256']} - not run")
    run(script, dict(os.environ, RIG_VERSION=pin["version"]))
    exe = find()
    if not exe:
        raise RigError(f"rig's install.sh ({pin['version']}) ran and left no rig: is ~/.local/bin on PATH?")
    return exe


def _port(rows, rid):
    for r in rows:
        if r.get("id") == rid and isinstance(r.get("port"), int):
            return r["port"]
    used = {r.get("port") for r in rows}
    p = PORT_BASE
    while p in used:
        p += 1
    return p


def register(home, cfg, api, cfg_path, default, row):
    """the row rig launches the pack by (returned); `api` the osmos-api binary, `cfg_path` the pack's osmos.json,
    `row` the pack's rig settings (packs.json)"""
    os.makedirs(home, exist_ok=True)
    mpath, spath = os.path.join(home, "models.json"), os.path.join(home, "settings.json")
    rows = _read(mpath, [])
    settings = _read(spath, {}) if default else None
    rid = f"osmos-{cfg['pack']}"
    window, reserve = cfg["max_context"], row["reserve"]
    if reserve >= window or window // KEEP_RECENT_SHARE >= window - reserve:
        raise RigError(f"a {window}-token context leaves no room for a {reserve}-token reply (rig would refuse the row)")
    out = {"id": rid, "window": window, "maxTokens": row["max_tokens"], "reserve": reserve,
           "keepRecent": window // KEEP_RECENT_SHARE, "role": "interactive", "efforts": list(row["efforts"]),
           "effort": row["effort"], "command": f"{api} --config {cfg_path} --port ${{PORT}}", "port": _port(rows, rid),
           "aliases": list(row["aliases"]), "stopGrace": STOP_GRACE, "places": 1}
    rows = [r for r in rows if r.get("id") != rid] + [out]
    _write(mpath, rows)
    if default:
        settings["model"] = rid
        _write(spath, settings)
    return out
