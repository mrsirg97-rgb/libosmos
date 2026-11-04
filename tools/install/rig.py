"""a pack registered in rig (the user's agent runtime: it launches a local model as a child and routes to it): a row
merged by id into <rig home>/models.json - the provider half and the child half rig starts (`command`, `${PORT}`
substituted by rig; `port`; `aliases`; `stopGrace`; `places`) - and, asked, the default model in settings.json.
no other row or key touched; a file replaced whole (written aside, then renamed); a file rig would refuse refused
here first, unwritten."""
import json
import os

PORT_BASE = 8091        # the first port tried for an osmos row (rig's default endpoint is 8090)
STOP_GRACE = 30         # seconds rig's SIGTERM waits: a conversation's state written out (osmos-serve's slots)
# the provider half: the context the pack runs, and what a turn may take of it (rig's row: all four required)
MAX_TOKENS, RESERVE = 8192, 8192
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


def _port(rows, rid):
    for r in rows:
        if r.get("id") == rid and isinstance(r.get("port"), int):
            return r["port"]
    used = {r.get("port") for r in rows}
    p = PORT_BASE
    while p in used:
        p += 1
    return p


def register(home, cfg, api, cfg_path, default):
    """the row rig launches the pack by (returned); `api` the osmos-api binary, `cfg_path` the pack's osmos.json"""
    os.makedirs(home, exist_ok=True)
    mpath, spath = os.path.join(home, "models.json"), os.path.join(home, "settings.json")
    rows = _read(mpath, [])
    settings = _read(spath, {}) if default else None
    rid = f"osmos-{cfg['pack']}"
    window = cfg["max_context"]
    row = {"id": rid, "window": window, "maxTokens": MAX_TOKENS, "reserve": RESERVE,
           "keepRecent": window // KEEP_RECENT_SHARE, "role": "interactive",
           "command": f"{api} --config {cfg_path} --port ${{PORT}}", "port": _port(rows, rid),
           "aliases": ["osmos"], "stopGrace": STOP_GRACE, "places": 1}
    rows = [r for r in rows if r.get("id") != rid] + [row]
    _write(mpath, rows)
    if default:
        settings["model"] = rid
        _write(spath, settings)
    return row
