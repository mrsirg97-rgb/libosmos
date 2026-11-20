"""a manifest file fetched at its pinned commit (the `hf` CLI: huggingface_hub's) into <models>/.hf/<repo>/ - or a
release asset (osmos's own GitHub release) into <models>/.releases/<tag>/ - then checked: its bytes and SHA-256 the
manifest's before anything uses it. a file there already and right is not fetched again; a wrong one is refused, never
used. a file past HF_CLI_MAX (the hub's 50 GB a file without Xet, which the CLI then refuses - and Xet's own
download holds what it fetched in memory: a 105 GB shard, a 128 GB box) is fetched over HTTPS at the same pinned
revision, resumed from its .part where a run stopped (0.11.0: Hy4-flash's Q4_K_M shards, 105 and 63 GB)."""
import os
import shutil
import subprocess
import urllib.request

import userfiles


class FetchError(Exception):
    pass


RELEASES = "https://github.com/mrsirg97-rgb/libosmos/releases/download"  # osmos's own: the NPU's kernels (0.7.0)
HUB = "https://huggingface.co"
HF_CLI_MAX = 50 * 10 ** 9  # (a file's bytes past which the CLI is not asked)
PIECE = 1 << 24


def cache(models, entry):
    if "release" in entry:
        return os.path.join(models, ".releases", entry["release"])
    return os.path.join(models, ".hf", entry["repo"].replace("/", "--"))


def _asset(entry, d, p):
    os.makedirs(d, exist_ok=True)
    url = f"{RELEASES}/{entry['release']}/{entry['path']}"
    print(f"  fetching {url} ({entry['bytes'] / 1e6:.1f} MB)", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(p + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
    except OSError as x:
        raise FetchError(f"{entry['path']}: the download failed ({x})")
    os.replace(p + ".part", p)


def _direct(entry, d, p, opener=None):
    """the file over HTTPS at its revision into p, a run's .part resumed (a Range from its size)"""
    os.makedirs(d, exist_ok=True)
    url = f"{HUB}/{entry['repo']}/resolve/{entry['revision']}/{entry['path']}"
    part = p + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > entry["bytes"]:
        os.remove(part)
        have = 0
    left = f", from {have / 1e9:.1f} GB" if have else ""
    print(f"  fetching {entry['repo']}/{entry['path']} ({entry['bytes'] / 1e9:.1f} GB) at {entry['revision'][:12]}{left}",
          flush=True)
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
            if have and getattr(r, "status", 206) != 206:   # (the server sent the whole file: start again)
                f.truncate(0)
            while True:
                b = r.read(PIECE)
                if not b:
                    break
                f.write(b)
    except OSError as x:
        raise FetchError(f"{entry['path']}: the download failed ({x}) - run again to resume it")
    got = os.path.getsize(part)
    if got != entry["bytes"]:
        raise FetchError(f"{entry['path']}: {got} bytes of {entry['bytes']} - run again to resume it")
    os.replace(part, p)


def fetcher(models, opener=None):
    """fetch(entry) -> the checked file's path (opener: urllib's, or a test's)"""
    exe = shutil.which("hf")

    def fetch(entry):
        d = cache(models, entry)
        p = os.path.join(d, entry["path"])
        if not (os.path.exists(p) and userfiles.matches(p, entry)) and "release" in entry:
            _asset(entry, d, p)
            if not userfiles.matches(p, entry):
                raise FetchError(f"{entry['path']}: its bytes or SHA-256 are not the manifest's - not used")
        elif not (os.path.exists(p) and userfiles.matches(p, entry)) and entry["bytes"] > HF_CLI_MAX:
            _direct(entry, d, p, opener)
            if not userfiles.matches(p, entry):
                os.remove(p)
                raise FetchError(f"{entry['path']}: its bytes or SHA-256 are not the manifest's - not used")
        elif not (os.path.exists(p) and userfiles.matches(p, entry)):
            if not exe:
                raise FetchError("the `hf` CLI is needed to download: pip install -U huggingface_hub")
            print(f"  fetching {entry['repo']}/{entry['path']} ({entry['bytes'] / 1e9:.1f} GB) at "
                  f"{entry['revision'][:12]}", flush=True)
            r = subprocess.run([exe, "download", entry["repo"], entry["path"], "--revision", entry["revision"],
                                "--local-dir", d])
            if r.returncode != 0 or not os.path.exists(p):
                raise FetchError(f"{entry['path']}: the download failed")
            if not userfiles.matches(p, entry):
                raise FetchError(f"{entry['path']}: its bytes or SHA-256 are not the manifest's - not used")
        return p

    return fetch
