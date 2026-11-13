"""a manifest file fetched at its pinned commit (the `hf` CLI: huggingface_hub's) into <models>/.hf/<repo>/ - or a
release asset (osmos's own GitHub release) into <models>/.releases/<tag>/ - then checked: its bytes and SHA-256 the
manifest's before anything uses it. a file there already and right is not fetched again; a wrong one is refused, never
used."""
import os
import shutil
import subprocess
import urllib.request

import userfiles


class FetchError(Exception):
    pass


RELEASES = "https://github.com/mrsirg97-rgb/libosmos/releases/download"  # osmos's own: the NPU's kernels (0.7.0)


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


def fetcher(models):
    """fetch(entry) -> the checked file's path"""
    exe = shutil.which("hf")

    def fetch(entry):
        d = cache(models, entry)
        p = os.path.join(d, entry["path"])
        if not (os.path.exists(p) and userfiles.matches(p, entry)) and "release" in entry:
            _asset(entry, d, p)
            if not userfiles.matches(p, entry):
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
