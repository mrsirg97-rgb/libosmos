"""a manifest file fetched at its pinned commit (the `hf` CLI: huggingface_hub's) into <models>/.hf/<repo>/, then
checked - its bytes and SHA-256 the manifest's - before anything uses it. a file there already and right is not
fetched again; a wrong one is refused, never used."""
import os
import shutil
import subprocess

import userfiles


class FetchError(Exception):
    pass


def cache(models, entry):
    return os.path.join(models, ".hf", entry["repo"].replace("/", "--"))


def fetcher(models):
    """fetch(entry) -> the checked file's path"""
    exe = shutil.which("hf")

    def fetch(entry):
        d = cache(models, entry)
        p = os.path.join(d, entry["path"])
        if not (os.path.exists(p) and userfiles.matches(p, entry)):
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
