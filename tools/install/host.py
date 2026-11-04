"""the machine checked before anything is fetched: a gfx1151 (rocminfo), the GPU memory the kernel lets the GPU map
(ttm's pages_limit - reported; raising it is a system setting, the user's), the tools the build and the download run,
free disk for what is missing. each check
a line; a failure says why and what would fix it."""
import os
import shutil
import subprocess

GFX = "gfx1151"
PAGE = 4096
# the GPU memory the packs were measured on (ttm's limit raised to ~113 GiB); the engine sizes its expert budget to
# what is free (osmos-serve --budget auto), so less runs - with fewer experts resident, slower, and not measured
GPU_GIB_NEEDED = 113


def gpu():
    exe = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False, "rocminfo did not run: ROCm 7.x is needed (https://rocm.docs.amd.com)"
    if GFX not in out:
        return False, f"no {GFX} (Ryzen AI MAX+ 395) found by rocminfo: osmos runs on Strix Halo"
    return True, f"{GFX} found"


def gpu_memory():
    try:
        with open("/sys/module/ttm/parameters/pages_limit") as f:
            gib = int(f.read()) * PAGE / (1 << 30)
    except (OSError, ValueError):
        return False, "ttm's pages_limit unreadable: is the amdgpu driver loaded?"
    if gib < GPU_GIB_NEEDED:
        pages = GPU_GIB_NEEDED * (1 << 30) // PAGE
        return False, (f"the GPU may map {gib:.0f} GiB; the packs need ~{GPU_GIB_NEEDED}: add `options ttm "
                       f"pages_limit={pages}` to /etc/modprobe.d/ttm.conf and reboot (a system setting: yours)")
    return True, f"the GPU may map {gib:.0f} GiB"


def disk(path, need_bytes):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need_bytes:
        return False, f"{path}: {free / 1e9:.0f} GB free, {need_bytes / 1e9:.0f} GB needed for the files to fetch"
    return True, f"{path}: {free / 1e9:.0f} GB free, {need_bytes / 1e9:.0f} GB to fetch"


# the build's tools (the submodule's git, make, gcc, ROCm's hipcc) and the download's (huggingface_hub's CLI); a tool
# found on PATH or at its usual place
TOOLS = {"git": None, "make": None, "gcc": None, "hipcc": "/opt/rocm/bin/hipcc"}
HF = {"hf": None}
FIX = {"git": "your distribution's git", "make": "your distribution's make", "gcc": "your distribution's gcc",
       "hipcc": "ROCm 7.x (https://rocm.docs.amd.com)", "hf": "pip install -U huggingface_hub"}


def tools(download, build=True, which=shutil.which, exists=os.path.exists):
    need = dict(TOOLS if build else {}, **(HF if download else {}))
    missing = [n for n, at in need.items() if not which(n) and not (at and exists(at))]
    if missing:
        return False, "missing: " + "; ".join(f"{n} ({FIX[n]})" for n in missing)
    return True, "the tools: " + (", ".join(need) or "none needed")
