"""the machine checked before anything is fetched: a gfx1151 (rocminfo), the GPU memory the kernel lets the GPU map
(ttm's pages_limit - reported; raising it is a system setting, the user's), free disk for what is missing. each check
a line; a failure says why and what would fix it."""
import os
import shutil
import subprocess

GFX = "gfx1151"
PAGE = 4096
# the budget the packs are measured at (110 GiB, plain) and what a step's buffers need beside it - below this, the
# packs' settings would be refused at start
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
