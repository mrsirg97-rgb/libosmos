"""the machine checked before anything is fetched: a gfx1151 (its PCI ids in sysfs), the GPU memory the kernel lets the GPU map
(ttm's pages_limit - reported; raising it is a system setting, the user's), the tools the build and the download run,
free disk for what is missing. each check
a line; a failure says why and what would fix it."""
import glob
import os
import shutil

GFX = "gfx1151"
PAGE = 4096
# the GPU memory the packs were measured on (ttm's limit raised to ~113 GiB); the engine sizes its expert budget to
# what is free (osmos-serve --budget auto), so less runs - with fewer experts resident, slower, and not measured
GPU_GIB_NEEDED = 113


AMD = 0x1002
STRIX_HALO = {0x1586}  # the Radeon 8060S / 8050S's PCI device: a gfx1151


def gpu(sys_root="/sys"):
    """the GPU by its PCI ids in sysfs - no ROCm needed to ask (issue #25: rocminfo's probe hung where none was)"""
    base = os.path.join(sys_root, "bus", "pci", "devices")
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return False, f"{base} unreadable: sysfs is how the GPU is found (is this Linux?)"
    amd = []
    for name in names:
        try:
            with open(os.path.join(base, name, "vendor")) as f:
                vendor = int(f.read(), 16)
            with open(os.path.join(base, name, "device")) as f:
                device = int(f.read(), 16)
        except (OSError, ValueError):
            continue
        if vendor == AMD:
            if device in STRIX_HALO:
                return True, f"{GFX} found (PCI {vendor:04x}:{device:04x} at {name})"
            amd.append(f"{vendor:04x}:{device:04x}")
    if not amd:
        return False, f"no AMD GPU in {base}: osmos runs on Strix Halo ({GFX}, Ryzen AI MAX+)"
    return False, f"no {GFX} (Ryzen AI MAX+'s Radeon, PCI 1002:1586): AMD devices {', '.join(amd)} - osmos runs on Strix Halo"


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


def npu(devices=None, libs=None, header="/usr/include/xrt/xrt_kernel.h"):
    """(ok, why): an AMD NPU osmos can use (0.7.0's --npu) - its device (the amdxdna driver's /dev/accel/accel*),
    XRT's runtime library and its headers (libxrt-dev: osmos builds against them)"""
    devices = glob.glob("/dev/accel/accel*") if devices is None else devices
    libs = (glob.glob("/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so*") + glob.glob("/opt/xilinx/xrt/lib/libxrt_coreutil.so*")
            if libs is None else libs)
    if not devices:
        return False, "no NPU (/dev/accel/accel*: the amdxdna driver's) - the GPU alone runs the packs"
    if not libs:
        return False, "no XRT runtime (libxrt_coreutil): install xrt / libxrt2"
    if not os.path.exists(header):
        return False, f"no XRT headers ({header}): install libxrt-dev"
    return True, f"an NPU ({devices[0]}) and XRT"

