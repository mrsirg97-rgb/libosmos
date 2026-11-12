"""osmos-install --doctor: the machine as osmos will run on it - host.py's checks, then the ones that cost speed or
memory without failing anything (each found the hard way: DESIGN.md, "osmos doctor"). each check a line, a level and
what would fix it; nothing written but a probe file, removed. every path a parameter: the tests' fake machines.

    ok    as osmos was measured
    note  works, but not as measured (slower, or a download slower) - the line says what to change
    FAIL  osmos will refuse or fail - the line says why"""
import os
import tempfile

import host

OK, NOTE, FAIL = "ok", "note", "FAIL"
TMPFS_GIB = 4  # /tmp's bytes past this, when it is RAM: the GPU's memory less by them (osmos's "GPU memory refused")
RAM_GIB = 120  # the packs were measured on 128 GB (MemTotal ~125 GiB)


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def power(sys_root="/sys"):
    """the platform profile: the packs measured at low-power and performance; balanced is where a boot with the
    monitor off leaves the Halo (slower than either)"""
    p = os.path.join(sys_root, "firmware", "acpi")
    now, choices = _read(os.path.join(p, "platform_profile")), _read(os.path.join(p, "platform_profile_choices"))
    if now is None:
        return NOTE, "no ACPI platform profile to read: its speed not the one measured, perhaps"
    if now == "balanced":
        return NOTE, (f"the power profile is balanced (of {choices or '?'}): slower than the profiles osmos is "
                      f"measured at - `echo performance | sudo tee {p}/platform_profile` (or low-power: the "
                      f"tokens a joule)")
    return OK, f"the power profile: {now}"


def mount_of(path, mounts="/proc/mounts"):
    """the mount a path lies on (the longest mount point that is its prefix): (point, fs type, device)"""
    path, best = os.path.realpath(path), ("/", "?", "?")
    for line in (_read(mounts) or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        dev, point, fs = parts[0], parts[1].replace("\\040", " "), parts[2]
        if (path == point or path.startswith(point.rstrip("/") + "/")) and len(point) >= len(best[0]):
            best = (point, fs, dev)
    return best


def tmp(path="/tmp", mounts="/proc/mounts", used=None):
    """/tmp as RAM (tmpfs): what lies in it the GPU cannot have - big scratch there and a model's build is refused"""
    point, fs, _ = mount_of(path, mounts)
    if fs != "tmpfs":
        return OK, f"{path}: {fs}, not RAM"
    if used is None:
        st = os.statvfs(path)
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
    if used > TMPFS_GIB << 30:
        return NOTE, (f"{path} is RAM (tmpfs) and holds {used / (1 << 30):.1f} GiB: memory the GPU cannot have - "
                      f"a model's build may be refused (\"GPU memory refused\"); move large files out of {path}")
    return OK, f"{path} is RAM (tmpfs), {used / (1 << 30):.1f} GiB in it"


def memory(meminfo="/proc/meminfo"):
    vals = {}
    for line in (_read(meminfo) or "").splitlines():
        k, _, v = line.partition(":")
        if v.strip().endswith("kB"):
            vals[k] = int(v.split()[0]) << 10
    total, avail = vals.get("MemTotal"), vals.get("MemAvailable")
    if not total:
        return NOTE, f"{meminfo} unreadable"
    gib = lambda b: b / (1 << 30)  # noqa: E731
    if gib(total) < RAM_GIB:
        return NOTE, (f"{gib(total):.0f} GiB of RAM: the packs were measured on 128 GB - fewer experts resident, "
                      f"slower (osmos sizes to what is free)")
    return OK, f"{gib(total):.0f} GiB of RAM, {gib(avail or 0):.0f} available"


def direct(models):
    """osmos reads the experts O_DIRECT (straight into memory the GPU reads): a filesystem that refuses it (some network
    and FUSE ones; tmpfs before Linux 6.6) refuses the model - asked of the folder itself, by a probe file"""
    if not os.path.isdir(models):
        return NOTE, f"{models}: not made yet (no pack installed)"
    fd, probe = tempfile.mkstemp(prefix=".osmos-probe-", dir=models)
    os.close(fd)
    try:
        d = os.open(probe, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        os.close(d)
    except OSError as e:
        return FAIL, (f"{models} refuses direct reads ({e.strerror}): osmos reads its experts O_DIRECT - put the "
                      f"packs on ext4, XFS or btrfs")
    finally:
        os.remove(probe)
    point, fs, _ = mount_of(models)
    clones = " (clones: a table shared between packs costs its bytes once)" if fs in ("btrfs", "xfs") else ""
    return OK, f"{models}: {fs}, direct reads{clones}"


def device_of(dev, sys_root="/sys"):
    """a /dev/<partition>'s whole disk in sysfs (nvme0n1p2 -> nvme0n1, sda1 -> sda), or None"""
    name = os.path.basename(dev)
    block = os.path.join(sys_root, "class", "block", name)
    if not os.path.exists(block):
        return None
    parent = os.path.realpath(os.path.join(block, ".."))
    whole = os.path.basename(parent)
    return whole if os.path.exists(os.path.join(sys_root, "block", whole)) else name


def ssd(models, mounts="/proc/mounts", sys_root="/sys"):
    """the experts stream from the disk a token: a spinning one is far too slow"""
    _, fs, dev = mount_of(models if os.path.exists(models) else os.path.dirname(models) or "/", mounts)
    whole = device_of(dev, sys_root) if dev.startswith("/dev/") else None
    if not whole:
        return NOTE, f"the models' disk ({dev}, {fs}): not one sysfs names - its speed unchecked"
    rot = _read(os.path.join(sys_root, "block", whole, "queue", "rotational"))
    if rot == "1":
        return FAIL, (f"the models' disk ({whole}) spins: the experts stream from it every token - put the packs on "
                      f"an NVMe SSD")
    return OK, f"the models' disk: {whole}, solid state"


def network(route="/proc/net/route"):
    """a pack is 120-200 GB: over Wi-Fi it is hours (this box's: ~30 MB/s)"""
    for line in (_read(route) or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) > 2 and parts[1] == "00000000":  # the default route
            iface = parts[0]
            if iface.startswith(("wl", "ww")):
                return NOTE, f"downloads go over {iface} (wireless): a pack is hours there - a cable if you have one"
            return OK, f"downloads go over {iface}"
    return NOTE, "no default route: downloads will fail (offline: --have your files)"


def rocm_version(header="/opt/rocm/include/rocm-core/rocm_version.h"):
    """ROCm's version from its own header (ROCM_VERSION_MAJOR, _MINOR, _PATCH), or None"""
    parts = {}
    for line in (_read(header) or "").splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "#define" and f[1] in ("ROCM_VERSION_MAJOR", "ROCM_VERSION_MINOR", "ROCM_VERSION_PATCH"):
            parts[f[1]] = f[2]
    if len(parts) != 3:
        return None
    return ".".join(parts[k] for k in ("ROCM_VERSION_MAJOR", "ROCM_VERSION_MINOR", "ROCM_VERSION_PATCH"))


def driver(sys_root="/sys", header="/opt/rocm/include/rocm-core/rocm_version.h"):
    if not os.path.isdir(os.path.join(sys_root, "module", "amdgpu")):
        return FAIL, "the amdgpu driver is not loaded"
    v = rocm_version(header)
    return OK, f"amdgpu loaded, kernel {os.uname().release}, ROCm {v or '(no rocm_version.h under /opt/rocm)'}"


def run(models):
    """every check in order: host.py's (ok or FAIL), then these"""
    checks = [(OK if ok else FAIL, why) for ok, why in (host.gpu(), host.gpu_memory())]
    checks += [driver(), power(), memory(), tmp(), direct(models), ssd(models), network()]
    tl = host.tools(True, True)
    checks.append((OK if tl[0] else NOTE, tl[1]))
    return checks
