#!/usr/bin/env python3
"""The Halo's package power while a command runs.

    tools/power.py [--every 0.5] [--csv FILE] -- <command> [args...]

Samples amdgpu's PPT (package power tracking: the APU's CPU, GPU and NPU together - not the wall: DRAM, the SSD, the
fans and the supply's loss are outside it) every `--every` seconds while the command runs, then prints the command's
wall time, the mean and peak watts, and the energy (the samples' mean times the wall). The command's exit code is
returned. A missing sensor is an error, not a zero.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import threading
import time
from pathlib import Path


def ppt_sensor() -> Path:
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        p = Path(h)
        if (p / "name").read_text().strip() == "amdgpu" and (p / "power1_label").exists() and \
                (p / "power1_label").read_text().strip() == "PPT":
            return p / "power1_average"
    raise SystemExit("power.py: no amdgpu PPT sensor (hwmon power1_label PPT)")


def main() -> int:
    ap = argparse.ArgumentParser(description="package power (amdgpu PPT) while a command runs")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--csv", default="", help="write the samples (seconds since start, watts)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- the command")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("a command after --")
    sensor = ppt_sensor()
    samples: list[tuple[float, float]] = []
    done = threading.Event()
    t0 = time.monotonic()

    def sample() -> None:
        while not done.is_set():
            samples.append((time.monotonic() - t0, int(sensor.read_text()) / 1e6))
            done.wait(a.every)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    rc = subprocess.call(cmd)
    wall = time.monotonic() - t0
    done.set()
    th.join()
    if not samples:
        raise SystemExit("power.py: no samples")
    watts = [w for _, w in samples]
    mean = sum(watts) / len(watts)
    if a.csv:
        Path(a.csv).write_text("s,watts\n" + "".join(f"{t:.2f},{w:.2f}\n" for t, w in samples))
    print(f"power: {wall:.1f} s, package mean {mean:.1f} W, peak {max(watts):.1f} W, "
          f"energy {mean * wall / 3600:.2f} Wh ({len(samples)} samples, amdgpu PPT; not wall power)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
