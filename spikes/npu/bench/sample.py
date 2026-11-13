#!/usr/bin/env python3
# package power (amdgpu PPT, W) and the GPU's current sclk (MHz) every 0.25 s until the stop file appears: "t W MHz"
import glob, sys, time, os
from pathlib import Path
ppt = next(Path(h) / "power1_average" for h in sorted(glob.glob("/sys/class/hwmon/hwmon*"))
           if (Path(h) / "name").read_text().strip() == "amdgpu" and (Path(h) / "power1_label").exists()
           and (Path(h) / "power1_label").read_text().strip() == "PPT")
sclk = Path("/sys/class/drm/card0/device/pp_dpm_sclk")
out, stop = open(sys.argv[1], "w"), sys.argv[2]
while not os.path.exists(stop):
    w = int(ppt.read_text()) / 1e6
    cur = next((l.split()[1] for l in sclk.read_text().splitlines() if l.endswith("*")), "0Mhz")
    out.write(f"{time.time():.3f} {w:.1f} {cur.lower().replace('mhz', '')}\n"); out.flush()
    time.sleep(0.25)
