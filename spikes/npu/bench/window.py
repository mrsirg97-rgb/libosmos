#!/usr/bin/env python3
# a run's prompt window (dbench's "window prompt start end") over sample.py's samples: mean W, mean / min MHz
import sys
log, smp = sys.argv[1], sys.argv[2]
w = next(l.split() for l in open(log) if l.startswith("window prompt") and l.split()[-1] != "64")
t0, t1 = float(w[2]), float(w[3])
s = [tuple(map(float, l.split())) for l in open(smp) if l.strip()]
s = [x for x in s if t0 <= x[0] <= t1]
if not s: sys.exit("no samples in the window")
print(f"prompt window {t1 - t0:.1f} s: {sum(x[1] for x in s) / len(s):.1f} W, GPU {sum(x[2] for x in s) / len(s):.0f} MHz "
      f"(min {min(x[2] for x in s):.0f}, {len(s)} samples)")
