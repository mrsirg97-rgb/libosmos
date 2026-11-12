#!/usr/bin/env python3
"""tools/bench/kld_compare.py - runs of tools/bench/kld scored against one reference, side by side (DESIGN.md, "0.6.6:
the reference, and how a quant is scored").

    python3 tools/bench/kld_compare.py <dir> <baseline> <run> [<run> ...] [--rows a:b]

each run is <dir>/<name>.dump (kld's DUMP: per scored row kl F32, nll F32, top I32, ref_top I32), all of one text,
one mask and one reference. for each: the mean KL and its standard error, the median and 99th percentile, the top-1
agreement with the reference, the perplexity; and against the baseline the paired difference of the KL a row, its
standard error, and the two halves' means (a difference the halves disagree on is not one to act on). --rows: the
scored rows a .. b - 1 alone (a split: the dev rows, say).
"""
import math
import os
import sys

import numpy as np

REC = np.dtype([("kl", "<f4"), ("nll", "<f4"), ("top", "<i4"), ("ref_top", "<i4")])


def load(d, name, rows):
    a = np.fromfile(os.path.join(d, name + ".dump"), dtype=REC)
    return a[rows[0]:rows[1]] if rows else a


def main():
    args = sys.argv[1:]
    rows = None
    if "--rows" in args:
        i = args.index("--rows")
        a, b = args[i + 1].split(":")
        rows = (int(a), int(b))
        del args[i:i + 2]
    if len(args) < 3:
        raise SystemExit(__doc__)
    d, base, runs = args[0], args[1], args[2:]
    b = load(d, base, rows)
    print(f"{len(b)} rows{'' if not rows else f' ({rows[0]}:{rows[1]})'}; paired against {base}")
    print(f"{'run':18s} {'KLD':>9s} {'+-':>8s} {'median':>9s} {'p99':>8s} {'top-1':>7s} {'ppl':>8s} "
          f"{'d KLD':>9s} {'+-':>8s} {'halves':>19s}")
    for name in [base] + runs:
        r = load(d, name, rows)
        if len(r) != len(b):
            print(f"{name:18s} {len(r)} rows, not {len(b)}: not the same text, mask or split")
            continue
        kl = r["kl"].astype(np.float64)
        n = len(kl)
        se = kl.std(ddof=1) / math.sqrt(n)
        top = 100.0 * float((r["top"] == r["ref_top"]).mean())
        ppl = math.exp(float(r["nll"].astype(np.float64).mean()))
        line = (f"{name:18s} {kl.mean():9.5f} {se:8.5f} {np.median(kl):9.5f} {np.quantile(kl, 0.99):8.4f} "
                f"{top:6.2f}% {ppl:8.4f}")
        if name != base:
            dk = kl - b["kl"].astype(np.float64)
            h = n // 2
            line += f" {dk.mean():+9.5f} {dk.std(ddof=1) / math.sqrt(n):8.5f} {dk[:h].mean():+9.5f} {dk[h:].mean():+9.5f}"
        print(line)


if __name__ == "__main__":
    main()
