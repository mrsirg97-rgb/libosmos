#!/usr/bin/env python3
"""tools/serve/fill_test.py - --fill on a real model: the RAM osmos-serve keeps free, through the engine to the sizing
(osmos/run/fit.h's fit_room) and the saved conversations' room.

  --fill 4     ready; its sizing NOTE and its slots NOTE each keep 4.00 GiB free, and the budget 3 GiB or more past
               the default's (8 GiB kept free: one more start, the default's NOTE 8.00)
  --fill 1000  more than any machine can keep free: refused, loud - not started, why and the levers said (0.16.0:
               the ledger never takes the RAM --fill keeps; before, it started on the smallest build anyway)

    python tools/serve/fill_test.py --model <first shard> [--picks <file>] [--exe bin/osmos-serve]

exit 0: every check passed; 1: a check failed; 2: the engine did not start.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Engine  # noqa: E402

failed = []


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def start(a, fill):
    """a start with --fill (None: the default): its budget and its NOTE lines"""
    args = ["--model", a.model, "--max-context", "65536", "--no-guess"] + (["--picks", a.picks] if a.picks else [])
    if fill is not None:
        args += ["--fill", str(fill)]
    e = Engine(a.exe, args)
    notes = list(e.notes)
    e.p.stdin.write("QUIT\n")
    e.p.stdin.flush()
    e.p.wait(timeout=120)
    sized = [n for n in notes if "the budget sized to this GPU" in n]
    budget = float(re.search(r"GPU: ([\d.]+) GiB", sized[-1]).group(1)) if sized else None
    return budget, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--picks")
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(__file__), "..", "..", "bin", "osmos-serve"))
    a = ap.parse_args()
    print("[fill_test] start", flush=True)

    b8, n8 = start(a, None)
    b4, n4 = start(a, 4)
    print(f"  budgets: {b8} GiB (8 kept free), {b4} GiB (--fill 4)", flush=True)
    check("the default keeps 8.00 GiB free", any("8.00 GiB of RAM kept free" in n for n in n8))
    check("--fill 4 keeps 4.00 GiB free in its sizing", any("4.00 GiB of RAM kept free" in n for n in n4))
    check("--fill 4's slots keep 4 free", any(n.startswith("NOTE slots:") and "4 kept free" in n for n in n4))
    # 4 GiB more room: the budget up by it, but for the sizing's steps (~0.93 GiB on this box: fit.h)
    check("--fill 4's budget 3 GiB or more past the default's", b4 is not None and b8 is not None and b4 >= b8 + 3)

    args = ["--model", a.model, "--max-context", "65536", "--no-guess", "--fill", "1000", "--serve"] + \
        (["--picks", a.picks] if a.picks else [])
    r = subprocess.run([a.exe, *args], input="QUIT\n", capture_output=True, text=True, timeout=1800)
    said = r.stdout + r.stderr
    print(f"  --fill 1000: exit {r.returncode}", flush=True)
    check("--fill 1000: not started", r.returncode != 0 and "READY" not in r.stdout)
    check("--fill 1000: the build ledger's refusal, the RAM kept named", "refused by the ledger" in said and "1000.0 GiB kept" in said)
    check("--fill 1000: the levers said", "a smaller --fill" in said and "--max-context)" in said)

    print(f"[fill_test] {'ok' if not failed else 'FAILED: ' + '; '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
