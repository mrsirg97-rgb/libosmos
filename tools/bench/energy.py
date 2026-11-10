"""tools/bench/energy.py - a bench's energy: tools/power.py's samples (its csv) matched to the bench log's windows.

    python3 tools/bench/energy.py <power.csv> <bench.log>

dbench's log: a "window prompt|decode <start> <end> <tokens>" line a prompt read and an answer - each window's mean
package power, its joules and its joules a token, then the decode windows' and the prompt windows' totals.
agent_bench's log: its "window <start> <end>" and "answer tokens <n>" - the bench's mean power and joules an answer
token. power is the APU package's (amdgpu PPT), not the wall's.
"""
import re
import sys


def samples(path):
    lines = open(path).read().splitlines()
    start = float(lines[0].split()[2])  # "# start <epoch> ..."
    out = []
    for line in lines[2:]:  # "s,watts" then the samples
        s, w = line.split(",")
        out.append((start + float(s), float(w)))
    return out


def window(smp, a, b):
    """the samples' mean watts in [a, b] and the joules over it. a window shorter than the samples' spacing (a
    43-token prompt's quarter second) between two of them: those two's mean. else none in it fails closed (the csv is
    not this run's)"""
    w = [x for t, x in smp if a <= t <= b]
    if not w:
        before = [(t, x) for t, x in smp if t < a]
        after = [(t, x) for t, x in smp if t > b]
        gaps = sorted(smp[i + 1][0] - smp[i][0] for i in range(len(smp) - 1))
        spacing = gaps[len(gaps) // 2] if gaps else 0
        if before and after and after[0][0] - before[-1][0] <= 2 * spacing:
            w = [before[-1][1], after[0][1]]
    if not w:
        sys.exit(f"no power samples in the window {a:.3f} - {b:.3f}: the csv is not this run's")
    mean = sum(w) / len(w)
    return mean, mean * (b - a)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    smp = samples(sys.argv[1])
    log = open(sys.argv[2]).read()
    rows = re.findall(r"^window (prompt|decode) ([0-9.]+) ([0-9.]+) ([0-9]+)$", log, re.M)
    if rows:
        tot = {}
        for kind, a, b, n in rows:
            mean, j = window(smp, float(a), float(b))
            print(f"{kind:6s} {int(n):6d} tokens {float(b) - float(a):7.1f} s  {mean:6.1f} W  {j / int(n):6.2f} J a token")
            t = tot.setdefault(kind, [0, 0.0, 0.0])
            t[0] += int(n)
            t[1] += j
            t[2] += float(b) - float(a)
        for kind, (n, j, sec) in tot.items():
            print(f"{kind} total: {n} tokens, {sec:.1f} s, {j / sec:.1f} W, {j / n:.2f} J a token")
        return
    m = re.search(r"window ([0-9.]+) ([0-9.]+)", log)
    n = re.search(r"answer tokens ([0-9]+)", log)
    if not (m and n):
        sys.exit("no windows in the log: neither dbench's nor agent_bench's")
    a, b = float(m.group(1)), float(m.group(2))
    mean, j = window(smp, a, b)
    print(f"agents: {b - a:.1f} s, {mean:.1f} W, {j / 1000:.1f} kJ, {j / int(n.group(1)):.2f} J an answer token")


if __name__ == "__main__":
    main()
