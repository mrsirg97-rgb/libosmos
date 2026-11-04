#!/usr/bin/env python3
"""tools/serve/parallel_test.py - several conversations at once (`--parallel N`; maya #587's parallel_test.py, ported):
each answers the bits it answers alone, a STOP ends one alone, a conversation stays in its lane, a short question is
answered while a long prompt is read.

    python tools/serve/parallel_test.py --model <first shard> [--picks <file>] [--exe bin/osmos-serve] [--lanes 4]

  alone     the reference: four prompts (real text, different from their second token), one at a time, --parallel 1
  together  --parallel <lanes>: the four sent at once - each answer's token hash the one it got alone
  drafted   (--dspark) the four at once all drafting, then two drafting and two greedy: each answer the greedy one
            alone (a verify's accepted tokens are greedy's), the drafts cut to keep a round a decode step
  stop      --parallel 2: two at once, the first STOPped after a few tokens: its DONE `cancel`, fewer tokens; the
            other's answer whole and its own
  slots     --parallel 2, OSMOS_SLOT_KEEP=1: A and B at once, then A's next turn - it reuses A from its lane
  reading   --parallel 2: a long prompt, then a short one: the short one read in one piece ahead of the long one's
            next and answering between its pieces - tokens of it before the long read's last piece
exit 0: every check passed; 1: a check failed; 2: the engine did not start.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Multi, hash_tokens, read_ids  # noqa: E402

failed = []
NEW = 48


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--picks")
    ap.add_argument("--dspark")
    ap.add_argument("--exe", default="bin/osmos-serve")
    ap.add_argument("--prompts", default=os.path.expanduser("~/bench/osmos-prompts"))
    ap.add_argument("--budget", default="100")
    ap.add_argument("--lanes", type=int, default=4)
    a = ap.parse_args()
    calib, gpl = read_ids(f"{a.prompts}/calib16k.txt"), read_ids(f"{a.prompts}/p8192.txt")
    prompts = [calib[0:1500], gpl[0:1200], calib[4000:5800], calib[9000:9900]]
    base = ["--model", a.model, "--budget", a.budget, "--max-tokens", "2048", "--max-context", "65536"]
    if a.picks:
        base += ["--picks", a.picks]
    if a.dspark:
        base += ["--dspark", a.dspark]
    tmp = tempfile.mkdtemp(prefix="osmos-par-")
    env = dict(os.environ, OSMOS_SLOT_KEEP="0", OSMOS_SLOT_DIR=f"{tmp}/none")
    try:
        print("alone: four prompts one at a time, --parallel 1", flush=True)
        e = Multi(a.exe, base + ["--parallel", "1"], env)
        alone = []
        for i, p in enumerate(prompts):
            e.send(i, p, NEW)
            alone.append(e.wait(i))
        e.quit()
        ref = [hash_tokens(r["tokens"]) for r in alone]
        print("        " + ", ".join(f"{len(p)} tok {h:016x}" for p, h in zip(prompts, ref)), flush=True)

        print(f"together: the four at once, --parallel {a.lanes}", flush=True)
        e = Multi(a.exe, base + ["--parallel", str(a.lanes)], env)
        t0 = time.monotonic()
        for i, p in enumerate(prompts):
            e.send(i, p, NEW)
        got = [e.wait(i) for i in range(len(prompts))]
        wall = time.monotonic() - t0
        e.quit()
        same = [hash_tokens(r["tokens"]) == h for r, h in zip(got, ref)]
        check(f"together: every answer the one it got alone ({sum(same)} of {len(same)}; all in {wall:.1f} s against "
              f"{sum(r['s'] for r in alone):.1f} one at a time)", all(same))

        if a.dspark:
            for name, specs in (("all drafting", [1, 1, 1, 1]), ("two drafting, two greedy", [1, 0, 1, 0])):
                print(f"drafted, {name}: the four at once, --parallel {a.lanes}", flush=True)
                e = Multi(a.exe, base + ["--parallel", str(a.lanes)], env)
                t0 = time.monotonic()
                for i, p in enumerate(prompts):
                    e.send(i, p, NEW, spec=specs[i])
                got = [e.wait(i) for i in range(len(prompts))]
                wall = time.monotonic() - t0
                e.quit()
                same = [hash_tokens(r["tokens"]) == h for r, h in zip(got, ref)]
                acc = sum(int(r["done"][7]) for r in got if "done" in r)
                dr = sum(int(r["done"][8]) for r in got if "done" in r)
                check(f"drafted, {name}: every answer the greedy one alone ({sum(same)} of {len(same)}; {acc} of {dr} "
                      f"drafts accepted; all in {wall:.1f} s)", all(same))

        print("stop: two at once, the first stopped after a few tokens, --parallel 2", flush=True)
        e = Multi(a.exe, base + ["--parallel", "2"], env)
        e.send(0, prompts[0], NEW)
        e.send(1, prompts[1], NEW)
        e.wait(0, until=lambda r: len(r["tokens"]) >= 5 or "done" in r)
        e.stop(0)
        r0, r1 = e.wait(0), e.wait(1)
        e.quit()
        check(f"stop: the stopped one `{r0.get('finish')}` after {len(r0['tokens'])} tokens",
              r0.get("finish") == "cancel" and len(r0["tokens"]) < NEW)
        check("stop: the other's answer whole, its own", hash_tokens(r1["tokens"]) == ref[1])

        print("slots: A and B at once, then A's next turn, --parallel 2, OSMOS_SLOT_KEEP=1", flush=True)
        e = Multi(a.exe, base + ["--parallel", "2"], dict(env, OSMOS_SLOT_KEEP="1", OSMOS_SLOT_DIR=f"{tmp}/s"))
        e.send(0, prompts[0], NEW)
        e.send(1, prompts[1], NEW)
        ra, rb = e.wait(0), e.wait(1)
        turn = prompts[0] + ra["tokens"] + gpl[3000:3040]
        e.send(2, turn, NEW)
        rt = e.wait(2)
        e.quit()
        held = len(prompts[0]) + len(ra["tokens"]) - 1
        check(f"slots: A's next turn reuses A from its lane ({rt.get('reused')} of {held})", rt.get("reused") == held)

        print("reading: a long prompt, then a short one, --parallel 2", flush=True)
        long_p = calib + gpl  # ~24K tokens of text
        e = Multi(a.exe, base + ["--parallel", "2"], env)
        e.send(0, long_p, 8)
        e.wait(0, until=lambda r: len(r["pp"]) >= 1)
        e.send(1, prompts[3], 16)
        rs = e.wait(1)
        rl = e.wait(0)
        e.quit()
        with e.cv:
            read_end = max(t for t, kind, rid in e.log if kind == "PP" and rid == 0)
            early = sum(1 for t, kind, rid in e.log if kind == "T" and rid == 1 and t < read_end)
        check(f"reading: the short one answered while the long one ({len(long_p)} tok, {len(rl['pp'])} pieces) was "
              f"read - {early} of its 16 tokens before the read's last piece", early >= 1)
        check("reading: the short one's answer its own", hash_tokens(rs["tokens"]) == hash_tokens(alone[3]["tokens"][:16]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'every check passed' if not failed else f'{len(failed)} failed'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
