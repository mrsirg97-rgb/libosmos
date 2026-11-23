#!/usr/bin/env python3
"""tools/serve/sample_test.py - sampling (0.14.0: GEN temp= top_p= seed=) on a real model: a token at position t is
drawn from its row with the uniform of (seed, t) alone, a draft accepted only where it is the token the row draws -
so a drafted answer is the plain one, token for token, at any seed (osmos-serve's one guarantee, kept sampled).

  drafted    a sampled answer plain and drafted the same (temperature 1, top_p 1 and 0.9; top_p 0.95 with top_k 20),
             the drafts mostly taken
  parallel   --parallel 2, one plain and one drafted at once: the one conversation's tokens
  seeds      the same seed the same tokens again; another seed other tokens; temp=0 the greedy answer exactly
  refused    a temperature or top_p out of range: ERR

    python tools/serve/sample_test.py --model <first shard> --ids <ids file> --dspark <drafter> [--exe bin/osmos-serve]
        [-- <more osmos-serve args>]

exit 0: every check passed; 1: a check failed; 2: the engine did not start.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Engine, Multi, read_ids  # noqa: E402

PROMPT = 400
ANSWER = 160
failed = []


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def keys(temp, top_p, seed, top_k=0):
    return f" temp={temp} top_p={top_p} seed={seed}" + (f" top_k={top_k}" if top_k else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dspark", required=True)
    ap.add_argument("--exe", default="bin/osmos-serve")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    base = ["--model", a.model, "--dspark", a.dspark, "--max-context", "8192"] + rest
    prompt = read_ids(a.ids)[:PROMPT]
    env = dict(os.environ, OSMOS_SLOT_KEEP="0")

    e = Engine(a.exe, base, env=env)
    greedy = e.gen(prompt, ANSWER, spec=0)["tokens"]
    for top_p, top_k in ((1, 0), (0.9, 0), (0.95, 20)):  # (0.15.2: Qwen's release sampling - top-k then top-p)
        k = keys(1.0, top_p, 1017, top_k)
        plain = e.gen(prompt, ANSWER, spec=0, extra=k)
        drafted = e.gen(prompt, ANSWER, spec=1, extra=k)
        for r in (plain, drafted):
            if "err" in r:
                print(f"  {r['err']}")
        acc, dr = (int(drafted["done"][6]), int(drafted["done"][7])) if "done" in drafted else (0, 0)
        what = f"temperature 1, top_p {top_p}" + (f", top_k {top_k}" if top_k else "")
        check(f"{what}: drafted == plain ({len(plain['tokens'])} tokens; drafts {acc} of {dr} taken)",
              plain["tokens"] == drafted["tokens"] and len(plain["tokens"]) > 0)
        check(f"{what}: the drafts mostly right under sampling (> 25%)", dr and acc / dr > 0.25)
        check(f"{what}: not the greedy answer", plain["tokens"] != greedy)
    k = keys(1.0, 1, 1017)
    again = e.gen(prompt, ANSWER, spec=1, extra=k)["tokens"]
    first = e.gen(prompt, ANSWER, spec=0, extra=k)["tokens"]
    other = e.gen(prompt, ANSWER, spec=0, extra=keys(1.0, 1, 1018))["tokens"]
    check("the same seed: the same tokens again", again == first)
    check("another seed: other tokens", other != first)
    check("temp=0: the greedy answer exactly", e.gen(prompt, ANSWER, spec=1, extra=keys(0, 1, 5))["tokens"] == greedy)
    check("temp=1 top_k=1: the greedy answer (the one most likely token kept: top_k reaches the draw)",
          e.gen(prompt, ANSWER, spec=1, extra=keys(1.0, 1, 5, 1))["tokens"] == greedy)
    for bad in (keys(-1, 1, 1), keys(1, 0, 1), keys(1, 1.5, 1)):
        check(f"refused:{bad}", "err" in e.gen(prompt, 4, spec=0, extra=bad))
    e.quit()

    m = Multi(a.exe, base + ["--parallel", "2"], env=env)
    m.send(1, prompt, ANSWER, spec=0, extra=k)
    m.send(2, prompt, ANSWER, spec=1, extra=k)
    p1, p2 = m.wait(1)["tokens"], m.wait(2)["tokens"]
    m.quit()
    check("--parallel 2, plain and drafted at once: the one conversation's tokens", p1 == first and p2 == first)

    print(f"{'FAILED: ' + '; '.join(failed) if failed else 'every check passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
