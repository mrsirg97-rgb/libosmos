#!/usr/bin/env python3
"""tools/serve/answer_test.py - the answer rule (--answer-after; osmos/text/answer.h) on a real model: a turn that
closed its reasoning does not end before its answer has begun (Qwen3.8 at small talk, 2026-11-08: </think>, then
the end of turn, every one of rig's retries the same).

the failure made on purpose - a close token C, and the token the model says after it (E) made an end (--eos E):

  first  C the prompt's last (</think>): the rule at the pick after the prompt
  mid    C a token of the free answer, a few in: the rule at a decode step's pick - a plain step's, a drafted
         round's verify, a --parallel round's

each: bare (--eos E alone) ends the turn at E; with --answer-after C it goes on, E never there, and every path -
one conversation plain and drafted, --parallel 2 plain and drafted at once - the same tokens.

    python tools/serve/answer_test.py --model <first shard> --ids <ids file> [--dspark <drafter>] [--exe bin/osmos-serve]

exit 0: every check passed; 1: a check failed; 2: the engine did not start.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Engine, Multi, read_ids  # noqa: E402

PROMPT = 200   # real text's first tokens
ANSWER = 16    # the free answer: room for C a few tokens in, then the rule's tokens past it
FIRST_C = 3    # C's earliest place in the free answer: past the prompt's pick, inside decode steps
failed = []


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def paths(a, base, prompt):
    """the answer of every path under the args: [(name, tokens, finish)]"""
    out = []
    e = Engine(a.exe, base)
    out.append(("plain", *pick(e.gen(prompt, ANSWER, spec=0))))
    if a.dspark:
        out.append(("drafted", *pick(e.gen(prompt, ANSWER, spec=1))))
    e.quit()
    m = Multi(a.exe, base + ["--parallel", "2"])
    m.send(1, prompt, ANSWER, spec=0)
    if a.dspark:
        m.send(2, prompt, ANSWER, spec=1)
    out.append(("parallel plain", *pick(m.wait(1))))
    if a.dspark:
        out.append(("parallel drafted", *pick(m.wait(2))))
    m.quit()
    return out


def pick(r):
    return r["tokens"], r.get("finish", r.get("err"))


def scenario(name, a, base, prompt, c, e, before):
    """C's place: `before` the free answer's tokens up to and with C; E the token after"""
    print(f"[answer_test] {name}: C {c}, E {e}, {len(before)} tokens before E")
    for p, toks, fin in paths(a, base + ["--eos", str(e)], prompt):
        check(f"{name} bare {p}: ends at E ({len(toks)} tokens, {fin})", toks == before + [e] and fin == "stop")
    got = paths(a, base + ["--eos", str(e), "--answer-after", f"#{c}"], prompt)
    ref = got[0][1]
    # the answer begins past C - E not its next token; past that a word, the turn may end at E as any turn may
    check(f"{name} rule: goes on past C ({len(ref)} tokens), E not after it",
          len(ref) > len(before) and ref[:len(before)] == before and ref[len(before)] != e)
    for p, toks, fin in got[1:]:
        check(f"{name} rule {p}: the plain answer's tokens", toks == ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dspark")
    ap.add_argument("--exe", default="bin/osmos-serve")
    a = ap.parse_args()
    base = ["--model", a.model, "--max-context", "4096"] + (["--dspark", a.dspark] if a.dspark else [])

    probe = Engine(a.exe, base + ["--answer-after", "</think>"])
    end = next((int(n.split()[-1]) for n in probe.notes if n.startswith("NOTE answer-after")), None)
    if end is None:
        print("osmos-serve printed no 'NOTE answer-after <id>'")
        return 2
    text = read_ids(a.ids)[:PROMPT]
    closed = text + [end]
    first = probe.gen(closed, 1)["tokens"][0]
    free = probe.gen(text, ANSWER)["tokens"]
    probe.quit()

    scenario("first", a, base, closed, end, first, [])
    # mid: C at j (>= FIRST_C), E = free[j + 1] not said before it - bare's answer then ends exactly there; C not
    # the prompt's last token, so the rule has nothing to do before it
    j = next((j for j in range(FIRST_C, len(free) - 2) if free[j + 1] not in free[:j + 1] and free[j] != text[-1]),
             None)
    if j is None:
        print("no place for C in the free answer")
        return 2
    scenario("mid", a, base, text, free[j], free[j + 1], free[:j + 1])
    print("[answer_test] " + ("ok" if not failed else f"{len(failed)} failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
