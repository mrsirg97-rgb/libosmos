#!/usr/bin/env python3
"""tools/serve/turn_test.py - a turn's next turn reuses its prompt wherever the client's history leaves it (0.7.5):

- DeepSeek's generation prompt ends <｜Assistant｜><think>, and the next turn renders that place </think> (earlier
  reasoning dropped without tools): the next prompt repeats this one's tokens but the last;
- a client that writes the answer back otherwise (no reasoning_content, arguments re-serialized: most harnesses)
  leaves the decoded answer at its first token - Qwen, which continues on decoded tokens, read the whole prompt again;
- a new user message after a tool loop, the template dropping that loop's reasoning, leaves the prompt at the last
  user turn (osmos-api's GEN mark=).

the server keeps its points (the prompt's one token short of its end, mark's) and goes back to the furthest the next
prompt continues - each answer the bits of the same prompt read whole.

    python tools/serve/turn_test.py --model <first shard> [--picks <file>] [--exe bin/osmos-serve]

  A, then A less its last token + another + new tokens   reused len(A) - 1 (0.7.4: 0), the whole read's bits
  the same prompt again (a retry)                        reused all but its last token, the same bits
  C with mark=m, then C's first m tokens + new tokens    reused m, the whole read's bits
  D, D + a tool round (mark=len(D) - 1: D's own point),  reused len(D) - 1: a tool loop's first request's point kept
    then D less its last token + new tokens               through its next rounds (a new user message's turn)
  B (another conversation, slots off), then each again   read from token 0: the bits compared

the prompts are real text (~/bench/osmos-prompts). exit 0: every check passed; 1: a check failed; 2: no engine.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Engine, hash_tokens, read_ids  # noqa: E402

failed = []


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--picks")
    ap.add_argument("--exe", default="bin/osmos-serve")
    ap.add_argument("--prompts", default=os.path.expanduser("~/bench/osmos-prompts"))
    a = ap.parse_args()
    calib, other = read_ids(f"{a.prompts}/calib16k.txt"), read_ids(f"{a.prompts}/p8192.txt")
    A, B, new = calib[:6000], other[:3000], other[3000:3064]
    swap = next(t for t in other if t != A[-1])  # the next turn's token where A's last was (</think> for <think>)
    turn = A[:-1] + [swap] + new
    args = ["--model", a.model, "--max-tokens", "2048", "--max-context", "32768"]
    if a.picks:
        args += ["--picks", a.picks]
    C, m = calib[6000:9000], 2000
    turn_c = C[:m] + other[4000:4064]
    D = calib[9000:12000]
    loop = D + other[5000:5200]  # the loop's next round: D, its answer and a tool's result (a continuation)
    turn_d = D[:-1] + [swap] + other[5200:5264]
    e = Engine(a.exe, args, dict(os.environ, OSMOS_SLOT_KEEP="0"))
    try:
        e.gen(A, 16)
        rt = e.gen(turn, 16)
        check(f"the next turn reuses A but its last token ({rt.get('reused')} of {len(A) - 1})",
              rt.get("reused") == len(A) - 1)
        rr = e.gen(turn, 16)  # the same prompt again (a retry): back to its own point one short of its end
        check(f"the same prompt again reuses all but its last token ({rr.get('reused')} of {len(turn) - 1}), the "
              f"same bits", rr.get("reused") == len(turn) - 1 and rr["tokens"] == rt["tokens"])
        e.gen(C, 16, extra=f" mark={m}")
        rc = e.gen(turn_c, 16)
        check(f"a turn leaving C at its mark reuses it ({rc.get('reused')} of {m})", rc.get("reused") == m)
        e.gen(D, 16)
        e.gen(loop, 16, extra=f" mark={len(D) - 1}")
        rd = e.gen(turn_d, 16)
        check(f"a new user turn after a tool round reuses the loop's first point ({rd.get('reused')} of "
              f"{len(D) - 1})", rd.get("reused") == len(D) - 1)
        e.gen(B, 16)
        r0 = e.gen(turn, 16)
        check(f"after B the turn reads from token 0 (reused {r0.get('reused')})", r0.get("reused") == 0)
        check(f"the continued turn's answer the whole read's bits ({hash_tokens(rt['tokens']):016x} / "
              f"{hash_tokens(r0['tokens']):016x})", rt["tokens"] == r0["tokens"] and len(rt["tokens"]) == 16)
        e.gen(B, 16)
        r1 = e.gen(turn_c, 16)
        check(f"C's turn from its mark the whole read's bits ({hash_tokens(rc['tokens']):016x} / "
              f"{hash_tokens(r1['tokens']):016x}; reused {r1.get('reused')})",
              r1.get("reused") == 0 and rc["tokens"] == r1["tokens"])
        e.gen(B, 16)
        r2 = e.gen(turn_d, 16)
        check(f"that turn the whole read's bits ({hash_tokens(rd['tokens']):016x} / {hash_tokens(r2['tokens']):016x})",
              r2.get("reused") == 0 and rd["tokens"] == r2["tokens"])
    finally:
        e.quit()
    print(f"[turn_test] {'ok' if not failed else 'FAILED: ' + '; '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
