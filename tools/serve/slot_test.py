#!/usr/bin/env python3
"""tools/serve/slot_test.py - the kept conversations (OSMOS_SLOT_KEEP; maya #356's slot_keep_test.py, ported):
a request that does not continue the conversation in the engine takes back the kept one it continues, so A, B, A's
next turn reads only A's new tokens - and answers with the bits of A never set aside (the answers' token hashes
compared; maya's test compared greedy tokens). a turn read from token 0 is another computation: its answer tokens
then go through a prompt step's experts (WMMA), not the decode's (the lane) - the same in both engines.

    python tools/serve/slot_test.py --model <first shard> [--picks <file>] [--dspark <dir>] [--exe bin/osmos-serve]

  reference A, its next turn in place, B, the turn again  slots off: the turn continues A in place (the bits a
                                                        conversation never set aside answers with); after B, the
                                                        turn reads from token 0
  in-run   A, B, A's next turn, one process             the turn reuses A and its answer (all but the answer's last
                                                        token), and answers the reference's bits
  kept     A, B, then QUIT                              the folder keeps both (2 .bin + 2 .meta) under OSMOS_SLOT_GB
  restart  A's next turn alone, a new process           the same reuse, from the folder; the same bits (greedy and,
                                                        with --dspark, drafted)
  evict    A, B, C into a folder capped at A+B's bytes   the least recently used (A) gone, B and C kept

the prompts are real text (the bench's token files: ~/bench/osmos-prompts), A and B different texts from their
second token. exit 0: every check passed; 1: a check failed; 2: the engine did not start.
"""
import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Engine, hash_tokens, read_ids  # noqa: E402

failed = []


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def files(d, ext):
    return sorted(f for f in os.listdir(d) if f.startswith("slot-") and f.endswith(ext))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--picks")
    ap.add_argument("--dspark")
    ap.add_argument("--exe", default="bin/osmos-serve")
    ap.add_argument("--prompts", default=os.path.expanduser("~/bench/osmos-prompts"))
    ap.add_argument("--budget", default="100")
    a = ap.parse_args()
    calib, gpl = read_ids(f"{a.prompts}/calib16k.txt"), read_ids(f"{a.prompts}/p8192.txt")
    A, B, C = calib[:12000], gpl[:8000], calib[12000:16000]
    new = gpl[8000:8064]  # A's next turn: its prompt, its answer, then these
    args = ["--model", a.model, "--budget", a.budget, "--max-tokens", "2048", "--max-context", "65536"]
    if a.picks:
        args += ["--picks", a.picks]
    if a.dspark:
        args += ["--dspark", a.dspark]
    tmp = tempfile.mkdtemp(prefix="osmos-slots-")

    def engine(keep, d, gb=None):
        env = dict(os.environ, OSMOS_SLOT_KEEP="1" if keep else "0", OSMOS_SLOT_DIR=d)
        if gb is not None:
            env["OSMOS_SLOT_GB"] = str(gb)
        return Engine(a.exe, args, env)

    try:
        print("reference: A, its next turn continued in place, then B and the turn again (slots off)", flush=True)
        e = engine(False, f"{tmp}/none")
        ra = e.gen(A, 32)
        turn = A + ra["tokens"] + new
        held = len(A) + len(ra["tokens"]) - 1  # A and its answer, all but its last token (printed, never fed)
        rt = e.gen(turn, 32)
        rb = e.gen(B, 32)
        r0 = e.gen(turn, 32)
        e.quit()
        check(f"reference: the turn continues A in place (reused {rt['reused']} of {held})", rt["reused"] == held)
        check(f"reference, slots off: after B the turn reads from token 0 (reused {r0['reused']}, {r0['s']:.1f} s)",
              r0["reused"] == 0)
        ref_a, ref_b, ref_t = (hash_tokens(r["tokens"]) for r in (ra, rb, rt))
        print(f"        A {len(A)} tok: {ref_a:016x}; B {len(B)}: {ref_b:016x}; the turn {len(turn)} tok: {ref_t:016x}"
              f" (read from token 0: {hash_tokens(r0['tokens']):016x} - a prompt step's experts, not the decode's)",
              flush=True)

        print("in-run: A, B, A's next turn, one process", flush=True)
        kept = f"{tmp}/kept"
        e = engine(True, kept, 0.05)
        r1a, r1b = e.gen(A, 32), e.gen(B, 32)
        r1t = e.gen(turn, 32)
        check(f"in-run: A and B answer as the reference", hash_tokens(r1a["tokens"]) == ref_a and
              hash_tokens(r1b["tokens"]) == ref_b)
        check(f"in-run: the turn reuses A and its answer ({r1t['reused']} of {held}), in {r1t['s']:.1f} s against "
              f"{r0['s']:.1f} read from token 0", r1t["reused"] == held)
        check(f"in-run: the turn's answer the reference's bits ({hash_tokens(r1t['tokens']):016x})",
              hash_tokens(r1t["tokens"]) == ref_t)
        e.quit()

        print("kept: A, B, then QUIT (a fresh folder)", flush=True)
        kept = f"{tmp}/kept2"
        e = engine(True, kept, 0.05)
        e.gen(A, 32), e.gen(B, 32)
        e.quit()
        bins, metas = files(kept, ".bin"), files(kept, ".meta")
        size = sum(os.path.getsize(os.path.join(kept, f)) for f in bins)
        check(f"kept: the folder holds A and B ({len(bins)} .bin, {len(metas)} .meta, {size / 2**20:.1f} MiB)",
              len(bins) == 2 and len(metas) == 2)
        check("kept: the folder under OSMOS_SLOT_GB=0.05", size <= 0.05 * 2**30)

        print("restart: A's next turn alone, a new process", flush=True)
        kept_d = f"{tmp}/kept-drafted"  # (the drafted restart's own copy: the first takes A back out of the folder)
        shutil.copytree(kept, kept_d)
        e = engine(True, kept, 0.05)
        note = next((n for n in e.notes if n.startswith("NOTE slots:")), "")
        r2 = e.gen(turn, 32)
        check(f"restart: {note[11:]}", "2 conversations" in note)
        check(f"restart: the turn reuses A from the folder ({r2['reused']} of {held}), in {r2['s']:.1f} s",
              r2["reused"] == held)
        check("restart: the turn's answer the reference's bits", hash_tokens(r2["tokens"]) == ref_t)
        e.quit()
        if a.dspark:
            e = engine(True, kept_d, 0.05)
            r3 = e.gen(turn, 32, spec=1)
            check(f"restart, drafted: the turn reuses A ({r3['reused']}), its answer the reference's bits",
                  r3["reused"] == held and hash_tokens(r3["tokens"]) == ref_t)
            e.quit()

        print("evict: A, B, C into a folder capped at A+B's bytes", flush=True)
        ev = f"{tmp}/evict"
        cap_gb = size / 2**30 * 1.02
        e = engine(True, ev, cap_gb)
        e.gen(A, 32), e.gen(B, 32), e.gen(C, 32)
        e.quit()
        e = engine(True, ev, cap_gb)
        ra2, rb2 = e.gen(turn, 8), e.gen(B + rb["tokens"] + new, 8)
        e.quit()
        check(f"evict: A gone (its turn reused {ra2['reused']}), B kept (its turn reused {rb2['reused']})",
              ra2["reused"] == 0 and rb2["reused"] == len(B) + len(rb["tokens"]) - 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'every check passed' if not failed else f'{len(failed)} failed'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
