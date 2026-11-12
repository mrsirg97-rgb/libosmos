#!/usr/bin/env python3
"""tools/bench/kld_tokens.py - where a quant's distance from the reference lies, token by token (DESIGN.md, "0.6.6:
the reference, and how a quant is scored").

    python3 tools/bench/kld_tokens.py <run.dump> <ids.txt> <maya root> <tokenizer dir> [--mask m.txt] [--worst 25]

the dump is kld's (kl, nll, top, ref_top a scored row); ids.txt and the mask the run's. prints how much of the KL the
worst rows hold (1%, 5%, 10% of them), the KL by the kind of token predicted (a word's start, a word's piece, digits,
whitespace, punctuation, a special token), by span in a DeepSeek chat (reasoning: inside <think> .. </think>; a tool
call: inside the DSML calls block; the answer: the rest of an assistant turn; a prompt: outside one), and the worst
rows with their text around them: the token predicted, the reference's pick, the run's.
"""
import os
import sys

import numpy as np

REC = np.dtype([("kl", "<f4"), ("nll", "<f4"), ("top", "<i4"), ("ref_top", "<i4")])


def read_list(p):
    return [int(x) for x in open(p).read().replace("\n", "").strip().strip(",").split(",") if x]


def kind(piece):
    if piece.startswith("<｜") or piece in ("<think>", "</think>"):
        return "special"
    s = piece.strip()
    if not s:
        return "whitespace"
    if s.isdigit():
        return "digits"
    if all(not c.isalnum() for c in s):
        return "punctuation"
    return "word start" if piece[:1].isspace() else "word piece"


def main():
    a = sys.argv[1:]
    if len(a) < 4:
        raise SystemExit(__doc__)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    dump, ids_p, maya, tokdir = a[:4]
    worst = int(opt("--worst", 25))
    sys.path.insert(0, os.path.join(maya, "tools"))
    import strata_tokenizer as ST  # noqa: E402
    tok = ST.Tokenizer.from_dir(tokdir)
    dec = lambda t: tok.decode([t]) if t >= 0 else "?"  # noqa: E731
    ids = read_list(ids_p)
    mask = read_list(opt("--mask", None)) if "--mask" in a else [1] * len(ids)
    rows = [r for r in range(len(ids) - 1) if mask[r + 1]]  # row r predicts token r + 1 (kld's order)
    d = np.fromfile(dump, dtype=REC)
    if len(d) != len(rows):
        raise SystemExit(f"{len(d)} records, {len(rows)} scored rows: not this text and mask")
    kl = d["kl"].astype(np.float64)
    total = kl.sum()
    print(f"{len(kl)} rows, mean KL {kl.mean():.5f}, median {np.median(kl):.5f}")
    srt = np.sort(kl)[::-1]
    for share in (0.01, 0.05, 0.10):
        k = max(1, int(len(srt) * share))
        print(f"  the worst {share:.0%} of rows ({k}): {100 * srt[:k].sum() / total:.1f}% of the KL")

    special = {s: tok.encode(s, parse_special=True)[0] for s in ("<think>", "</think>", "<｜Assistant｜>",
                                                                 "<｜end▁of▁sentence｜>", "<｜User｜>")}
    calls_open = tok.encode("<｜DSML｜", parse_special=True)
    span, state, prev = [], "prompt", []
    for i, t in enumerate(ids):  # the span the token at i lies in
        if t == special["<｜Assistant｜>"]:
            state = "answer"
        elif t in (special["<｜end▁of▁sentence｜>"], special["<｜User｜>"]):
            state = "prompt"
        elif t == special["<think>"] and state != "prompt":
            state = "reasoning"
        elif t == special["</think>"] and state == "reasoning":
            state = "answer"
        elif calls_open and t == calls_open[0] and state == "answer":
            state = "tool call"
        span.append(state)

    def table(title, keys):
        print(f"\nby {title}:")
        print(f"  {'':14s} {'rows':>7s} {'mean KL':>9s} {'share of KL':>12s} {'top-1':>7s}")
        for k in sorted(set(keys), key=lambda k: -sum(kl[i] for i in range(len(keys)) if keys[i] == k)):
            idx = [i for i in range(len(keys)) if keys[i] == k]
            agree = float((d["top"][idx] == d["ref_top"][idx]).mean())
            print(f"  {k:14s} {len(idx):7d} {kl[idx].mean():9.4f} {100 * kl[idx].sum() / total:11.1f}% "
                  f"{100 * agree:6.1f}%")

    table("the token predicted", [kind(dec(ids[r + 1])) for r in rows])
    table("span", [span[r + 1] for r in rows])
    print(f"\nthe {worst} worst rows (the token predicted, the reference's pick, this run's; text before it):")
    for j in np.argsort(-kl)[:worst]:
        r = rows[j]
        ctx = tok.decode(ids[max(0, r - 12):r + 1]).replace("\n", "\\n")[-60:]
        print(f"  {kl[j]:7.3f}  {dec(ids[r + 1])!r:14s} ref {dec(int(d['ref_top'][j]))!r:14s} "
              f"run {dec(int(d['top'][j]))!r:14s} | ...{ctx}")


if __name__ == "__main__":
    main()
