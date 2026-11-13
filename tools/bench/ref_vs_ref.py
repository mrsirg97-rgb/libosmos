#!/usr/bin/env python3
"""tools/bench/ref_vs_ref.py - one osmosref file scored against another, as kld scores a run: KL(reference || candidate)
a row over the whole vocabulary, in kld's DUMP records (kl, nll, top, ref_top), so kld_compare.py reads another engine's
run beside osmos's (DESIGN.md, "0.6.10": llama.cpp's UD-IQ3_XXS against the BF16 reference, both llama_ref's).

    python3 tools/bench/ref_vs_ref.py <reference.ref> <candidate.ref> <ids.txt> <out.dump> [--mask m.txt]

both files of one text, mask and vocabulary (their headers checked); a row's nll the candidate's of the next token.
prints the mean KL, the top-1 agreement and the candidate's perplexity.
"""
import math
import sys

import numpy as np

REC = np.dtype([("kl", "<f4"), ("nll", "<f4"), ("top", "<i4"), ("ref_top", "<i4")])


def head(path):
    with open(path, "rb") as f:
        b = f.read(24)
    assert b[:8] == b"osmosref", path
    vocab, n, step, rows = np.frombuffer(b, "<u4", 4, 8)
    return int(vocab), int(n), int(rows)


def read_list(p):
    return [int(x) for x in open(p).read().replace("\n", "").strip().strip(",").split(",") if x]


def main():
    a = sys.argv[1:]
    if len(a) < 4:
        raise SystemExit(__doc__)
    ref, cand, ids_p, out = a[:4]
    hr, hc = head(ref), head(cand)
    assert hr == hc, (hr, hc)
    vocab, n, rows = hr
    ids = read_list(ids_p)[:n]
    mask = read_list(a[a.index("--mask") + 1])[:n] if "--mask" in a else [1] * n
    want = [r for r in range(n - 1) if mask[r + 1]]
    assert len(want) == rows, (len(want), rows)
    R = np.memmap(ref, "<f4", "r", 24, (rows, vocab))
    C = np.memmap(cand, "<f4", "r", 24, (rows, vocab))
    rec = np.zeros(rows, REC)
    for k in range(0, rows, 256):
        r, c = R[k:k + 256].astype(np.float64), C[k:k + 256].astype(np.float64)
        p = np.exp(r)
        rec["kl"][k:k + 256] = np.maximum((p * (r - c)).sum(1), 0)
        rec["top"][k:k + 256] = c.argmax(1)
        rec["ref_top"][k:k + 256] = r.argmax(1)
        nxt = np.array([ids[want[i] + 1] for i in range(k, min(k + 256, rows))])
        rec["nll"][k:k + 256] = -c[np.arange(len(nxt)), nxt]
    rec.tofile(out)
    print(f"{out}: kld mean {rec['kl'].mean():.6f}, top-1 agreement {100 * (rec['top'] == rec['ref_top']).mean():.2f}%, "
          f"ppl {math.exp(rec['nll'].mean()):.4f} over {rows} rows")


if __name__ == "__main__":
    main()
