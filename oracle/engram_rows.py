"""reference Engram rows for tests/test_engram.c on a GGUF whose hash is its metadata (llama.cpp's ple.*: unsloth's
UD): the layout, the primes, their offsets and the multipliers derived from DeepSeek's recipe (engram_hash.py: the
primes above engram.vocab_size - 1, the multipliers from 10007 * layer seeds) - not read from the file's ple.* lists,
which are what test_engram checks - and the file's own token map (engram.token_map: the tokenizer's normalization,
which needs the tokenizers package to derive). T token ids (seeded, the vocabulary's ends among them) and the rows
engram_hash.Hasher gives each position of them as one sequence. one file: b"SUBE", u32 version, u32 T, u32 layers,
u32 cols, the ids (i32 [T]), the rows (i64 [T][layers][cols]).
run: PYTHONPATH=<llama.cpp>/gguf-py python -I engram_rows.py --model FIRST_SHARD --out FILE"""
import argparse
import os
import struct
import sys

import numpy as np
from gguf import GGUFReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engram_hash as eh  # noqa: E402

T = 64
SEED = 1017


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kv = GGUFReader(a.model).fields
    g = lambda k: kv["deepseek41." + k].contents()  # noqa: E731
    layers, mx, nh = list(g("ple.layers")), g("ple.ngram_size"), g("ple.heads_per_ngram")
    cfg = {"engram_layer_ids": layers, "engram_max_ngram_size": mx, "engram_n_heads": nh,
           "engram_vocab_size": g("engram.vocab_size"), "engram_head_dim": g("embedding_length_per_layer_input")}
    # the layers' rows: the sum of the recipe's own primes (from_config checks the layout against it)
    rows = []
    seen = set()
    for _ in layers:
        total = 0
        for _ in range(mx - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(nh):
                cur = eh._next_prime(cur, seen)
                seen.add(cur)
                total += cur
        rows.append(total)
    lay = eh.Layout.from_config({**cfg, "engram_num_embeddings": rows})
    tm = np.asarray(g("engram.token_map"), dtype=np.int64)
    compressed = g("engram.compressed_vocab_size")
    assert int(tm.max()) < compressed, "the map's entries past the compressed vocabulary"
    h = eh.Hasher(lay, tm, eh.multipliers(lay, compressed), pad=tm[g("engram.pad_token_id")])
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(tm), T).astype(np.int32)
    ids[[5, 17, 40]] = [0, len(tm) - 1, g("engram.pad_token_id")]
    out = h.ids(ids).astype(np.int64)
    assert out.shape == (T, len(layers), (mx - 1) * nh)
    with open(a.out, "wb") as f:
        f.write(b"SUBE" + struct.pack("<IIII", 1, T, len(layers), (mx - 1) * nh))
        f.write(ids.tobytes())
        f.write(out.tobytes())
    print("%d positions, %d layers x %d columns; the layers' rows %s (the recipe's primes)" %
          (T, len(layers), (mx - 1) * nh, rows))


main()
