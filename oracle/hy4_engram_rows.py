"""reference Engram rows for tests/test_hy4_engram.c: the reference's hash (modeling_hy_v4_flash.py of the
transformers PR #50903, head c2ec8505 - HYV4FlashEngramLayer.hash_ngrams and draw_hash_multipliers,
HYV4FlashTextModel.build_ngram_tokens) transcribed below in numpy, on the release's own inputs: its config (the primes, the seed, the layers) and its lookup_table (the safetensors shard holding layer 11's),
never the GGUF's lists, which are what the test checks. T token ids (seeded; the vocabulary's ends, the pad and the
tokens past the lookup table among them) as one sequence from position 0, and each position's rows: a head's hash
plus its offset in its layer's table (the layer's heads' primes summed before it).
one file: b"HY4E", u32 version, u32 T, u32 layers, u32 cols, the ids (i32 [T]), the rows (i64 [T][layers][cols]).
run: python -I hy4_engram_rows.py --config config.json --shard
     model-00147-of-00148.safetensors --out FILE"""
import argparse
import json
import struct

import numpy as np

T = 512
SEED = 1108
VERSION = 1
INT64_MAX = np.iinfo(np.int64).max


def draw_hash_multipliers(seed, layer_id, n_gram, compressed):
    """odd multipliers: numpy's default generator seeded seed + 10007 * layer_id, bounded so a product with any
    compressed id stays within int64"""
    half_bound = max(1, INT64_MAX // compressed // 2)
    drawn = np.random.default_rng(seed + 10007 * layer_id).integers(0, half_bound, size=n_gram, dtype=np.int64)
    return drawn * 2 + 1


def build_ngram_tokens(ids, n_gram, pad):
    """[T, n_gram]: the token at each position and the n_gram - 1 before it, the pad before the sequence's start"""
    out = np.full((len(ids), n_gram), pad, dtype=np.int64)
    for k in range(n_gram):
        out[k:, k] = ids[:len(ids) - k]
    return out


def hash_ngrams(ngram, table, pad, mult, sizes, heads):
    """a hash id a head: the tokens through the lookup table (past it: the pad's), times the multipliers, xor-folded
    over orders 2 .. n_gram, each order's fold mod its heads' primes"""
    size = len(table)
    compressed = np.where(ngram < size, table[np.minimum(ngram, size - 1)], table[pad]).astype(np.int64)
    mixed = compressed * mult
    out = []
    for order in range(2, ngram.shape[1] + 1):
        mix = mixed[:, 0].copy()
        for k in range(1, order):
            mix ^= mixed[:, k]
        start = (order - 2) * heads
        out.append(mix[:, None] % np.array(sizes[start:start + heads], dtype=np.int64))
    return np.concatenate(out, axis=1)


def lookup_table(shard, layer):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        key = f"model.layers.{layer}.engram_layer.lookup_table"
        if key not in head or head[key]["dtype"] != "I32":
            raise SystemExit(f"{shard}: no I32 {key}")
        a, b = head[key]["data_offsets"]
        f.seek(8 + n + a)
        return np.frombuffer(f.read(b - a), dtype=np.int32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))["text_config"]
    layer_ids = sorted(int(i) for i in cfg["engram_layer_ids"])
    n_gram, heads = cfg["engram_max_ngram_size"], cfg["engram_n_head_per_ngram"]
    pad, vocab = cfg["engram_pad_id"], cfg["vocab_size"]
    table = lookup_table(a.shard, layer_ids[1] - 1).astype(np.int64)

    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, vocab, size=T)
    specials = [0, 1, pad, vocab - 1, len(table) - 1, len(table), *range(len(table), vocab)]
    ids[1:1 + len(specials)] = specials
    ids[T // 2:T // 2 + 3] = pad
    ngram = build_ngram_tokens(ids, n_gram, pad)

    cols = (n_gram - 1) * heads
    rows = np.zeros((T, len(layer_ids), cols), dtype=np.int64)
    for li, layer_id in enumerate(layer_ids):
        sizes = [s for order in cfg["engram_layer_head_vocab_sizes"][str(layer_id)] for s in order]
        mult = draw_hash_multipliers(cfg.get("engram_seed", 0), layer_id, n_gram, int(table.max()) + 1)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        rows[:, li, :] = hash_ngrams(ngram, table, pad, mult, sizes, heads) + offsets
        print(f"layer {layer_id - 1}: multipliers {[int(x) for x in mult]}")
    with open(a.out, "wb") as f:
        f.write(b"HY4E" + struct.pack("<4I", VERSION, T, len(layer_ids), cols))
        f.write(ids.astype(np.int32).tobytes())
        f.write(rows.tobytes())
    print(f"{a.out}: {T} positions, {len(layer_ids)} layers x {cols} columns, table {len(table)} tokens")


if __name__ == "__main__":
    main()
