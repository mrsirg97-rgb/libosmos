"""every intermediate of ref_gguf.py's first --layers layers, for the engine's per-op parity checks: the oracle's own
methods wrapped (nothing of its forward copied, so nothing here can drift from it). per layer L, in the .npz:

  L.h_in [T, hc, d]    the residual stream into the layer (after Engram, where the layer has it)
  L.a_pre/post/comb    the attention's hyper-connection mixes ([T, hc], [T, hc], [T, hc, hc])
  L.attn_x [T, d]      the attention's input (hc_pre with the lagged pre, normed)
  L.q [T, H, D]        q as the sparse attention takes it (q_b, rope)
  L.kv [N, D]          the keys / values it reads (the window's kv, roped and FP8-rounded; then compressed rows)
  L.idx [T, K]         the rows each position reads (-1 none)
  L.o [T, H, D]        the attention before the inverse rope
  L.attn_out [T, d]    the attention block's output (wo_a, wo_b)
  L.h_mid              the stream after the attention's hc_post
  L.f_pre/post/comb    the FFN's mixes
  L.ffn_x [T, d]       the FFN's input
  L.route_idx/w [T, k] the experts each position takes and their weights
  L.moe_out [T, d]     the FFN's output
  L.h_out              the stream out of the layer

run: PYTHONPATH=<llama.cpp>/gguf-py ../.oracle-venv/bin/python layer_dump.py --pack <maya-ds4> --ids ... --layers N
     --out X.npz [--dev cpu]"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ref as R  # noqa: E402
import ref_gguf as RG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="cpu")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    r = RG.RefGguf(pathlib.Path(a.pack).expanduser(), a.dev)
    r.nl = a.layers
    rec, cur = {}, {"li": -1, "mix": 0}

    def put(name, v):
        rec["%d.%s" % (cur["li"], name)] = v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)

    mixes, attention, sparse, moe = r.hc_mixes, r.attention, r.sparse_attn, r.moe

    def w_mixes(x, fn, scale, base):
        out = mixes(x, fn, scale, base)
        if cur["mix"] == 0:  # the attention's: a new layer begins
            cur["li"] += 1
            put("h_in", x)
            for n, v in zip(("a_pre", "a_post", "a_comb"), out):
                put(n, v)
        else:
            put("h_mid", x)
            for n, v in zip(("f_pre", "f_post", "f_comb"), out):
                put(n, v)
        cur["mix"] ^= 1
        return out

    def w_attention(li, x, S):
        assert li == cur["li"]
        put("attn_x", x)
        y = attention(li, x, S)
        put("attn_out", y)
        return y

    def w_sparse(q, kv, sink, idx, scale):
        o = sparse(q, kv, sink, idx, scale)
        put("q", q), put("kv", kv), put("idx", idx), put("o", o)
        return o

    def w_moe(li, x):
        put("ffn_x", x)
        y = moe(li, x)
        put("moe_out", y)
        put("route_idx", r.last_route[0]), put("route_w", r.last_route[1])
        return y

    r.hc_mixes, r.attention, r.sparse_attn, r.moe = w_mixes, w_attention, w_sparse, w_moe
    dump = []
    with torch.inference_mode():
        r.forward(ids, dump)
    for li, h in enumerate(dump):
        rec["%d.h_out" % li] = h
    rec["ids"] = np.array(ids)
    np.savez(a.out, **rec)
    print("layers %d, T %d: %d arrays -> %s" % (a.layers, len(ids), len(rec), a.out))


main()
