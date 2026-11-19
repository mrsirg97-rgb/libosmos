"""which 64 dims of LFM3's heads vibes1.7b's LSA maps read (their W_K / W_Q: [256, 64] - 4 KV heads x 64): the
reference model (lfm3_forward.py's: Liquid's modeling code in F32 on the GGUF's values) over a document's first
--tokens tokens; at each full-attention layer, each candidate input's index vectors - a key's normalize(k_in W_K), a
query's normalize(q_in W_Q), q_in each KV group's query heads summed (q35's LSA) - scored query x key, and the
attention mass (the model's own weights, a KV group's heads averaged) the top --top keys by that score hold, over
the queries past --min and the keys before their last --window positions (the read's part the index decides). the
maps' own input holds far more than the others; random maps the floor.

  lfm3_lsa_probe.py --ref DIR --gguf FILE --maps FILE --text FILE [--tokens 4096 --top 256 --window 512]"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm3_forward import build  # noqa: E402


def maps(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        meta = head.pop("__metadata__", {})
        out = {}
        for k, v in head.items():
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = torch.from_numpy(np.frombuffer(f.read(b - a), np.float32).reshape(v["shape"]).copy())
    return out, meta


# a candidate input: the dims of a 128-dim head it takes (LFM3's rotation: pairs (i, i + 64), i < 32 rotated)
CANDIDATES = {
    "unrotated (32..63, 96..127)": list(range(32, 64)) + list(range(96, 128)),
    "unrotated by pair (32, 96, 33, 97..)": [d for i in range(32, 64) for d in (i, i + 64)],
    "unrotated (96..127, 32..63)": list(range(96, 128)) + list(range(32, 64)),
    "the last 64 (64..127)": list(range(64, 128)),
    "the first 64 (0..63)": list(range(64)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--top", type=int, default=256)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--min", type=int, default=2048)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(os.path.join(a.ref, "tokenizer.json")).encode(open(a.text).read()).ids[: a.tokens]
    T = len(ids)
    model, config = build(a.ref, a.gguf)
    model.config._attn_implementation = "eager"
    W, meta = maps(a.maps)
    layers = json.loads(meta["layers"])
    H, KV, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    G = H // KV
    caught = {}

    def grab(l, what):
        def hook(mod, args, out):  # q_norm / k_norm's output: [1, T, heads, hd] (the rotation after: the tail unmoved)
            caught[(l, what)] = out[0].float()
        return hook

    def grab_attn(l):
        def hook(mod, args, kwargs, out):
            caught[(l, "w")] = out[1][0].float()  # [H, T, T]: the model's weights
        return hook

    for l in layers:
        at = model.model.layers[l].self_attn
        at.q_norm.register_forward_hook(grab(l, "q"))
        at.k_norm.register_forward_hook(grab(l, "k"))
        at.register_forward_hook(grab_attn(l), with_kwargs=True)
    # the rotation's (q, k after it) - the last-64 and first-64 candidates read rotated dims
    rot = {}
    for l in layers:
        mod = sys.modules[type(model).__module__]
        if not hasattr(mod, "_probe_rope"):
            orig = mod.apply_rotary_pos_emb

            def apply(q, k, cos, sin, unsqueeze_dim=1, _orig=orig):
                qr, kr = _orig(q, k, cos, sin, unsqueeze_dim)
                rot.setdefault("seq", []).append((qr[0].transpose(0, 1).float(), kr[0].transpose(0, 1).float()))
                return qr, kr

            mod.apply_rotary_pos_emb = apply
            mod._probe_rope = True
    model.model(input_ids=torch.tensor([ids]), use_cache=False)
    # apply_rotary_pos_emb runs on the attention layers in order: the full ones are layers' positions among them
    attn_layers = [i for i, t in enumerate(config.layer_types) if t != "conv"]
    rng = np.random.default_rng(0)
    qs = torch.arange(a.min, T)
    print(f"{T} ids; queries {a.min}..{T - 1}, the top {a.top} keys by score before each query's last {a.window}: "
          f"the share of the model's attention mass they hold (a KV group's heads averaged)")
    tau_line = []
    for l in layers:
        qn, kn, w = caught[(l, "q")], caught[(l, "k")], caught[(l, "w")]
        qr, kr = rot["seq"][attn_layers.index(l)]
        wk, wq = W[f"layers.{l}.W_K_idx"], W[f"layers.{l}.W_Q_idx"]
        line = [f"  layer {l:2d} (tau {json.loads(meta['tau'])[str(l)]:.3f}):"]
        for name, dims in list(CANDIDATES.items()) + [("random maps", None)]:
            src_k, src_q = (kn, qn) if name.startswith("unrotated") or dims is None else (kr, qr)
            d = dims or CANDIDATES["unrotated (32..63, 96..127)"]
            k_in = src_k[:, :, d].reshape(T, KV * 64)
            q_in = src_q[:, :, d].reshape(T, KV, G, 64).sum(2).reshape(T, KV * 64)
            mk, mq = (wk, wq) if dims is not None else (torch.from_numpy(rng.standard_normal(wk.shape).astype(np.float32)),
                                                       torch.from_numpy(rng.standard_normal(wq.shape).astype(np.float32)))
            ik = torch.nn.functional.normalize(k_in @ mk, dim=-1)
            iq = torch.nn.functional.normalize(q_in @ mq, dim=-1)
            held = []
            for p in qs[::64].tolist():
                cut = p + 1 - a.window
                if cut <= a.top:
                    continue
                s = ik[:cut] @ iq[p]
                top = torch.topk(s, a.top).indices
                mass = w[:, p, :cut].reshape(KV, G, cut).mean(1)  # [KV, cut]
                held.append((mass[:, top].sum() / mass.sum()).item())
            line.append(f"{name} {np.mean(held):.3f}")
            if name.startswith("unrotated (32"):  # tau against this input's scores: where it falls, what it keeps
                tau = json.loads(meta["tau"])[str(l)]
                sc, kept, frac = [], [], []
                for p in qs[::64].tolist():
                    cut = p + 1 - a.window
                    if cut <= a.top:
                        continue
                    s_ = ik[:cut] @ iq[p]
                    mass = w[:, p, :cut].reshape(KV, G, cut).mean(1)
                    sel = s_ >= tau
                    sc.append(s_), frac.append(sel.float().mean().item())
                    kept.append((mass[:, sel].sum() / mass.sum()).item())
                allsc = torch.cat(sc)
                q = torch.quantile(allsc, torch.tensor([0.5, 0.9, 0.99]))
                tau_line.append(f"    layer {l:2d}: scores median {q[0]:.3f}, p90 {q[1]:.3f}, p99 {q[2]:.3f}; "
                                f"tau {tau:.3f} keeps {np.mean(frac):.3f} of the keys, {np.mean(kept):.3f} of the mass")
        print("; ".join(line))
    print("  tau, as a cut on the (unrotated) index score:")
    for t in tau_line:
        print(t)
    print(f"  (chance: {a.top} of the ~{a.min + (T - a.min) // 2 - a.window} keys a query scores - "
          f"{a.top / (a.min + (T - a.min) // 2 - a.window):.3f} of a uniform mass)")


if __name__ == "__main__":
    main()
