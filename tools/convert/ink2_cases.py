"""ink2_cases.py - reference cases for Inkling-2's new GPU ops, each run through transformers' own module at the
model's real sizes: an op's inputs, its output and the stages between, for the kernels to be held to from their
first compile.

    PYTHONPATH=<transformers main>:<llama.cpp>/gguf-py python tools/convert/ink2_cases.py \
        --real ~/Models/inkling2/Inkling-2-Small [--out ~/osmos-oracle/ink2-cases]

--real: the checkpoint's folder (config.json: the shapes; no weights read). The inputs are seeded random, scaled so
every edge is crossed: q, k and the relative states straddle their L2 clips, the gate and up straddle their softcaps,
a sequence runs past the window and the relative extent (1,088 > 1,024), a full layer is run at position 0 (no log
scaling) and at 300,000 (tau 1.085). The projections are not the cases' (osmos's matvec and matmul, tested): an
attention case starts at its rows of q, k, v, the relative states, the gate and the sinks, transformers' module
given them in place of its projections, and ends before o_proj.

Every array a QDMP file (tests/gpu/q4x_ref.h's ref_read: "QDMP", u32 dims, 4 x i64 ne, F32 values) named
<case>/<array>.bin, ne GGUF's order (the innermost first: torch's shape reversed) - a weight as its GGUF tensor
holds it. The expert case's weights are a GGUF of their own (experts.gguf: blk.0.ffn_{gate,up,down}_exps.{weight,
scale}, 16 experts in NVFP4 as ink2_ref_gguf.py writes the checkpoint's), its reference the oracle's expert path.
cases.json: each case's arrays and scalars.

  sliding     layer 0's attention: the k and v convolutions (static), the clips, v's norm, the relative bias
              (window 1,024), the sinks, the gate; every 8th row from 1,024 on flat (q and rel zero: each key in
              reach weighted alike - the window's edge seen). out: the gated output; ungated: the same with every gate 1
  full_p0     layer 5's (full): the dynamic k and v convolutions (static + per-token taps), the relative bias out to
              rel_extent, causal over all keys, tau 1
  full_p300k  layer 5's at positions 300,000 on: q and the relative states scaled by tau
  conv        the post-attention convolution (4,608 channels, 4 taps): (conv + x) / sqrt(5)
  res1_nb1, res1_nb8, res3_nb1, res3_nb8
              the attention residual: 1 query (a sliding layer's, the MoE's) or 3 (a full layer's q, k, v) over 1 or
              8 block states and the partial one
  router      sigmoid scores, the bias for the pick alone, the picked weights normalized x 8; the picks sorted by
              expert (ties none: the scores are continuous)
  swiglu      tanh(g / 4) * 4 * sigmoid(g) * tanh(u / 25) * 25
  softcap     the logits' tanh(x / 30) * 30, a vocabulary wide
  experts     16 tokens through 16 NVFP4 experts, 8 each: the expert's scale applied before the activation (gate
              and up) and after down, the routing weights, the sum
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True)
ap.add_argument("--out", default=os.path.expanduser("~/osmos-oracle/ink2-cases"))
ap.add_argument("--only", default="", help="these cases, comma-separated")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
argv, sys.argv = sys.argv, [sys.argv[0], "--src", "x", "--tokens", "x", "--out", "x"]   # (both parse at import)
import ink2_oracle as O  # noqa: E402
sys.argv = [argv[0], "--src", "x"]
import ink2_ref_gguf as R  # noqa: E402
sys.argv = argv
import gguf  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.inkling2 import modeling_inkling2 as M  # noqa: E402

T_ATTN = 1088   # past the window and the relative extent (1,024)
T = 64          # the per-token ops' tokens
FAR = 300_000   # a full layer's positions with log scaling (tau = 1 + 0.1 ln(300,001 / 128,000))
N_EXP = 16      # the expert case's experts (the real 384 the same per expert)
manifest = {}


def dump(case, name, t):
    t = t.detach().float().contiguous()
    while t.dim() > 1 and t.shape[0] == 1:   # (the batch's 1)
        t = t[0]
    assert t.dim() <= 4, (case, name, t.shape)
    ne = list(reversed(t.shape)) + [1] * (4 - t.dim())
    d = os.path.join(args.out, case)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".bin"), "wb") as f:
        f.write(b"QDMP" + struct.pack("<I", t.dim()) + struct.pack("<4q", *ne))
        f.write(t.numpy().tobytes())
    manifest.setdefault(case, {"arrays": {}, "scalars": {}})["arrays"][name] = ne[:t.dim()]


def scalars(case, **kv):
    manifest.setdefault(case, {"arrays": {}, "scalars": {}})["scalars"].update(kv)


class Given(nn.Module):
    """a projection replaced by its output: the case's rows in place of a matmul"""

    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return self.t


def rn(g, *shape, sd=1.0):
    return torch.randn(*shape, generator=g) * sd


def un(g, *shape, lo, hi):
    return lo + (hi - lo) * torch.rand(*shape, generator=g)


def mask(n, window):
    d = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    ok = (d >= 0) & ((d < window) if window else True)
    return torch.zeros(n, n).masked_fill(~ok, torch.finfo(torch.float32).min)[None, None]


def attention(cfg, case, L, pos0):
    full = cfg.layer_types[L] != "hybrid_sliding"
    A = (M.Inkling2FullAttention if full else M.Inkling2SlidingAttention)(cfg, L).float().eval()
    H, KV, D, DR, K = A.num_heads, A.num_key_value_heads, A.head_dim, cfg.d_rel, cfg.sconv_kernel_size
    g = torch.Generator().manual_seed(1000 + L)
    n = T_ATTN
    # (sizes putting about half of each past its clip: |q| ~ sd sqrt(128) x 1.04 (the scale's rms) against 22.6; |k|
    # after its convolution ~ sd sqrt(128) sqrt(1 + sum w^2) / sqrt(5) sliding, / sqrt(9) full (static + dynamic
    # taps) against 22.6; |rel| ~ sd 4 x 1.53 against 8)
    k_sd = 4.6 if full else 3.1
    x = dict(q_raw=rn(g, n, H * D, sd=1.9), k_raw=rn(g, n, KV * D, sd=k_sd), v_raw=rn(g, n, KV * D),
             rel_raw=rn(g, n, H * DR, sd=1.3), gate_raw=rn(g, n, H * D, sd=2.0), sink_raw=rn(g, n, H, sd=2.0))
    # (flat rows: q and rel zero - every logit 0, each key in reach weighted alike, so the window's last key and the
    # first past it are seen; with random logits (sd ~7) a key at the edge carries e^-19 and is not. a clip of zero)
    flat = torch.arange(cfg.sliding_window_size, n, 8)
    x["q_raw"][flat], x["rel_raw"][flat] = 0.0, 0.0
    w = dict(q_clip=un(g, H, D, lo=0.5, hi=1.5), rel_clip=un(g, H, DR, lo=0.5, hi=2.5),
             rel_proj=rn(g, DR, A.rel_extent))
    A.q_l2_clip.scale.data, A.rel_l2_clip.scale.data = w["q_clip"], w["rel_clip"]
    A.rel_logits_proj.proj.data = w["rel_proj"]
    if full:
        w.update(k_conv_static=rn(g, K, KV * D, sd=0.3), v_conv_static=rn(g, K, KV * D, sd=0.3))
        x.update(k_conv_dyn=rn(g, n, KV * K, sd=0.3), v_conv_dyn=rn(g, n, KV * K, sd=0.3))
        A.k_sconv.static_weight.data, A.v_sconv.static_weight.data = w["k_conv_static"], w["v_conv_static"]
        A.k_sconv.weight_proj, A.v_sconv.weight_proj = Given(x["k_conv_dyn"][None]), Given(x["v_conv_dyn"][None])
    else:
        w.update(k_conv=rn(g, KV * D, K, sd=0.5), v_conv=rn(g, KV * D, K, sd=0.5))
        A.k_sconv.conv1d.weight.data, A.v_sconv.conv1d.weight.data = w["k_conv"][:, None], w["v_conv"][:, None]
    for proj, k in (("q_proj", "q_raw"), ("k_proj", "k_raw"), ("v_proj", "v_raw"), ("r_proj", "rel_raw"),
                    ("gate_proj", "gate_raw"), ("sink_proj", "sink_raw")):
        setattr(A, proj, Given(x[k][None]))
    A.o_proj = nn.Identity()
    got = {}
    for mod, name in ((A.q_l2_clip, "q_clipped"), (A.k_l2_clip, "k"), (A.v_norm, "v"), (A.rel_l2_clip, "rel_clipped"),
                      (A.k_sconv, "k_conv_out"), (A.v_sconv, "v_conv_out")):
        mod.register_forward_hook(lambda m, i, o, name=name: got.__setitem__(name, o))
    h = torch.zeros(1, n, cfg.hidden_size)
    pos = (torch.arange(n) + pos0)[None]
    kw = dict(attention_mask=mask(n, None if full else cfg.sliding_window_size), position_ids=pos)
    run = (lambda: A(h, h, h, **kw)[0]) if full else (lambda: A(h, **kw)[0])
    out = run()
    stages = dict(got)
    A.gate_proj = Given(torch.full((1, n, H * D), 100.0))   # (sigmoid(100) = 1 in fp32: the attention ungated)
    ungated = run()
    for k, v in {**x, **w, **stages}.items():
        dump(case, k, v)
    dump(case, "out", out)
    dump(case, "ungated", ungated)
    tau = 1 + cfg.log_scaling_alpha * np.log(max(1.0, (pos0 + 1) / cfg.log_scaling_n_floor)) if full else 1.0
    scalars(case, layer=L, full=full, tokens=n, pos0=pos0, heads=H, kv_heads=KV, head_dim=D, d_rel=DR,
            rel_extent=A.rel_extent, window=0 if full else cfg.sliding_window_size, taps=K, scaling=A.scaling,
            q_clip=cfg.qk_l2_clip_threshold, k_clip=cfg.qk_l2_clip_threshold, rel_clip=cfg.rel_l2_clip_threshold,
            rms_eps=cfg.rms_norm_eps, tau_first=float(tau),
            conv_scale=1 / np.sqrt((2 if full else 1) * K + 1))
    clip = lambda t, th: float((torch.linalg.vector_norm(t, dim=-1) > th).float().mean())  # noqa: E731
    qn = x["q_raw"].view(n, H, D) * w["q_clip"]
    print(f"{case}: q over its clip {clip(qn, cfg.qk_l2_clip_threshold):.0%}, k "
          f"{clip(stages['k_conv_out'].view(n, KV, D), cfg.qk_l2_clip_threshold):.0%}, rel "
          f"{clip(x['rel_raw'].view(n, H, DR) * w['rel_clip'], cfg.rel_l2_clip_threshold):.0%}; tau {tau:.4f}")


def conv(cfg):
    g = torch.Generator().manual_seed(2)
    C = M.Inkling2ShortConvolution(cfg.hidden_size, cfg.conv_kernel_size, layer_idx=0, conv_idx=2).float()
    w = rn(g, cfg.hidden_size, cfg.conv_kernel_size, sd=0.5)
    C.conv1d.weight.data = w[:, None]
    x = rn(g, 1, T, cfg.hidden_size)
    dump("conv", "x", x), dump("conv", "w", w), dump("conv", "out", C(x))
    scalars("conv", tokens=T, channels=cfg.hidden_size, taps=cfg.conv_kernel_size,
            scale=1 / np.sqrt(cfg.conv_kernel_size + 1))


def residual(cfg, nq, nb):
    case = f"res{nq}_nb{nb}"
    g = torch.Generator().manual_seed(3 + 10 * nq + nb)
    R_ = M.Inkling2BlockAttentionResidual(cfg, num_queries=nq).float()
    q = rn(g, *R_.query.shape, sd=3.0)
    R_.query.data = q
    # (the block states' scales apart, as a residual stream's grow: the norm takes them out)
    blocks = rn(g, 1, T, nb, cfg.hidden_size) * un(g, 1, 1, nb, 1, lo=0.1, hi=10.0)
    partial = rn(g, 1, T, cfg.hidden_size) * 4.0
    outs = R_(blocks, partial)
    dump(case, "query", q), dump(case, "blocks", blocks), dump(case, "partial", partial)
    for i, o in enumerate(outs):
        dump(case, f"out{i}", o)
    scalars(case, tokens=T, queries=nq, blocks=nb, eps=R_.norm.eps, scale=cfg.hidden_size ** -0.5)


def router(cfg):
    g = torch.Generator().manual_seed(4)
    Rt = M.Inkling2TopKRouter(cfg).float()
    Rt.weight.data = rn(g, cfg.num_local_experts, cfg.hidden_size, sd=4 / np.sqrt(cfg.hidden_size))
    Rt.e_score_correction_bias = rn(g, cfg.num_local_experts, sd=0.1)
    x = rn(g, T, cfg.hidden_size)
    logits, wts, idx = Rt(x)
    order = idx.argsort(-1)
    dump("router", "x", x), dump("router", "weight", Rt.weight), dump("router", "bias", Rt.e_score_correction_bias)
    dump("router", "logits", logits)
    dump("router", "picks", idx.gather(-1, order)), dump("router", "weights", wts.gather(-1, order))
    scalars("router", tokens=T, experts=cfg.num_local_experts, used=cfg.num_experts_per_tok,
            scale=cfg.routed_scaling_factor)


def swiglu(cfg):
    g = torch.Generator().manual_seed(5)
    S = M.Inkling2SoftcappedSwiGLU(cfg)
    gate, up = rn(g, T, cfg.moe_intermediate_size, sd=8.0), rn(g, T, cfg.moe_intermediate_size, sd=30.0)
    dump("swiglu", "gate", gate), dump("swiglu", "up", up), dump("swiglu", "out", S(gate, up))
    scalars("swiglu", tokens=T, width=cfg.moe_intermediate_size, gate_softcap=cfg.mlp_gate_softcap,
            up_softcap=cfg.mlp_up_softcap)


def softcap(cfg):
    g = torch.Generator().manual_seed(6)
    x = rn(g, 4, cfg.unpadded_vocab_size, sd=40.0)
    dump("softcap", "x", x), dump("softcap", "out", M.Inkling2Softcap(cfg.final_logit_softcapping)(x))
    scalars("softcap", rows=4, width=cfg.unpadded_vocab_size, cap=cfg.final_logit_softcapping)


class FakeSrc:
    """a checkpoint in memory: ink2_ref_gguf's and the oracle's readers over it"""

    def __init__(self, t):
        self.t = t

    def get(self, k):
        return self.t[k]

    def meta(self, k):
        t = self.t[k]
        return {torch.uint8: "U8", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3"}[t.dtype], list(t.shape)


def experts(cfg):
    g = torch.Generator().manual_seed(7)
    H, F_, n = cfg.hidden_size, cfg.moe_intermediate_size, T // 4
    p = "model.llm.layers.0.mlp.experts."
    ck = {}
    for proj, rows, cols in (("w13", 2 * F_, H), ("w2", H, F_)):
        # (each expert's own amax: a spread of 4x across them, so a scale wrongly shared or skipped shows)
        sd = 0.03 * un(g, N_EXP, lo=0.5, hi=2.0)
        enc = [O.nvfp4(rn(g, rows, cols) * sd[e]) for e in range(N_EXP)]
        ck[p + proj + "_weight"] = torch.stack([e[0] for e in enc])
        ck[p + proj + "_weight_scale"] = torch.stack([e[1] for e in enc])
        ck[p + proj + "_weight_amax"] = torch.stack([e[2] for e in enc]).float()
        ck[p + proj + "_weight_outer_scale_denominator"] = torch.tensor([2688.0])
    src = FakeSrc(ck)
    x = rn(g, n, H)
    scores = torch.rand(n, N_EXP, generator=g)
    idx = scores.topk(cfg.num_experts_per_tok, -1).indices.sort(-1).values
    wts = torch.rand(n, cfg.num_experts_per_tok, generator=g) + 0.1
    wts = wts / wts.sum(-1, keepdim=True) * cfg.routed_scaling_factor
    E = O.StreamExperts(src, 0, torch.device("cpu"), torch.float32, M.Inkling2SoftcappedSwiGLU(cfg))
    out = E(x, idx, wts)
    dump("experts", "x", x), dump("experts", "picks", idx), dump("experts", "weights", wts), dump("experts", "out", out)
    # the GGUF: the converter's own expert bytes and scales
    path = os.path.join(args.out, "experts", "experts.gguf")
    w = gguf.GGUFWriter(path, "inkling2")
    parts = list(R.experts(src, 0))
    for name, sh, t, data in parts:
        w.add_tensor(name, data.view(np.int8) if data.dtype == np.uint8 else data, raw_shape=sh, raw_dtype=t)
    w.write_header_to_file(), w.write_kv_data_to_file(), w.write_tensors_to_file()
    w.close()
    scalars("experts", tokens=n, experts=N_EXP, used=cfg.num_experts_per_tok, hidden=H, ff=F_,
            gate_softcap=cfg.mlp_gate_softcap, up_softcap=cfg.mlp_up_softcap, gguf="experts.gguf",
            tensors=[nm for nm, *_ in parts])


def main():
    cfg = AutoConfig.from_pretrained(args.real).text_config
    cfg._attn_implementation = "eager"
    torch.set_grad_enabled(False)
    first_full = next(L for L in range(cfg.num_hidden_layers) if cfg.layer_types[L] != "hybrid_sliding")
    cases = {
        "sliding": lambda: attention(cfg, "sliding", 0, 0),
        "full_p0": lambda: attention(cfg, "full_p0", first_full, 0),
        "full_p300k": lambda: attention(cfg, "full_p300k", first_full, FAR),
        "conv": lambda: conv(cfg),
        "res1_nb1": lambda: residual(cfg, 1, 1), "res1_nb8": lambda: residual(cfg, 1, 8),
        "res3_nb1": lambda: residual(cfg, 3, 1), "res3_nb8": lambda: residual(cfg, 3, 8),
        "router": lambda: router(cfg), "swiglu": lambda: swiglu(cfg), "softcap": lambda: softcap(cfg),
        "experts": lambda: experts(cfg),
    }
    only = args.only.split(",") if args.only else list(cases)
    os.makedirs(args.out, exist_ok=True)
    mf = os.path.join(args.out, "cases.json")
    if os.path.exists(mf):
        manifest.update(json.load(open(mf)))
    for c in only:
        cases[c]()
        print(f"{c}: {len(manifest[c]['arrays'])} arrays", flush=True)
        json.dump(manifest, open(mf, "w"), indent=1)


if __name__ == "__main__":
    main()
