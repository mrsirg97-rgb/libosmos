"""test_cases.py - ink2_cases.py's files against the formulas, written out in float64 from the files alone (no
transformers): what each array means, in the layout the kernels will read, the second implementation the cases are
checked by. Every case's output within 1e-5 relative L2 (fp32's own rounding through the op) of transformers'.

    PYTHONPATH=<llama.cpp>/gguf-py python tools/convert/test_cases.py [~/osmos-oracle/ink2-cases]

The conventions a kernel can get wrong, each pinned by its case:
  - the static convolutions (conv, a sliding layer's k and v): w [C, taps] (GGUF ne taps, C), tap taps-1 the
    current token, tap 0 the oldest: y[t] = (sum_j w[c, j] x[t - (taps-1) + j] + x[t]) / sqrt(taps + 1)
  - the dynamic ones (a full layer's): static [taps, C] (ne C, taps), tap 0 the current token; dyn [T, kv_heads *
    taps], head h's taps at h * taps: y[t] = (sum_j (static[j, c] + dyn[t, h(c) * taps + j]) x[t - j] + x[t]) / 3
  - q's clip after its scale, k's with none, rel's after its scale; tau (full layers) after the clips, on q and rel
  - the relative bias rel[t, h] . rel_proj[:, t - s] for 0 <= t - s < rel_extent, else 0; the sink a logit of its
    own with a value of zero; the gate sigmoid(gate_raw) on the attention's output
  - the router's picks by sigmoid + bias, the weights the sigmoid alone, normalized (+ 1e-20) and x 8
"""
import json
import math
import os
import struct
import sys

import numpy as np
import torch

import gguf

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/osmos-oracle/ink2-cases")
cases = json.load(open(os.path.join(D, "cases.json")))
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
TOL = 1e-5
fails = 0


def load(case, name):
    with open(os.path.join(D, case, name + ".bin"), "rb") as f:
        assert f.read(4) == b"QDMP"
        dims, = struct.unpack("<I", f.read(4))
        ne = struct.unpack("<4q", f.read(32))[:dims]
        v = np.frombuffer(f.read(), dtype=np.float32)
    return torch.from_numpy(v.copy()).double().reshape(tuple(reversed(ne)))


def agree(case, what, got, want):
    global fails
    want = want.double()
    err = float((got - want).norm() / want.norm().clamp(min=1e-30))
    ok = err < TOL
    fails += not ok
    print(f"  {case:11s} {what:12s} rel L2 {err:.2e} {'ok' if ok else 'FAIL'}")


def l2clip(x, th):
    return x * (th / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)


def static_conv(x, w):   # x [T, C], w [C, taps]: tap taps-1 the current token
    taps = w.shape[1]
    xp = torch.cat([x.new_zeros(taps - 1, x.shape[1]), x])
    y = sum(w[:, j] * xp[j:j + x.shape[0]] for j in range(taps))
    return (y + x) / math.sqrt(taps + 1)


def dynamic_conv(x, static, dyn, heads):   # static [taps, C] tap 0 current; dyn [T, heads * taps]
    taps, C = static.shape
    T = x.shape[0]
    w = static.T[None] + dyn.view(T, heads, 1, taps).expand(T, heads, C // heads, taps).reshape(T, C, taps)
    xp = torch.cat([x.new_zeros(taps - 1, C), x])
    y = sum(w[:, :, j] * xp[taps - 1 - j:taps - 1 - j + T] for j in range(taps))
    return (y + x) / math.sqrt(2 * taps + 1)


def attention(case):
    s = cases[case]["scalars"]
    T, H, KV, Dh, DR, P0 = s["tokens"], s["heads"], s["kv_heads"], s["head_dim"], s["d_rel"], s["pos0"]
    g = lambda n: load(case, n)  # noqa: E731
    pos = torch.arange(T, dtype=torch.float64) + P0
    tau = (1 + 0.1 * torch.log(((pos + 1) / 128000).clamp(min=1))) if s["full"] else torch.ones(T, dtype=torch.float64)
    q = l2clip(g("q_raw").view(T, H, Dh) * g("q_clip"), s["q_clip"])
    agree(case, "q clipped", q, g("q_clipped"))
    if s["full"]:
        kc = dynamic_conv(g("k_raw"), g("k_conv_static"), g("k_conv_dyn"), KV)
        vc = dynamic_conv(g("v_raw"), g("v_conv_static"), g("v_conv_dyn"), KV)
    else:
        kc, vc = static_conv(g("k_raw"), g("k_conv")), static_conv(g("v_raw"), g("v_conv"))
    agree(case, "k conv", kc, g("k_conv_out"))
    agree(case, "v conv", vc, g("v_conv_out"))
    k = l2clip(kc.view(T, KV, Dh), s["k_clip"])
    v = vc.view(T, KV, Dh)
    v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + s["rms_eps"])
    agree(case, "k", k, g("k"))
    agree(case, "v", v, g("v"))
    rel = l2clip(g("rel_raw").view(T, H, DR) * g("rel_clip"), s["rel_clip"])
    agree(case, "rel clipped", rel, g("rel_clipped"))
    q, rel = q * tau[:, None, None], rel * tau[:, None, None]
    rel_proj, ext = g("rel_proj"), s["rel_extent"]
    kk, vv = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)   # [T, H, Dh]
    logits = torch.einsum("thd,shd->hts", q, kk) * s["scaling"]
    dist = torch.arange(T)[:, None] - torch.arange(T)[None, :]
    rl = torch.einsum("thr,re->hte", rel, rel_proj)   # [H, T, ext]
    bias = rl.gather(-1, dist.clamp(0, ext - 1).expand(H, T, T))
    bias = bias.masked_fill((dist < 0) | (dist >= ext), 0.0)
    ok = (dist >= 0) & ((dist < s["window"]) if s["window"] else True)
    logits = (logits + bias).masked_fill(~ok, -math.inf)
    sink = g("sink_raw").T[:, :, None]   # [H, T, 1]
    p = torch.softmax(torch.cat([logits, sink], -1), -1)[..., :-1]
    att = torch.einsum("hts,shd->thd", p, vv).reshape(T, H * Dh)
    agree(case, "ungated", att, g("ungated"))
    agree(case, "out", att * torch.sigmoid(g("gate_raw")), g("out"))


def conv(case):
    agree(case, "out", static_conv(load(case, "x"), load(case, "w")), load(case, "out"))


def residual(case):
    s = cases[case]["scalars"]
    q = load(case, "query").reshape(s["queries"], -1)
    st = torch.cat([load(case, "blocks"), load(case, "partial")[:, None]], 1)   # [T, nb + 1, C]
    st = st * torch.rsqrt(st.square().mean(-1, keepdim=True) + s["eps"])
    p = torch.softmax(torch.einsum("qc,tbc->tqb", q, st) * s["scale"], -1)
    o = torch.einsum("tqb,tbc->tqc", p, st)
    for i in range(s["queries"]):
        agree(case, f"out{i}", o[:, i], load(case, f"out{i}"))


def router(case):
    s = cases[case]["scalars"]
    logits = load(case, "x") @ load(case, "weight").T
    agree(case, "logits", logits, load(case, "logits"))
    sc = torch.sigmoid(logits)
    picks = (sc + load(case, "bias")).topk(s["used"], -1).indices.sort(-1).values
    global fails
    same = bool((picks == load(case, "picks").long()).all())
    fails += not same
    print(f"  {case:11s} {'picks':12s} {'the same' if same else 'DIFFER'}")
    w = sc.gather(-1, picks)
    agree(case, "weights", w / (w.sum(-1, keepdim=True) + 1e-20) * s["scale"], load(case, "weights"))


def swiglu(case):
    s = cases[case]["scalars"]
    gt, up = load(case, "gate"), load(case, "up")
    cg, cu = s["gate_softcap"], s["up_softcap"]
    agree(case, "out", torch.tanh(gt / cg) * cg * torch.sigmoid(gt) * torch.tanh(up / cu) * cu, load(case, "out"))


def softcap(case):
    c = cases[case]["scalars"]["cap"]
    agree(case, "out", torch.tanh(load(case, "x") / c) * c, load(case, "out"))


def nvfp4_rows(t):
    """a GGUF NVFP4 tensor's values, decoded as ggml does: [experts, rows, K]"""
    n_exp, rows, K = (int(d) for d in reversed(t.shape))
    b = np.asarray(t.data).view(np.uint8).reshape(n_exp, rows, K // 64, 36)
    d = torch.from_numpy(b[..., :4].copy()).view(torch.float8_e4m3fn).double()   # (UE4M3: every one positive here)
    qs = torch.from_numpy(b[..., 4:].copy()).long().view(n_exp, rows, K // 64, 4, 8)
    codes = torch.stack([qs & 0x0F, qs >> 4], -2).reshape(n_exp, rows, K // 64, 4, 16)
    lut = torch.tensor(E2M1, dtype=torch.float64)
    return (lut[codes] * d[..., None]).reshape(n_exp, rows, K)


def experts(case):
    s = cases[case]["scalars"]
    r = gguf.GGUFReader(os.path.join(D, case, s["gguf"]))
    t = {x.name: x for x in r.tensors}
    W, S = {}, {}
    for nm in ("gate", "up", "down"):
        W[nm] = nvfp4_rows(t[f"blk.0.ffn_{nm}_exps.weight"])
        S[nm] = torch.from_numpy(np.asarray(t[f"blk.0.ffn_{nm}_exps.scale"].data).copy()).double()
    x, picks, wts = load(case, "x"), load(case, "picks").long(), load(case, "weights")
    out = torch.zeros_like(x)
    cg, cu = s["gate_softcap"], s["up_softcap"]
    for tkn in range(x.shape[0]):
        for j in range(s["used"]):
            e = int(picks[tkn, j])
            gt = (W["gate"][e] @ x[tkn]) * S["gate"][e]   # (the scale after the block sum, as llama.cpp)
            up = (W["up"][e] @ x[tkn]) * S["up"][e]
            h = torch.tanh(gt / cg) * cg * torch.sigmoid(gt) * torch.tanh(up / cu) * cu
            out[tkn] += wts[tkn, j] * (W["down"][e] @ h) * S["down"][e]
    agree(case, "out", out, load(case, "out"))


RUN = {"sliding": attention, "full_p0": attention, "full_p300k": attention, "conv": conv, "router": router,
       "swiglu": swiglu, "softcap": softcap, "experts": experts}
for c in cases:
    RUN.get(c, residual if c.startswith("res") else None)(c)
print("[test_cases]", "ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
