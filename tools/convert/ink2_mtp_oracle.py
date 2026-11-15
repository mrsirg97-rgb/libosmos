"""ink2_mtp_oracle.py - Inkling-2-Small's drafter (mtp.safetensors: DSpark, the RNN Markov head) in plain PyTorch,
F32, held to TokenSpeed's inkling2_mtp.py (the model card's runtimes; transformers has none): over a text's taps
(ink2_run TAPS=: h entering layers 5 .. 41 and the final normed hidden), at each anchor position p a block - the
token at p then 7 mask rows at positions p .. p + 7 - through the 5 layers over the context (positions < p: each
layer's own mixture of the 8 taps, normed, its ctx K / V), the target's head on rows 1 .. 7, then the Markov steps
(a GRU-like state over each step's previous token). its greedy drafts against the text's next tokens: the acceptance
(the matching prefix's length, 0 .. 7), and the drafts and logits for osmos's drafter to be held to.

    python tools/convert/ink2_mtp_oracle.py <mtp.safetensors> <target first shard> <ids> <taps.f32> <first p> <last p>
        [--out cases.npz]
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("mtp")
ap.add_argument("target")
ap.add_argument("ids")
ap.add_argument("taps")
ap.add_argument("p0", type=int)
ap.add_argument("p1", type=int)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--out")
ap.add_argument("--ctx-through-anchor", action="store_true")  # convention probes: the context through p itself
ap.add_argument("--causal", action="store_true")              # the block causal
a = ap.parse_args()
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/Projects/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

W = {k: v.float() for k, v in load_file(a.mtp).items()}
H, NH, NKV, D, RANK, BLOCK, WIN = 4608, 48, 8, 128, 256, 8, 1024
LOCAL, EPS, THETA, CAP, VOCAB = {0, 1, 3, 4}, 1e-6, 1e6, 30.0, 200058


def tensor(name):
    for fn in sorted(glob.glob(a.target.replace("-00001-of-", "-*-of-"))):
        for t in GGUFReader(fn).tensors:
            if t.name == name:
                return torch.from_numpy(dequantize(t.data, t.tensor_type).reshape([int(x) for x in t.shape][::-1]).astype(np.float32))
    sys.exit(f"{name}: not in the target")


embed, embed_norm, head = tensor("token_embd.weight"), tensor("token_embd_norm.weight"), tensor("output.weight")
ids = [int(x) for x in open(a.ids).read().replace("\n", ",").split(",") if x.strip()]
taps = torch.from_numpy(np.fromfile(a.taps, dtype=np.float32).reshape(len(ids), 8, H))
alpha = torch.softmax(W["fuse_weight"], dim=-1)  # [6 stages][8 taps]: the 5 layers', then the Markov head's


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):  # x [n][heads][D], NeoX halves
    inv = THETA ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    ang = pos[:, None].double() * inv[None]
    c, s = torch.cos(ang).float()[:, None], torch.sin(ang).float()[:, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


fold_w = head @ W["markov_head.hidden_out.weight"]  # [V][rank]: the head folded (TokenSpeed's in FP64, here F32)
fold_b = head @ W["markov_head.hidden_out.bias"]


def mix(rows, stage, norm):  # rows [n][8][H] -> [n][H]
    return rms((rows * alpha[stage][None, :, None]).sum(1), W[norm])


def draft(p):
    nc = p + 1 if a.ctx_through_anchor else p
    ctx = taps[:nc]
    pos_c = torch.arange(nc)
    x = W["mask_embed"].expand(BLOCK, H).clone()
    x[0] = rms(embed[ids[p]], embed_norm)
    pos_b = torch.arange(p, p + BLOCK)
    residual, hidden = None, x
    for i in range(5):
        pre = f"layers.{i}."
        if residual is None:
            residual, normed = hidden, rms(hidden, W[pre + "input_layernorm.weight"])
        else:
            residual = hidden + residual
            normed = rms(residual, W[pre + "input_layernorm.weight"])
        m = mix(ctx, i, pre + "dflare_norm.weight")
        kc = rms((m @ W[pre + "self_attn.k_proj_ctx.weight"].T).view(-1, NKV, D), W[pre + "self_attn.k_norm_ctx.weight"])
        kc, vc = rope(kc, pos_c), (m @ W[pre + "self_attn.v_proj_ctx.weight"].T).view(-1, NKV, D)
        q = rope(rms((normed @ W[pre + "self_attn.q_proj.weight"].T).view(BLOCK, NH, D), W[pre + "self_attn.q_norm.weight"]), pos_b)
        kb = rope(rms((normed @ W[pre + "self_attn.k_proj.weight"].T).view(BLOCK, NKV, D), W[pre + "self_attn.k_norm.weight"]), pos_b)
        vb = (normed @ W[pre + "self_attn.v_proj.weight"].T).view(BLOCK, NKV, D)
        if i in LOCAL:  # the window: keys within WIN + BLOCK - 1 of the query (TokenSpeed's sliding_window_size)
            keep = pos_c >= p - (WIN + BLOCK - 1) + 1  # (a window of WIN + BLOCK - 1 back of the anchor)
            kc, vc, kpos = kc[keep], vc[keep], pos_c[keep]
        else:
            kpos = pos_c
        k, v = torch.cat((kc, kb)), torch.cat((vc, vb))
        allpos = torch.cat((kpos, pos_b))
        g = NH // NKV
        out = torch.empty(BLOCK, NH, D)
        for hh in range(NH):
            s = (q[:, hh] @ k[:, hh // g].T) / math.sqrt(D)  # non-causal over the block
            if a.causal:
                s = s.masked_fill(allpos[None, :] > pos_b[:, None], float("-inf"))
            if i in LOCAL:
                s = s.masked_fill((pos_b[:, None] - allpos[None, :]) >= WIN + BLOCK - 1, float("-inf"))
            out[:, hh] = torch.softmax(s, -1) @ v[:, hh // g]
        att = out.reshape(BLOCK, NH * D) @ W[pre + "self_attn.o_proj.weight"].T
        residual = att + residual
        normed = rms(residual, W[pre + "post_attention_layernorm.weight"])
        hidden = (torch.nn.functional.silu(normed @ W[pre + "mlp.gate_proj.weight"].T) *
                  (normed @ W[pre + "mlp.up_proj.weight"].T)) @ W[pre + "mlp.down_proj.weight"].T
    final = rms(hidden + residual, W["norm.weight"])
    rows = final[1:]
    base = rows @ head.T
    f_rnn = mix(taps[nc - 1:nc], 5, "markov_head.dflare_norm.weight")[0]
    state, prev, drafts, logits = torch.zeros(RANK), ids[p], [], []
    for s_ in range(BLOCK - 1):
        inp = torch.cat((state, W["markov_head.markov_w1.weight"][prev], rows[s_], f_rnn))
        gate, cand, outp = (W["markov_head.rnn_proj.weight"] @ inp + W["markov_head.rnn_proj.bias"]).chunk(3)
        gs = torch.sigmoid(gate)
        state = gs * state + (1 - gs) * torch.tanh(cand)
        lg = base[s_] + fold_w @ torch.tanh(outp) + fold_b
        lg = CAP * torch.tanh(lg / CAP)
        lg = lg[:VOCAB]
        prev = int(torch.argmax(lg))
        drafts.append(prev), logits.append(lg)
    return drafts, torch.stack(logits)


acc, cases = [], {}
for p in range(a.p0, a.p1 + 1, a.every):
    d, lg = draft(p)
    k = 0
    while k < len(d) and p + 1 + k < len(ids) and d[k] == ids[p + 1 + k]:
        k += 1
    acc.append(k)
    if a.out and len(cases) < 3 * 2:
        cases[f"drafts_{p}"], cases[f"logits_{p}"] = np.array(d), lg.numpy()
    print(f"p {p}: {k} accepted  drafts {d}  text {ids[p + 1:p + 8]}", flush=True)
print(f"mean acceptance {np.mean(acc):.2f} of {BLOCK - 1} over {len(acc)} anchors (+1 a round: the verify's own token)")
if a.out:
    np.savez(a.out, **cases)
