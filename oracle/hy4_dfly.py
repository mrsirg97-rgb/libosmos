"""reference DFly2 drafts for tests/gpu/test_hy4_dfly_gpu.hip: Hy4-flash's drafter as vLLM serves it
(vllm/models/hy_v4/nvidia/dfly.py, dfly_speculator.py, dfly_inputs.py at 076fe951: HYV4DFlyQueryModel) - transcribed
in torch, F32, on the release's own BF16 weights (model-00148-of-00148.safetensors) and the GGUF's token_embd and
output - run on osmos's own target state at the last accepted token (the test's dump: each target layer's MLA cache
rows below the anchor, its indexer's groups for the position before it, the head's normed input there; the anchor
token y): the block [y, mask x 5] at positions anchor .. anchor + 5; the anchor's row fused (anchors + gate *
eh_proj([enorm(e); hnorm(h)])); five pre-norm layers, each gated MLA (q_a, its norm, q_b; kv_a, its norm, the rope
GPT-J's at base 1e7; scores (q_nope k_b-absorbed . latent + q_pe . k_pe) / 16 over the target's picked keys and the
block's own rows up to the row's, the head's learnable sink a logit with no value; sigmoid(linear_gate) times the
heads' outputs; o_proj; post_proj_norm) then the dense SwiGLU; the final norm; then the chain: draft j from row j's
hidden corrected (hidden_norm, embed_norm of the token before, a SwiGLU, added) through the target's head, greedy.
prints the drafts and their softmax probabilities and writes them with the corrected hiddens.
run: python -I hy4_dfly.py --dfly model-00148-of-00148.safetensors --gguf Hy4-flash-Q4_K_M-00001-of-00002.gguf
     --dump DIR --out FILE"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4_gguf import Gguf  # noqa: E402
from hy4_pieces import write  # noqa: E402

TARGETS = [25, 29, 33, 37, 41]  # config.json: dfly_target_kv_layers = dfly_target_index_layer
MASK = 120009
EPS = 1e-5
H, NOPE, ROPE, LAT, VD, POOL = 32, 192, 64, 512, 256, 4
BASE = 1e7
P = "model.dfly_layers."


def st_load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in head.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16).astype(np.uint32) << 16
            out[k] = torch.from_numpy(raw.view(np.float32).reshape(v["shape"]).copy())
    return out


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):
    """GPT-J's: pairs (2i, 2i + 1) of the last dim, freq base^(-2i / 64)"""
    i = torch.arange(ROPE // 2, dtype=torch.float32)
    inv = 1.0 / (BASE ** (2 * i / ROPE))
    ang = pos[:, None].float() * inv[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    shape = x.shape
    x2 = x.reshape(*shape[:-1], ROPE // 2, 2)
    a, b = x2[..., 0], x2[..., 1]
    while c.dim() < a.dim():
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    return torch.stack((a * c - b * s, a * s + b * c), -1).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    for k in ("dfly", "gguf", "dump", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    W = st_load(a.dfly)
    g = Gguf(a.gguf)
    anchor, y, _, top = np.fromfile(os.path.join(a.dump, "meta.i32"), dtype=np.int32)
    osmos_drafts = np.fromfile(os.path.join(a.dump, "drafts.i32"), dtype=np.int32)
    R = 6
    pos = torch.arange(anchor, anchor + R)
    ids = [int(y)] + [MASK] * (R - 1)
    emb_rows = lambda t: torch.from_numpy(g.rows("token_embd.weight", t))  # noqa: E731
    head = torch.from_numpy(g.tensor("output.weight"))  # [V][D]
    h_t = torch.from_numpy(np.fromfile(os.path.join(a.dump, "hidden.f32"), dtype=np.float32))
    h = emb_rows(ids).clone()
    # the fusion, on the anchor's row
    e0 = h[0]
    fused = W[P + "mtp_fusion.eh_proj.weight"] @ torch.cat((rms(e0, W[P + "mtp_fusion.enorm.weight"]),
                                                           rms(h_t, W[P + "mtp_fusion.hnorm.weight"])))
    h[0] = e0 + W[P + "mtp_fusion.gate"] * fused
    for li, L in enumerate(TARGETS):
        p = f"{P}{li}."
        x = rms(h, W[p + "input_layernorm.weight"])
        q = (rms(x @ W[p + "self_attn.q_a_proj.weight"].T, W[p + "self_attn.q_a_layernorm.weight"]) @
             W[p + "self_attn.q_b_proj.weight"].T).view(R, H, NOPE + ROPE)
        kva = x @ W[p + "self_attn.kv_a_proj_with_mqa.weight"].T
        c_blk = rms(kva[:, :LAT], W[p + "self_attn.kv_a_layernorm.weight"])
        kpe_blk = rope(kva[:, LAT:], pos)
        q_nope, q_pe = q[..., :NOPE], rope(q[..., NOPE:], pos)
        kvb = W[p + "self_attn.kv_b_proj.weight"].view(H, NOPE + VD, LAT)
        wk, wv = kvb[:, :NOPE], kvb[:, NOPE:]  # [H][NOPE][LAT], [H][VD][LAT]
        cache = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"cache{L}.f32"), dtype=np.float32)
                                 .reshape(-1, LAT + ROPE))
        sel = np.fromfile(os.path.join(a.dump, f"sel{L}.i32"), dtype=np.int32)
        n = int(np.fromfile(os.path.join(a.dump, f"nsel{L}.i32"), dtype=np.int32)[0])
        hist = sorted({t for gp in sel[:n] for t in range(gp * POOL, gp * POOL + POOL) if t < anchor})
        c_hist, kpe_hist = cache[hist, :LAT], cache[hist, LAT:]
        sink = W[p + "self_attn.learnable_sink_param"]
        outs = []
        for j in range(R):
            c = torch.cat((c_hist, c_blk[:j + 1]))
            kp = torch.cat((kpe_hist, kpe_blk[:j + 1]))
            k_nope = torch.einsum("hnl,sl->hsn", wk, c)
            v = torch.einsum("hvl,sl->hsv", wv, c)
            sc = (torch.einsum("hn,hsn->hs", q_nope[j], k_nope) + q_pe[j] @ kp.T) / 16.0
            sc = torch.cat((sc, sink[:, None]), 1)
            pr = torch.softmax(sc, 1)[:, :-1]
            outs.append(torch.einsum("hs,hsv->hv", pr, v).reshape(-1))
        o = torch.stack(outs) * torch.sigmoid(x @ W[p + "self_attn.linear_gate.weight"].T)
        o = rms(o @ W[p + "self_attn.o_proj.weight"].T, W[p + "self_attn.post_proj_norm.weight"])
        h = h + o
        x2 = rms(h, W[p + "post_attention_layernorm.weight"])
        h = h + (F.silu(x2 @ W[p + "mlp.gate_proj.weight"].T) * (x2 @ W[p + "mlp.up_proj.weight"].T)) @ \
            W[p + "mlp.down_proj.weight"].T
    z = rms(h, W[P + "final_norm.weight"])
    prev, drafts, probs, corr = int(y), [], [], []
    hc = P + "hidden_correction."
    for j in range(1, R):
        cat = torch.cat((rms(z[j], W[hc + "hidden_norm.weight"]), rms(emb_rows([prev])[0], W[hc + "embed_norm.weight"])))
        c = z[j] + (F.silu(W[hc + "gate_proj.weight"] @ cat) * (W[hc + "up_proj.weight"] @ cat)) @ W[hc + "down_proj.weight"].T
        logits = head @ c
        prev = int(logits.argmax())
        drafts.append(prev), probs.append(float(torch.softmax(logits, 0)[prev])), corr.append(c.numpy())
    print("reference drafts:", drafts, "probabilities", [round(p, 3) for p in probs])
    print("osmos's drafts:  ", osmos_drafts.tolist(), "- the same" if drafts == osmos_drafts.tolist() else "- DIFFER")
    write(a.out, {"drafts": np.array(drafts, dtype=np.float32), "probs": np.array(probs, dtype=np.float32),
                  "z": z.numpy(), "corrected": np.stack(corr)})


if __name__ == "__main__":
    main()
