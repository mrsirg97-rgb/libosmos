"""reference MLPs for tests/gpu/test_hy4_moe_gpu.hip: Hy4-flash's latent MoE and its dense MLP by the reference's own
code - the transformers PR #50903 modeling_hy_v4_flash.py's HYV4FlashMoE.forward (HYV4FlashTopkRouter: sigmoid scores,
the bias for the pick alone, the picked weights normalized and scaled; the routed experts in the latent, their SwiGLU
clamped; the shared expert; the combine norm) on a sparse layer, and HYV4FlashMLP (its post_fc2_norm) on the dense
layer 0 - their source compiled alone (hy4_pieces.reference), in F32 on the GGUF's own weights dequantised (the
experts the file's - Q4_K_M's gate and up Q4_K, layer 3's down Q6_K -, the rest Q8_0 or F32; only the experts the tokens pick read: the reference's loop reads no other).
half the tokens loud (x 8: the routed SwiGLU's clamp acting on some of their values - counted, some
required). each step kept: the router's logits, picks and weights, the latent's input, the experts' sum, the shared expert's
output, the block's output; the file's bytes beside them (each picked expert's own), as the GPU reads them.
run: python -I hy4_moe.py --modeling modeling_hy_v4_flash.py --config config.json --gguf Hy4-flash-Q4_K_M-00001-of-00002.gguf
     --out FILE"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy4_pieces as P  # noqa: E402
from hy4_gguf import Gguf  # noqa: E402

T = 8  # a decode step's tokens at most (mv's MV_MAX_TOKENS)
SEED = 1121
LAYER, DENSE = 3, 0
LOUD = 8.0  # the loud tokens' scale: enough that the routed SwiGLU's clamp (10) acts on some of their values
NAMES = ["HYV4FlashRMSNorm", "HYV4FlashMLP", "HYV4FlashExperts", "HYV4FlashTopkRouter", "HYV4FlashMoE"]


def param(t):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)), requires_grad=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeling", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    R = P.reference(a.modeling, NAMES, {"ACT2FN": {"silu": nn.SiLU()}, "use_experts_implementation": None})
    c = json.load(open(a.config))["text_config"]
    g = Gguf(a.gguf)
    config = types.SimpleNamespace(
        hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"], hidden_act=c["hidden_act"],
        rms_norm_eps=c["rms_norm_eps"], num_local_experts=c["n_routed_experts"],
        moe_intermediate_size=c["moe_intermediate_size"], routed_expert_hidden_size=c["routed_expert_hidden_size"],
        swiglu_limit=c["swiglu_limit"], num_experts_per_tok=c["num_experts_per_tok"],
        routed_scaling_factor=c["routed_scaling_factor"], n_group=c["n_group"], topk_group=c["topk_group"],
        norm_topk_prob=c["norm_topk_prob"], n_shared_experts=c["n_shared_experts"])
    rng = np.random.default_rng(SEED)
    x = torch.from_numpy(rng.normal(0, 1.0, size=(1, T, config.hidden_size)).astype(np.float32))
    x[0, T // 2:] *= LOUD  # half the tokens loud: their experts' gate and up past the clamp (counted below)
    blk = f"blk.{LAYER}."

    with torch.device("meta"):
        moe = R["HYV4FlashMoE"](config)
    moe.gate.weight = param(g.tensor(blk + "ffn_gate_inp.weight"))
    moe.gate.e_score_correction_bias = torch.from_numpy(g.tensor(blk + "exp_probs_b.bias"))
    moe.routed_expert_down_proj.weight = param(g.tensor(blk + "ffn_routed_down.weight"))
    moe.routed_expert_up_proj.weight = param(g.tensor(blk + "ffn_routed_up.weight"))
    sh = moe.shared_experts
    sh.gate_proj.weight = param(g.tensor(blk + "ffn_gate_shexp.weight"))
    sh.up_proj.weight = param(g.tensor(blk + "ffn_up_shexp.weight"))
    sh.down_proj.weight = param(g.tensor(blk + "ffn_down_shexp.weight"))
    moe.combine_output_norm.weight = param(g.tensor(blk + "post_ffw_norm.weight"))
    logits, weights, picks = moe.gate(x)  # the experts the tokens pick: only theirs read into the reference's tables
    hit = sorted(set(picks.reshape(-1).tolist()))
    E, I, L = config.num_local_experts, config.moe_intermediate_size, config.routed_expert_hidden_size
    gate_up, down = torch.zeros(E, 2 * I, L), torch.zeros(E, L, I)
    for e in hit:
        gate_up[e, :I] = torch.from_numpy(g.expert(blk + "ffn_gate_exps.weight", e))
        gate_up[e, I:] = torch.from_numpy(g.expert(blk + "ffn_up_exps.weight", e))
        down[e] = torch.from_numpy(g.expert(blk + "ffn_down_exps.weight", e))
    moe.experts.gate_up_proj, moe.experts.down_proj = param(gate_up), param(down)
    seen = {}
    hook = lambda key: (lambda m, i, o: seen.__setitem__(key, o.detach().clone()))  # noqa: E731
    for mod, key in ((moe.routed_expert_down_proj, "routed_in"), (moe.experts, "experts"),
                     (moe.routed_expert_up_proj, "routed_out"), (moe.shared_experts, "shared")):
        mod.register_forward_hook(hook(key))
    out = moe(x)[0]

    dblk = f"blk.{DENSE}."
    dense = R["HYV4FlashMLP"](types.SimpleNamespace(**{**vars(config), "hidden_size": config.hidden_size}))
    dense.gate_proj.weight = param(g.tensor(dblk + "ffn_gate.weight"))
    dense.up_proj.weight = param(g.tensor(dblk + "ffn_up.weight"))
    dense.down_proj.weight = param(g.tensor(dblk + "ffn_down.weight"))
    dense.post_fc2_norm.weight = param(g.tensor(dblk + "post_ffw_norm.weight"))
    dense_out = dense(x)[0]

    clamped = 0  # the values the routed SwiGLU's clamp changed: the test's check of it needs some
    for t in range(T):
        for e in picks[t].tolist():
            gu = gate_up[e] @ seen["routed_in"][t]
            clamped += int((gu[:I] > config.swiglu_limit).sum() + (gu[I:].abs() > config.swiglu_limit).sum())
    assert clamped > 0, "no value past the clamp: the test would not see it"
    slot = {e: i for i, e in enumerate(hit)}
    arrays = {
        "x": x[0].numpy(), "logits": logits.numpy(), "picks": picks.numpy().astype(np.float32),
        "weights": weights.numpy(), "routed_in": seen["routed_in"].numpy(), "experts": seen["experts"].numpy(),
        "routed_out": seen["routed_out"].numpy(), "shared": seen["shared"].numpy(), "out": out.numpy(),
        "dense_out": dense_out.numpy(),
        "router": g.tensor(blk + "ffn_gate_inp.weight"), "router_bias": g.tensor(blk + "exp_probs_b.bias"),
        "routed_down_q8": g.raw(blk + "ffn_routed_down.weight").view(np.float32),
        "routed_up_q8": g.raw(blk + "ffn_routed_up.weight").view(np.float32),
        "sh_gate_q8": g.raw(blk + "ffn_gate_shexp.weight").view(np.float32),
        "sh_up_q8": g.raw(blk + "ffn_up_shexp.weight").view(np.float32),
        "sh_down_q8": g.raw(blk + "ffn_down_shexp.weight").view(np.float32),
        "post_norm": g.tensor(blk + "post_ffw_norm.weight"),
        "hit": np.array(hit, dtype=np.float32),
        "exp_types": np.array([g.tensors[blk + f"ffn_{r}_exps.weight"][1] for r in ("gate", "up", "down")],
                              dtype=np.float32),
        "exp_gate": np.stack([g.expert_raw(blk + "ffn_gate_exps.weight", e) for e in hit]).view(np.float32),
        "exp_up": np.stack([g.expert_raw(blk + "ffn_up_exps.weight", e) for e in hit]).view(np.float32),
        "exp_down": np.stack([g.expert_raw(blk + "ffn_down_exps.weight", e) for e in hit]).view(np.float32),
        "d_gate_q8": g.raw(dblk + "ffn_gate.weight").view(np.float32),
        "d_up_q8": g.raw(dblk + "ffn_up.weight").view(np.float32),
        "d_down_q8": g.raw(dblk + "ffn_down.weight").view(np.float32),
        "d_post_norm": g.tensor(dblk + "post_ffw_norm.weight"),
        "shape": np.array([E, config.num_experts_per_tok, I, L, config.n_shared_experts * I, config.intermediate_size,
                           config.swiglu_limit, config.routed_scaling_factor, config.rms_norm_eps],
                          dtype=np.float32),
    }
    assert all(e in slot for e in picks.reshape(-1).tolist())
    P.write(a.out, arrays)
    print(f"{a.out}: {len(arrays)} arrays; {T} tokens ({T // 2} loud: {clamped} values clamped), {len(hit)} experts "
          f"picked; out |mean| "
          f"{float(out.abs().mean()):.4g}, dense |mean| {float(dense_out.abs().mean()):.4g}")


if __name__ == "__main__":
    main()
