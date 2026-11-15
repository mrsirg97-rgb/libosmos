"""test_oracle.py - ink2_oracle.py against transformers' own Inkling2ForCausalLM on a small random model.

The small model's experts are encoded into the checkpoint's formats (NVFP4 for layers 0-9, MXFP8 for 10-11, w13 with
gate and up interleaved), the transformers model given the decoded values, a checkpoint written with the real names.
Then the oracle (layers streamed, attention in chunks against a cache, experts decoded) must give transformers' full
forward's log-probabilities.

    PYTHONPATH=<transformers main> python tools/convert/test_oracle.py --real ~/Models/inkling2/Inkling-2-Small [--out DIR]

--real: the checkpoint's folder (its config.json and model.safetensors.index.json: the names; no weights read).
"""
import json
import os
import subprocess
import sys

import torch
from safetensors.torch import save_file

import argparse  # noqa: E402
import tempfile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True)
ap.add_argument("--out", default="")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--src", "x", "--tokens", "x", "--out", "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ink2_oracle as O  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.inkling2 import modeling_inkling2 as M  # noqa: E402

REAL = os.path.expanduser(args.real)
OUT = args.out or tempfile.mkdtemp(prefix="ink2_tiny_")
torch.manual_seed(0)


def main():
    real = json.load(open(os.path.join(REAL, "config.json")))
    tc = real["text_config"]
    tc.update(hidden_size=256, num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
              swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=64, n_routed_experts=16,
              num_experts_per_tok=4, intermediate_size=128, moe_intermediate_size=128, vocab_size=1024,
              unpadded_vocab_size=1000, sliding_window_size=64, rel_extent=64, log_scaling_n_floor=100)
    tc.pop("local_layer_ids", None)
    real.get("vision_config", {})["text_hidden_size"] = 256
    real.get("mtp_config", {})["hidden_size"] = 256
    os.makedirs(OUT, exist_ok=True)
    json.dump(real, open(os.path.join(OUT, "config.json"), "w"))
    cfg = AutoConfig.from_pretrained(OUT).text_config
    cfg._attn_implementation = "eager"
    model = M.Inkling2ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():   # (random everywhere, the zero-initialized queries and convs too)
            if "norm" in n and p.dim() == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 1 / p.shape[-1] ** 0.5))
        for L, layer in enumerate(model.model.layers):
            layer.mlp.gate.e_score_correction_bias.copy_(0.01 * torch.randn(16))
    # the checkpoint: the real names (the oracle's renames inverted over the real index), experts encoded
    idx = json.load(open(os.path.join(REAL, "model.safetensors.index.json")))["weight_map"]
    inv = {O.rename(k): k for k in idx}
    sd = {}
    with torch.no_grad():
        for n, t in model.state_dict().items():
            if ".mlp.experts." in n:
                continue
            sd[inv[n]] = t.to(torch.bfloat16) if t.dtype == torch.float32 and "attn_res" not in inv[n] and \
                "gate.bias" not in inv[n] else t
        for L, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            p = f"model.llm.layers.{L}.mlp.experts."
            gu, dn = ex.gate_up_proj.data, ex.down_proj.data
            I = gu.shape[1] // 2
            w13 = torch.stack([gu[:, :I], gu[:, I:]], 2).reshape(gu.shape)   # (gate row i at 2i, up at 2i + 1)
            for proj, W, tgt in (("w13", w13, "gate_up"), ("w2", dn, "down")):
                if L < 10:
                    parts = [O.nvfp4(W[e]) for e in range(W.shape[0])]
                    sd[p + proj + "_weight"] = torch.stack([x[0] for x in parts])
                    sd[p + proj + "_weight_scale"] = torch.stack([x[1] for x in parts])
                    sd[p + proj + "_weight_amax"] = torch.stack([x[2] for x in parts]).float()
                    sd[p + proj + "_weight_outer_scale_denominator"] = torch.tensor([2688.0])
                else:
                    parts = [O.mxfp8(W[e]) for e in range(W.shape[0])]
                    sd[p + proj + "_weight"] = torch.stack([x[0] for x in parts])
                    sd[p + proj + "_weight_scale"] = torch.stack([x[1] for x in parts])
            # the transformers model gets the decoded values (deinterleaved), the dense weights their bf16 values
            se = O.StreamExperts.__new__(O.StreamExperts)
            nn_init = torch.nn.Module.__init__
            nn_init(se)
            se.nv, se.dtype = L < 10, torch.float32
            se.lut = torch.tensor(O.E2M1)
            se.t = {k[len(p):].replace("_weight_scale", "_s").replace("_weight_amax", "_amax").replace("_weight", ""): v
                    for k, v in sd.items() if k.startswith(p)}
            if se.nv:
                for proj in ("w13", "w2"):
                    se.t[proj + "_o"] = se.t[proj + "_amax"] / 2688.0
            dec13 = torch.stack([se.weight("w13", e) for e in range(16)])
            gu.copy_(torch.cat([dec13[:, 0::2], dec13[:, 1::2]], 1))
            dn.copy_(torch.stack([se.weight("w2", e) for e in range(16)]))
        for n, t in model.state_dict().items():   # (the dense weights as the checkpoint stores them)
            if ".mlp.experts." not in n and inv[n] in sd and sd[inv[n]].dtype == torch.bfloat16:
                t.copy_(sd[inv[n]].float())
    shards = {"model-00001-of-00001.safetensors": sd}
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(OUT, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}},
              open(os.path.join(OUT, "model.safetensors.index.json"), "w"))
    ids = torch.randint(0, 1000, (300,))
    json.dump(ids.tolist(), open(os.path.join(OUT, "tokens.json"), "w"))
    with torch.no_grad():
        ref = torch.log_softmax(model(input_ids=ids[None]).logits[0].float(), -1)
    for chunk in (300, 128, 37):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ink2_oracle.py"),
                        "--src", OUT, "--tokens", os.path.join(OUT, "tokens.json"), "--out", f"{OUT}/oracle.{chunk}",
                        "--chunk", str(chunk), "--gpu", "0"], check=True, stderr=subprocess.DEVNULL)
        got = torch.frombuffer(bytearray(open(f"{OUT}/oracle.{chunk}.logprobs.f32", "rb").read()),
                               dtype=torch.float32).reshape(300, 1000)
        d = (got - ref).abs()
        kl = (ref.exp() * (ref - got)).sum(-1)
        print(f"chunk {chunk:3d}: max |dlogp| {d.max():.2e}, mean KL {kl.mean():.2e}, top-1 agree "
              f"{(got.argmax(-1) == ref.argmax(-1)).float().mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
