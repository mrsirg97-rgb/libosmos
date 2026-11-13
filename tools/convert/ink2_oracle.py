"""ink2_oracle.py - Inkling-2-Small's forward pass from the original checkpoint, a layer at a time: the oracle every
Inkling pack is scored against.

    PYTHONPATH=<transformers main> python tools/convert/ink2_oracle.py --src ~/Models/inkling2/Inkling-2-Small \
        --tokens t.json --out ref [--gpu 0] [--chunk 2048]

transformers' inkling2 is in its main branch from 109a7d19bf (2026-11-12), in no release yet: installed into a
folder of its own (pip install --no-deps --target DIR git+https://github.com/huggingface/transformers@109a7d19bf,
huggingface_hub>=1.31, tokenizers 0.23, regex>=2025.10) and put first on PYTHONPATH, any torch 2.x beside it.

The architecture is transformers' own (models/inkling2: Inkling2SlidingDecoderLayer / Inkling2FullDecoderLayer, its
embedding, final attention residual, norm, head and softcap) - only the routed experts are ours, decoded from the
checkpoint's NVFP4 (layers 0-39: e2m1 x e4m3 x amax[e] / 2688) or MXFP8 (40-47: e4m3 x 2^(e - 127)) on the GPU, an
expert at a time, every value exact in fp32. One layer's weights are resident at once; its experts' bytes (6 GB
NVFP4, 12 GB MXFP8) go to the GPU whole. Attention runs in --chunk-token steps against a cache of the layer's own,
the mask from that cache's positions (full: causal; sliding: distance < 1024).

--dtype float32 (default): every weight upcast exactly, every activation fp32 - the model's function with no
rounding of ours beyond fp32's. bfloat16: transformers' recommended run.

Writes OUT.logprobs.f32 (the log-softmax over the 200,058 real tokens, a row per position, row i predicting token
i + 1) and OUT.json (the tokens, the shape, the seconds).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--tokens", required=True, help="a JSON list of token ids")
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
ap.add_argument("--layers", type=int, default=0, help="only the first N layers (a test); 0: all")
a = ap.parse_args()

from transformers import AutoConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.inkling2 import modeling_inkling2 as M  # noqa: E402

# (code 8, E2M1's -0, as +0: ggml's and gguf-py's table is integers)
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# the checkpoint's names -> transformers' text model's (conversion_mapping.py "inkling2_text", in order)
RENAMES = [
    (r"model\.llm\.unembed\.weight", "lm_head.weight"),
    (r"model\.llm\.", "model."),
    (r"embed\.weight", "embed_tokens.weight"),
    (r"embed_norm\.", "embed_tokens.embed_norm."),
    (r"mlp\.gate\.bias", "mlp.gate.e_score_correction_bias"),
    (r"attn\.wq_du", "self_attn.q_proj"), (r"attn\.wk_dv", "self_attn.k_proj"), (r"attn\.wv_dv", "self_attn.v_proj"),
    (r"attn\.wr_du", "self_attn.r_proj"), (r"attn\.wo_ud", "self_attn.o_proj"),
    (r"attn\.gate_attn", "self_attn.gate_proj"), (r"attn\.attn_sink_proj", "self_attn.sink_proj"),
    (r"\.attn\.(q_l2_clip|k_l2_clip|rel_l2_clip|k_sconv|v_sconv|rel_logits_proj)\.", r".self_attn.\1."),
    (r"attn_sconv\.", "post_attention_convolution."), (r"mlp_sconv\.", "post_mlp_convolution."),
    (r"(convolution|sconv)\.weight(?!_)", r"\1.conv1d.weight"),
    (r"(convolution|sconv)\.static_w", r"\1.static_weight"),
    (r"(convolution|sconv)\.weight_proj", r"\1.weight_proj.weight"),
    (r"mlp_norm", "post_attention_layernorm"), (r"attn_norm", "input_layernorm"),
    (r"(q|k|v)_input_norm", r"\1_input_layernorm"),
]


def rename(k):
    m = re.fullmatch(r"model\.llm\.attn_res\.attn_queries\.(\d+)", k)
    if m:
        return f"model.layers.{int(m[1]) + 1}.attention_input.query"
    m = re.fullmatch(r"model\.llm\.attn_res\.mlp_queries\.(\d+)", k)
    if m:
        return f"model.layers.{m[1]}.mlp_input.query"
    if k == "model.llm.attn_res.output_query":
        return "model.final_input.query"
    for pat, rep in RENAMES:
        k = re.sub(pat, rep, k)
    return k


class Src:
    def __init__(self, path):
        self.path = path
        self.where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, k):
        f = self.where[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), "pt")
        return self.files[f].get_tensor(k)

    def layer_keys(self, L):
        p = f"model.llm.layers.{L}."
        ks = [k for k in self.where if k.startswith(p) and ".mlp.experts." not in k]
        q = f"model.llm.attn_res.attn_queries.{L - 1}"
        if L > 0 and q in self.where:
            ks.append(q)
        ks.append(f"model.llm.attn_res.mlp_queries.{L}")
        return ks


class StreamExperts(nn.Module):
    """Inkling2Experts' forward over the checkpoint's own expert bytes, each expert decoded when a token picks it"""

    def __init__(self, src, L, device, dtype, act_fn):
        super().__init__()
        p = f"model.llm.layers.{L}.mlp.experts."
        self.nv = src.get(p + "w13_weight").dtype == torch.uint8
        self.t = {}
        for proj in ("w13", "w2"):
            self.t[proj] = src.get(p + proj + "_weight").to(device)
            self.t[proj + "_s"] = src.get(p + proj + "_weight_scale").to(device)
            if self.nv:
                self.t[proj + "_o"] = (src.get(p + proj + "_weight_amax").float().to(device) /
                                       src.get(p + proj + "_weight_outer_scale_denominator").float().to(device))
        self.lut = torch.tensor(E2M1, device=device)
        self.dtype, self.act_fn = dtype, act_fn
        self.num_experts = self.t["w13"].shape[0]

    def weight(self, proj, e):
        w, s = self.t[proj][e], self.t[proj + "_s"][e]
        if self.nv:
            codes = torch.stack([w & 0x0F, w >> 4], -1).reshape(w.shape[0], -1).long()
            return (self.lut[codes] * s.float().repeat_interleave(16, dim=1) * self.t[proj + "_o"][e]).to(self.dtype)
        return (w.float() * torch.exp2(s.view(torch.uint8).float() - 127).repeat_interleave(32, dim=1)).to(self.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        for e in torch.unique(top_k_index).tolist():
            tok, pos = torch.where(top_k_index == e)
            x = hidden_states[tok]
            gu = F.linear(x, self.weight("w13", e))
            h = self.act_fn(gu[:, 0::2], gu[:, 1::2])      # (w13's rows: gate even, up odd)
            y = F.linear(h, self.weight("w2", e)) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, y.to(torch.float32))
        return final.to(hidden_states.dtype)


def mask(cache, L, q_pos, sliding, window, device, dtype):
    """the additive mask of a chunk at q_pos against layer L's cache (after this chunk's keys are in)"""
    kv_len, kv_off = cache.get_mask_sizes(len(q_pos), L)
    kv = torch.arange(kv_len, device=device) + kv_off
    d = q_pos[:, None] - kv[None, :]
    ok = (d >= 0) & ((d < window) if sliding else True)
    return torch.zeros(ok.shape, device=device, dtype=dtype).masked_fill(~ok, torch.finfo(dtype).min)[None, None]


def main():
    dev = torch.device(f"cuda:{a.gpu}")
    dtype = getattr(torch, a.dtype)
    cfg = AutoConfig.from_pretrained(a.src).text_config
    cfg._attn_implementation = "eager"
    src = Src(a.src)
    ids = torch.tensor(json.load(open(a.tokens)), device=dev)
    T, t0 = len(ids), time.time()
    torch.set_grad_enabled(False)

    def load(module, keys, prefix):
        sd = {rename(k)[len(prefix):]: src.get(k).to(dev, dtype) for k in keys}
        missing, unexpected = module.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("mlp.experts.") and "e_score_correction_bias" not in m]
        assert not missing and not unexpected, (missing, unexpected)
        if "mlp.gate.e_score_correction_bias" in sd:   # (a buffer: load_state_dict sets it, kept fp32)
            module.mlp.gate.e_score_correction_bias = src.get(
                [k for k in keys if k.endswith("mlp.gate.bias")][0]).float().to(dev)

    emb = M.Inkling2NormedEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, cfg.rms_norm_eps)
    load(emb.to(dev, dtype), ["model.llm.embed.weight", "model.llm.embed_norm.weight"], "model.embed_tokens.")
    hidden = emb(ids[None])
    del emb
    block_states = hidden.new_zeros(1, T, 0, cfg.hidden_size)
    n_layers = a.layers or cfg.num_hidden_layers
    for L in range(n_layers):
        tl = time.time()
        full = cfg.layer_types[L] != "hybrid_sliding"
        layer = (M.Inkling2FullDecoderLayer if full else M.Inkling2SlidingDecoderLayer)(cfg, L)
        layer = layer.to(dev, dtype).eval()
        load(layer, src.layer_keys(L), f"model.layers.{L}.")
        layer.mlp.experts = StreamExperts(src, L, dev, dtype, layer.mlp.experts.act_fn)
        cache = DynamicCache(config=cfg)
        out = torch.empty_like(hidden)
        for c0 in range(0, T, a.chunk):
            q = torch.arange(c0, min(T, c0 + a.chunk), device=dev)
            # (the mask needs this chunk's keys in the cache: a dry update would change it, so the sizes are the
            # cache's after the update - get_mask_sizes is asked before it, as the model asks, with the chunk's length)
            m = mask(cache, L, q, not full, cfg.sliding_window_size, dev, dtype)
            out[:, q] = layer(hidden[:, q], block_states[:, q], attention_mask=m, conv_mask=None,
                              past_key_values=cache, position_ids=q[None])
        if layer.is_block_boundary:
            block_states = torch.cat([block_states, hidden.unsqueeze(-2)], dim=-2)
        hidden = out
        del layer, cache
        torch.cuda.empty_cache()
        print(f"layer {L} ({'full' if full else 'sliding'}) {time.time() - tl:.1f} s", file=sys.stderr, flush=True)
    fin = M.Inkling2BlockAttentionResidual(cfg).to(dev, dtype)
    fin.query.data = src.get("model.llm.attn_res.output_query").to(dev, dtype)
    norm = M.Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev, dtype)
    norm.weight.data = src.get("model.llm.norm.weight").to(dev, dtype)
    hidden = norm(fin(block_states, hidden)[0])
    head = src.get("model.llm.unembed.weight").to(dev, dtype)
    V = cfg.unpadded_vocab_size
    with open(a.out + ".logprobs.f32", "wb") as f:
        for c0 in range(0, T, 512):
            lg = F.linear(hidden[0, c0:c0 + 512], head).float()
            lg = torch.tanh(lg / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
            f.write(torch.log_softmax(lg[:, :V], -1).cpu().numpy().tobytes())
    json.dump({"tokens": ids.tolist(), "rows": T, "vocab": V, "dtype": a.dtype, "layers": n_layers,
               "seconds": round(time.time() - t0, 1)}, open(a.out + ".json", "w"))
    print(f"{T} rows in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
