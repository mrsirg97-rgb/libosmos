"""LFM3's reference forward (tests/gpu/test_lfm3_forward_gpu.hip's fixture): Liquid's own modeling code
(modeling_lfm3.py, the checkpoint's: trust_remote_code) built in F32 with eager attention, its weights the GGUF's own
values - gguf-py's dequantization (Q4_0, BF16, F32) - so a mismatch is osmos's forward, not a grid's. the ids of a
text's first --tokens tokens (the checkpoint's tokenizer.json) through the model in one pass: each layer's output at
the kept rows (the first 16, the last 64), the final norm's, the kept rows' logits (the tied head). run with
PYTHONPATH=<llama.cpp>/gguf-py and a venv holding transformers (5.12.1, the checkpoint's) and torch.

  lfm3_forward.py --ref DIR --gguf FILE --text FILE --tokens N --out FILE [--check-layer L --osmos DIR]

--check-layer L: the reference's layer L alone on osmos's own input (DIR/h<L-1>.f32, osmos's tap of layer L - 1's
output, n x D), against osmos's layer L (DIR/h<L>.f32): a divergence's layer, not its cause's downstream."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixture import write  # noqa: E402

KEEP_FIRST, KEEP_LAST = 16, 64


def hf_name(name):
    """a GGUF tensor's name in modeling_lfm3.py's state dict (None: not the model's - rope_freqs, the rotation's)"""
    top = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.embedding_norm.weight"}
    if name in top:
        return top[name]
    if not name.startswith("blk."):
        return None
    _, l, rest = name.split(".", 2)
    roles = {
        "attn_norm.weight": "operator_norm.weight", "post_attention_norm.weight": "post_operator_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight", "post_ffw_norm.weight": "post_ffn_norm.weight",
        "ffn_gate.weight": "feed_forward.w1.weight", "ffn_up.weight": "feed_forward.w3.weight",
        "ffn_down.weight": "feed_forward.w2.weight",
        "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_gate.weight": "self_attn.gate_proj.weight",
        "shortconv.in_proj.weight": "conv.in_proj.weight", "shortconv.out_proj.weight": "conv.out_proj.weight",
        "shortconv.conv.weight": "conv.conv.weight",
    }
    if rest not in roles:
        raise SystemExit(f"{name}: no role in modeling_lfm3.py")
    return f"model.layers.{l}.{roles[rest]}"


def gguf_state(path):
    """the GGUF's every tensor as F32, numpy's shape (ggml's dims reversed), by modeling_lfm3.py's names"""
    from gguf import GGUFReader
    from gguf.quants import dequantize

    state = {}
    for t in GGUFReader(path).tensors:
        n = hf_name(t.name)
        if n is None:
            continue
        v = dequantize(t.data, t.tensor_type).astype(np.float32).reshape([int(d) for d in reversed(t.shape)])
        if n.endswith("conv.conv.weight"):  # ggml [taps, D] -> torch's depthwise [D, 1, taps]
            v = v.reshape(v.shape[0], 1, v.shape[1])
        state[n] = torch.from_numpy(np.ascontiguousarray(v))
    return state


def build(ref, gguf):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(ref, trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
    # Liquid's class, never transformers' built-in Lfm3 (it rotates all 64 pairs: not the model - DESIGN.md)
    if not type(model).__module__.startswith("transformers_modules"):
        raise SystemExit(f"{type(model).__module__}: not the checkpoint's own modeling code")
    state = gguf_state(gguf)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied: the embedding's
    if missing or unexpected:
        raise SystemExit(f"weights not the model's: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model.tie_weights()
    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    return model.eval(), config


def q8_0(x):
    """ggml's quantize_row_q8_0 then its dequantization, blocks of 32 along the last dim: d = amax / 127 (its f16
    stored), q = round(x / d) half away from zero"""
    shape = x.shape
    b = x.float().reshape(-1, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.sign(b * inv) * torch.floor(torch.abs(b * inv) + 0.5)
    return (q * d.half().float()).reshape(shape)


def kv_q8(model):
    """the attention's K and V rounded as osmos's cache rounds them (its eager attention patched: the same call)"""
    mod = sys.modules[type(model).__module__]
    eager = mod.eager_attention_forward

    def rounded(module, query, key, value, *args, **kwargs):
        return eager(module, query, q8_0(key), q8_0(value), *args, **kwargs)

    mod.eager_attention_forward = rounded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the checkpoint's config.json, configuration / modeling code")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-q8", action="store_true",
                    help="K and V through ggml's Q8_0 before the attention (osmos's cache: q4x_kv_put's rounding)")
    ap.add_argument("--check-layer", type=int, default=-1)
    ap.add_argument("--osmos", help="osmos's dumped streams (DIR/h<L>.f32)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(os.path.join(a.ref, "tokenizer.json")).encode(open(a.text).read()).ids[: a.tokens]
    T = len(ids)
    model, config = build(a.ref, a.gguf)
    if a.kv_q8:
        kv_q8(model)
    D, L = config.hidden_size, config.num_hidden_layers

    if a.check_layer >= 0:
        l = a.check_layer
        prev = np.fromfile(os.path.join(a.osmos, f"h{l - 1}.f32"), np.float32).reshape(T, D) if l else None
        got = np.fromfile(os.path.join(a.osmos, f"h{l}.f32"), np.float32).reshape(T, D)
        outs = {}

        def grab(i):
            def hook(mod, args, kwargs, out):
                outs[i] = out[0] if isinstance(out, tuple) else out
            return hook

        def feed(mod, args, kwargs):  # layer l's input: osmos's
            if prev is None:
                return None
            hs = torch.from_numpy(prev)[None]
            return (hs,) + tuple(args[1:]), kwargs if "hidden_states" not in kwargs else {**kwargs, "hidden_states": hs}

        layer = model.model.layers[l]
        layer.register_forward_hook(grab(l), with_kwargs=True)
        layer.register_forward_pre_hook(feed, with_kwargs=True)
        model.model(input_ids=torch.tensor([ids]), use_cache=False)
        ref = outs[l][0].numpy()
        err = np.abs(ref - got).max()
        print(f"layer {l} on osmos's input: max error {err:.3g} of {np.abs(ref).max():.4g} "
              f"({err / np.abs(ref).max():.2g})")
        return

    rows = sorted(set(range(min(KEEP_FIRST, T))) | set(range(max(0, T - KEEP_LAST), T)))
    taps = np.zeros((L, len(rows), D), np.float32)

    def tap(i):
        def hook(mod, args, out):
            taps[i] = (out[0] if isinstance(out, tuple) else out)[0, rows].float().numpy()
        return hook

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(tap(i))
    h = model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, rows]
    logits = (h @ model.lm_head.weight.T).numpy()
    lp = torch.log_softmax(torch.from_numpy(logits).double(), -1)
    nxt = [(k, r) for k, r in enumerate(rows) if r + 1 < T]
    nll = -np.mean([lp[k, ids[r + 1]].item() for k, r in nxt])
    top1 = np.mean([lp[k].argmax().item() == ids[r + 1] for k, r in nxt])
    print(f"{T} ids; the kept rows' next id: mean NLL {nll:.3f} ({len(nxt)} rows), top-1 {top1:.3f}")
    write(a.out, {"ids": np.array(ids, np.float32), "rows": np.array(rows, np.float32), "taps": taps,
                  "final": h.numpy(), "logits": logits})
    print(f"{a.out}: {T} ids, {len(rows)} rows kept, {L} layers' taps")


if __name__ == "__main__":
    main()
