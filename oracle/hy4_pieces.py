"""reference pieces for tests/gpu/test_hy4_hc_gpu.hip: Hy4-flash's iHC (the mixes, pre, post), its head's collapse
and Engram's layer, by the reference's own code - the transformers PR #50903 modeling_hy_v4_flash.py's classes
(HYV4FlashDecoderLayer.forward with its attention and MLP stubbed to fixed outputs: the iHC, the norms, Engram and
the residuals as it runs them; HYV4FlashHyperHead; HYV4FlashEngramLayer), their source compiled alone with torch and
numpy (the module's other imports are transformers' internals: stubs). in F32 throughout, on the GGUF's own weights
dequantised (hy4_gguf.py) - the oracle and osmos read the same values, so what differs is the summation order alone.
Engram's rows are the GGUF table's at the reference's hash (its lookup_table the release's).

one file of named F32 arrays: b"HY4P", u32 version, u32 count, then each: u32 name length, the name, u32 dims, the
dims (u64, outermost first), the values.
run: python -I hy4_pieces.py --modeling modeling_hy_v4_flash.py --config config.json --shard
     model-00147-of-00148.safetensors --gguf Hy4-flash-SPARK-4Bit.gguf --out FILE"""
import argparse
import ast
import contextlib
import json
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy4_engram_rows as hashes  # noqa: E402
from hy4_gguf import Gguf  # noqa: E402

VERSION = 1
T = 16
SEED = 1118
LAYER = 1  # a sparse layer with Engram before its iHC
NAMES = ["HYV4FlashRMSNorm", "HYV4FlashUnweightedRMSNorm", "HYV4FlashHyperConnection", "HYV4FlashHyperHead",
         "HYV4FlashEngramEmbedding", "_fp8_round_trip", "HYV4FlashEngramLayer", "HYV4FlashDecoderLayer"]


class Stub:
    """a name the reference's signatures use (annotations, bases' arguments): subscriptable, unionable"""

    def __class_getitem__(cls, item):
        return cls


def reference(path):
    """the classes and functions in NAMES from the reference's file, compiled alone, their decorators dropped"""
    scope = {"torch": torch, "nn": nn, "F": F, "np": np, "maybe_autocast": lambda **k: contextlib.nullcontext(),
             "GradientCheckpointingLayer": nn.Module, "Cache": Stub, "Unpack": Stub, "TransformersKwargs": Stub,
             "HYV4FlashConfig": Stub, "HYV4FlashTextConfig": Stub, "deprecate_kwarg": None}
    nodes = []
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NAMES:
            node.decorator_list = []
            for fn in getattr(node, "body", []):
                if isinstance(fn, ast.FunctionDef):
                    fn.decorator_list = []
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)
    missing = [n for n in NAMES if n not in scope]
    if missing:
        raise SystemExit(f"{missing} not in {path}")
    return scope


class Fixed(nn.Module):
    """a sublayer stubbed: its input kept, a fixed output returned (the attention's as a tuple)"""

    def __init__(self, out, pair):
        super().__init__()
        self.out, self.pair, self.seen = out, pair, None

    def forward(self, hidden_states=None, **kwargs):
        self.seen = hidden_states.detach().clone()
        return (self.out, None) if self.pair else self.out


def param(t):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)), requires_grad=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeling", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    R = reference(a.modeling)
    cfg = json.load(open(a.config))["text_config"]
    g = Gguf(a.gguf)
    D, hc = cfg["hidden_size"], cfg["hc_mult"]
    n_gram, heads, pad, vocab = (cfg["engram_max_ngram_size"], cfg["engram_n_head_per_ngram"], cfg["engram_pad_id"],
                                 cfg["vocab_size"])
    table = hashes.lookup_table(a.shard, 11).astype(np.int64)
    config = types.SimpleNamespace(hc_mult=hc, hc_eps=cfg["hc_eps"], hc_magnitude=cfg["hc_magnitude"],
                                   rms_norm_eps=cfg["rms_norm_eps"], hidden_size=D,
                                   engram_max_ngram_size=n_gram, engram_n_head_per_ngram=heads,
                                   engram_fp8_embedding=cfg["engram_fp8_embedding"], engram_pad_id=pad,
                                   engram_seed=cfg.get("engram_seed", 0),
                                   engram_layer_head_vocab_sizes=cfg["engram_layer_head_vocab_sizes"],
                                   engram_n_embed_per_ngram=cfg["engram_n_embed_per_ngram"],
                                   engram_lookup_table_size=len(table))
    blk = f"blk.{LAYER}."

    # the tokens, the streams entering the layer (each a token's embedding, the four apart by seeded noise), the
    # attention's and the MLP's fixed outputs (the embedding's scale)
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(table), size=T)
    ids[3], ids[7] = pad, vocab - 1
    emb = g.rows("token_embd.weight", ids)
    sd = float(emb.std())
    h = emb[:, None, :] + rng.normal(0, 0.1 * sd, size=(T, hc, D)).astype(np.float32)
    y_attn = rng.normal(0, sd, size=(T, D)).astype(np.float32)
    y_ffn = rng.normal(0, sd, size=(T, D)).astype(np.float32)

    # iHC's weights (fn Q8_0 in the file: its dequantised values, the same both sides)
    def hc_module(prefix):
        m = R["HYV4FlashHyperConnection"](config)
        m.fn = param(g.tensor(blk + prefix + "_fn.weight"))
        m.base = param(g.tensor(blk + prefix + "_base.weight"))
        m.scale = param(g.tensor(blk + prefix + "_scale.weight"))
        return m

    def rms(name):
        m = R["HYV4FlashRMSNorm"](D, eps=cfg["rms_norm_eps"])
        m.weight = param(g.tensor(name))
        return m

    # Engram: built on the meta device (its table is 63 GB), then its small parts set from the file; the table's
    # rows the reference asks for read from the GGUF's, at the GGUF's offsets for the reference's hash
    lid = LAYER + 1
    with torch.device("meta"):
        eg = R["HYV4FlashEngramLayer"](config, lid)
    sizes = [s for order in cfg["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
    eg.head_vocab_sizes = torch.tensor(sizes)
    eg.lookup_table = torch.from_numpy(table.astype(np.int32))
    eg.hash_multipliers = torch.from_numpy(hashes.draw_hash_multipliers(config.engram_seed, lid, n_gram,
                                                                        int(table.max()) + 1))
    first = g.kv["hy_v4_flash.ple.head_offsets"][g.kv["hy_v4_flash.ple.layers"].index(LAYER) * len(sizes)]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]) + first
    gathered = {}

    def lookup(hash_ids):
        rows = (hash_ids.clamp_min(0).numpy() + offsets).reshape(-1)
        out = torch.from_numpy(g.rows("per_layer_token_embd.weight", rows)).reshape(*hash_ids.shape, -1)
        gathered["rows"] = out.clone()
        return out.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)

    del eg.multi_head_embedding
    eg.multi_head_embedding = lookup
    norms = g.tensor(blk + "engram_lookup_norm.weight")
    for c, n in enumerate(eg.lookup_per_head_norms):
        n.weight = param(norms[c])
    kv_w, kv_b = g.tensor(blk + "engram_kv.weight"), g.tensor(blk + "engram_kv_b.weight")
    for c in range(hc):
        eg.key_projs[c].weight, eg.key_projs[c].bias = param(kv_w[c * D:(c + 1) * D]), param(kv_b[c])
        eg.norm_key[c].weight = param(g.tensor(blk + "engram_gate.weight")[c])  # the file's: key's x query's
        eg.norm_query[c].weight = param(np.ones(D))
    eg.value_proj.weight, eg.value_proj.bias = param(kv_w[hc * D:]), param(kv_b[hc])
    eg.unified_pre_gamma = param(g.tensor(blk + "engram_pre_gamma.weight"))
    for c, s in enumerate(g.tensor(blk + "engram_hc_scale.weight")):
        eg.unified_per_hc_scales[c] = param(s)

    # the decoder layer by the reference's forward: its attention and MLP the fixed outputs
    layer = R["HYV4FlashDecoderLayer"].__new__(R["HYV4FlashDecoderLayer"])
    nn.Module.__init__(layer)
    layer.layer_idx, layer.attention_type = LAYER, "indexed_attention"
    layer.attn_hc, layer.ffn_hc = hc_module("hc_attn"), hc_module("hc_ffn")
    layer.input_layernorm, layer.post_attention_layernorm = rms(blk + "attn_norm.weight"), rms(blk + "ffn_norm.weight")
    layer.self_attn = Fixed(torch.from_numpy(y_attn)[None], True)
    layer.mlp = Fixed(torch.from_numpy(y_ffn)[None], False)
    layer.engram_layer = eg
    ngram = torch.from_numpy(hashes.build_ngram_tokens(ids, n_gram, pad))[None]
    h_in = torch.from_numpy(h)[None]
    out = layer.forward(h_in, ngram_tokens=ngram, engram_mask=torch.ones(1, T, dtype=torch.bool))

    # the pieces between: Engram's output alone, its embedding after the round trip and the norms, iHC's pre and post
    eg_out = eg.forward(h_in, ngram, torch.ones(1, T, dtype=torch.bool))[0]
    h_eg = h_in + eg_out[None]
    post_attn, _ = layer.attn_hc(h_eg)
    hc_head = R["HYV4FlashHyperHead"](config)
    hc_head.hc_fn = param(g.tensor("output_hc_fn.weight"))
    hc_head.hc_base = param(g.tensor("output_hc_base.weight"))
    hc_head.hc_scale = param(g.tensor("output_hc_scale.weight"))
    head = rms("output_norm.weight")(hc_head(h_in))[0]

    arrays = {
        "ids": ids.astype(np.float32), "h": h, "y_attn": y_attn, "y_ffn": y_ffn,
        "hc_attn_fn": g.tensor(blk + "hc_attn_fn.weight"), "hc_attn_base": g.tensor(blk + "hc_attn_base.weight"),
        "hc_attn_scale": g.tensor(blk + "hc_attn_scale.weight"), "attn_norm": g.tensor(blk + "attn_norm.weight"),
        "hc_ffn_fn": g.tensor(blk + "hc_ffn_fn.weight"), "hc_ffn_base": g.tensor(blk + "hc_ffn_base.weight"),
        "hc_ffn_scale": g.tensor(blk + "hc_ffn_scale.weight"), "ffn_norm": g.tensor(blk + "ffn_norm.weight"),
        "out_hc_fn": g.tensor("output_hc_fn.weight"), "out_hc_base": g.tensor("output_hc_base.weight"),
        "out_hc_scale": g.tensor("output_hc_scale.weight"), "output_norm": g.tensor("output_norm.weight"),
        "eg_rows": gathered["rows"][0].numpy(), "eg_lookup_norm": norms, "eg_kv": kv_w, "eg_kv_b": kv_b,
        "eg_gate": g.tensor(blk + "engram_gate.weight"), "eg_pre_gamma": g.tensor(blk + "engram_pre_gamma.weight"),
        "eg_hc_scale": g.tensor(blk + "engram_hc_scale.weight"),
        # the reference's results
        "eg_out": eg_out.numpy(), "h_engram": h_eg[0].numpy(),
        "attn_in": layer.self_attn.seen[0].numpy(), "attn_post": post_attn[0].numpy(),
        "ffn_in": layer.mlp.seen[0].numpy(), "h_out": out[0].numpy(), "head": head.numpy(),
    }
    with open(a.out, "wb") as f:
        f.write(b"HY4P" + struct.pack("<2I", VERSION, len(arrays)))
        for name, v in arrays.items():
            v = np.ascontiguousarray(v, dtype=np.float32)
            f.write(struct.pack("<I", len(name)) + name.encode() + struct.pack("<I", v.ndim))
            f.write(struct.pack(f"<{v.ndim}Q", *v.shape) + v.tobytes())
    print(f"{a.out}: {len(arrays)} arrays; {T} tokens, layer {LAYER}; h_out |max| {np.abs(arrays['h_out']).max():.4g}")


if __name__ == "__main__":
    main()
