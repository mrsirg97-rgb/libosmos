"""reference MLA for tests/gpu/test_hy4_mla_gpu.hip: Hy4-flash's gated MLA by the reference's own code - the
transformers PR #50903 modeling_hy_v4_flash.py's HYV4FlashAttention.forward (its rotary, eager attention with the
learned sinks, the sigmoid gate, the sandwich norm; a sparse layer's indexer present, dense at these lengths), its
source compiled alone (hy4_pieces.reference), in F32 on the GGUF's own weights dequantised (kv_b_proj rebuilt from
the file's k_b and v_b as the converter split it). T tokens from position 0 through a sliding layer (past its 512-key
window) and a sparse one; each step's tensors kept by hooks: the projections' outputs (q_b, kv_a, gate), the gated
attention entering o_proj, the layer's output. the file's k_b and v_b bytes (Q8_0) beside them, as the GPU reads them.
run: python -I hy4_mla.py --modeling modeling_hy_v4_flash.py --config config.json --gguf Hy4-flash-Q3_K-00001-of-00002.gguf
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

T = 600
SEED = 1119
LAYERS = (2, 5)  # a sliding layer, a sparse one
NAMES = ["HYV4FlashRMSNorm", "HYV4FlashRotaryEmbedding", "rotate_half", "apply_rotary_pos_emb",
         "_apply_partial_rotary_pos_emb", "HYV4FlashCompressedIndexer", "repeat_kv", "eager_attention_forward",
         "HYV4FlashAttention"]


class Interfaces:
    """ALL_ATTENTION_FUNCTIONS: the eager path, the reference's own"""

    @staticmethod
    def get_interface(name, default):
        return default


def param(t):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)), requires_grad=False)


def make_config(c):
    """the reference's text config as its attention reads it (config.json's text_config, its layer kinds
    renamed as the configuration class does)"""
    sliding = c["swa_pattern"]
    rope = {"indexed_attention": c["rope_parameters"]["deepseek_sparse_attention"],
            "sliding_attention": c["rope_parameters"]["sliding_attention"]}
    rope = {k: {"rope_type": "default", "rope_theta": v["rope_theta"], "mrope_section": v["mrope_section"]}
            for k, v in rope.items()}
    config = types.SimpleNamespace(
        layer_types=["sliding_attention" if s else "indexed_attention" for s in sliding], rope_parameters=rope,
        max_position_embeddings=c["max_position_embeddings"], hidden_size=c["hidden_size"],
        rms_norm_eps=c["rms_norm_eps"], attention_bias=False, attention_dropout=0.0,
        num_attention_heads=c["num_attention_heads"], swa_num_attention_heads=c["swa_num_attention_heads"],
        qk_nope_head_dim=c["qk_nope_head_dim"], swa_qk_nope_head_dim=c["swa_qk_nope_head_dim"],
        qk_rope_head_dim=c["qk_rope_head_dim"], swa_qk_rope_head_dim=c["swa_qk_rope_head_dim"],
        v_head_dim=c["v_head_dim"], swa_v_head_dim=c["swa_v_head_dim"], q_lora_rank=c["q_lora_rank"],
        kv_lora_rank=c["kv_lora_rank"], learnable_sink_init=0.0, _attn_implementation="eager",
        compressed_indexer_compress_ratio=c["compressed_indexer_compress_ratio"],
        compressed_indexer_num_attention_heads=c["compressed_indexer_num_attention_heads"],
        compressed_indexer_head_dim=c["compressed_indexer_head_dim"],
        compressed_indexer_group_topk=c["compressed_indexer_group_topk"],
        candidate_topk=c["compressed_indexer_group_topk"] * c["compressed_indexer_compress_ratio"],
        sliding_window=c["sliding_window"])
    return config


def attention(R, config, g, L):
    """layer L's HYV4FlashAttention on the GGUF's weights (kv_b_proj rebuilt from k_b and v_b)"""
    blk = f"blk.{L}."
    att = R["HYV4FlashAttention"](config, L)
    H, nope, vd = att.num_heads, att.qk_nope_head_dim, att.v_head_dim
    set_ = lambda m, name, t: setattr(m, name, param(t))  # noqa: E731
    set_(att.q_a_proj, "weight", g.tensor(blk + "attn_q_a.weight"))
    set_(att.q_a_layernorm, "weight", g.tensor(blk + "attn_q_a_norm.weight"))
    set_(att.q_b_proj, "weight", g.tensor(blk + "attn_q_b.weight"))
    set_(att.kv_a_proj_with_mqa, "weight", g.tensor(blk + "attn_kv_a_mqa.weight"))
    set_(att.kv_a_layernorm, "weight", g.tensor(blk + "attn_kv_a_norm.weight"))
    k_b, v_b = g.tensor(blk + "attn_k_b.weight"), g.tensor(blk + "attn_v_b.weight")  # [H][512][192], [H][256][512]
    set_(att.kv_b_proj, "weight", np.concatenate([np.concatenate([k_b[h].T, v_b[h]]) for h in range(H)]))
    set_(att.gate_proj, "weight", g.tensor(blk + "attn_gate.weight"))
    set_(att.o_proj, "weight", g.tensor(blk + "attn_output.weight"))
    set_(att.post_proj_norm, "weight", g.tensor(blk + "post_attention_norm.weight"))
    att.sinks = param(g.tensor(blk + "attn_sinks.weight"))
    if att.compressed_indexer is not None:
        ix = att.compressed_indexer
        set_(ix.query_projection, "weight", g.tensor(blk + "indexer.attn_q_b.weight"))
        set_(ix.head_weight_projection, "weight", g.tensor(blk + "indexer.proj.weight"))
        set_(ix.token_key_projection, "weight", g.tensor(blk + "indexer.attn_k.weight"))
        set_(ix.token_key_norm, "weight", g.tensor(blk + "indexer.k_norm.weight"))
        set_(ix.key_value_projection, "weight", g.tensor(blk + "indexer_compressor_value.weight"))
        set_(ix.key_gate_projection, "weight", g.tensor(blk + "indexer_compressor_gate.weight"))
        set_(ix.key_norm, "weight", g.tensor(blk + "indexer_compressor_norm.weight"))
        ix.intra_group_ape = param(g.tensor(blk + "indexer_compressor_ape.weight"))
    return att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeling", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    R = P.reference(a.modeling, NAMES, {"ALL_ATTENTION_FUNCTIONS": Interfaces, "ROPE_INIT_FUNCTIONS": {},
                                        "dynamic_rope_update": None, "Callable": object})
    c = json.load(open(a.config))["text_config"]
    g = Gguf(a.gguf)
    config = make_config(c)
    rotary = R["HYV4FlashRotaryEmbedding"](config)
    rng = np.random.default_rng(SEED)
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    arrays = {}
    for L in LAYERS:
        blk, kind = f"blk.{L}.", config.layer_types[L]
        att = attention(R, config, g, L)
        H, nope, vd = att.num_heads, att.qk_nope_head_dim, att.v_head_dim
        seen = {}
        hook = lambda key: (lambda m, i, o: seen.__setitem__(key, o.detach().clone()))  # noqa: E731
        att.q_b_proj.register_forward_hook(hook("q"))
        att.kv_a_proj_with_mqa.register_forward_hook(hook("kva"))
        att.gate_proj.register_forward_hook(hook("gate"))
        att.o_proj.register_forward_pre_hook(lambda m, i: seen.__setitem__("o_in", i[0].detach().clone()))
        x = torch.from_numpy(rng.normal(0, 1.0, size=(1, T, config.hidden_size)).astype(np.float32))
        i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
        allowed = (j <= i) & ((i - j < config.sliding_window) if kind == "sliding_attention" else True)
        mask = torch.zeros(1, 1, T, T).masked_fill(~allowed, torch.finfo(torch.float32).min)
        out, _ = att(hidden_states=x, position_embeddings=rotary(x, pos, kind), attention_mask=mask, cache=None,
                     position_ids=pos)
        p = f"L{L}."
        arrays.update({p + "x": x[0].numpy(), p + "q": seen["q"][0].numpy(), p + "kva": seen["kva"][0].numpy(),
                       p + "gate": seen["gate"][0].numpy(), p + "o_in": seen["o_in"][0].numpy(),
                       p + "out": out[0].numpy(), p + "kv_a_norm": g.tensor(blk + "attn_kv_a_norm.weight"),
                       p + "sinks": g.tensor(blk + "attn_sinks.weight"),
                       p + "k_b_q8": g.raw(blk + "attn_k_b.weight").view(np.float32),
                       p + "v_b_q8": g.raw(blk + "attn_v_b.weight").view(np.float32),
                       p + "rope_base": np.array([config.rope_parameters[kind]["rope_theta"]], dtype=np.float32),
                       p + "shape": np.array([H, nope, vd, config.kv_lora_rank, config.qk_rope_head_dim,
                                              config.sliding_window if kind == "sliding_attention" else 0],
                                             dtype=np.float32)})
        print(f"layer {L} ({kind}, {H} heads): o_in |mean| {seen['o_in'].abs().mean():.4g}, "
              f"out |mean| {out.abs().mean():.4g}")
    P.write(a.out, arrays)
    print(f"{a.out}: {len(arrays)} arrays, {T} tokens")


if __name__ == "__main__":
    main()
