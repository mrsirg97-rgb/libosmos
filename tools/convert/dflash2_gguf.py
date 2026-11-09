#!/usr/bin/env python3
"""tools/convert/dflash2_gguf.py - incoai/Qwen3.8-Flash-Next-DFlash2 (model.safetensors, config.json: BF16, 1.2 GB) as one
GGUF osmos's Qwen forward drafts with (osmos/q4x/dflash.h): its linear weights Q8_0 (the MTP head's type), its norms and
the dynamic convolutions' base kernels F32, the selector's two codebooks Q8_0 rows (a token's 256 gathered by its id).
every shape the config's, checked; every size osmos needs in the metadata - nothing implied by a name.

    python3 tools/convert/dflash2_gguf.py <the drafter's directory> <out.gguf> [--type Q8_0|BF16]

the names (llama.cpp's where it has one, the MTP head's GGUF the same):
    fc.weight [12800 -> 2560]  hidden_norm.weight  output_norm.weight (config's norm)
    blk.N.attn_norm (input_layernorm)  .attn_q .attn_k .attn_v .attn_output  .attn_q_norm .attn_k_norm
    blk.N.ffn_norm (post_attention_layernorm)  .ffn_gate .ffn_up .ffn_down
    blk.N.attn_conv (attention_conv.base_kernel, [2][2][2560])  .attn_conv_proj (its kernel_projection [2560 -> 640])
    blk.N.ffn_conv  .ffn_conv_proj (mlp_conv's)
    sel_proj.weight (candidate_selector.hidden_projection [2560 -> 256])  sel_pred.weight, sel_succ.weight (the
    predecessor / successor codebooks, [vocab][256])
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.environ.get("OSMOS_GGUF_PY", os.path.expanduser("~/Projects/llama.cpp/gguf-py")))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

ARCH = "dflash2"


def safetensors(path):
    """name -> f32 array, from a safetensors file of BF16 tensors (parsed: no code of the download's runs)"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        out = {}
        for k, v in head.items():
            if k == "__metadata__":
                continue
            if v["dtype"] != "BF16":
                raise SystemExit(f"{k}: {v['dtype']}, not BF16")
            a, b = v["data_offsets"]
            f.seek(8 + n + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
            out[k] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
        return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    qtype = sys.argv[sys.argv.index("--type") + 1] if "--type" in sys.argv else "Q8_0"
    linear = {"Q8_0": gguf.GGMLQuantizationType.Q8_0, "BF16": gguf.GGMLQuantizationType.BF16}[qtype]
    cfg = json.load(open(os.path.join(src, "config.json")))
    dc = cfg["dflash_config"]
    if cfg["architectures"] != ["DFlash2DraftModel"] or cfg["model_type"] != "qwen3":
        raise SystemExit(f"not a DFlash2 draft model: {cfg['architectures']} {cfg['model_type']}")
    W = safetensors(os.path.join(src, "model.safetensors"))
    D, L, H, KV, HD, FF = (cfg[k] for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                            "num_key_value_heads", "head_dim", "intermediate_size"))
    V, T, G, K, R = cfg["vocab_size"], len(dc["target_layer_ids"]), dc["conv_group_size"], dc["conv_kernel_size"], dc["selector_rank"]
    if not all(t == "sliding_attention" for t in cfg["layer_types"]) or cfg.get("is_causal") is not False:
        raise SystemExit("not the layers osmos's drafter runs: every one sliding, none causal")
    if cfg.get("attention_bias") or cfg.get("tie_word_embeddings") or cfg["rope_parameters"].get("rope_type") != "default":
        raise SystemExit("biases, tied embeddings or a rope osmos's drafter does not run")

    w = gguf.GGUFWriter(dst, ARCH)
    w.add_name("Qwen3.8-Flash-Next-DFlash2")
    w.add_block_count(L)
    w.add_embedding_length(D)
    w.add_feed_forward_length(FF)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_rope_freq_base(cfg["rope_parameters"]["rope_theta"])
    w.add_vocab_size(V)
    w.add_uint32(f"{ARCH}.attention.sliding_window", cfg["sliding_window"])
    w.add_uint32(f"{ARCH}.block_size", dc["block_size"])
    w.add_uint32(f"{ARCH}.mask_token_id", dc["mask_token_id"])
    w.add_array(f"{ARCH}.target_layers", [int(x) for x in dc["target_layer_ids"]])
    w.add_uint32(f"{ARCH}.target_block_count", cfg["num_target_layers"])
    w.add_uint32(f"{ARCH}.conv.kernel", K)
    w.add_uint32(f"{ARCH}.conv.group", G)
    w.add_uint32(f"{ARCH}.selector.rank", R)
    w.add_uint32(f"{ARCH}.selector.top_k", dc["selector_top_k"])

    def put(name, a, shape, kind):
        a = np.ascontiguousarray(W.pop(a) if isinstance(a, str) else a, dtype=np.float32)
        if list(a.shape) != list(shape):
            raise SystemExit(f"{name}: {list(a.shape)}, the config's {list(shape)}")
        if kind == "f32":
            w.add_tensor(name, a)
        else:
            w.add_tensor(name, quants.quantize(a, linear), raw_dtype=linear)

    put("fc.weight", "fc.weight", [D, T * D], "q")
    put("hidden_norm.weight", "hidden_norm.weight", [D], "f32")
    put("output_norm.weight", "norm.weight", [D], "f32")
    for i in range(L):
        p, b = f"layers.{i}.", f"blk.{i}."
        put(b + "attn_norm.weight", p + "input_layernorm.weight", [D], "f32")
        put(b + "attn_q.weight", p + "self_attn.q_proj.weight", [H * HD, D], "q")
        put(b + "attn_k.weight", p + "self_attn.k_proj.weight", [KV * HD, D], "q")
        put(b + "attn_v.weight", p + "self_attn.v_proj.weight", [KV * HD, D], "q")
        put(b + "attn_output.weight", p + "self_attn.o_proj.weight", [D, H * HD], "q")
        put(b + "attn_q_norm.weight", p + "self_attn.q_norm.weight", [HD], "f32")
        put(b + "attn_k_norm.weight", p + "self_attn.k_norm.weight", [HD], "f32")
        put(b + "ffn_norm.weight", p + "post_attention_layernorm.weight", [D], "f32")
        put(b + "ffn_gate.weight", p + "mlp.gate_proj.weight", [FF, D], "q")
        put(b + "ffn_up.weight", p + "mlp.up_proj.weight", [FF, D], "q")
        put(b + "ffn_down.weight", p + "mlp.down_proj.weight", [D, FF], "q")
        for ours, theirs in (("attn_conv", "attention_conv"), ("ffn_conv", "mlp_conv")):
            put(b + ours + ".weight", p + theirs + ".base_kernel", [2, K, D], "f32")
            put(b + ours + "_proj.weight", p + theirs + ".kernel_projection.weight", [2 * K * (D // G), D], "q")
    put("sel_proj.weight", "candidate_selector.hidden_projection.weight", [R, D], "q")
    put("sel_pred.weight", "candidate_selector.predecessor_codebook", [V, R], "q")
    put("sel_succ.weight", "candidate_selector.successor_codebook", [V, R], "q")
    if W:
        raise SystemExit(f"tensors left that osmos's drafter does not read: {sorted(W)[:6]}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"dflash2_gguf: {dst} ({qtype}), {L} layers, {os.path.getsize(dst) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
