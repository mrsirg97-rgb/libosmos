"""ink2_mtp_gguf.py - Inkling-2's drafter (Small's, Mini's) (the checkpoint's mtp.safetensors: DSpark, the RNN Markov head) as a
GGUF osmos reads (general.architecture inkling2_mtp): every tensor under its checkpoint name, the matrices BF16 (the
checkpoint's values) or Q8_0, the norms, mask, fuse weights and the Markov head's small tensors F32, and mtp_config's
geometry as inkling2_mtp.* keys. the drafter's forward: tools/convert/ink2_mtp_oracle.py (TokenSpeed's math).

    PYTHONPATH=<llama.cpp>/gguf-py python tools/convert/ink2_mtp_gguf.py <mtp.safetensors> <config.json> <out.gguf>
        [--q8]
"""
import argparse
import json
import sys

import numpy as np
from safetensors.torch import load_file

import gguf
from gguf import GGMLQuantizationType as T
from gguf.quants import quantize

ap = argparse.ArgumentParser()
ap.add_argument("mtp")
ap.add_argument("config")
ap.add_argument("out")
ap.add_argument("--q8", action="store_true")
ap.add_argument("--source", default="thinkingmachines/Inkling-2-Small", help="the checkpoint's name (provenance)")
a = ap.parse_args()
cfg = json.load(open(a.config))
m = cfg["mtp_config"]
need = {"head_dim": 128, "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden",
        "sliding_window_size": 1024}   # (the kernels'; its width and heads osmos checks against the target's)
for k, v in need.items():
    if m.get(k) != v:
        sys.exit(f"mtp_config.{k} {m.get(k)!r}: this converter writes {v!r} (osmos's drafter's shapes)")
W = load_file(a.mtp)
MATS = ("q_proj", "k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx", "o_proj", "gate_proj", "up_proj", "down_proj")
w = gguf.GGUFWriter(a.out, "inkling2_mtp")
w.add_uint32("inkling2_mtp.block_count", m["num_hidden_layers"])
w.add_uint32("inkling2_mtp.embedding_length", m["hidden_size"])
w.add_uint32("inkling2_mtp.feed_forward_length", m["intermediate_size"])
w.add_uint32("inkling2_mtp.attention.head_count", m["num_attention_heads"])
w.add_uint32("inkling2_mtp.attention.head_count_kv", m["num_key_value_heads"])
w.add_uint32("inkling2_mtp.attention.key_length", m["head_dim"])
w.add_float32("inkling2_mtp.attention.layer_norm_rms_epsilon", m["rms_norm_eps"])
w.add_float32("inkling2_mtp.rope.freq_base", m["rope_theta"])
w.add_uint32("inkling2_mtp.attention.sliding_window", m["sliding_window_size"])
w.add_array("inkling2_mtp.local_layers", [int(x) for x in m["local_layer_ids"]])
w.add_uint32("inkling2_mtp.block_size", m["block_size"])
w.add_uint32("inkling2_mtp.markov_rank", m["markov_rank"])
w.add_array("inkling2_mtp.target_layers", [int(x) for x in m["target_layer_ids"]])
w.add_string("inkling2_mtp.source", f"{a.source} mtp.safetensors")
n_bytes = 0
for name in sorted(W):
    t = W[name].float().numpy()
    if any(name.endswith(f".{k}.weight") for k in MATS):
        if a.q8:
            q = quantize(t.astype(np.float32), T.Q8_0)
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=T.Q8_0)
            n_bytes += q.nbytes
        else:  # BF16: the checkpoint's own values (the top 16 bits of each F32)
            b = (t.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            w.add_tensor(name, b.view(np.int16), raw_shape=b.shape, raw_dtype=T.BF16)
            n_bytes += b.nbytes
    elif name == "markov_head.markov_w1.weight":  # a row gather: BF16 as given
        b = (t.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        w.add_tensor(name, b.view(np.int16), raw_shape=b.shape, raw_dtype=T.BF16)
        n_bytes += b.nbytes
    else:
        w.add_tensor(name, t.astype(np.float32))
        n_bytes += t.nbytes
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{a.out}: {len(W)} tensors, {n_bytes / 1e9:.2f} GB ({'Q8_0' if a.q8 else 'BF16'} matrices)")
