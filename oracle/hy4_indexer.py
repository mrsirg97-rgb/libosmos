"""reference sparse attention for tests/gpu/test_hy4_indexer_gpu.hip: Hy4-flash's compressed indexer by the
reference's own code - the transformers PR #50903 modeling_hy_v4_flash.py's HYV4FlashAttention.forward on a sparse
layer past 2,048 keys (its _mask_unrecalled_tokens: the token keys, HYV4FlashCompressedIndexer.pool_groups and
forward - the groups each query recalls - then eager attention over them), its source compiled alone
(hy4_pieces.reference), in F32 on the GGUF's own weights (hy4_mla.attention). T tokens from position 0; each step
kept: the MLA projections' outputs as hy4_mla.py's, the indexer's (its query, its heads' weights, the token keys
before their norm), the pooled group keys (pool_groups' output, before their rope), the groups each query recalled
(group_recall: the 511 best complete groups and its own), the gated attention entering o_proj. the file's Q8_0 bytes
of k_b, v_b and the pooling's value and gate projections beside them.
run: python -I hy4_indexer.py --modeling modeling_hy_v4_flash.py --config config.json --gguf Hy4-flash-Q3_K-00001-of-00002.gguf
     --out FILE"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy4_mla as M  # noqa: E402
import hy4_pieces as P  # noqa: E402
from hy4_gguf import Gguf  # noqa: E402

T = 2600  # past 2,048 keys: the last ~140 queries' complete groups past 511, chosen among
SEED = 1120
LAYER = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeling", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    R = P.reference(a.modeling, M.NAMES, {"ALL_ATTENTION_FUNCTIONS": M.Interfaces, "ROPE_INIT_FUNCTIONS": {},
                                          "dynamic_rope_update": None, "Callable": object})
    c = json.load(open(a.config))["text_config"]
    g = Gguf(a.gguf)
    config = M.make_config(c)
    kind = config.layer_types[LAYER]
    assert kind == "indexed_attention"
    att = M.attention(R, config, g, LAYER)
    ix = att.compressed_indexer
    seen = {}
    hook = lambda key: (lambda m, i, o: seen.__setitem__(key, o.detach().clone()))  # noqa: E731
    for mod, key in ((att.q_b_proj, "q"), (att.kv_a_proj_with_mqa, "kva"), (att.gate_proj, "gate"),
                     (ix.query_projection, "ixq"), (ix.head_weight_projection, "ixw"),
                     (ix.token_key_projection, "ixk")):
        mod.register_forward_hook(hook(key))
    att.o_proj.register_forward_pre_hook(lambda m, i: seen.__setitem__("o_in", i[0].detach().clone()))
    pool, recall = ix.pool_groups, ix.forward

    def pool_kept(token_keys):
        seen["groups"] = pool(token_keys).detach().clone()
        return seen["groups"]

    def recall_kept(*args, **kwargs):
        seen["recall"] = recall(*args, **kwargs).detach().clone()
        return seen["recall"]

    ix.pool_groups, ix.forward = pool_kept, recall_kept
    rotary = R["HYV4FlashRotaryEmbedding"](config)
    rng = np.random.default_rng(SEED)
    x = torch.from_numpy(rng.normal(0, 1.0, size=(1, T, config.hidden_size)).astype(np.float32))
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    mask = torch.zeros(1, 1, T, T).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1),
                                               torch.finfo(torch.float32).min)
    att(hidden_states=x, position_embeddings=rotary(x, pos, kind), attention_mask=mask, cache=None, position_ids=pos)
    assert "recall" in seen, "the indexer did not run (the keys not past candidate_topk)"
    blk = f"blk.{LAYER}."
    rope = config.rope_parameters[kind]["rope_theta"]
    arrays = {k: seen[k][0].numpy() for k in ("q", "kva", "gate", "o_in", "ixq", "ixw", "ixk", "groups", "recall")}
    arrays.update({
        "kv_a_norm": g.tensor(blk + "attn_kv_a_norm.weight"), "sinks": g.tensor(blk + "attn_sinks.weight"),
        "k_b_q8": g.raw(blk + "attn_k_b.weight").view(np.float32),
        "v_b_q8": g.raw(blk + "attn_v_b.weight").view(np.float32),
        "ix_k_norm": g.tensor(blk + "indexer.k_norm.weight"),
        "pool_value_q8": g.raw(blk + "indexer_compressor_value.weight").view(np.float32),
        "pool_gate_q8": g.raw(blk + "indexer_compressor_gate.weight").view(np.float32),
        "pool_ape": g.tensor(blk + "indexer_compressor_ape.weight"),
        "pool_norm": g.tensor(blk + "indexer_compressor_norm.weight"),
        "rope_base": np.array([rope], dtype=np.float32),
        "shape": np.array([att.num_heads, att.qk_nope_head_dim, att.v_head_dim, config.kv_lora_rank,
                           config.qk_rope_head_dim, ix.num_heads, ix.head_dim, ix.compress_ratio, ix.group_topk],
                          dtype=np.float32)})
    P.write(a.out, arrays)
    r = arrays["recall"]
    print(f"{a.out}: {len(arrays)} arrays, {T} tokens; recall {r.shape}, groups {arrays['groups'].shape}; "
          f"o_in |mean| {np.abs(arrays['o_in']).mean():.4g}")


if __name__ == "__main__":
    main()
