"""the whole-forward reference for tests/gpu/test_hy4_forward_gpu.hip: Hy4-flash's text model by the reference's own
code - the transformers PR #50903 modeling_hy_v4_flash.py's HYV4FlashDecoderLayer.forward, layer after layer, each
with its real attention (HYV4FlashAttention: MLA, the sinks, the gate; a sparse layer's indexer), its MLP (the dense
HYV4FlashMLP, or HYV4FlashMoE) and its Engram layer (the table's rows at the reference's hash), then
HYV4FlashHyperHead, the final norm and the head - their source compiled alone (hy4_pieces.reference), on CPU in F32
on the GGUF's own weights dequantised (hy4_mla.attention's, hy4_moe's, hy4_pieces' setups). the ids: BOS, then the
public wiki text (osmos-eval's wiki10k_text.txt) split by the release's vocabulary, longest match first over GPT-2's
byte alphabet (realistic ids for routing and the indexer, not the tokenizer's merges: osmos is given the same ids).
kept at the rows rows_of(T): every layer's streams after it (taps) and the logits.
run: python -I hy4_forward.py --modeling modeling_hy_v4_flash.py --config config.json --tokenizer tokenizer.json
     --shard model-00147-of-00148.safetensors --gguf Hy4-flash-SPARK-4Bit.gguf --text wiki10k_text.txt --out FILE"""
import argparse
import gc
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy4_engram_rows as hashes  # noqa: E402
import hy4_mla as M  # noqa: E402
import hy4_pieces as P  # noqa: E402
from hy4_gguf import Gguf  # noqa: E402

TOKENS = 2600  # past the window (512) and the indexer's dense length (2,048)


def rows_of(T):
    """the rows kept: the first 32, every 37th, the last 64"""
    return sorted(set(range(min(32, T))) | set(range(0, T, 37)) | set(range(max(0, T - 64), T)))
NAMES = sorted(set(P.NAMES + M.NAMES + ["HYV4FlashMLP", "HYV4FlashExperts", "HYV4FlashTopkRouter", "HYV4FlashMoE"]))


def param(t):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)), requires_grad=False)


def byte_alphabet():
    """GPT-2's bytes -> printable characters (byte-level BPE's alphabet)"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b), cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def ids_of(text, vocab, count):
    """the text's bytes split by the vocabulary, longest match first, `count` ids"""
    alpha = byte_alphabet()
    s = "".join(alpha[b] for b in text.encode("utf-8"))
    longest = max(len(k) for k in vocab)
    out, i = [], 0
    while i < len(s) and len(out) < count:
        for n in range(min(longest, len(s) - i), 0, -1):
            if s[i:i + n] in vocab:
                out.append(vocab[s[i:i + n]])
                i += n
                break
        else:
            raise SystemExit(f"no token for {s[i]!r}")
    if len(out) < count:
        raise SystemExit("the text too short")
    return out


class ExpertTables:
    """the routed experts' tables the reference indexes ([E][2I][L], [E][L][I]: 14.5 GB in F32), allocated once and
    shared by every MoE layer - each layer fills the experts its tokens pick (the reference's loop reads no other)"""

    def __init__(self, config):
        E, I, Lt = config.num_local_experts, config.moe_intermediate_size, config.routed_expert_hidden_size
        self.I = I
        self.gate_up = nn.Parameter(torch.zeros(E, 2 * I, Lt), requires_grad=False)
        self.down = nn.Parameter(torch.zeros(E, Lt, I), requires_grad=False)

    def fill(self, g, blk, experts):
        for e in experts:
            self.gate_up.data[e, :self.I] = torch.from_numpy(g.expert(blk + "ffn_gate_exps.weight", e))
            self.gate_up.data[e, self.I:] = torch.from_numpy(g.expert(blk + "ffn_up_exps.weight", e))
            self.down.data[e] = torch.from_numpy(g.expert(blk + "ffn_down_exps.weight", e))


def moe(R, config, g, L, tables):
    """layer L's HYV4FlashMoE on the shared expert tables: the router first, then the experts it picked read"""
    blk = f"blk.{L}."
    with torch.device("meta"):
        m = R["HYV4FlashMoE"](config)
    m.gate.weight = param(g.tensor(blk + "ffn_gate_inp.weight"))
    m.gate.e_score_correction_bias = torch.from_numpy(g.tensor(blk + "exp_probs_b.bias"))
    m.routed_expert_down_proj.weight = param(g.tensor(blk + "ffn_routed_down.weight"))
    m.routed_expert_up_proj.weight = param(g.tensor(blk + "ffn_routed_up.weight"))
    m.shared_experts.gate_proj.weight = param(g.tensor(blk + "ffn_gate_shexp.weight"))
    m.shared_experts.up_proj.weight = param(g.tensor(blk + "ffn_up_shexp.weight"))
    m.shared_experts.down_proj.weight = param(g.tensor(blk + "ffn_down_shexp.weight"))
    m.combine_output_norm.weight = param(g.tensor(blk + "post_ffw_norm.weight"))
    m.experts.gate_up_proj, m.experts.down_proj = tables.gate_up, tables.down
    return m


def route_then_fill(m, g, L, tables):
    """the router's picks for the layer's input, their experts read into the tables before the MoE runs"""
    real = m.forward

    def forward(hidden_states):
        tables.fill(g, f"blk.{L}.", sorted(set(m.gate(hidden_states)[2].reshape(-1).tolist())))
        return real(hidden_states)

    return forward


def dense(R, config, g, L):
    blk = f"blk.{L}."
    d = R["HYV4FlashMLP"](config)
    d.gate_proj.weight = param(g.tensor(blk + "ffn_gate.weight"))
    d.up_proj.weight = param(g.tensor(blk + "ffn_up.weight"))
    d.down_proj.weight = param(g.tensor(blk + "ffn_down.weight"))
    d.post_fc2_norm.weight = param(g.tensor(blk + "post_ffw_norm.weight"))
    return d


def engram(R, config, g, table, L, c):
    """layer L's HYV4FlashEngramLayer: built on the meta device, its parts the file's, its rows the table's"""
    blk, lid, D, hc = f"blk.{L}.", L + 1, c["hidden_size"], c["hc_mult"]
    with torch.device("meta"):
        eg = R["HYV4FlashEngramLayer"](config, lid)
    sizes = [s for order in c["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
    eg.head_vocab_sizes = torch.tensor(sizes)
    eg.lookup_table = torch.from_numpy(table.astype(np.int32))
    eg.hash_multipliers = torch.from_numpy(hashes.draw_hash_multipliers(config.engram_seed, lid,
                                                                        config.engram_max_ngram_size,
                                                                        int(table.max()) + 1))
    first = g.kv["hy_v4_flash.ple.head_offsets"][g.kv["hy_v4_flash.ple.layers"].index(L) * len(sizes)]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]) + first

    def lookup(hash_ids):
        rows = (hash_ids.clamp_min(0).numpy() + offsets).reshape(-1)
        out = torch.from_numpy(g.rows("per_layer_token_embd.weight", rows)).reshape(*hash_ids.shape, -1)
        return out.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)

    del eg.multi_head_embedding
    eg.multi_head_embedding = lookup
    norms = g.tensor(blk + "engram_lookup_norm.weight")
    for k, n in enumerate(eg.lookup_per_head_norms):
        n.weight = param(norms[k])
    kv_w, kv_b, gate = (g.tensor(blk + "engram_kv.weight"), g.tensor(blk + "engram_kv_b.weight"),
                        g.tensor(blk + "engram_gate.weight"))
    for k in range(hc):
        eg.key_projs[k].weight, eg.key_projs[k].bias = param(kv_w[k * D:(k + 1) * D]), param(kv_b[k])
        eg.norm_key[k].weight, eg.norm_query[k].weight = param(gate[k]), param(np.ones(D))
    eg.value_proj.weight, eg.value_proj.bias = param(kv_w[hc * D:]), param(kv_b[hc])
    eg.unified_pre_gamma = param(g.tensor(blk + "engram_pre_gamma.weight"))
    for k, s in enumerate(g.tensor(blk + "engram_hc_scale.weight")):
        eg.unified_per_hc_scales[k] = param(s)
    return eg


def main():
    ap = argparse.ArgumentParser()
    for k in ("modeling", "config", "tokenizer", "shard", "gguf", "text", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--tokens", type=int, default=TOKENS)
    a = ap.parse_args()
    T = a.tokens
    ROWS = rows_of(T)
    torch.set_grad_enabled(False)
    R = P.reference(a.modeling, NAMES, {"ALL_ATTENTION_FUNCTIONS": M.Interfaces, "ROPE_INIT_FUNCTIONS": {},
                                        "dynamic_rope_update": None, "Callable": object,
                                        "ACT2FN": {"silu": nn.SiLU()}, "use_experts_implementation": None})
    c = json.load(open(a.config))["text_config"]
    g = Gguf(a.gguf)
    config = M.make_config(c)
    for k, v in dict(hc_mult=c["hc_mult"], hc_eps=c["hc_eps"], hc_magnitude=c["hc_magnitude"],
                     intermediate_size=c["intermediate_size"], hidden_act=c["hidden_act"],
                     num_local_experts=c["n_routed_experts"], moe_intermediate_size=c["moe_intermediate_size"],
                     routed_expert_hidden_size=c["routed_expert_hidden_size"], swiglu_limit=c["swiglu_limit"],
                     num_experts_per_tok=c["num_experts_per_tok"], routed_scaling_factor=c["routed_scaling_factor"],
                     n_group=c["n_group"], topk_group=c["topk_group"], norm_topk_prob=c["norm_topk_prob"],
                     n_shared_experts=c["n_shared_experts"], engram_max_ngram_size=c["engram_max_ngram_size"],
                     engram_n_head_per_ngram=c["engram_n_head_per_ngram"],
                     engram_fp8_embedding=c["engram_fp8_embedding"], engram_pad_id=c["engram_pad_id"],
                     engram_seed=c.get("engram_seed", 0),
                     engram_layer_head_vocab_sizes=c["engram_layer_head_vocab_sizes"],
                     engram_n_embed_per_ngram=c["engram_n_embed_per_ngram"]).items():
        setattr(config, k, v)
    table = hashes.lookup_table(a.shard, 11).astype(np.int64)
    config.engram_lookup_table_size = len(table)
    vocab = json.load(open(a.tokenizer))["model"]["vocab"]
    ids = np.array([c["bos_token_id"]] + ids_of(open(a.text).read(), vocab, T - 1), dtype=np.int64)
    D, hc, n_layers = c["hidden_size"], c["hc_mult"], c["num_hidden_layers"]
    engram_layers = [i - 1 for i in c["engram_layer_ids"]]

    h = torch.from_numpy(g.rows("token_embd.weight", ids))[None].unsqueeze(2).expand(-1, -1, hc, -1).contiguous()
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    rotary = R["HYV4FlashRotaryEmbedding"](config)
    ngram = torch.from_numpy(hashes.build_ngram_tokens(ids, c["engram_max_ngram_size"], c["engram_pad_id"]))[None]
    emask = torch.ones(1, T, dtype=torch.bool)
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    masks = {}
    for kind in ("sliding_attention", "indexed_attention"):
        allowed = (j <= i) & ((i - j < config.sliding_window) if kind == "sliding_attention" else True)
        masks[kind] = torch.zeros(1, 1, T, T).masked_fill(~allowed, torch.finfo(torch.float32).min)
    taps = np.zeros((n_layers, len(ROWS), hc, D), dtype=np.float32)
    tables = ExpertTables(config)
    for L in range(n_layers):
        t0 = time.time()
        kind = config.layer_types[L]
        blk = f"blk.{L}."
        layer = R["HYV4FlashDecoderLayer"].__new__(R["HYV4FlashDecoderLayer"])
        nn.Module.__init__(layer)
        layer.layer_idx, layer.attention_type = L, kind
        for name, pre in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
            m = R["HYV4FlashHyperConnection"](config)
            m.fn, m.base = param(g.tensor(blk + pre + "_fn.weight")), param(g.tensor(blk + pre + "_base.weight"))
            m.scale = param(g.tensor(blk + pre + "_scale.weight"))
            setattr(layer, name, m)
        for name, t in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
            m = R["HYV4FlashRMSNorm"](D, eps=c["rms_norm_eps"])
            m.weight = param(g.tensor(blk + t + ".weight"))
            setattr(layer, name, m)
        layer.self_attn = M.attention(R, config, g, L)
        if L < 1:
            layer.mlp = dense(R, config, g, L)
        else:
            layer.mlp = moe(R, config, g, L, tables)
            layer.mlp.forward = route_then_fill(layer.mlp, g, L, tables)
        layer.engram_layer = engram(R, config, g, table, L, c) if L in engram_layers else None
        h = layer.forward(h, attention_mask=masks[kind], position_ids=pos, cache=None,
                          position_embeddings=rotary(h, pos, kind), indexer_position_ids=pos, ngram_tokens=ngram,
                          engram_mask=emask)
        taps[L] = h[0, ROWS].numpy()
        del layer
        gc.collect()
        print(f"layer {L:2d} ({kind}): {time.time() - t0:5.1f} s, |h| {float(h.abs().mean()):.4g}", flush=True)
    head = R["HYV4FlashHyperHead"](config)
    head.hc_fn, head.hc_base = param(g.tensor("output_hc_fn.weight")), param(g.tensor("output_hc_base.weight"))
    head.hc_scale = param(g.tensor("output_hc_scale.weight"))
    norm = R["HYV4FlashRMSNorm"](D, eps=c["rms_norm_eps"])
    norm.weight = param(g.tensor("output_norm.weight"))
    x = norm(head(h[:, ROWS]))[0]
    logits = (x @ torch.from_numpy(g.tensor("output.weight")).T).numpy()
    nxt = ids[np.minimum(np.array(ROWS) + 1, T - 1)]
    lp = logits - logits.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    nll = -lp[np.arange(len(ROWS)), nxt][np.array(ROWS) < T - 1]
    print(f"logits: the next id's mean NLL {nll.mean():.3f} ({len(nll)} rows), top-1 "
          f"{(logits.argmax(-1) == nxt)[np.array(ROWS) < T - 1].mean():.3f}")
    P.write(a.out, {"ids": ids.astype(np.float32), "rows": np.array(ROWS, dtype=np.float32), "taps": taps,
                    "logits": logits})
    print(f"{a.out}: {T} ids, {len(ROWS)} rows kept, {n_layers} layers' taps")


if __name__ == "__main__":
    main()
