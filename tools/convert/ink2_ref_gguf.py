"""ink2_ref_gguf.py - thinkingmachines/Inkling-2-Small (or -Mini) as a reference GGUF: every weight the original's, no
value changed.

    PYTHONPATH=<llama.cpp>/gguf-py python tools/convert/ink2_ref_gguf.py --src ~/Models/inkling2/Inkling-2-Small \
        --out ~/Models/inkling2/ref/ink2-ref [--layers 0,1]       (shards of whole layers, ~48 GB, 393 GB)
    ... --check 0,40          (decode sampled expert rows both ways, compare the bits)

- the routed experts of layers 0-39 (Small's; Mini's 0-19 - by the checkpoint's dtype): NVFP4 (ggml type 40). The
  checkpoint's E2M1 codes two to a byte (the even column the low nibble), an E4M3 scale every 16, and an outer scale a expert, amax[e] / denominator (vLLM's Inkling2MoE: the
  weight's share of the GEMM's (x_amax * w_amax) / (1536 * denominator)). In ggml's layout a block is 64 values: 4
  scales as UE4M3 (the E4M3 bits, which must have no sign and no NaN), then 32 bytes, sub-block s's value j in byte
  8s + j's low nibble and j + 8 in its high one. The outer scale is a tensor of its own, `<name>.scale` (F32, one a
  expert), as llama.cpp's converter writes ModelOpt's: a value is e2m1 x ue4m3 x scale[e].
- the routed experts of layers 40-47: MXFP8 (ggml type 43, osmos's MXFP8): an E8M0 scale, then 32 E4M3 codes, a row's
  every 32 values - the checkpoint's bytes rearranged.
- gate and up: the checkpoint's w13 interleaves them, gate rows even, up rows odd (transformers' Interleave(dim=1));
  here two tensors, ffn_gate_exps and ffn_up_exps.
- everything else as stored: BF16 or F32. The vision tower is left out (a text pack); the drafter (mtp.safetensors)
  is in, under mtp.*.
- the output: <out>-0000k-of-0000N.gguf, the first every tensor not a layer's and the metadata, then whole layers to
  about --split-gb a shard; each written as .part and renamed when whole (0.8.0), a shard there skipped - a stopped run
  goes on where it was, and only the source shards its layers live in are read.
- --delete-source (0.8.0: a disk that holds the checkpoint or the reference, not both): each source shard removed once
  every tensor it holds is in a finished output shard; the plan written first (<out>.plan.json), so a resumed run
  needs no header of a removed file. the whole model only (not --layers or --check).
- Inkling-2-Mini's loop (its loop_config): the keys inkling2.num_loops and inkling2.loop.* (unsloth's), block_count
  the cache slots (24 layers + 2 more passes of the 12-layer core: 48); each core layer's controllers and the two
  cross-pass query tables as stored (BF16, F32), under unsloth's GGUF's names. --name Inkling-2-Mini.
- --vocab-from: llama.cpp's tokenizer arrays (tokenizer.ggml.*: what osmos's tokenizer reads) copied from a GGUF it
  converted from the same tokenizer.json (unsloth's), its tokens checked against the checkpoint's first.
- names: GGUF style where one fits (blk.N.attn_q ...), Inkling's own words where none does; the map is NAMES below.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

import gguf  # noqa: E402  (llama.cpp's gguf-py: NVFP4's type and gguf-py's writer)

NVFP4 = gguf.GGMLQuantizationType.NVFP4
MXFP8 = gguf.GGMLQuantizationType(43)   # (unsloth's and osmos's MXFP8, 33 bytes a 32 values; gguf-py names 43 F8_E4M3)
SIZES = {NVFP4: (64, 36), MXFP8: (32, 33), gguf.GGMLQuantizationType.BF16: (1, 2),
         gguf.GGMLQuantizationType.F32: (1, 4)}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--vocab-from", default="",
                help="a GGUF llama.cpp converted from the same tokenizer.json (unsloth's): its tokenizer.ggml.* copied, "
                     "checked against the checkpoint's tokenizer.json first")
ap.add_argument("--name", default="Inkling-2-Small", help="the model's name (general.name: <name> (reference: ...))")
ap.add_argument("--out", default="", help="the output's path prefix: <prefix>-0000N-of-0000M.gguf")
ap.add_argument("--layers", default="", help="only these layers (a test), comma-separated")
ap.add_argument("--split-gb", type=float, default=48.0)
ap.add_argument("--check", default="", help="layers whose experts to check, decoded both ways")
ap.add_argument("--delete-source", action="store_true",
                help="remove each source shard once every tensor it holds is in a finished output shard (a disk "
                     "that cannot hold both; the whole model only: the plan kept as <out>.plan.json for a resume)")
a = ap.parse_args()

# (checkpoint pattern, GGUF name); N the layer
NAMES = [
    (r"model\.llm\.embed\.weight", "token_embd.weight"),
    (r"model\.llm\.embed_norm\.weight", "token_embd_norm.weight"),
    (r"model\.llm\.norm\.weight", "output_norm.weight"),
    (r"model\.llm\.unembed\.weight", "output.weight"),
    (r"model\.llm\.attn_res\.output_query", "output_res_query"),
    (r"model\.llm\.attn_res\.mlp_queries\.(\d+)", "blk.{0}.ffn_res_query"),
    (r"model\.llm\.attn_res\.attn_queries\.(\d+)", "blk.{0+1}.attn_res_query"),   # (layer i + 1's: transformers)
    (r"model\.llm\.layers\.(\d+)\.attn\.wq_du\.weight", "blk.{0}.attn_q.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.wk_dv\.weight", "blk.{0}.attn_k.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.wv_dv\.weight", "blk.{0}.attn_v.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.wr_du\.weight", "blk.{0}.attn_rel.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.wo_ud\.weight", "blk.{0}.attn_output.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.gate_attn\.weight", "blk.{0}.attn_gate.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.attn_sink_proj\.weight", "blk.{0}.attn_sink.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn\.q_l2_clip\.scale", "blk.{0}.attn_q_clip"),
    (r"model\.llm\.layers\.(\d+)\.attn\.rel_l2_clip\.scale", "blk.{0}.attn_rel_clip"),
    (r"model\.llm\.layers\.(\d+)\.attn\.rel_logits_proj\.proj", "blk.{0}.attn_rel_proj"),
    (r"model\.llm\.layers\.(\d+)\.attn\.([kv])_sconv\.weight", "blk.{0}.attn_{1}_conv"),
    (r"model\.llm\.layers\.(\d+)\.attn\.([kv])_sconv\.static_w", "blk.{0}.attn_{1}_conv_static"),
    (r"model\.llm\.layers\.(\d+)\.attn\.([kv])_sconv\.weight_proj", "blk.{0}.attn_{1}_conv_proj.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn_norm\.weight", "blk.{0}.attn_norm.weight"),
    (r"model\.llm\.layers\.(\d+)\.([qkv])_input_norm\.weight", "blk.{0}.attn_{1}_norm.weight"),
    (r"model\.llm\.layers\.(\d+)\.attn_sconv\.weight", "blk.{0}.attn_out_conv"),
    (r"model\.llm\.layers\.(\d+)\.mlp_sconv\.weight", "blk.{0}.ffn_out_conv"),
    (r"model\.llm\.layers\.(\d+)\.mlp_norm\.weight", "blk.{0}.ffn_norm.weight"),
    (r"model\.llm\.layers\.(\d+)\.mlp\.gate\.weight", "blk.{0}.ffn_gate_inp.weight"),
    (r"model\.llm\.layers\.(\d+)\.mlp\.gate\.bias", "blk.{0}.exp_probs_b.bias"),
    # the loop's (Inkling-2-Mini): a core layer's controls, the message queries (unsloth's GGUF's names)
    (r"model\.llm\.layers\.(\d+)\.controllers\.(attn|k|v|mlp)\.(down|gain|forget)_weight", "blk.{0}.loop_{1}_{2}.weight"),
    (r"model\.llm\.middle_cycle_cross_pass_queries", "loop_res_query.weight"),
    (r"model\.llm\.middle_cycle_cross_pass_kv_queries", "loop_kv_res_query.weight"),
    # the drafter (mtp.safetensors)
    (r"layers\.(\d+)\.(.+)", "mtp.blk.{0}.{1}"),
    (r"markov_head\.(.+)", "mtp.markov.{0}"),
    (r"fuse_weight", "mtp.fuse"),
    (r"mask_embed", "mtp.mask_embed"),
    (r"norm\.weight", "mtp.output_norm.weight"),
]
SKIP = re.compile(r"model\.visual\.|mlp\.experts\.")   # (the experts are written by experts())


def gguf_name(k):
    for pat, out in NAMES:
        m = re.fullmatch(pat, k)
        if m:
            g = m.groups()
            return re.sub(r"\{(\d)(\+1)?\}", lambda x: str(int(g[int(x[1])]) + 1) if x[2] else g[int(x[1])], out)
    raise KeyError(k)


class Src:
    """the checkpoint's tensors by name, opened lazily from their shards"""

    def __init__(self, path):
        self.path = path
        ip = os.path.join(path, "model.safetensors.index.json")
        idx = json.load(open(ip))["weight_map"]
        self.where = dict(idx)
        mtp = os.path.join(path, "mtp.safetensors")
        if os.path.exists(mtp):   # (the drafter: absent from a test's checkpoint)
            with safe_open(mtp, "pt") as f:
                self.where.update({k: "mtp.safetensors" for k in f.keys()})
        self.files = {}
        self.cfg = json.load(open(os.path.join(path, "config.json")))

    def get(self, k):
        f = self.where[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), "pt")
        return self.files[f].get_tensor(k)

    def meta(self, k):   # (dtype, shape) without reading the data
        f = self.where[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), "pt")
        s = self.files[f].get_slice(k)
        return s.get_dtype(), s.get_shape()


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def nvfp4_block(w, s):
    """rows of checkpoint NVFP4 (w: [R, K/2] U8, s: [R, K/16] E4M3) -> ggml NVFP4 bytes [R, K/64 * 36]"""
    R = w.shape[0]
    d = s.view(torch.uint8)
    bad = ((d & 0x80) != 0) | ((d & 0x7F) == 0x7F)
    assert not bad.any(), f"{int(bad.sum())} scales negative or NaN: not UE4M3"
    v = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(R, -1, 16)   # (each block of 16, in order)
    qs = (v[:, :, :8] | (v[:, :, 8:] << 4)).to(torch.uint8)           # [R, K/16, 8]
    nb = qs.shape[1] // 4
    out = torch.cat([d.reshape(R, nb, 4), qs.reshape(R, nb, 32)], dim=-1)
    return out.reshape(R, nb * 36)


def mxfp8_block(w, s):
    """rows of checkpoint MXFP8 (w: [R, K] E4M3, s: [R, K/32] UE8M0) -> ggml MXFP8 bytes [R, K/32 * 33]"""
    R, K = w.shape
    out = torch.cat([s.view(torch.uint8).reshape(R, K // 32, 1), w.view(torch.uint8).reshape(R, K // 32, 32)], -1)
    return out.reshape(R, K // 32 * 33)


def experts(src, L):
    """layer L's routed experts as (name, logical shape, ggml type, bytes as numpy uint8 / f32) - gate, up, down"""
    p = f"model.llm.layers.{L}.mlp.experts."
    nv = src.meta(p + "w13_weight")[0] == "U8"
    for proj, rows in (("w13", None), ("w2", None)):
        w, s = src.get(p + proj + "_weight"), src.get(p + proj + "_weight_scale")
        n_exp, R = w.shape[0], w.shape[1]
        K = w.shape[2] * (2 if nv else 1)
        parts = [("gate", slice(0, None, 2)), ("up", slice(1, None, 2))] if proj == "w13" else [("down", slice(None))]
        for nm, sl in parts:
            blocks = []
            for e in range(n_exp):
                we, se = w[e, sl], s[e, sl]
                blocks.append((nvfp4_block(we, se) if nv else mxfp8_block(we, se)).numpy())
            data = np.stack(blocks)
            if not nv:   # (the prompt matmul reads f16: an MXFP8 value is exact in it only where its scale allows)
                f = torch.from_numpy(data).reshape(-1, 33)
                v = f[:, 1:].contiguous().view(torch.float8_e4m3fn).float() * torch.exp2(f[:, :1].float() - 127)
                inexact = int((v.half().float() != v).sum())
                if inexact:
                    print(f"blk.{L}.ffn_{nm}_exps: {inexact} MXFP8 values not exact in f16", file=sys.stderr)
            yield (f"blk.{L}.ffn_{nm}_exps.weight", [n_exp, data.shape[1], K],
                   NVFP4 if nv else MXFP8, data)
        if nv:   # (the outer scale: gate and up share w13's)
            sc = (src.get(p + proj + "_weight_amax").float() /
                  src.get(p + proj + "_weight_outer_scale_denominator").float()).numpy().astype(np.float32)
            for nm, _ in parts:
                yield f"blk.{L}.ffn_{nm}_exps.scale", [n_exp], gguf.GGMLQuantizationType.F32, sc


def plain(src, k):
    t = src.get(k)
    if t.dim() == 3 and t.shape[1] == 1:   # (the depthwise convs' (C, 1, k): (C, k))
        t = t[:, 0, :]
    if t.dtype == torch.bfloat16:
        return list(t.shape), gguf.GGMLQuantizationType.BF16, t.contiguous().view(torch.int16).numpy()
    assert t.dtype == torch.float32, (k, t.dtype)
    return list(t.shape), gguf.GGMLQuantizationType.F32, t.contiguous().numpy()


def check(src, L, rows=8):
    """sampled expert rows: the checkpoint's values (e2m1 x e4m3 x amax / den; or e4m3 x 2^(e - 127)) against the
    GGUF bytes' values decoded as ggml decodes them - every bit"""
    p = f"model.llm.layers.{L}.mlp.experts."
    nv = src.meta(p + "w13_weight")[0] == "U8"
    w, s = src.get(p + "w13_weight"), src.get(p + "w13_weight_scale")
    g = torch.Generator().manual_seed(L)
    worst = 0
    for e in torch.randint(0, w.shape[0], (4,), generator=g).tolist():
        for r in torch.randint(0, w.shape[1], (rows,), generator=g).tolist():
            if nv:
                outer = (src.get(p + "w13_weight_amax")[e].float() /
                         src.get(p + "w13_weight_outer_scale_denominator").float()[0])
                codes = torch.stack([w[e, r] & 0x0F, w[e, r] >> 4], -1).reshape(-1).long()
                ref = E2M1[codes] * s[e, r].float().repeat_interleave(16) * outer
                b = torch.from_numpy(nvfp4_block(w[e, r:r + 1], s[e, r:r + 1]).numpy()).reshape(-1, 36)
                q = b[:, 4:].reshape(-1, 4, 8)
                lo, hi = (q & 0x0F).long(), (q >> 4).long()
                c = torch.cat([lo, hi], -1).reshape(-1)        # (sub-block: 8 low nibbles then 8 high)
                ue = b[:, :4].reshape(-1)
                exp, man = ((ue >> 3) & 0xF).float(), (ue & 7).float()
                sc = torch.where(exp == 0, man * 2.0 ** -9, (1 + man / 8) * 2.0 ** (exp - 7))
                sc = torch.where((ue == 0) | (ue == 0x7F), torch.zeros_like(sc), sc)
                got = E2M1[c] * sc.repeat_interleave(16) * outer
            else:
                ref = w[e, r].float() * (2.0 ** (s[e, r].view(torch.uint8).float() - 127)).repeat_interleave(32)
                b = torch.from_numpy(mxfp8_block(w[e, r:r + 1], s[e, r:r + 1]).numpy()).reshape(-1, 33)
                got = (b[:, 1:].contiguous().view(torch.float8_e4m3fn).float() *
                       (2.0 ** (b[:, :1].float() - 127))).reshape(-1)
            same = torch.equal(ref.view(torch.int32), got.view(torch.int32))
            worst = max(worst, 0 if same else int((ref != got).sum()))
    amax = src.get(p + "w13_weight_amax")[0].item() if nv else None
    dec = None
    if nv:   # (the outer scale's meaning: expert 0's decoded max against its amax)
        outer = amax / src.get(p + "w13_weight_outer_scale_denominator").item()
        codes = torch.stack([w[0] & 0x0F, w[0] >> 4], -1).reshape(w.shape[1], -1).long()
        dec = (E2M1[codes].abs() * s[0].float().repeat_interleave(16, dim=1)).max().item() * outer
    print(json.dumps({"layer": L, "format": "nvfp4" if nv else "mxfp8", "rows": 4 * rows, "differing_values": worst,
                      "expert0_amax": amax, "expert0_decoded_max": dec}))


def vocab(w, tc):
    """llama.cpp's tokenizer arrays (osmos's tokenizer reads them: src/text/tokenizer.c) from --vocab-from, as written
    there - after its tokens are checked: the unpadded vocabulary's count, each of tokenizer.json's BPE tokens at its id"""
    r = gguf.GGUFReader(a.vocab_from)
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    tj = json.load(open(os.path.join(a.src, "tokenizer.json"), encoding="utf-8"))
    assert len(toks) == tc["unpadded_vocab_size"], f"{len(toks)} tokens, the vocabulary {tc['unpadded_vocab_size']}"
    bad = [t for t, i in tj["model"]["vocab"].items() if toks[i] != t]
    assert not bad, f"{len(bad)} of tokenizer.json's tokens not at their ids (first {bad[:3]})"
    for k, f in r.fields.items():
        if k.startswith("tokenizer.ggml."):
            sub = f.types[1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(k, f.contents(), f.types[0], sub_type=sub)


def meta(w, tc):
    """the text model's shape as llama.cpp-style keys (src/ink2/geo.c reads them). what this build implements is
    asserted, not written: the sliding layers' heads the full ones', no q / o bias, no pre-affine norms, the value norm
    unweighted, attn_res's values normalized, the router's bias for selection, no shared experts"""
    same = [("swa_num_attention_heads", "num_attention_heads"), ("swa_num_key_value_heads", "num_key_value_heads"),
            ("swa_head_dim", "head_dim")]
    for x, y in same:
        assert tc[x] == tc[y], f"{x} {tc[x]} != {y} {tc[y]}: the sliding layers' attention another shape"
    want = {"q_bias": False, "o_bias": False, "pre_norm_pre_affine": False, "final_norm_pre_affine": False,
            "use_value_norm_no_gamma": True, "attn_res_normalize_values": True, "use_gate_bias": True,
            "n_shared_experts": 0}
    for k, v in want.items():
        assert tc.get(k, v) == v, f"{k} = {tc.get(k)}: not what the code implements ({v})"
    A, L = "inkling2", tc["num_hidden_layers"]
    local = tc.get("local_layer_ids") or [i for i in range(L) if (i + 1) % 6 != 0]
    slots = L
    lc = tc.get("loop_config")
    if lc:   # (the loop: block_count its cache slots, a core layer's passes after the first one each - geo.c's loop())
        assert set(lc) == {"core_start", "core_end", "num_passes", "gate_rank", "is_varloop"}, f"loop_config {lc}"
        assert not lc["is_varloop"], "a variable loop: not what the code implements"
        blk = tc["n_layers_per_attn_res_block"]
        assert 0 < lc["core_start"] < lc["core_end"] <= L and lc["core_start"] % blk == 0 == lc["core_end"] % blk
        assert lc["num_passes"] >= 2 and lc["gate_rank"] >= 1, f"loop_config {lc}"
        slots = L + (lc["num_passes"] - 1) * (lc["core_end"] - lc["core_start"])
    u32 = {"block_count": slots, "embedding_length": tc["hidden_size"], "context_length": tc["model_max_length"],
           "vocab_size": tc["unpadded_vocab_size"], "attention.head_count": tc["num_attention_heads"],
           "attention.head_count_kv": tc["num_key_value_heads"], "attention.key_length": tc["head_dim"],
           "attention.value_length": tc["head_dim"], "attention.sliding_window": tc["sliding_window_size"],
           "attention.relative_dim": tc["d_rel"], "attention.relative_extent": tc["rel_extent"],
           "attention.log_scaling_floor": tc["log_scaling_n_floor"], "conv_kernel": tc["sconv_kernel_size"],
           "attn_res.block_size": tc["n_layers_per_attn_res_block"], "expert_count": tc["n_routed_experts"],
           "expert_used_count": tc["num_experts_per_tok"], "expert_feed_forward_length": tc["intermediate_size"],
           "expert_gating_func": 2}   # (2: sigmoid, llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID)
    f32 = {"attention.layer_norm_rms_epsilon": tc["rms_norm_eps"], "attention.log_scaling_alpha": tc["log_scaling_alpha"],
           "attention.q_clip": tc["qk_l2_clip_threshold"], "attention.k_clip": tc["qk_l2_clip_threshold"],
           "attention.relative_clip": tc["rel_l2_clip_threshold"], "expert_weights_scale": tc["route_scale"],
           "expert_gate_softcap": tc["mlp_gate_softcap"], "expert_up_softcap": tc["mlp_up_softcap"],
           "final_logit_softcapping": tc["final_logit_softcapping"]}
    if lc:
        u32.update({"num_loops": lc["num_passes"], "loop.core_start": lc["core_start"],
                    "loop.core_end": lc["core_end"], "loop.gate_rank": lc["gate_rank"]})
    for k, v in u32.items():
        w.add_uint32(f"{A}.{k}", int(v))
    for k, v in f32.items():
        w.add_float32(f"{A}.{k}", float(v))
    w.add_bool(f"{A}.expert_weights_norm", True)
    w.add_array(f"{A}.attention.sliding_layers", [int(x) for x in local])
    w.add_string(f"{A}.text_config.json", json.dumps(tc))   # (provenance: the checkpoint's own config, whole)


def main():
    src = Src(a.src)
    if a.check:
        for L in map(int, a.check.split(",")):
            check(src, L)
        return
    tc = src.cfg["text_config"]
    if a.delete_source and a.layers:
        sys.exit("--delete-source converts the whole model only (a test's --layers would leave the rest unread)")
    plan_path = a.out + ".plan.json"
    if a.delete_source and os.path.exists(plan_path):   # (a resume: the source's removed shards' headers gone)
        shards = [[tuple(e[:4]) + (gguf.GGMLQuantizationType(e[4]),) for e in plan] for plan in json.load(open(plan_path))]
        write(src, tc, shards)
        return
    layers = list(map(int, a.layers.split(","))) if a.layers else list(range(tc["num_hidden_layers"]))
    keys = []
    for k in src.where:
        if SKIP.search(k):
            continue
        m = re.match(r"model\.llm\.layers\.(\d+)\.", k)
        if m and int(m[1]) not in layers:
            continue
        m = re.match(r"model\.llm\.attn_res\.(attn|mlp)_queries\.(\d+)", k)
        if m and int(m[2]) + (m[1] == "attn") not in layers:
            continue
        keys.append(k)
    # pass 1: every tensor's info (shapes from the headers; the experts' from their shapes), grouped into shards:
    # the first every tensor not a layer's (the embedding, the head, attn_res's queries, the drafter) and the
    # metadata, then whole layers to about --split-gb a shard (their dense tensors, then their experts)
    glob, per_layer = [], {L: [] for L in layers}
    for k in keys:
        dt, sh = src.meta(k)
        sh = [sh[0], sh[2]] if len(sh) == 3 and sh[1] == 1 else list(sh)
        t = gguf.GGMLQuantizationType.BF16 if dt == "BF16" else gguf.GGMLQuantizationType.F32
        m = re.match(r"model\.llm\.layers\.(\d+)\.", k)
        (per_layer[int(m[1])] if m else glob).append(("plain", k, gguf_name(k), sh, t))
    for L in layers:
        p = f"model.llm.layers.{L}.mlp.experts."
        dt, (n_exp, R, Kb) = src.meta(p + "w13_weight")
        nv = dt == "U8"
        for nm, rows, kb in (("gate", R // 2, Kb), ("up", R // 2, Kb), ("down",) + tuple(src.meta(p + "w2_weight")[1][1:])):
            K = kb * 2 if nv else kb
            t = NVFP4 if nv else MXFP8
            per_layer[L].append(("expert", L, f"blk.{L}.ffn_{nm}_exps.weight", [n_exp, rows, K], t))
        if nv:
            for nm in ("gate", "up", "down"):
                per_layer[L].append(("expert", L, f"blk.{L}.ffn_{nm}_exps.scale", [n_exp], gguf.GGMLQuantizationType.F32))

    shards, cur, size = [glob], [], 0
    for L in layers:
        lb = sum(map(nbytes, per_layer[L]))
        if cur and size + lb > a.split_gb * 1e9:
            shards.append(cur)
            cur, size = [], 0
        cur += per_layer[L]
        size += lb
    if cur:
        shards.append(cur)
    if a.delete_source:
        with open(plan_path + ".part", "w") as f:
            json.dump([[list(e[:4]) + [int(e[4])] for e in plan] for plan in shards], f)
        os.rename(plan_path + ".part", plan_path)
    write(src, tc, shards)


def nbytes(e):
    bs, ts = SIZES[e[4]]
    return int(np.prod(e[3])) // bs * ts


def sources(src, plan):
    """the source files a shard's tensors are read from: a plain tensor's own, a layer's experts' all"""
    files = set()
    for kind, k, *_ in plan:
        if kind == "plain":
            files.add(src.where[k])
        else:
            p = f"model.llm.layers.{k}.mlp.experts."
            files |= {f for name, f in src.where.items() if name.startswith(p)}
    return files


def write(src, tc, shards):
    count = len(shards)
    need = [sources(src, plan) for plan in shards]   # (from the index alone: no file opened)
    # pass 2: each shard a GGUF of its own, written as .part and renamed when whole - a shard already there skipped,
    # so a stopped run goes on where it was (the plan the same: the shards the same)
    for no, plan in enumerate(shards):
        path = f"{a.out}-{no + 1:05d}-of-{count:05d}.gguf"
        if os.path.exists(path):
            print(f"shard {no + 1} of {count}: there, skipped", file=sys.stderr, flush=True)
            drop_sources(src, need, no)
            continue
        w = gguf.GGUFWriter(path + ".part", "inkling2")
        w.add_uint16("split.no", no)
        w.add_uint16("split.count", count)
        w.add_int32("split.tensors.count", sum(map(len, shards)))
        if no == 0:
            w.add_name(f"{a.name} (reference: the original weights)")
            w.add_string("general.quantized_by", "osmos ink2_ref_gguf.py: no value changed")
            meta(w, tc)
            if a.vocab_from:
                vocab(w, tc)
            w.add_string("inkling2.mtp_config.json", json.dumps(src.cfg.get("mtp_config", {})))
            for key, f in (("tokenizer.huggingface.json", "tokenizer.json"), ("tokenizer.chat_template", "chat_template.jinja")):
                if os.path.exists(os.path.join(a.src, f)):
                    w.add_string(key, open(os.path.join(a.src, f), encoding="utf-8").read())
        for e in plan:
            # (the shape in values, as written: a dtype other than uint8 keeps gguf-py from reading it as bytes
            # through its own size table - which has no MXFP8)
            dt = {gguf.GGMLQuantizationType.F32: np.float32, gguf.GGMLQuantizationType.BF16: np.int16}.get(e[4], np.int8)
            w.add_tensor_info(e[2], e[3], np.dtype(dt), nbytes(e), raw_dtype=e[4])
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        done_layers, pending = set(), {}
        for kind, k, name, sh, t in plan:
            if kind == "plain":
                _, _, data = plain(src, k)
                w.write_tensor_data(data.view(np.int8) if data.dtype == np.uint8 else data)
            else:
                if k not in done_layers:
                    pending = {n: d for n, _, _, d in experts(src, k)}
                    done_layers.add(k)
                d = pending.pop(name)
                w.write_tensor_data(d.view(np.int8) if d.dtype == np.uint8 else d)
            print(f"wrote {name} {sh} {t.name}", file=sys.stderr, flush=True)
        w.close()
        pending = {}
        os.rename(path + ".part", path)
        print(f"shard {no + 1} of {count}: {len(plan)} tensors", file=sys.stderr, flush=True)
        drop_sources(src, need, no)


def drop_sources(src, need, no):
    """--delete-source: the source files no shard after `no` reads, removed (their handles closed first)"""
    if not a.delete_source:
        return
    later = set().union(*need[no + 1:]) if no + 1 < len(need) else set()
    for f in sorted(need[no] - later):
        path = os.path.join(src.path, f)
        if os.path.exists(path):
            src.files.pop(f, None)
            os.remove(path)
            print(f"removed the source's {f} (every tensor it holds written)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
