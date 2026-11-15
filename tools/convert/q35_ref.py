"""the qwen35 forward's reference (src/q35, tools/q35_check): HF transformers' Qwen3.5 on the NVFP4 file's own values -
its BF16 checkpoint with every NVFP4 matrix replaced by its dequantized blocks (E2M1 x E4M3 x the tensor's scale), the
W4A16 model osmos runs - over the first T tokens of a held session. out: ids.i32 (T), h.f32 (layers x T x n_embd, the
residual after each layer), logits.f32 (T x n_vocab), meta.json.

run: PYTHONPATH=<transformers> python q35_ref.py BF16_DIR NVFP4_DIR OUT --T 64 [--session N]
(the held sessions: /mnt/data/incarnate/qwen's steer.HELD; any token file works with --ids)"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/mnt/data/incarnate/qwen")


def dequant(packed, s, gs):  # ModelOpt NVFP4: low nibble first, code sign<<3 | E2M1 index, E4M3 a 16
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=packed.device)
    c = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    v = grid[c & 7] * (1 - 2 * (c >> 3)).float()
    return (v.reshape(v.shape[0], -1, 16) * s.float().unsqueeze(-1) * gs).reshape(v.shape[0], -1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("nvfp4")
    ap.add_argument("out")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--ids")
    ap.add_argument("--no-h", action="store_true")  # logits alone (a long T)
    a = ap.parse_args()
    import nvfp4
    tok, model = nvfp4.load(a.bf16)
    idx = json.load(open(f"{a.nvfp4}/model.safetensors.index.json"))["weight_map"]
    mods = dict(model.named_modules())
    n = 0
    files = {}
    for name, shard in idx.items():
        if not name.endswith(".weight") or f"{name[:-7]}.weight_scale_2" not in idx:
            continue
        f = files.setdefault(shard, safe_open(f"{a.nvfp4}/{shard}", "pt"))
        b = name[:-7]
        w = mods[nvfp4.module(name)].weight
        w.copy_(dequant(f.get_tensor(name).to(w.device), f.get_tensor(f"{b}.weight_scale").to(w.device),
                        f.get_tensor(f"{b}.weight_scale_2").float().item()).to(w.dtype))
        n += 1
    print("NVFP4 matrices in place:", n, flush=True)
    if a.ids:
        ids = torch.from_numpy(np.fromfile(a.ids, dtype=np.int32).astype(np.int64))[: a.T]
    else:
        import steer
        held = torch.load(steer.HELD)
        ids = held[a.session][0][: a.T]
    T = len(ids)
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    hs = []
    hooks = [] if a.no_h else [L.register_forward_hook(lambda m, i, o: hs.append((o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
                               for L in layers]
    dev = next(model.parameters()).device
    logits = model(input_ids=ids[None].to(dev)).logits[0].float().cpu()
    for h in hooks:
        h.remove()
    os.makedirs(a.out, exist_ok=True)
    ids.numpy().astype(np.int32).tofile(f"{a.out}/ids.i32")
    if hs:
        torch.stack(hs).numpy().astype(np.float32).tofile(f"{a.out}/h.f32")
    logits.numpy().astype(np.float32).tofile(f"{a.out}/logits.f32")
    json.dump({"T": T, "layers": len(hs), "n_embd": int(hs[0].shape[-1]) if hs else 0,
               "n_vocab": int(logits.shape[1]), "nvfp4": n}, open(f"{a.out}/meta.json", "w"))
    print("wrote", a.out, "T", T, "layers", len(hs), flush=True)


if __name__ == "__main__":
    main()
