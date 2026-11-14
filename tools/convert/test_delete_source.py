"""test_delete_source.py - ink2_ref_gguf.py --delete-source (0.8.0): the same shards as a run that keeps its source,
each source file removed once no later shard reads it, and a run stopped part-way resumed to the same bytes.

    PYTHONPATH=<llama.cpp>/gguf-py python tools/convert/test_delete_source.py CKPT_DIR SCRATCH_DIR

CKPT_DIR a small checkpoint (test_oracle.py --out DIR). its tensors are first spread over several files the way
Inkling-2-Small's are: each layer's in a file of its own but one of its tensors in a last file every layer reads (the
real checkpoint's shard 24: attn_res queries and norms), the rest in the first.
"""
import filecmp
import glob
import json
import os
import re
import shutil
import subprocess
import sys

from safetensors import safe_open
from safetensors.torch import save_file

ckpt, scratch = sys.argv[1], sys.argv[2]
conv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ink2_ref_gguf.py")
SPLIT = "0.0012"   # GB a shard: several shards of the small checkpoint
failed = []


def check(what, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failed.append(what)


def spread(dst):
    """the checkpoint's tensors over several files, as the real one's lie"""
    os.makedirs(dst)
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    tensors = {}
    for f in sorted(set(idx.values())):
        with safe_open(os.path.join(ckpt, f), "pt") as h:
            tensors.update({k: h.get_tensor(k) for k in h.keys()})
    files, shared = {}, set()
    for k in tensors:
        m = re.match(r"model\.llm\.layers\.(\d+)\.", k)
        if not m:
            files[k] = "model-00001.safetensors"
        elif k.endswith("norm.weight") and k not in shared and not any(
                s.startswith(f"model.llm.layers.{m[1]}.") for s in shared):
            files[k] = "model-shared.safetensors"
            shared.add(k)
        else:
            files[k] = f"model-layer{int(m[1]):03d}.safetensors"
    for f in sorted(set(files.values())):
        save_file({k: t.contiguous() for k, t in tensors.items() if files[k] == f}, os.path.join(dst, f))
    json.dump({"metadata": {}, "weight_map": files}, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    for f in os.listdir(ckpt):
        if f.endswith(".json") and f != "model.safetensors.index.json" and not f.startswith("oracle"):
            shutil.copy(os.path.join(ckpt, f), dst)
    return sorted(set(files.values())), len(shared)


def run(src, out, *extra, stop_after=None):
    p = subprocess.Popen([sys.executable, conv, "--src", src, "--out", out, "--split-gb", SPLIT, *extra],
                         stderr=subprocess.PIPE, text=True)
    shards = 0
    for line in p.stderr:
        if re.match(r"shard \d+ of \d+: \d+ tensors", line):
            shards += 1
            if stop_after and shards == stop_after:
                p.kill()   # (stopped between shards, as a crash would leave it: the next one's .part not begun)
                break
    p.wait()
    return p.returncode


shutil.rmtree(scratch, ignore_errors=True)
os.makedirs(scratch)
base_src = os.path.join(scratch, "src-kept")
names, n_shared = spread(base_src)
print(f"the checkpoint spread over {len(names)} files ({n_shared} layers' norm in the shared one)", flush=True)
check("a run keeping its source finishes", run(base_src, os.path.join(scratch, "kept")) == 0)
want = sorted(glob.glob(os.path.join(scratch, "kept-*.gguf")))
check(f"it writes several shards ({len(want)})", len(want) >= 3)

del_src = os.path.join(scratch, "src-del")
shutil.copytree(base_src, del_src)
check("--delete-source with --layers refused",
      run(del_src, os.path.join(scratch, "x"), "--layers", "0", "--delete-source") != 0 and
      all(os.path.exists(os.path.join(del_src, f)) for f in names))
out = os.path.join(scratch, "del")
run(del_src, out, "--delete-source", stop_after=2)
left = set(os.listdir(del_src))
check(f"stopped after 2 shards: some source files removed, the shared one kept ({len(names) - len(left & set(names))}"
      f" of {len(names)} removed)", 0 < len(set(names) - left) < len(names) and "model-shared.safetensors" in left)
check("the plan kept for the resume", os.path.exists(out + ".plan.json"))
check("the resume finishes", run(del_src, out, "--delete-source") == 0)
got = sorted(glob.glob(out + "-*.gguf"))
check(f"the same shards ({len(got)} of {len(want)})", len(got) == len(want))
check("each shard the same bytes as the run that kept its source",
      len(got) == len(want) and all(filecmp.cmp(g, w, shallow=False) for g, w in zip(got, want)))
check("every source tensor file removed, the config kept",
      not glob.glob(os.path.join(del_src, "*.safetensors")) and os.path.exists(os.path.join(del_src, "config.json")))
print(f"[test_delete_source] {'ok' if not failed else 'FAILED: ' + '; '.join(failed)}")
sys.exit(1 if failed else 0)
