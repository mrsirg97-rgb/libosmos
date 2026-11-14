"""ink2_patch.py - requant_ink2's built tensors as one GGUF of replacements (gguf_assemble.py's input): a pack's
experts swapped for ours, every other tensor the base's.

    PYTHONPATH=<llama.cpp>/gguf-py python tools/convert/ink2_patch.py <base first shard> <out.gguf> <plan> <dir> [...]

each plan's lines a tensor and a type (requant_ink2's plan); its bytes <dir>/<name>.bin, the dims the base's tensor of
that name (the same model: an expert tensor's experts, rows and cols), the type the plan's. every name must be the
base's, its bytes the type's size for those dims - refused otherwise, nothing written. pairs of plan and dir, as
many as there are.
"""
import glob
import os
import sys

import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
from gguf.constants import GGML_QUANT_SIZES

base, out, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
assert rest and len(rest) % 2 == 0, __doc__
dims = {}
for fn in sorted(glob.glob(base.replace("-00001-of-", "-*-of-"))):
    for t in GGUFReader(fn).tensors:
        dims[t.name] = [int(x) for x in t.shape]   # ne, innermost first
want = []
for plan, d in zip(rest[0::2], rest[1::2]):
    for line in open(plan):
        if not line.strip():
            continue
        name, ty = line.split()
        if name not in dims:
            sys.exit(f"{name}: not the base's")
        qt = T[ty.upper()]
        bs, ts = GGML_QUANT_SIZES[qt]
        n = int(np.prod(dims[name]))
        path = os.path.join(d, name + ".bin")
        size = os.path.getsize(path) if os.path.exists(path) else -1
        if size != n // bs * ts:
            sys.exit(f"{path}: {size} bytes, {qt.name} of {dims[name]} is {n // bs * ts}")
        want.append((name, qt, path, dims[name]))
w = GGUFWriter(out + ".part", "inkling2")
w.add_string("osmos.patch", "requant_ink2's experts (unsloth's imatrix) for " + os.path.basename(base))
for name, qt, path, ne in want:
    w.add_tensor_info(name, list(reversed(ne)), np.dtype(np.uint8), os.path.getsize(path), raw_dtype=qt)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for name, qt, path, ne in want:
    w.write_tensor_data(np.fromfile(path, dtype=np.uint8))
    print(f"  {name} {qt.name} {os.path.getsize(path) / 1e9:.2f} GB", flush=True)
w.close()
os.replace(out + ".part", out)
print(f"{out}: {len(want)} tensors")
