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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_assemble as A  # noqa: E402  (clone_or_copy: FICLONERANGE where the filesystem has clones)

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
# every tensor's data at a 4,096-byte boundary (the .bin's own offset 0 mod 4,096): its whole blocks cloned from the
# .bin (no space where the filesystem has clones), only a last partial block copied
w = GGUFWriter(out + ".part", "inkling2")
w.add_custom_alignment(A.BLOCK)
w.add_string("osmos.patch", "requant_ink2's experts (unsloth's imatrix) for " + os.path.basename(base))
for name, qt, path, ne in want:
    # (int8, not uint8: gguf-py reads a uint8 shape as bytes; the shape here is values, ink2_ref_gguf.py's way)
    w.add_tensor_info(name, list(reversed(ne)), np.dtype(np.int8), os.path.getsize(path), raw_dtype=qt)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
w.close()
cloned = 0
with open(out + ".part", "r+b") as f:
    at = (f.seek(0, os.SEEK_END) + A.BLOCK - 1) // A.BLOCK * A.BLOCK
    for name, qt, path, ne in want:
        n = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        cloned += A.clone_or_copy(fd, 0, f.fileno(), at, n)
        os.close(fd)
        at = (at + n + A.BLOCK - 1) // A.BLOCK * A.BLOCK
        print(f"  {name} {qt.name} {n / 1e9:.2f} GB", flush=True)
    f.truncate(at)
os.replace(out + ".part", out)
print(f"{out}: {len(want)} tensors ({cloned / 1e9:.1f} GB cloned)")
