"""an .npz (layer_dump.py's) as one file an array for the GPU tests: b"SUBA", u32 dtype (0 F32, 1 I32), u32 ndim,
u64 dims[ndim], the values. integers become I32, everything else F32. run: python npz2bin.py X.npz DIR"""
import os
import struct
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
z = np.load(src)
for k in z.files:
    a = z[k]
    i = np.issubdtype(a.dtype, np.integer)
    a = np.ascontiguousarray(a.astype(np.int32 if i else np.float32))
    with open(os.path.join(out, k + ".bin"), "wb") as f:
        f.write(b"SUBA" + struct.pack("<II", int(i), a.ndim) + struct.pack("<%dQ" % a.ndim, *a.shape))
        f.write(a.tobytes())
print("%d arrays -> %s" % (len(z.files), out))
