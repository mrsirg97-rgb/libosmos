"""the GPU tests' oracle fixtures (tests/gpu/oracle_fixture.h's reader): b"ORCL", u32 version, u32 count, then each
array's name (u32 length, bytes), dims (u32 count, u64 each) and F32 values"""
import struct

import numpy as np

VERSION = 1


def write(path, arrays):
    with open(path, "wb") as f:
        f.write(b"ORCL" + struct.pack("<2I", VERSION, len(arrays)))
        for name, v in arrays.items():
            v = np.ascontiguousarray(v, dtype=np.float32)
            f.write(struct.pack("<I", len(name)) + name.encode() + struct.pack("<I", v.ndim))
            f.write(struct.pack(f"<{v.ndim}Q", *v.shape) + v.tobytes())


def read(path):
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"ORCL"
        version, n = struct.unpack("<2I", f.read(8))
        assert version == VERSION
        for _ in range(n):
            (ln,) = struct.unpack("<I", f.read(4))
            name = f.read(ln).decode()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            out[name] = np.frombuffer(f.read(4 * int(np.prod(dims))), np.float32).reshape(dims)
    return out
