#!/usr/bin/env python3
"""tools/convert/qtypes.py - the ggml types osmos reads, as include/osmos/model/qtype.h's QTYPES lists them (its one
table: X(name, ggml id, a block's values, a block's bytes, paths)) - read from the header, never copied, so a Python
tool's sizes cannot drift from the engine's (0.7.1: gguf_assemble's copy lacked Q5_1 and Q2_0, Qwen's experts).

    SIZES: {ggml id: (values a block, bytes a block)}   NAMES: {ggml id: name}
"""
import os
import re

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "include", "osmos", "model", "qtype.h")
_X = re.compile(r"X\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,")


def _read(path=HEADER):
    text = open(path).read()
    table = text[text.index("#define QTYPES(X)"):]
    table = table[:table.index("\n\n")]  # the macro's lines (each ending in a backslash), to its blank line
    rows = _X.findall(table)
    if not rows:
        raise RuntimeError(f"{path}: no QTYPES rows")
    return {int(i): (int(v), int(b)) for _, i, v, b in rows}, {int(i): n for n, i, _, _ in rows}


SIZES, NAMES = _read()
