"""tests/fixtures/json.txt: Python's own json over inputs chosen at its corners - what json.loads reads and
json.dumps(ensure_ascii=False) writes back (DeepSeek's encoding writes tool schemas and arguments so: the prompt's
bytes are these). floats at repr's switch between fixed and exponent, -0.0, ints past 64 bits, every escape, \\u
pairs, raw Unicode, duplicate keys, nesting; and inputs it refuses. the C module's test reads it (tests/test_json.c).
run: python3 tools/gen_json_fixtures.py > tests/fixtures/json.txt

a case a line: `O <input hex> <output hex>` (read and written back) or `X <input hex>` (refused)."""
import json
import sys

GOOD = [
    '0', '-0', '7', '-12', '123456789012345678901234567890', '1.0', '-0.0', '0.5', '0.1', '1.5', '2.25', '3.14159',
    '1e0', '1E2', '1e15', '1e16', '1.5e16', '9999999999999998', '1e21', '1e-4', '1e-5', '0.0001', '0.00001234',
    '-3e-7', '1.7976931348623157e308', '5e-324', '123456789.123456789', '0.30000000000000004', '100.0', '1e-07',
    'true', 'false', 'null', '""', '"a"', '"say \\"hi\\""', '"back\\\\slash"', '"\\/"', '"\\b\\f\\n\\r\\t"',
    '"\\u0001\\u001f\\u007f"', '"\\u00e9\\u4e2d"', '"\\ud83d\\ude00"', '"é中😀"', '"tab\\there"', '[]', '{}',
    '[1,2,3]', '[ 1 , [ 2 , [ 3 ] ] ]', '{"a":1,"b":[true,null],"c":{"d":"e"}}', '{"a":1,"a":2,"b":3}',
    '{"z":1,"y":2}', '{"ü":"√","emoji":"😀"}', ' \n\t {"k" : "v"} \r\n', '[1.0, 2, -0, 1e2, 2.5e-3]',
]
BAD = ['', '{', '[1,]', '{"a":}', '01', '1.', '.5', '+1', '"\\x41"', '"\\ud800"', '"a\nb"',
       '"raw\ttab"', "'single'", 'tru', '[1] 2', '{"a" 1}', '"\\u12"', '{1:2}']


def main():
    out = []
    for s in GOOD:
        out.append("O " + s.encode().hex() + " " + json.dumps(json.loads(s), ensure_ascii=False).encode().hex())
    for s in BAD:
        try:
            v = json.loads(s)
            enc = json.dumps(v, ensure_ascii=False).encode()  # a lone surrogate reads but writes no UTF-8
            raise SystemExit(f"python reads {s!r}: not a refusal case ({enc!r})")
        except (ValueError, UnicodeEncodeError):
            out.append("X " + s.encode().hex())
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
