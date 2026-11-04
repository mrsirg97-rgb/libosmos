"""tests/fixtures/tokenizer.txt: llama.cpp's own tokenization (llama-tokenize, its vocabulary read from the model's
first shard) of cases chosen to cross every branch of DeepSeek-V3's pre-tokenizer - numbers in threes, CJK and kana
runs, a punctuation mark before letters, letters with marks, every whitespace shape, newlines, text that ends in
spaces, Unicode spaces and symbols, special tokens - each with special tokens parsed and not. the C tokenizer's test
reads it (tests/test_tokenizer.c). run: python3 tools/gen_tok_fixtures.py <llama-tokenize> <first shard>
> tests/fixtures/tokenizer.txt

the file: a case a block - `S 0|1` (special tokens parsed), `T <the text's UTF-8 bytes in hex>`, `I <ids>`."""
import subprocess
import sys

CASES = [
    'Hello world!', 'hello', ' hello', '  hello', "Hello, World. How's it going?", "don't can't I'm we'll they've",
    '1', '12', '123', '1234', '12345678', '3.14159', 'v1.2.3', 'a1b22c333d4444', '$100,000.00', '1e10', 'x=-42',
    '日本語のテスト', '中文分词测试', 'カタカナとひらがな', '한국어 텍스트', '日本語abc123中文', '龥一', 'ゟ゠ヿ', 'café naïve résumé', 'é', 'à́b',
    'Ⅷ ½ ² ٣ ⅷ', 'αβγ δ', 'Привет, мир', 'مرحبا بالعالم', '!hello', '#define', '@user', '(paren)', '[x]', '{y}',
    'a.b', 'foo_bar', '--flag', '...', '!!!', '?!', '"quoted"', "it's", "'s", 'end.', 'x;\ny', 'a\n', 'a\n\n', '\n',
    '\n\n\n', '\r\n', 'a\r\nb', ' \n ', '  \n\n  x', '\t\tx', 'a  b', 'a   b', 'a    ', '    ', ' ', '  ', 'x\t',
    '\t', 'a \tb', 'a\xa0b', 'a\u3000b', 'a\u2003\u2003b', 'x\u2028y', '😀 emoji 🎉🎉', '→ ← ↑', '€£¥', '©®™', '∑∫√',
    '≤≥≠', '©x', 'a©b', '✓done', '°C', 'def f(x):\n    return x**2  # square\n', 'if (a && b) { return; }',
    '<div class="x">hi</div>', 'https://example.com/path?q=1&r=2', 'C:\\Users\\name',
    'snake_case camelCase PascalCase kebab-case', 'The quick brown fox jumps over the lazy dog. 0123456789',
    '<｜User｜>hi<｜Assistant｜>', '<｜begin▁of▁sentence｜>x', 'a<｜end▁of▁sentence｜>b', '<think>reason</think>',
    'Large language models (LLMs) are trained on 15T tokens; their 128K-token context windows -'
    ' and MoE layers with 384 experts - let them reason over long documents.\n\nSecond paragraph:'
    '\n  - item one\n  - item two\n',
]


def ids(exe, model, text, special):
    cmd = [exe, "-m", model, "--stdin", "--ids", "--no-escape", "--log-disable"]
    if not special:
        cmd.append("--no-parse-special")
    r = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, check=True)
    line = r.stdout.decode().strip().splitlines()[-1]
    return [int(x) for x in line.strip("[]").split(",") if x.strip()]


def main():
    exe, model = sys.argv[1], sys.argv[2]
    out = ["# llama.cpp's llama-tokenize over " + model.rsplit("/", 1)[-1] + " (tools/gen_tok_fixtures.py)"]
    for text in CASES:
        for special in (0, 1):
            out += [f"S {special}", "T " + text.encode("utf-8").hex(),
                    "I " + " ".join(map(str, ids(exe, model, text, special)))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
