"""tests/fixtures/tokenizer.txt (DeepSeek's) and tokenizer_qwen35.txt: llama.cpp's own tokenization (llama-tokenize,
its vocabulary read from the model's first shard) of cases chosen to cross every branch of the model's pre-tokenizer -
DeepSeek-V3's numbers in threes, CJK and kana runs, a punctuation mark before letters; Qwen3.5's contractions in any
case, single digits, marks in a letter run, a lone mark, punctuation runs into newlines; both: every whitespace shape,
newlines, text that ends in spaces, Unicode spaces and symbols, the family's special tokens - each with special tokens
parsed and not. the C tokenizer's test reads them (tests/test_tokenizer.c). run:
python3 tools/gen_tok_fixtures.py <llama-tokenize> <first shard> [qwen35] > tests/fixtures/tokenizer[_qwen35].txt
Inkling-2's (o200k's expression: an upper run before lowers, title case, marks and other letters in both runs, digits
in threes, a punctuation run into slashes and newlines) from its own tokenizer.json (Hugging Face's tokenizers - the
model's reference; llama.cpp's inkling pre-tokenizer is not in this tree):
python3 tools/gen_tok_fixtures.py --hf <tokenizer.json> inkling > tests/fixtures/tokenizer_inkling.txt
LFM3's (vibes1.7b: pre lfm2 - the Llama-3 expression, ignore_merges: digits in threes, contractions in any case,
marks outside a letter run, a word the vocabulary holds whole) from its checkpoint's tokenizer.json:
python3 tools/gen_tok_fixtures.py --hf <tokenizer.json> lfm2 > tests/fixtures/tokenizer_lfm2.txt

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
    '<think>reason</think>',
    'Large language models (LLMs) are trained on 15T tokens; their 128K-token context windows -'
    ' and MoE layers with 384 experts - let them reason over long documents.\n\nSecond paragraph:'
    '\n  - item one\n  - item two\n',
]
SPECIALS = {
    'deepseek': ['<｜User｜>hi<｜Assistant｜>', '<｜begin▁of▁sentence｜>x', 'a<｜end▁of▁sentence｜>b'],
    'qwen35': ['<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n', 'a<|endoftext|>b',
               '<tool_call>\n<function=f>\n</function>\n</tool_call>', "I'M HE'S WE'RE THEY'VE YOU'LL I'D 'T",
               "'x 'S' ''s", 'a\u0301b \u0301x \u0301 x\u0301\u0302', 'नमस्ते दुनिया', 'ภาษาไทย', '?!\n\n.x', ' ...\r\n',
               '-\u0301', '1\u0301', '12,345.67', '½x', 'a\x00b\x01', '\ufffd', '\U0010FFFD \U000E0001',
               "'Day", "'Time", "'Model", "x'Very", "'Return", "'Llama", 'end.\n\nNext', '{\n}\n', '});\n\n', 'a:\n  b', '"\u0301', '.\u0301\u0301',
               ' \u093e', '(\u0301)', 'x \u0301\u0301y'],
    'lfm2': ['<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>', 'a<|im_end|>b', '<|startoftext|>x',
             '<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>',
             '<tool_response>\nok\n</tool_response>', "I'M HE'S WE'RE THEY'VE YOU'LL I'D 'T", "'x 'S' ''s",
             'a\u0301b \u0301x \u0301 x\u0301\u0302', 'नमस्ते दुनिया', 'ภาษาไทย', '?!\n\n.x', ' ...\r\n',
             '-\u0301', '1\u0301', '12,345.67 1234567', '½x ²3', 'a\x00b\x01', '\ufffd', '\U0010FFFD \U000E0001',
             "x'Very 'Return 's'", 'end.\n\nNext', '{\n}\n', '});\n\n', 'a:\n  b', '"\u0301', '.\u0301\u0301',
             ' information internationalization antidisestablishmentarianism', ' the The THE tHe',
             '                ', '\n\n\n\n\n\n\n\n\n\n\n', 'ʰello ˈstress', 'ǅ ᾈᾉ ΣΑΣ σας İstanbul ı'],
    'inkling': ['<|message_user|><|content_text|>hi<|end_message|><|message_model|>', 'a<|end_message|>b',
                '<|content_thinking|>x<|content_invoke_tool_json|>{"a":1}', "I'M HE'S WE'RE THEY'VE YOU'LL I'D 'T",
                'HTTPServer XMLHttpRequest ABCdef aBC AbC', '\u01c5ungla \u01c4UNGLA \u01c6ungla', 'MacDonald McDONALD',
                'a\u0301b \u0301x \u0301 x\u0301\u0302 X\u0301Y', 'ʰello ʰELLO ˈstress', '中文abc 日本語ABC', 'ﬁne ﬀ',
                'a//b\n', ' ...//\r\n', '?!/\n\nx', '12,345.67 1234567', '½x ²3', 'नमस्ते दुनिया', 'ภาษาไทย',
                'a\x00b\x01', '\ufffd', '\U0010FFFD \U000E0001', "x'Very 'Return 's'", 'end.\n\nNext', '});\n\n',
                '"\u0301', '.\u0301\u0301', '-\u0301', '1\u0301', 'ǅ', 'ᾈᾉ', 'ΣΑΣ σας', 'İstanbul ı'],
}


def ids(exe, model, text, special):
    cmd = [exe, "-m", model, "--stdin", "--ids", "--no-escape", "--log-disable"]
    if not special:
        cmd.append("--no-parse-special")
    r = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, check=True)
    line = r.stdout.decode().strip().splitlines()[-1]
    return [int(x) for x in line.strip("[]").split(",") if x.strip()]


def hf_ids(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)

    def ids(_exe, _model, text, special):
        tok.encode_special_tokens = not special
        return tok.encode(text, add_special_tokens=False).ids
    return ids


def main():
    global ids
    if sys.argv[1] == "--hf":
        ids = hf_ids(sys.argv[2])
        sys.argv[1:3] = ["hf", sys.argv[2]]
    exe, model = sys.argv[1], sys.argv[2]
    family = sys.argv[3] if len(sys.argv) > 3 else 'deepseek'
    who = "Hugging Face's tokenizers over " if exe == "hf" else "llama.cpp's llama-tokenize over "
    out = ["# " + who + model.rsplit("/", 1)[-1] + " (tools/gen_tok_fixtures.py)"]
    for text in CASES + SPECIALS[family]:
        for special in (0, 1):
            out += [f"S {special}", "T " + text.encode("utf-8").hex(),
                    "I " + " ".join(map(str, ids(exe, model, text, special)))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
