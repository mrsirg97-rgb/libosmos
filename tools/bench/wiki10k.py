"""tools/bench/wiki10k.py MAYA_ROOT TOKENIZER_DIR OUT: the reference's Wiki text (DESIGN.md, "0.6.6: the reference, and
how a quant is scored") as token ids, comma separated - BOS, then the first 10,239 tokens of WikiText-2's test split
(Salesforce/wikitext, wikitext-2-raw-v1: its rows from Hugging Face's dataset viewer, joined as they come), tokenized
with maya's tokenizer as tools/bench/prompts.py's. rows 2,048 .. 10,239 are the working rows, 0 .. 2,047 the dev rows.

run: python tools/bench/wiki10k.py ~/maya ~/models/ds1017/maya-ds4/tokenizer bench-eval/wiki10k.txt
Qwen's (0.6.9: its reference, DESIGN.md): the same text by its own tokenizer through llama.cpp's llama-tokenize, no
BOS (Qwen's GGUF adds none) - the first 10,240 tokens, the same rows the working and dev rows:
     python tools/bench/wiki10k.py --qwen <llama-tokenize> <Qwen first shard> bench-eval/wiki10k_qwen.txt"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

N, CHARS = 10240, 200000


def rows(chars):
    text, off = "", 0
    while len(text) < chars:
        q = urllib.parse.urlencode({"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                                    "offset": off, "length": 100})
        for attempt in range(6):
            try:
                rq = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{q}",
                                            headers={"User-Agent": "osmos-bench"})
                data = json.load(urllib.request.urlopen(rq, timeout=60))
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        got = [r["row"]["text"] for r in data["rows"]]
        if not got:
            break
        text += "".join(got)
        off += len(got)
    return text[:chars]


def qwen(tokenize, shard, out):
    import json
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as t:
        t.write(rows(CHARS))
    try:
        p = subprocess.run([tokenize, "-m", shard, "--ids", "--log-disable", "--no-bos", "-f", t.name],
                           capture_output=True, check=True)
    finally:
        os.remove(t.name)
    ids = json.loads(p.stdout.decode().strip().splitlines()[-1])
    assert len(ids) >= N, len(ids)
    with open(out, "w") as f:
        f.write(",".join(map(str, ids[:N])))
    print(f"{out}: {N} tokens")


def main():
    if sys.argv[1] == "--qwen":
        return qwen(*sys.argv[2:5])
    root, tok_dir, out = sys.argv[1:4]
    sys.path.insert(0, os.path.join(root, "tools"))
    import strata_tokenizer as ST  # noqa: E402
    tok = ST.Tokenizer.from_dir(tok_dir)
    ids = tok.encode("<｜begin▁of▁sentence｜>", parse_special=True) + tok.encode(rows(CHARS))
    assert ids[0] == 0 and len(ids) >= N, (ids[:3], len(ids))
    with open(out, "w") as f:
        f.write(",".join(map(str, ids[:N])))
    print(f"{out}: {N} tokens")


if __name__ == "__main__":
    main()
