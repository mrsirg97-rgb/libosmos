"""tools/bench/prompts.py MAYA_ROOT TOKENIZER_DIR OUT_DIR: the bench's prompts as token ids (comma separated) with
maya's tokenizer (its tools/strata_tokenizer.py) - DeepSeek-V4.1's chat turn, `<|begin|><|User|>q<|Assistant|></think>`.

  warmup.txt                 a short unrelated request (the warm-up: unlike the questions)
  q1.txt q2.txt q3.txt       maya's --bench questions (code, an explanation, a web page)
  q4.txt q5.txt              text a drafter knows: a CSV to JSON, a class's getters and setters
  p2048.txt p8192.txt        long prompts: the GPL-3's text (/usr/share/common-licenses/GPL-3) after a summarize
                             request, cut to exactly 2,048 and 8,192 tokens

run: python tools/bench/prompts.py ~/maya ~/models/ds1017/maya-ds4/tokenizer bench-prompts"""
import os
import sys

root, tok_dir, out = sys.argv[1:4]
sys.path.insert(0, os.path.join(root, "tools"))
import strata_tokenizer as ST  # noqa: E402

tok = ST.Tokenizer.from_dir(tok_dir)
BEGIN, USER, ASSISTANT = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜></think>"


def turn(q):
    return tok.encode(BEGIN + USER + q + ASSISTANT, parse_special=True)


rows = ["id,name,city,age"] + [f"{i},{n},{c},{a}" for i, (n, c, a) in enumerate([
    ("Alice", "Paris", 31), ("Bob", "Berlin", 45), ("Chen", "Shanghai", 28), ("Dana", "Toronto", 39),
    ("Eve", "Madrid", 52), ("Farid", "Cairo", 24), ("Grace", "Dublin", 36), ("Hiro", "Osaka", 41),
    ("Ines", "Lisbon", 29), ("Jon", "Oslo", 47), ("Kim", "Seoul", 33), ("Luis", "Lima", 26), ("Mia", "Vienna", 38),
    ("Nia", "Accra", 30), ("Omar", "Dubai", 44)], 1)]
prompts = {
    "warmup": "Say hello in five languages.",
    "q1": "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a "
          "docstring.",
    "q2": "Explain how photosynthesis works, step by step, for a high-school student.",
    "q3": "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.",
    "q4": "Convert this CSV to a JSON array of objects, one object per row, with id and age as numbers:\n\n" +
          "\n".join(rows),
    "q5": "Write a Python class Customer with private fields first_name, last_name, email, phone, street, city, state, "
          "zip_code, country, company, title, and notes, an __init__ taking all of them, and a getter and a setter "
          "method for each field.",
}
os.makedirs(out, exist_ok=True)
for name, q in prompts.items():
    with open(os.path.join(out, name + ".txt"), "w") as f:
        f.write(",".join(map(str, turn(q))))
text = open("/usr/share/common-licenses/GPL-3").read() * 2
head = tok.encode(BEGIN + USER + "Summarize this license in three sentences.\n\n", parse_special=True)
tail = tok.encode(ASSISTANT, parse_special=True)
body = tok.encode(text, parse_special=False)
for n in (2048, 8192):
    ids = head + body[: n - len(head) - len(tail)] + tail
    assert len(ids) == n
    with open(os.path.join(out, f"p{n}.txt"), "w") as f:
        f.write(",".join(map(str, ids)))
print("wrote", ", ".join(list(prompts) + ["p2048", "p8192"]), "to", out)
