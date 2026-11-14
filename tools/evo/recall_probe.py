#!/usr/bin/env python3
"""recall_probe.py - a short probe of one recalled fact from a long recall stream (DESIGN.md, "0.7.6: finding the
layers that cost a pack its recall"): the stream's tail (its questions, each answered) after only the last W tokens
of its body, the fact planted D tokens before the tail - so a recall that wavers at 1M is seen in minutes, at any
window and distance, instead of in an hour's whole read.

    python3 tools/evo/recall_probe.py --stream <prefix> --model <first shard> --fact <name> W D <out prefix>

--stream: the recall stream's <prefix>.txt (ids), <prefix>.mask (its scored rows) and <prefix>.json (its facts: name
under "colour", "code", "position" of its note in the body, "rows" its answer's rows) - tools of a private corpus
(the operator's sessions), never committed. --model: the pack the ids are of (osmos-api --tokenize reads its
tokenizer). --user / --assistant / --think-end: the template's markers (DeepSeek's by default). writes
<out>.txt and <out>.mask for kld (MASK=), the first scored row the answer's first token.
"""
import argparse
import json
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--stream", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--fact", required=True)
ap.add_argument("--api", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bin", "osmos-api"))
ap.add_argument("--user", default="<｜User｜>")
ap.add_argument("--assistant", default="<｜Assistant｜>")
ap.add_argument("--think-end", default="</think>")
ap.add_argument("--note", default="\n\nNote for later: the access code for the {name} locker is {code}.")
ap.add_argument("W", type=int)
ap.add_argument("D", type=int)
ap.add_argument("out")
a = ap.parse_args()


def tokenize(text):
    r = subprocess.run([a.api, "--tokenize", "--model", a.model], input=text.encode(), capture_output=True, check=True)
    return [int(x) for x in r.stdout.decode().split(",") if x.strip()]


ids = [int(x) for x in open(a.stream + ".txt").read().split(",")]
mask = [int(x) for x in open(a.stream + ".mask").read().split(",")]
meta = json.load(open(a.stream + ".json"))
USER, ASSIST, THINK_END = (tokenize(x)[-1] for x in (a.user, a.assistant, a.think_end))
specials = {USER, ASSIST, THINK_END}
first = min(f["rows"][0] for f in meta["facts"])
done = tokenize("Done.")
# the tail: from the turn that answers "Done." to the stream's end (its questions and answers)
ts = max(i for i in range(first - 200, first) if ids[i:i + 2] == [ASSIST, THINK_END] and ids[i + 2:i + 2 + len(done)] == done)
body, tail, tmask = ids[:ts], ids[ts:], mask[ts:]
fact = next(f for f in meta["facts"] if f["colour"] == a.fact)
note = tokenize(a.note.format(name=a.fact, code=fact["code"]))
p = fact["position"]  # the body's own note out: a window that reached it would hold two
assert body[p:p + len(note)] == note, "the fact's note not at its position (another stream, or another tokenizer)"
body = body[:p] + body[p + len(note):]
s = next(i for i in range(len(body) - a.W, len(body)) if body[i] == USER)  # the window: from a user turn
win = body[s:]
at = len(win) - a.D  # the note at the end of a plain user message's text about D tokens before the tail
spot = next(i for i in range(max(at, 1), len(win)) if win[i] == ASSIST and win[i - 1] not in specials and
            win[max(j for j in range(i) if win[j] in specials)] == USER)
win = win[:spot] + note + win[spot:]
stream = [ids[0]] + win + tail
smask = [0] * (1 + len(win)) + tmask
open(a.out + ".txt", "w").write(",".join(map(str, stream)))
open(a.out + ".mask", "w").write(",".join(map(str, smask)))
print(f"{a.out}: {len(stream)} tokens, {a.fact} {len(win) - spot} back of the tail, {sum(smask)} scored rows")
