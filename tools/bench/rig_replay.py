#!/usr/bin/env python3
"""tools/bench/rig_replay.py - a drafter A/B's prompts from real agent work: points of rig's saved sessions (an
assistant turn's context: the conversation before it, its tool calls and their results) rendered with the model's own
chat template (tokenizer.chat_template in its GGUF, as tools/gen_chat_fixtures_qwen.py renders it) and tokenized by
llama.cpp's llama-tokenize (the vocabulary alone), an ids file a point - q4x_bench's input.

    python3 -I tools/bench/rig_replay.py <rig dir> <out dir> --shard <Qwen first shard> --gguf-py <llama.cpp>/gguf-py \\
        --tokenize <llama.cpp>/build-hip/bin/llama-tokenize [--points 16] [--min 1000] [--max 8000] [--model 3.8] [--seed 1]

a point a session (those run on a model whose name holds --model, 6 messages or more), its turn one of the
session's first 8 assistant turns after a user message or a tool result, its context --min .. --max tokens (a
bench's prompt time bounded: the earlier turns of long sessions). the sessions are read only; the ids files hold
the users' conversations - keep them out of the repository."""
import argparse
import glob
import json
import os
import random
import sqlite3
import subprocess
import sys

MIN_MESSAGES = 6  # a session long enough to hold a turn after a tool's result
FIRST_TURNS = 8   # a point among a session's first assistant turns: their context within a bench's prompt


def conversation(c, sid):
    """the session's messages as the template takes them: an assistant's reasoning and tool calls, then the calls'
    results as tool messages"""
    calls = {}
    for seq, name, args, res in c.execute("select message_seq, name, args, coalesce(result, '') from tool_calls "
                                          "where session_id = ? order by rowid", (sid,)):
        calls.setdefault(seq, []).append((name, args, res))
    msgs = []
    for seq, role, content, reasoning in c.execute("select seq, role, content, coalesce(reasoning, '') from messages "
                                                   "where session_id = ? order by seq", (sid,)):
        if role == "user":
            msgs.append({"role": "user", "content": content})
            continue
        m = {"role": "assistant", "content": content}
        if reasoning:
            m["reasoning_content"] = reasoning
        cs = calls.get(seq, [])
        if cs:
            tcs = []
            for name, args, _ in cs:
                try:
                    a = json.loads(args)
                except ValueError:
                    a = {"raw": args}
                tcs.append({"type": "function", "function": {"name": name, "arguments": a}})
            m["tool_calls"] = tcs
        msgs.append(m)
        msgs += [{"role": "tool", "content": res} for _, _, res in cs]
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rig")
    ap.add_argument("out")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--tokenize", required=True)
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--min", type=int, default=1000)
    ap.add_argument("--max", type=int, default=8000)
    ap.add_argument("--model", default="3.8")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from gen_chat_fixtures_qwen import render, template
    tpl = template(a.shard, a.gguf_py)
    os.makedirs(a.out, exist_ok=True)
    scratch = os.path.join(a.out, ".text")

    def tokens(text):
        with open(scratch, "w") as f:
            f.write(text)
        p = subprocess.run([a.tokenize, "-m", a.shard, "--ids", "--log-disable", "-f", scratch], capture_output=True,
                           check=True)
        return json.loads(p.stdout.decode().strip().splitlines()[-1])

    rng = random.Random(a.seed)
    sessions = []
    for path in sorted(glob.glob(os.path.join(a.rig, "sessions", "*.sqlite"))):
        c = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        q = ("select s.id from sessions s join messages m on m.session_id = s.id where s.model like ? "
             "group by s.id having count(*) >= ?")
        sessions += [(path, sid) for (sid,) in c.execute(q, (f"%{a.model}%", MIN_MESSAGES))]
    rng.shuffle(sessions)
    made = 0
    for path, sid in sessions:
        if made >= a.points:
            break
        c = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        c.text_factory = lambda b: b.decode("utf-8", "replace")  # a tool's binary output
        msgs = conversation(c, sid)
        turns = [k for k in range(1, len(msgs)) if msgs[k]["role"] == "assistant" and
                 msgs[k - 1]["role"] in ("user", "tool")][:FIRST_TURNS]
        rng.shuffle(turns)
        for k in turns:
            ids = tokens(render(tpl, msgs[:k], None, True, None, None))
            if a.min <= len(ids) <= a.max:
                with open(os.path.join(a.out, f"p{made:02d}.ids"), "w") as f:
                    f.write(",".join(map(str, ids)) + "\n")
                print(f"p{made:02d}: {len(ids)} tokens, {k} messages", flush=True)
                made += 1
                break
    if os.path.exists(scratch):
        os.remove(scratch)
    if made < a.points:
        sys.exit(f"rig_replay: {made} points of {a.points} (too few sessions with a turn of {a.min} .. {a.max} tokens)")


if __name__ == "__main__":
    main()
