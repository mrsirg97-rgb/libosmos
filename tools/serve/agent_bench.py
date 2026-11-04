#!/usr/bin/env python3
"""tools/serve/agent_bench.py - agents on --parallel N, each engine driven alike (maya's tools/ds4/agent_bench.py,
ported: its seeded workload, its timing): each agent a conversation of turns - a system prompt and task, then the
conversation so far with the model's answer and a tool's result - a tool's time between an answer and the next turn,
every agent at once. per turn: its first token and its last from its send.

    python tools/serve/agent_bench.py --model <first shard> --picks <file> --dspark <dir> \\
        [--maya-exe <maya-ds4> --maya-pack <maya pack>] [--agents 4] [--turns 8] [--seed 1]

  turn 1      the system prompt and task: 4,000-12,000 tokens
  turn 2..    a tool's result: 150-2,000 tokens nine times in ten, 5,000-20,000 (a file read) one in ten
  answers     64-400 tokens;  tool time 0.5-5 s between an answer and the next turn

the text: real text (the bench's token files, ~/bench/osmos-prompts), each agent's and turn's its own slice
(maya's: its pack's tokenizer over a corpus). every turn drafts (spec=1) on both engines; a conversation stays in its
lane, so a turn reads only its new tokens. exit 0: every turn of every run answered.
"""
import argparse
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import Multi, read_ids  # noqa: E402


def workload(seed, agents, turns):
    """maya's: per agent, its turns - new tokens, the answer's max_new, the tool time after it."""
    rng = random.Random(seed)
    plan = []
    for _ in range(agents):
        ts = []
        for k in range(turns):
            if k == 0:
                new = rng.randint(4000, 12000)
            elif rng.random() < 0.1:
                new = rng.randint(5000, 20000)
            else:
                new = int(150 * (2000 / 150) ** rng.random())
            ts.append({"new": new, "max_new": rng.randint(64, 400), "tool_s": rng.uniform(0.5, 5.0)})
        plan.append(ts)
    return plan


class Text:
    """real text's token ids, a slice of n from a seeded place (wrapping): every agent's and turn's its own"""

    def __init__(self, ids):
        self.ids = ids

    def take(self, key, n):
        at = random.Random(key).randrange(len(self.ids))
        return [self.ids[(at + i) % len(self.ids)] for i in range(n)]


def run(label, exe, args, env, plan, text):
    eng = Multi(exe, args, env)
    send_mu = threading.Lock()  # (a prompt's line is far past a pipe's atomic write: one sender at a time)
    turns, errors = [], []
    rid_next = [0]
    warm = threading.Barrier(len(plan) + 1)
    t0 = [0.0]
    done_at = [0.0] * len(plan)

    def agent(i):
        history = []
        warm.wait()
        for k, t in enumerate(plan[i]):
            prompt = history + text.take(f"{i}.{k}", t["new"])
            with send_mu:
                rid = rid_next[0] = rid_next[0] + 1
                ts = time.monotonic()
                eng.send(rid, prompt, t["max_new"], spec=1)
            r = eng.wait(rid)
            if "err" in r or not r["tokens"]:
                errors.append(f"agent {i} turn {k}: {r.get('err', 'no tokens')}")
                return
            with eng.cv:
                times = [tt for tt, kind, rr in eng.log if kind == "T" and rr == rid]
            turns.append({"agent": i, "turn": k, "new": t["new"], "reused": r.get("reused", 0),
                          "first": times[0] - ts, "done": times[-1] - ts, "tokens": len(r["tokens"])})
            history = prompt + r["tokens"]
            if k + 1 < len(plan[i]):
                time.sleep(t["tool_s"])
        done_at[i] = time.monotonic() - t0[0]

    threads = [threading.Thread(target=agent, args=(i,)) for i in range(len(plan))]
    for th in threads:
        th.start()
    warm.wait()
    t0[0] = time.monotonic()
    for th in threads:
        th.join()
    wall = time.monotonic() - t0[0]
    eng.quit()
    return {"label": label, "turns": turns, "errors": errors, "wall": wall, "agents": done_at}


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else 0.0


def report(res, total):
    print(f"\n{res['label']}", flush=True)
    for e in res["errors"]:
        print(f"  ERR {e}", flush=True)
    ts = res["turns"]
    print(f"  turns answered {len(ts)} of {total}; wall {res['wall']:.1f} s; agents done "
          + ", ".join(f"{x:.0f}" for x in res["agents"]) + " s", flush=True)
    if not ts:
        return False
    f, d = [t["first"] for t in ts], [t["done"] for t in ts]
    print(f"  first token  mean {statistics.mean(f):6.2f}  p50 {pct(f, 50):6.2f}  p95 {pct(f, 95):6.2f}  max {max(f):6.2f} s")
    print(f"  turn done    mean {statistics.mean(d):6.2f}  p50 {pct(d, 50):6.2f}  p95 {pct(d, 95):6.2f}  max {max(d):6.2f} s")
    short = [t["done"] for t in ts if t["new"] <= 2000]
    if short:
        print(f"  turns of <= 2,000 new tokens: done mean {statistics.mean(short):.2f}, p95 {pct(short, 95):.2f} s "
              f"({len(short)} turns)")
    later = [t for t in ts if t["turn"] > 0]
    print(f"  later turns that continued their conversation in place: {sum(1 for t in later if t['reused'] > 0)} of "
          f"{len(later)}; answer tokens {sum(t['tokens'] for t in ts)}", flush=True)
    return not res["errors"] and len(ts) == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--picks")
    ap.add_argument("--dspark", required=True)
    ap.add_argument("--exe", default="bin/osmos-serve")
    ap.add_argument("--budget", default="96")
    ap.add_argument("--max-tokens", default="2048")
    ap.add_argument("--maya-exe")
    ap.add_argument("--maya-pack")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompts", default=os.path.expanduser("~/bench/osmos-prompts"))
    ap.add_argument("--only", choices=("both", "osmos", "maya"), default="both")
    a = ap.parse_args()
    text = Text(read_ids(f"{a.prompts}/calib16k.txt") + read_ids(f"{a.prompts}/p8192.txt"))
    plan = workload(a.seed, a.agents, a.turns)
    news = [t["new"] for p in plan for t in p]
    total = sum(len(p) for p in plan)
    print(f"{a.agents} agents x {a.turns} turns (seed {a.seed}): new tokens a turn {min(news)}-{max(news)} (median "
          f"{int(statistics.median(news))}), {sum(news)} in all; answers up to {sum(t['max_new'] for p in plan for t in p)}",
          flush=True)
    ok = True
    if a.only in ("both", "osmos"):
        args = ["--model", a.model, "--dspark", a.dspark, "--budget", a.budget, "--max-tokens", a.max_tokens,
                "--max-context", str(a.ctx), "--parallel", str(a.agents)] + (["--picks", a.picks] if a.picks else [])
        env = dict(os.environ, OSMOS_SLOT_KEEP="0")
        ok &= report(run(f"osmos-serve (budget {a.budget} GiB, {a.max_tokens}-token steps)", a.exe, args, env,
                         plan, text), total)
    if a.only in ("both", "maya") and a.maya_exe:
        args = ["--pack", a.maya_pack, "--max-context", str(a.ctx), "--tp", "1", "--threads", "on", "--reserve-mib",
                "1024", "--vision", "off", "--dspark", "on", "--parallel", str(a.agents)]
        env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
                   MAYA_DS4_SLOT_KEEP="0")
        ok &= report(run("maya-ds4 (its config's settings)", a.maya_exe, args, env, plan, text), total)
    print("\nevery turn answered" if ok else "\na turn failed (above)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
