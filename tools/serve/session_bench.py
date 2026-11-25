#!/usr/bin/env python3
"""tools/serve/session_bench.py - a recorded agent session replayed through a pack's osmos-api, step by step (agent_bench's
sibling: one real session, not a seeded workload). each of the session's assistant steps asks the model for its next
step given the recording to that point - the original model's own calls and their real results - and the answer is
checked, then dropped: the recording goes on. no tool runs. what it measures is the engine's, not the model's: a step
continues the conversation the engine keeps (its prompt read only past what it holds), its answer parses in the pack's
own dialect, it does not loop, the same answer twice (--twice: a second replay), drafted the same as plain (--plain: a
second replay with the drafter off), the GPU's memory where it started.

    python tools/serve/session_bench.py --fixture rig-resume-oneshot.json --config ~/Models/osmos/<pack>/osmos.json \\
        [--api bin/osmos-api] [--max-tokens 192] [--steps 0] [--twice] [--plain] [--answers OUT.json] [--against BASE.json]

the fixture: tools/serve/rig_session.py's (a rig session, sanitized). each step greedy (temperature 0), at most
--max-tokens; a fresh slot folder a run (nothing from an earlier one). --answers: each run's answers saved (a
baseline); --against: the replay's answers each step the baseline's (a change meant to keep the bits - a kernel moved
- needs only the drafted replay against a baseline that was drafted == plain; a change of bits, drafted and --plain,
its answers the new baseline). exit 0: every step answered and parsed, and the same as asked.

SIGTERM or SIGINT: the server stopped and its folder removed, as at the end."""
import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

GPU = "/sys/class/drm/card0/device"


def gpu_mib():
    try:
        return sum(int(open(f"{GPU}/{f}").read()) for f in ("mem_info_gtt_used", "mem_info_vram_used")) // 1048576
    except OSError:
        return -1


class Api:
    """an osmos-api on the pack's config (its slots in a fresh folder; drafter off for --plain), started and waited for"""

    def __init__(self, exe, config, port, plain, log, settings):
        self.dir = tempfile.mkdtemp(prefix="session-bench-")
        cfg = json.load(open(config))
        cfg.update(settings)  # the pack's, as the installer writes them (layout.py: cfg.update(pack["settings"]))
        cfg["slots"] = dict(cfg.get("slots") or {}, dir=os.path.join(self.dir, "slots"))
        if plain:
            cfg.pop("drafter", None)
        self.cfg = os.path.join(self.dir, "osmos.json")
        json.dump(cfg, open(self.cfg, "w"))
        self.port, self.log = port, open(log, "w")
        self.p = subprocess.Popen([exe, "--config", self.cfg, "--port", str(port)], stdout=self.log, stderr=self.log)

    def wait_ready(self, log):
        for _ in range(3600):
            if self.p.poll() is not None:
                sys.exit(f"session_bench: osmos-api exited (see {log})")
            try:
                m = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=5).read())
                if any(x.get("osmos", {}).get("ready") for x in m.get("data", [])):  # (a model's own readiness)
                    return
            except OSError:
                pass
            time.sleep(1)
        sys.exit("session_bench: osmos-api not ready in an hour")

    def step(self, body):
        """one streamed request: (its answer: content, reasoning, tool calls, finish), its first token's and last
        token's seconds from the send, its usage and timings, its loop header"""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        t0 = time.monotonic()
        first = None
        out = {"content": "", "reasoning": "", "calls": {}, "finish": None}
        usage = timings = None
        with urllib.request.urlopen(req, timeout=7200) as r:
            loop = r.headers.get("x-osmos-loop", "")
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[6:])
                usage, timings = ch.get("usage") or usage, ch.get("timings") or timings
                for c in ch.get("choices", []):
                    d = c.get("delta", {})
                    if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                        first = time.monotonic() - t0
                    out["content"] += d.get("content") or ""
                    out["reasoning"] += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        e = out["calls"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        f = tc.get("function", {})
                        e["name"] += f.get("name") or ""
                        e["arguments"] += f.get("arguments") or ""
                    out["finish"] = c.get("finish_reason") or out["finish"]
        return out, first, time.monotonic() - t0, usage or {}, timings or {}, loop

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        shutil.rmtree(self.dir, ignore_errors=True)


def parsed(out):
    """the answer in the pack's dialect: tool calls each a name and JSON arguments, or text"""
    for c in out["calls"].values():
        try:
            json.loads(c["arguments"] or "{}")
        except ValueError:
            return False
        if not c["name"]:
            return False
    return bool(out["calls"] or out["content"].strip() or out["finish"] == "length")


def last_call(messages):
    """the recording's last tool call before a step (name, arguments), or None"""
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            f = m["tool_calls"][-1]["function"]
            return f["name"], f["arguments"]
    return None


def replay(fx, api, max_tokens, steps):
    rows, held = [], 0  # held: the last step's prompt (what the engine keeps of it, at least)
    for i, at in enumerate(steps):
        body = {"messages": fx["messages"][:at], "tools": fx["tools"], "max_tokens": max_tokens, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, **fx.get("request", {})}
        out, first, total, usage, t, _ = api.step(body)
        before = last_call(fx["messages"][:at])
        repeat = any((c["name"], c["arguments"]) == before for c in out["calls"].values())
        prompt, cached = usage.get("prompt_tokens", 0), usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        # continued: the engine held all of the last step's prompt but its last token (read again for its logits) - a
        # template that renders an earlier turn anew (an old turn's reasoning dropped) reads from that point: reported
        rows.append({"step": i, "prompt": prompt, "cached": cached, "new": prompt - cached, "held": held,
                     "continued": i == 0 or cached + 1 >= held, "first": first, "total": total, "finish": out["finish"],
                     "parsed": parsed(out), "calls": [c["name"] for c in out["calls"].values()],
                     "decode_tps": t.get("predicted_per_second", 0), "repeat": repeat,
                     "answer": json.dumps([out["content"], sorted(out["calls"].items())])})
        held = prompt
        r = rows[-1]
        print(f"  step {i:3d}: prompt {prompt:6d} (cached {cached:6d}, new {r['new']:5d}){'' if r['continued'] else ' re-read'}"
              f"  first {first or 0:6.2f} s  done {total:6.2f} s  {out['finish'] or '-':10s} {','.join(r['calls']) or 'text'}"
              f"{'' if r['parsed'] else '  UNPARSED'}{'  REPEATS THE LAST CALL' if repeat else ''}", flush=True)
    return rows


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def report(rows, wall, mem0, mem1):
    n = len(rows)
    print(f"  steps {n}; wall {wall:.1f} s; GPU memory {mem0} -> {mem1} MiB")
    print(f"  first token  p50 {pct([r['first'] for r in rows], .5):6.2f}  p95 {pct([r['first'] for r in rows], .95):6.2f} s")
    print(f"  step done    p50 {pct([r['total'] for r in rows], .5):6.2f}  p95 {pct([r['total'] for r in rows], .95):6.2f} s")
    print(f"  decode       median {statistics.median([r['decode_tps'] for r in rows] or [0]):6.1f} tok/s")
    rr = [r["step"] for r in rows if not r["continued"]]
    print(f"  prompt read  {sum(r['new'] for r in rows)} new tokens, {sum(r['cached'] for r in rows)} reused; "
          f"{n - len(rr)} of {n} steps continued in place" + (f" (read again from an earlier point: steps {rr})" if rr else ""))
    print(f"  answers      {sum(r['parsed'] for r in rows)} of {n} parsed; "
          f"{sum(1 for r in rows if r['calls'])} tool calls, {sum(1 for r in rows if not r['calls'])} text; "
          f"{sum(r['repeat'] for r in rows)} repeating the recording's last call")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--config", required=True, help="the pack's osmos.json")
    ap.add_argument("--api", default="bin/osmos-api")
    ap.add_argument("--port", type=int, default=8250)
    ap.add_argument("--max-tokens", type=int, default=192, help="a step's answer at most")
    ap.add_argument("--steps", type=int, default=0, help="the first N steps (0: all)")
    ap.add_argument("--twice", action="store_true", help="a second replay: every step's answer the same")
    ap.add_argument("--plain", action="store_true", help="a second replay drafter off: every step's answer the same")
    ap.add_argument("--log", default=None)
    ap.add_argument("--answers", help="each run's answers saved here (JSON: run -> step -> answer)")
    ap.add_argument("--against", help="a saved --answers: the replay's every step its replay's answer")
    ap.add_argument("--pack", help="the pack's id: its settings from --packs over the config, as the installer's")
    ap.add_argument("--packs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "install", "packs.json"))
    a = ap.parse_args()
    fx = json.load(open(a.fixture))
    steps = fx["steps"][: a.steps] if a.steps else fx["steps"]
    log = a.log or os.path.join(tempfile.gettempdir(), "session_bench.api.log")
    settings = {}
    if a.pack:
        packs = json.load(open(a.packs))["packs"]
        if a.pack not in packs:
            sys.exit(f"session_bench: no pack {a.pack!r} in {a.packs}")
        settings = packs[a.pack]["settings"]
    print(f"session_bench: {fx['source']} - {len(steps)} steps, {a.config}" + (f" with {a.pack}'s settings {settings}" if a.pack else ""))
    runs = [("replay", False)] + ([("again", False)] if a.twice else []) + ([("plain", True)] if a.plain else [])
    base = json.load(open(a.against)) if a.against else None
    def stop_on(sig, _):  # (a background job's SIGINT is ignored: SIGTERM too) - finally stops the server
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, stop_on)
    signal.signal(signal.SIGINT, stop_on)
    results, ok = {}, True
    for name, plain in runs:
        print(f"{name}:")
        api = Api(a.api, a.config, a.port, plain, log, settings)
        try:
            api.wait_ready(log)
            mem0, t0 = gpu_mib(), time.monotonic()
            rows = replay(fx, api, a.max_tokens, steps)
            mem1 = gpu_mib()
        finally:
            api.stop()
        report(rows, time.monotonic() - t0, mem0, mem1)
        results[name] = rows
        ok &= all(r["parsed"] for r in rows)
    for name in ("again", "plain"):
        if name in results:
            same = sum(x["answer"] == y["answer"] for x, y in zip(results["replay"], results[name]))
            print(f"  {name}: {same} of {len(steps)} steps the same answer as the replay's")
            ok &= same == len(steps)
    if base is not None:
        was = base.get("replay", [])
        diff = [r["step"] for r, w in zip(results["replay"], was) if r["answer"] != w]
        n = min(len(was), len(steps))
        print(f"  against {a.against}: {n - len(diff)} of {n} steps the baseline's answer"
              + (f" (differ: steps {diff[:20]}{' ..' if len(diff) > 20 else ''})" if diff else "")
              + ("" if len(was) == len(steps) else f" - the baseline {len(was)} steps, this {len(steps)}"))
        ok &= not diff and len(was) == len(steps)
    if a.answers:
        json.dump({"fixture": fx["source"], "config": a.config, "pack": a.pack,
                   **{k: [r["answer"] for r in v] for k, v in results.items()}}, open(a.answers, "w"), indent=0)
        print(f"  answers saved: {a.answers}")
    print("every step answered and parsed" + (" and as asked" if a.against or a.twice or a.plain else "")
          + ("" if ok else " - NOT ALL (see above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
