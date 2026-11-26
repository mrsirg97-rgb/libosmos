"""rig_session.py - a rig session (its sqlite: messages, tool_calls) to a session bench fixture (session_bench.py):
the request's preamble as rig sends it (a captured request: its system prompt, tools, reasoning effort), then the
session in OpenAI chat form - each user turn, each assistant step (its reasoning as reasoning_content, its text, its
tool calls) and each tool call's result - and the steps: the assistant messages a replay asks for. Sanitized for
publishing: the email addresses, the given names (a username, a GitHub handle), the home directory and anything
token-shaped each replaced by a placeholder, and counted in the report (review it before the fixture goes anywhere).

    rig_session.py --db SESSIONS.sqlite --session ID_PREFIX --request CAPTURED.json --out FIXTURE.json
                   [--name sirg --name mrsirg97-rgb ...] [--home /home/sirg]

The captured request: rig run once against a local server that records its body (rig -base-url URL -model ID -p
TEXT, a model row with no launch command) - its system prompt and tool definitions are rig's own, today's."""
import argparse
import json
import re
import sqlite3
import sys

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SCOPES = re.compile(r"(Token scopes:)[^\n]*")  # gh auth status: what a token may do
TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})\b")


class Sanitizer:
    """each value's replacements counted by kind; the same rules on every string of the fixture"""

    def __init__(self, names, home):
        self.names = sorted({n for n in names if n}, key=len, reverse=True)  # the longest first: a handle before its stem
        self.home = home.rstrip("/") if home else None
        self.count = {"email": 0, "token": 0, "scopes": 0, "home": 0, "name": 0}

    def __call__(self, s):
        if not isinstance(s, str):
            return s
        s, n = EMAIL.subn("user@example.com", s)
        self.count["email"] += n
        s, n = TOKEN.subn("<token>", s)
        self.count["token"] += n
        s, n = SCOPES.subn(r"\1 <scopes>", s)
        self.count["scopes"] += n
        if self.home:
            n = s.count(self.home)
            s = s.replace(self.home, "/home/user")
            self.count["home"] += n
        for name in self.names:
            s, n = re.subn(re.escape(name), "user", s)
            self.count["name"] += n
        return s

    def walk(self, x):
        if isinstance(x, dict):
            return {k: self.walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.walk(v) for v in x]
        return self(x)


def session_id(db, prefix):
    ids = [r[0] for r in db.execute("select id from sessions where id like ?", (prefix + "%",))]
    if len(ids) != 1:
        sys.exit(f"rig_session: {len(ids)} sessions start {prefix!r} (one wanted)")
    return ids[0]


def messages_of(db, sid):
    """the session in chat form, and the indices of its assistant messages (the replay's steps)"""
    calls = {}
    for seq, cid, name, args, result, err in db.execute(
            "select message_seq, id, name, args, result, err from tool_calls where session_id = ? order by started_at", (sid,)):
        calls.setdefault(seq, []).append((cid, name, args, result if err is None else f"error: {err}"))
    out, steps = [], []
    for seq, role, content, reasoning in db.execute(
            "select seq, role, content, reasoning from messages where session_id = ? order by seq", (sid,)):
        if role == "user":
            out.append({"role": "user", "content": content or ""})
            continue
        if role != "assistant":
            sys.exit(f"rig_session: message {seq}: a role {role!r} not read")
        m = {"role": "assistant", "content": content or ""}
        if reasoning:
            m["reasoning_content"] = reasoning
        mine = calls.get(seq, [])
        if mine:
            m["tool_calls"] = [{"id": c, "type": "function", "function": {"name": n, "arguments": a or "{}"}} for c, n, a, _ in mine]
        steps.append(len(out))
        out.append(m)
        out.extend({"role": "tool", "tool_call_id": c, "content": r or ""} for c, _, _, r in mine)
    return out, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--session", required=True, help="the session id, or a prefix of it")
    ap.add_argument("--request", required=True, help="a request rig sent (its system prompt and tools)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", action="append", default=[], help="a name to replace (a username, a handle): repeatable")
    ap.add_argument("--home", help="the home directory to replace (default: none)")
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    sid = session_id(db, a.session)
    model, started, version = db.execute("select model, started_at, version from sessions where id = ?", (sid,)).fetchone()
    req = json.load(open(a.request))
    system = [m for m in req["messages"] if m["role"] == "system"]
    if len(system) != 1:
        sys.exit("rig_session: the captured request has no single system message")
    convo, steps = messages_of(db, sid)
    clean = Sanitizer(a.name, a.home)
    fixture = clean.walk({
        "source": {"harness": f"rig {version}", "model": model, "started": started[:10],
                   "preamble": "the captured request's (rig today's): its system prompt and tools"},
        "request": {k: req[k] for k in ("reasoning_effort", "chat_template_kwargs") if k in req},
        "tools": req.get("tools", []),
        "messages": system + convo,
        "steps": [s + 1 for s in steps],  # (past the system message)
    })
    json.dump(fixture, open(a.out, "w"), indent=1)
    users = sum(1 for m in fixture["messages"] if m["role"] == "user")
    tools = sum(1 for m in fixture["messages"] if m["role"] == "tool")
    print(f"{a.out}: {len(fixture['messages'])} messages ({users} user, {len(steps)} assistant steps, {tools} tool "
          f"results), {len(fixture['tools'])} tools; replaced: " + ", ".join(f"{k} {v}" for k, v in clean.count.items()))


if __name__ == "__main__":
    main()
