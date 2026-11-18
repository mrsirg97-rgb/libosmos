"""tests/fixtures/chat_hy4.jsonl: Hy4-flash's own chat template (tokenizer.chat_template in its GGUF) as transformers
renders it - jinja2's sandbox with trim_blocks, lstrip_blocks, loopcontrols, its tojson (json.dumps, ensure_ascii
False) - over chats chosen to cross the template's branches: a leading system message or none (one synthesized for
the tools and the reasoning effort), each effort (no_think, high, max; none: high), preserved thinking on and off
(its default: on with tools), earlier reasoning kept or dropped (the last user query), `reasoning` over
`reasoning_content`, tools (their `defer_loading` / `strict` keys dropped, `function` wrappers or bare), tool calls
(every argument type: a string raw, the rest JSON), consecutive tool results clustered and reordered by the calls'
ids (or kept in order when the ids do not match), text parts, other roles, and the template's refusals. then
completions parsed back: each assistant message the template renders, cut after the generation prompt, read back to
itself (its reasoning, content and calls - an argument a string where the tool's schema says so, else its JSON); and
completions out of the format (refused). the C port's test reads it (tests/test_chat_hy4.c).
run: python3 tools/gen_chat_fixtures_hy4.py <Hy4-flash GGUF | chat_template.jinja> [name] > tests/fixtures/chat_<name>.jsonl
(and its template beside it, tests/fixtures/chat_<name>.template; name: hy4, AngelSlim's GGUF's - or hy4_release,
the release's chat_template.jinja: ':opensource', no 'low')

a line a case: {"kind": "render", "messages", "tools", "effort", "preserve", "out" | "error"} or
{"kind": "parse", "text", "effort", "tools", "out" | "error"}."""
import json
import os
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from hy4_gguf import Gguf  # noqa: E402


def template(path):
    src = open(path).read() if path.endswith(".jinja") else Gguf(path).kv["tokenizer.chat_template"]

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src), src


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city.", "strict": True,
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "units": {"type": "string", "enum": ["c", "f"]},
        "detail": {"type": "boolean"}, "spots": {"type": "array", "items": {"type": "string"}},
        "where": {"type": "object"}, "ratio": {"type": "number"}, "note": {"anyOf": [{"type": "string"},
                                                                                      {"type": "null"}]}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Search the web. 中文",
                                           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
HIDDEN = {"type": "function", "function": {"name": "later", "defer_loading": True, "parameters": {"type": "object"}}}
BARE = {"name": "bare_tool", "description": "no function wrapper", "parameters": {"type": "object",
                                                                                  "properties": {"x": {}}}}
TOOLS = [WEATHER, SEARCH]

CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {
    "city": "Paris", "days": 3, "units": "c", "detail": True, "spots": ["a", "b"], "where": {"lat": 1.5},
    "ratio": 0.25, "note": None}}}
CALL2 = {"id": "c2", "type": "function", "function": {"name": "search", "arguments": {"q": "x <y> \"z\"\nw"}}}
BARE_CALL = {"name": "search", "arguments": {"q": "bare call"}}


def m(role, content=None, **kw):
    out = {"role": role, "content": content}
    out.update(kw)
    return out


RENDERS = [
    ([m("user", "hi")], None, None, None),
    ([m("user", "hi")], None, "no_think", None),
    ([m("user", "hi")], None, "max", None),
    ([m("system", "be brief"), m("user", "hi")], None, None, None),
    ([m("system", "be brief"), m("user", "hi")], TOOLS, "high", None),
    ([m("user", "hi")], TOOLS, None, None),
    ([m("user", "hi")], TOOLS + [HIDDEN], None, None),
    ([m("user", "hi")], [BARE], None, None),
    ([m("system", ""), m("user", "hi")], TOOLS, None, None),
    ([m("user", "q1"), m("assistant", "a1", reasoning_content="r1"), m("user", "q2")], None, None, None),
    ([m("user", "q1"), m("assistant", "a1", reasoning_content="r1"), m("user", "q2")], None, None, True),
    ([m("user", "q1"), m("assistant", "a1", reasoning_content="r1"), m("user", "q2")], None, "no_think", True),
    ([m("user", "q1"), m("assistant", "a1", reasoning="r-new", reasoning_content="r-old")], None, None, True),
    ([m("user", "q1"), m("assistant", "a1", reasoning="", reasoning_content="r-old")], None, None, True),
    ([m("user", "q1"), m("assistant", "a1", reasoning_content="r1")], None, None, None),
    ([m("user", "w?"), m("assistant", "let me look", reasoning_content="need tool", tool_calls=[CALL, CALL2]),
      m("tool", "sunny", tool_call_id="c1"), m("tool", "found", tool_call_id="c2"), m("user", "thanks")], TOOLS,
     None, None),
    ([m("user", "w?"), m("assistant", None, tool_calls=[CALL, CALL2]), m("tool", "found", tool_call_id="c2"),
      m("tool", "sunny", tool_call_id="c1")], TOOLS, None, None),
    ([m("user", "w?"), m("assistant", None, tool_calls=[CALL, CALL2]), m("tool", "one", tool_call_id="c9"),
      m("tool", "two", tool_call_id="c1")], TOOLS, None, None),
    ([m("user", "w?"), m("assistant", "", tool_calls=[BARE_CALL]), m("tool", "r")], TOOLS, None, None),
    ([m("user", [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}])], None, None, None),
    ([m("user", "hi"), m("assistant", "partial")], None, None, None),
    ([m("user", "hi"), m("developer", "dev note"), m("user", "again")], None, None, None),
    ([m("user", "  spaced  \n"), m("assistant", " a ", reasoning_content=" r ")], None, None, True),
    ([m("user", "hi")], None, "low", None),
]

PARSES = [  # assistant messages: rendered by the template, cut after the generation prompt, read back
    (m("assistant", "plain answer", reasoning_content="some reasoning"), None, "high"),
    (m("assistant", "plain answer"), None, "no_think"),
    (m("assistant", "", reasoning_content="r", tool_calls=[CALL]), TOOLS, "high"),
    (m("assistant", "before calls", reasoning_content="r", tool_calls=[CALL, CALL2]), TOOLS, "max"),
    (m("assistant", "x", tool_calls=[CALL2]), TOOLS, "no_think"),
    (m("assistant", "多语言 text\n\nwith lines", reasoning_content="思考"), None, "high"),
]
BAD = [  # completions out of the format: (text after the generation prompt, effort)
    ("no end of turn", "no_think"),
    ("reasoning without its end<｜hy_end:{T}｜>", "high"),
    ("</think:{T}>a<tool_calls:{T}><tool_call:{T}>f<arg_key:{T}>k</arg_key:{T}><｜hy_end:{T}｜>", "high"),
    ("</think:{T}>text after<tool_calls:{T}></tool_calls:{T}>more<｜hy_end:{T}｜>", "high"),
    ("</think:{T}>a<｜hy_start:{T}｜>b<｜hy_end:{T}｜>", "high"),
    ("</think:{T}><tool_calls:{T}><tool_call:{T}>f<arg_key:{T}>k</arg_key:{T}><arg_value:{T}>1</arg_value:{T}>"
     "<arg_key:{T}>k</arg_key:{T}><arg_value:{T}>2</arg_value:{T}></tool_call:{T}></tool_calls:{T}><｜hy_end:{T}｜>",
     "high"),
]


def tag(src):
    """the template's special tokens' suffix (HYTK: ':opensource' in the release, ':6124c78e' in AngelSlim's GGUF)"""
    key = "{%- set HYTK = '"
    at = src.index(key) + len(key)
    return src[at:src.index("'", at)]


def main():
    t, src = template(sys.argv[1])
    T = tag(src).lstrip(":")
    out = []
    for messages, tools, effort, preserve in RENDERS:
        kw = {"messages": messages, "tools": tools, "add_generation_prompt": True}
        if effort is not None:
            kw["reasoning_effort"] = effort
        if preserve is not None:
            kw["preserved_thinking"] = preserve
        case = {"kind": "render", "messages": messages, "tools": tools, "effort": effort, "preserve": preserve}
        try:
            case["out"] = t.render(**kw)
        except jinja2.exceptions.TemplateError as e:
            case["error"] = str(e)
        out.append(case)
    for msg, tools, effort in PARSES:
        convo = [m("user", "q"), msg]
        prompt = t.render(messages=[m("user", "q")], tools=tools, reasoning_effort=effort, add_generation_prompt=True)
        full = t.render(messages=convo, tools=tools, reasoning_effort=effort, preserved_thinking=True,
                        add_generation_prompt=False)
        assert full.startswith(prompt[:prompt.rindex("<｜hy_start")]), "the prompt not the conversation's start"
        head = full[len(prompt[:prompt.rindex("<｜hy_start")]):]  # the assistant turn: its header, then the model's
        gen = prompt[prompt.rindex("<｜hy_start"):]
        assert head.startswith(gen), "the assistant turn not the generation prompt's continuation"
        want = {"role": "assistant", "content": msg["content"] or "", "reasoning_content":
                (msg.get("reasoning_content") or "") if effort != "no_think" else "",
                "tool_calls": [{"type": "function", "function": {"name": c["function"]["name"], "arguments":
                                json.dumps(c["function"]["arguments"], ensure_ascii=False)}}
                               for c in msg.get("tool_calls", [])]}
        out.append({"kind": "parse", "text": head[len(gen):], "effort": effort, "tools": tools, "out": want})
    for text, effort in BAD:
        out.append({"kind": "parse", "text": text.replace("{T}", T), "effort": effort, "tools": TOOLS, "error": True})
    for case in out:
        sys.stdout.write(json.dumps(case, ensure_ascii=False) + "\n")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures",
                           "chat_" + (sys.argv[2] if len(sys.argv) > 2 else "hy4") + ".template"), "w") as f:
        f.write(src)  # the template the cases were rendered by: the port reads its suffix and efforts from it


if __name__ == "__main__":
    main()
