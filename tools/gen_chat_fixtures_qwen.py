"""tests/fixtures/chat_qwen.jsonl: Qwen3.8-Flash-Next's own chat template (tokenizer.chat_template in its GGUF) as
transformers renders it - jinja2's sandbox with trim_blocks, lstrip_blocks, loopcontrols, its tojson
(json.dumps, ensure_ascii False) - over chats chosen to cross the template's branches: system and developer messages
merged, thinking on and off, each reasoning effort, earlier reasoning kept or dropped (preserve_thinking, the last
real query), tools and their schemas, tool calls (every argument type, content before them or none, a call without
its `function` wrapper), tool results grouped into one user turn, text parts, Python's whitespace trimmed, and the
template's refusals. then completions parsed back: each assistant turn the template renders, cut at the generation
prompt, read back to its message (reasoning, content, tool calls - a parameter a string where its schema says so,
else its JSON); and completions out of the format (refused). the C port's test reads it (tests/test_chat_qwen.c).
run: python3 tools/gen_chat_fixtures_qwen.py <first shard> <llama.cpp>/gguf-py > tests/fixtures/chat_qwen.jsonl

a line a case: {"kind": "render", "messages", "tools", "thinking", "effort", "preserve", "out" | "error", "text_only"?} or
{"kind": "parse", "text", "thinking", "tools", "out" | "error"}."""
import json
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

EOS = "<|im_end|>"


def template(shard, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    src = GGUFReader(shard).fields["tokenizer.chat_template"].contents()

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src)


def render(tpl, messages, tools, thinking, effort, preserve):
    kw = {"messages": messages, "add_generation_prompt": True, "enable_thinking": thinking}
    if tools is not None:
        kw["tools"] = tools
    if effort is not None:
        kw["reasoning_effort"] = effort
    if preserve is not None:
        kw["preserve_thinking"] = preserve
    return tpl.render(**kw)


# the parse: a completion after the generation prompt, through its end of turn, as the assistant message
def no_constant(s):
    raise ValueError(s)


def value_of(name, raw, schema):
    """the template writes a string raw and anything else as its JSON: the text read as JSON when that is a value
    other than a string, of a type the parameter's schema allows (its "type", or an anyOf / oneOf's: pydantic's
    Optional[T]) - else the text itself. no tool, parameter or type: a string, as vLLM's qwen3coder parser reads it"""
    s = schema.get(name) if isinstance(schema.get(name), dict) else {}
    alts = [a for k in ("anyOf", "oneOf") if isinstance(s.get(k), list) for a in s[k] if isinstance(a, dict)]
    allowed = set()
    for t in [s.get("type")] + [a.get("type") for a in alts]:
        allowed.update(x for x in (t if isinstance(t, list) else [t]) if isinstance(x, str))
    allowed = allowed or {"string"}
    try:
        v = json.loads(raw, parse_constant=no_constant)
    except ValueError:
        return raw
    kind = ("boolean" if isinstance(v, bool) else "null" if v is None else "integer" if isinstance(v, int)
            else "number" if isinstance(v, float) else "object" if isinstance(v, dict)
            else "array" if isinstance(v, list) else "string")
    ok = kind != "string" and (kind in allowed or (kind == "integer" and "number" in allowed))
    return v if ok else raw


def schemas(tools):
    out = {}
    for t in tools or []:
        f = t.get("function", t)
        out[f["name"]] = f.get("parameters", {}).get("properties", {})
    return out


def take(rest, lit):
    rest = rest.lstrip(" \t\r\n")
    if not rest.startswith(lit):
        raise ValueError("expected " + lit)
    return rest[len(lit):]


def upto(rest, stop):
    k = rest.find(stop)
    if k < 0:
        raise ValueError("no " + stop)
    return rest[:k], rest[k + len(stop):]


def parse(text, thinking, tools):
    if not text.endswith(EOS):
        raise ValueError("no end of turn")
    body = text[:-len(EOS)]
    if "<|im_start|>" in body or EOS in body or "<|endoftext|>" in body:
        raise ValueError("a special token in the text")
    reasoning = ""
    if thinking:
        reasoning, body = upto(body, "</think>")
        reasoning = reasoning.strip()
    k = body.find("<tool_call>")
    content = (body if k < 0 else body[:k]).strip()
    rest, calls, sch = ("" if k < 0 else body[k:]), [], schemas(tools)
    while rest.strip(" \t\r\n"):
        rest = take(rest, "<tool_call>")
        rest = take(rest, "<function=")
        name, rest = upto(rest, ">")
        if not name or "\n" in name or "<" in name:
            raise ValueError("a function's name")
        args = {}
        while True:
            r2 = rest.lstrip(" \t\r\n")
            if not r2.startswith("<parameter="):
                break
            pname, r2 = upto(r2[len("<parameter="):], ">")
            raw, rest = upto(r2, "</parameter>")
            if not pname or "\n" in pname or pname in args:
                raise ValueError("a parameter's name")
            raw = raw[1:] if raw.startswith("\n") else raw
            raw = raw[:-1] if raw.endswith("\n") else raw
            args[pname] = value_of(pname, raw, sch.get(name, {}))
        rest = take(rest, "</function>")
        rest = take(rest, "</tool_call>")
        calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
                                            "parameters": {"type": "object", "properties": {
                                                "city": {"type": "string", "description": "the city"},
                                                "days": {"type": "integer", "minimum": 1, "maximum": 7}},
                                                "required": ["city"]}}}
RUN = {"type": "function", "function": {"name": "run", "description": "Run a command — carefully.",
                                        "parameters": {"type": "object", "properties": {
                                            "cmd": {"type": "string"}, "env": {"type": "object"},
                                            "args": {"type": "array"}, "dry": {"type": "boolean"},
                                            "rate": {"type": "number"}, "note": {"type": ["string", "null"]}}}}}
TOOLS = [WEATHER, RUN]


def call(name, args, wrap=True):
    f = {"name": name, "arguments": args}
    return {"id": "c_" + name, "type": "function", "function": f} if wrap else f


U = {"role": "user", "content": "Hello"}
A_CALLS = {"role": "assistant", "content": "", "reasoning_content": "two cities",
           "tool_calls": [call("get_weather", {"city": "Paris", "days": 3}), call("get_weather", {"city": "Tōkyō"})]}
TOOL_TURNS = [{"role": "user", "content": "Weather in Paris and Tōkyō?"}, A_CALLS,
              {"role": "tool", "tool_call_id": "c1", "content": "sun, 21°C"},
              {"role": "tool", "tool_call_id": "c2", "content": "rain"}]
HISTORY = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": " add \n"},
           {"role": "user", "content": "and 3+3?"}]

# (messages, tools, thinking, effort, preserve)
RENDER = [
    ([U], None, True, None, None),
    ([U], None, False, None, None),
    ([U], None, True, "low", None),
    ([U], None, True, "medium", None),
    ([U], None, True, "high", None),
    ([U], None, True, "xhigh", None),
    ([U], None, True, "max", None),
    ([U], None, False, "max", None),
    ([], None, True, None, None),
    ([{"role": "system", "content": "Be brief."}, U], None, True, None, None),
    ([{"role": "system", "content": "Be brief."}, U], None, False, None, None),
    ([{"role": "system", "content": "Be brief."}, {"role": "developer", "content": "  Use metric.\n"},
      {"role": "system", "content": " 　 "}, U], None, True, "low", None),
    ([{"role": "system", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "two"}]}, U],
     None, True, None, None),
    ([{"role": "system", "content": None}, U], None, False, None, None),
    ([U, {"role": "assistant", "content": "hi"}, {"role": "system", "content": "late"}, U], None, True, None, None),
    (HISTORY, None, True, None, None),
    (HISTORY, None, True, None, False),
    (HISTORY, None, False, None, False),
    (HISTORY, None, True, None, True),
    ([{"role": "user", "content": "\x1c\x85 padded \xa0 "}], None, True, None, None),
    ([{"role": "user", "content": [{"type": "text", "text": "a "}, {"text": "b"}]}], None, True, None, None),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}], None, True, None, None),
    ([{"role": "user", "content": [{"type": "video", "video": "x"}]}], None, True, None, None),
    ([{"role": "user", "content": [{"type": "audio"}]}], None, True, None, None),
    ([{"role": "user", "content": 3}], None, True, None, None),
    ([{"role": "critic", "content": "x"}], None, True, None, None),
    ([U], TOOLS, True, None, None),
    ([U], TOOLS, False, None, None),
    ([U], TOOLS, True, "medium", None),
    ([{"role": "system", "content": "You are helpful."}, U], TOOLS, True, None, None),
    ([{"role": "system", "content": "You are helpful."}, U], [], True, None, None),
    ([{"role": "system", "content": "You are helpful."}] + TOOL_TURNS, TOOLS, True, None, None),
    (TOOL_TURNS, TOOLS, True, None, False),
    (TOOL_TURNS + [{"role": "assistant", "content": "Paris sun, Tōkyō rain.", "reasoning_content": "both in"},
                   {"role": "user", "content": "thanks"}], TOOLS, True, None, False),
    ([{"role": "user", "content": "go"},
      {"role": "assistant", "content": " Running it. ", "tool_calls": [
          call("run", {"cmd": "ls -la\n/tmp", "env": {"A": "1", "B": [1, 2.5, None]}, "args": ["x", 2],
                       "dry": True, "rate": 0.25, "note": None}),
          call("run", {"cmd": "echo \"hi\""}, wrap=False), call("run", {"cmd": "\"quoted\"", "rate": 2})]},
      {"role": "tool", "content": "ok"}], TOOLS, True, None, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [call("run", {})]}],
     TOOLS, False, None, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": "x", "tool_calls": [call("run", "")]}],
     TOOLS, True, None, None),
    ([{"role": "user", "content": "go"},
      {"role": "assistant", "content": "x", "tool_calls": [call("run", "{\"cmd\": \"ls\"}")]}], TOOLS, True, None, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": "x", "tool_calls": [call("run", [1])]}],
     TOOLS, True, None, None),
    ([{"role": "user", "content": "go"},
      {"role": "assistant", "content": "x", "tool_calls": [{"type": "function", "function": {"arguments": {}}}]}],
     TOOLS, True, None, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": "x", "tool_calls": [call("run", None)]}],
     TOOLS, True, None, None),
    ([{"role": "tool", "content": "orphan"}, U], None, True, None, None),
    ([{"role": "system", "content": "s"}, {"role": "tool", "content": "after system"}], None, True, None, None),
    ([U, {"role": "assistant", "content": "a", "reasoning_content": 7}], None, True, None, None),
    ([{"role": "user", "content": "<tool_response>\nx\n</tool_response>"},
      {"role": "assistant", "content": "y", "reasoning_content": "r"}], None, True, None, False),
    ([{"role": "user", "content": "Unicode: 日本語 — ü 😀 \"quotes\" \\ back"}], TOOLS, True, None, None),
]

# completions out of the format, or at its edges: (text, thinking, tools)
PARSE = [
    ("Just text." + EOS, False, None),
    ("let me think\n</think>\n\nThe answer is 4." + EOS, True, None),
    ("no end" , False, None),
    ("no think close" + EOS, True, None),
    ("a<|im_start|>b" + EOS, False, None),
    ("</think>\n\nempty reasoning" + EOS, True, None),
    ("x\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n3\n</parameter>\n<parameter=days>\n3"
     "\n</parameter>\n</function>\n</tool_call>" + EOS, True, TOOLS),
    ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
     "<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n<parameter=rate>\nNaN\n</parameter>\n"
     "<parameter=args>\n[1, \"a\"]\n</parameter>\n<parameter=extra>\n{\"k\": 1}\n</parameter>\n"
     "<parameter=x>\nnot json\n</parameter>\n</function>\n</tool_call>" + EOS, False, TOOLS),
    ("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>" + EOS, False, None),
    ("<tool_call>\n<function=save>\n<parameter=text>\n{\"a\": 1}\n</parameter>\n<parameter=n>\n5\n</parameter>\n"
     "<parameter=tag>\nnull\n</parameter>\n<parameter=raw>\ntrue\n</parameter>\n</function>\n</tool_call>" + EOS, False,
     [{"type": "function", "function": {"name": "save", "parameters": {"type": "object", "properties": {
         "text": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
         "tag": {"oneOf": [{"type": "string"}, {"type": "null"}]}, "raw": {"description": "no type"}}}}}]),
    ("<tool_call>\n<function=f>\n</function>\n</tool_call>\ntrailing" + EOS, False, None),
    ("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n"
     "</tool_call>" + EOS, False, None),
    ("<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>\n</tool_call>" + EOS, False, None),
    ("<tool_call>\n<function=>\n</function>\n</tool_call>" + EOS, False, None),
    ("<tool_call>\n<function=f>\n</function>" + EOS, False, None),
    ("text\n<tool_call>\n<function=f>\n<parameter=s>\n\n two lines \n\n</parameter>\n</function>\n</tool_call>"
     + EOS, False, None),
]


def main():
    tpl = template(sys.argv[1], sys.argv[2])
    out = []
    for messages, tools, thinking, effort, preserve in RENDER:
        case = {"kind": "render", "messages": messages, "tools": tools, "thinking": thinking, "effort": effort,
                "preserve": preserve}
        try:
            case["out"] = render(tpl, messages, tools, thinking, effort, preserve)
        except jinja2.exceptions.TemplateError as e:
            case["error"] = str(e)
        if media(messages):
            case["text_only"] = True  # rendered by the template, refused here
        out.append(case)
        # each assistant turn rendered, cut at the generation prompt: its completion, parsed back to it
        for i, m in enumerate(messages):
            if m.get("role") != "assistant" or "error" in case or preserve is False:
                continue
            calls = m.get("tool_calls") or []
            if any(not isinstance(c.get("function", c).get("arguments"), dict) for c in calls):
                continue
            out.append(roundtrip(tpl, messages, i, tools, effort))
    for text, thinking, tools in PARSE:
        case = {"kind": "parse", "text": text, "thinking": thinking, "tools": tools}
        try:
            case["out"] = parse(text, thinking, tools)
        except ValueError as e:
            case["error"] = str(e)
        out.append(case)
    sys.stdout.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out))


def media(messages):
    parts = [p for m in messages if isinstance(m.get("content"), list) for p in m["content"]]
    return any("image" in p or "image_url" in p or "video" in p or p.get("type") in ("image", "video") for p in parts)


def roundtrip(tpl, messages, i, tools, effort):
    """message i (an assistant's), rendered alone after its history in thinking mode: the template's text for it
    after the generation prompt's "<think>\\n" is the completion; parsed, the message the template read"""
    head = render(tpl, messages[:i], tools, True, effort, True)
    whole = tpl.render(messages=messages[:i + 1], tools=tools, add_generation_prompt=False,
                       reasoning_effort=effort or "xhigh")
    gen = "<|im_start|>assistant\n<think>\n"
    assert head.endswith(gen) and whole.startswith(head[:-len(gen)])
    turn = whole[len(head) - len(gen):]
    assert turn.startswith(gen) and turn.endswith(EOS + "\n")
    text = turn[len(gen):-1]
    m = messages[i]
    got = parse(text, True, tools)
    calls = [c.get("function", c) for c in m.get("tool_calls") or []]
    want_calls = [{"type": "function", "function": {"name": c["name"],
                                                     "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                  for c in calls]
    rc = m.get("reasoning_content")
    assert got == {"role": "assistant", "content": (m.get("content") or "").strip(),
                   "reasoning_content": rc.strip() if isinstance(rc, str) else "", "tool_calls": want_calls}, got
    return {"kind": "parse", "text": text, "thinking": True, "tools": tools, "out": got}


if __name__ == "__main__":
    main()
