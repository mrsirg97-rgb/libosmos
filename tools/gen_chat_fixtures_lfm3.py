"""tests/fixtures/chat_lfm3.jsonl: LFM3's own chat template (tokenizer.chat_template in vibes1.7b's GGUF: LiquidAI's
LFM3-1.7B's chat_template.jinja) as transformers renders it - jinja2's sandbox with trim_blocks, lstrip_blocks,
loopcontrols, its tojson (json.dumps, ensure_ascii False), {% generation %} (transformers' assistant-mask tag: a block
that renders its body) - over chats chosen to cross the template's branches: a leading system message and one later,
tools (a deferred one skipped), thinking kept after the last user message or preserved, dropped before it (its text
to the last </think> cut), an assistant's reasoning under its three names, tool calls (every argument type, content
before them), tool results merged by call id or each its own, a result already wrapped, text parts, a role as given;
the template's refusals and osmos's (images, videos, tool-search forms). then completions parsed back (qchat_parse's
format: each assistant turn the template renders, cut at the generation prompt) and out of the format (refused).
the C port's test reads it (tests/test_chat_lfm3.c).
run: python3 tools/gen_chat_fixtures_lfm3.py <the GGUF> <llama.cpp>/gguf-py > tests/fixtures/chat_lfm3.jsonl

a line a case: {"kind": "render", "messages", "tools", "thinking", "preserve", "out" | "error", "osmos_refuses"?} or
{"kind": "parse", "text", "thinking", "tools", "out" | "error"}."""
import json
import sys

import jinja2
import jinja2.ext
from jinja2 import nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment

EOS = "<|im_end|>"
BOS = "<|startoftext|>"


class Generation(jinja2.ext.Extension):
    """transformers' {% generation %}: its body rendered as is (the assistant mask's markers, not text)"""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.Scope(body).set_lineno(lineno)


def template(path, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    src = GGUFReader(path).fields["tokenizer.chat_template"].contents()

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols, Generation])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src)


def render(tpl, messages, tools, thinking, preserve, gen=True):
    kw = {"messages": messages, "add_generation_prompt": gen, "enable_thinking": thinking, "bos_token": BOS}
    if tools is not None:
        kw["tools"] = tools
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
    if "<|im_start|>" in body or EOS in body or "<|endoftext|>" in body or "<|startoftext|>" in body:
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
MERGED = [{"role": "user", "content": "go"}, A_CALLS,
          {"role": "tool", "tool_call_id": "c1", "content": "part one"},
          {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "part two"}, "three"]},
          {"role": "tool", "tool_call_id": "c1", "content": ""}]
HISTORY = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": " add \n"},
           {"role": "user", "content": "and 3+3?"}]
DEFERRED = {"type": "function", "defer_loading": True, "function": {"name": "later", "parameters": {}}}

# (messages, tools, thinking, preserve)
RENDER = [
    ([U], None, True, None),
    ([U], None, False, None),
    ([{"role": "system", "content": "Be brief."}, U], None, True, None),
    ([{"role": "system", "content": "  Be brief.\n"}, U], None, False, None),
    ([{"role": "system", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "two"}]}, U],
     None, True, None),
    ([{"role": "system", "content": None}, U], None, False, None),
    ([{"role": "system", "content": ""}, U], TOOLS, True, None),
    ([U, {"role": "assistant", "content": "hi"}, {"role": "system", "content": "late"}, U], None, True, None),
    ([{"role": "developer", "content": "dev"}, U], None, True, None),
    (HISTORY, None, True, None),
    (HISTORY, None, True, False),
    (HISTORY, None, False, None),
    (HISTORY, None, True, True),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "r1</think>  final ", "thinking": "t"},
      {"role": "user", "content": "q2"}], None, True, None),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a</think>b</think> c "}], None, True, None),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "x", "reasoning": "via reasoning"}],
     None, True, None),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "x", "thinking": None}], None, True, None),
    ([{"role": "user", "content": " \x1c padded \xa0 "}], None, True, None),
    ([{"role": "user", "content": [{"type": "text", "text": "a "}, "b", {"k": 1}]}], None, True, None),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}], None, True, None),
    ([{"role": "user", "content": [{"type": "video", "video": "x"}]}], None, True, None),
    ([{"role": "user", "content": 3}], None, True, None),
    ([{"role": "critic", "content": "x"}], None, True, None),
    ([U], TOOLS, True, None),
    ([U], TOOLS, False, None),
    ([U], TOOLS + [DEFERRED], True, None),
    ([{"role": "system", "content": "You are helpful."}, U], TOOLS, True, None),
    ([{"role": "system", "content": "You are helpful."}, U], [], True, None),
    ([{"role": "system", "content": "You are helpful."}] + TOOL_TURNS, TOOLS, True, None),
    (TOOL_TURNS, TOOLS, True, False),
    (MERGED, TOOLS, True, None),
    (TOOL_TURNS + [{"role": "assistant", "content": "Paris sun, Tōkyō rain.", "reasoning_content": "both in"},
                   {"role": "user", "content": "thanks"}], TOOLS, True, False),
    ([{"role": "user", "content": "go"},
      {"role": "assistant", "content": " Running it. ", "tool_calls": [
          call("run", {"cmd": "ls -la\n/tmp", "env": {"A": "1", "B": [1, 2.5, None]}, "args": ["x", 2],
                       "dry": True, "rate": 0.25, "note": None}),
          call("run", {"cmd": "echo \"hi\""}, wrap=False), call("run", {"cmd": "\"quoted\"", "rate": 2})]},
      {"role": "tool", "content": "ok"}], TOOLS, True, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [call("run", {})]}],
     TOOLS, False, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": "x", "tool_calls": [call("run", "")]}],
     TOOLS, True, None),
    ([{"role": "user", "content": "go"}, {"role": "assistant", "content": "x", "tool_calls": [call("run", "{}")]}],
     TOOLS, True, None),
    ([{"role": "user", "content": "go"},
      {"role": "assistant", "content": "x", "tool_calls": [call("run", "{\"cmd\": \"ls\"}")]}], TOOLS, True, None),
    ([{"role": "user", "content": "go"}, {"role": "tool", "content": "<tool_response>\nwrapped\n</tool_response>"}],
     None, True, None),
    ([{"role": "user", "content": "go"}, {"role": "tool", "content": None}], None, True, None),
    ([{"role": "user", "content": "go"}, {"role": "tool", "content": {"type": "text", "text": "a part"}}], None, True, None),
    ([{"role": "user", "content": "go"}, {"role": "tool", "content": [{"type": "tool_reference", "name": "run"}]}],
     TOOLS, True, None),
    ([{"role": "user", "content": "go"}, {"role": "tool", "additional_tools": [WEATHER["function"]], "content": ""}],
     None, True, None),
    ([{"role": "user", "content": "Unicode: 日本語 — ü 😀 \"quotes\" \\ back"}], TOOLS, True, None),
]

PARSE = [
    ("Just text." + EOS, False, None),
    ("let me think\n</think>\n\nThe answer is 4." + EOS, True, None),
    ("no end", False, None),
    ("no think close" + EOS, True, None),
    ("a<|im_start|>b" + EOS, False, None),
    ("a<|startoftext|>b" + EOS, False, None),
    ("</think>empty reasoning" + EOS, True, None),
    ("x</think><tool_call>\n<function=get_weather>\n<parameter=city>\n3\n</parameter>\n<parameter=days>\n3"
     "\n</parameter>\n</function>\n</tool_call>" + EOS, True, TOOLS),
    ("<tool_call>\n<function=f>\n</function>\n</tool_call>\ntrailing" + EOS, False, None),
]


def osmos_refuses(messages):
    """osmos's own refusals: text only, no tool-search forms"""
    for m in messages:
        if m.get("additional_tools"):
            return True
        c = m.get("content")
        parts = c if isinstance(c, list) else [c] if isinstance(c, dict) else []
        for p in parts:
            if isinstance(p, dict) and ("image" in p or "image_url" in p or "video" in p or
                                        p.get("type") in ("image", "video", "image_url", "tool_reference") or
                                        "result" in p):
                return True
    return False


def roundtrip(tpl, messages, i, tools):
    """message i (an assistant's), the last: the template's text for it after the generation prompt's "<think>" is
    the completion; parsed, the message the template read"""
    head = render(tpl, messages[:i], tools, True, None)
    whole = render(tpl, messages[:i + 1], tools, True, None, gen=False)
    gen = "<|im_start|>assistant\n<think>"
    assert head.endswith(gen) and whole.startswith(head[:-len(gen)])
    turn = whole[len(head) - len(gen):]
    assert turn.startswith(gen) and turn.endswith(EOS + "\n"), turn
    text = turn[len(gen):-1]
    return {"kind": "parse", "text": text, "thinking": True, "tools": tools, "out": parse(text, True, tools)}


def main():
    tpl = template(sys.argv[1], sys.argv[2])
    out = []
    for messages, tools, thinking, preserve in RENDER:
        case = {"kind": "render", "messages": messages, "tools": tools, "thinking": thinking, "preserve": preserve}
        try:
            case["out"] = render(tpl, messages, tools, thinking, preserve)
        except jinja2.exceptions.TemplateError as e:
            case["error"] = str(e)
        if osmos_refuses(messages):
            case["osmos_refuses"] = True
        out.append(case)
        last = len(messages) - 1
        m = messages[last] if messages else {}
        cont = m.get("content")
        if (m.get("role") == "assistant" and "error" not in case and not case.get("osmos_refuses") and
                not (isinstance(cont, str) and "</think>" in cont) and
                all(isinstance(c.get("function", c).get("arguments"), dict) for c in m.get("tool_calls") or [])):
            out.append(roundtrip(tpl, messages, last, tools))
    for text, thinking, tools in PARSE:
        case = {"kind": "parse", "text": text, "thinking": thinking, "tools": tools}
        try:
            case["out"] = parse(text, thinking, tools)
        except ValueError as e:
            case["error"] = str(e)
        out.append(case)
    sys.stdout.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out))


if __name__ == "__main__":
    main()
