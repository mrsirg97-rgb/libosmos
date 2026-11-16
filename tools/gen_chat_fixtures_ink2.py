"""tests/fixtures/chat_ink2.jsonl: Inkling-2-Small's own chat template (the checkpoint's chat_template.jinja) as
transformers renders it (apply_chat_template over the checkpoint's tokenizer) - over chats chosen to cross its
branches: the effort line's place (before the first message past the leading system ones, else last) and its every
spelling (named, numbers, the template's refusals), tool declarations (their defaults, sorted at every depth), each
role's header (a tool result's name, or every call its id answers; an assistant's name), reasoning blocks, content as
a string, text parts or none (an assistant's empty content with calls or reasoning not rendered), tool calls (their
arguments' top keys sorted, values compact and unsorted), the end of every model turn, and the template's refusals.
the C port's test reads it (tests/test_chat_ink2.c). run:
python3 tools/gen_chat_fixtures_ink2.py <Inkling-2-Small checkpoint dir> > tests/fixtures/chat_ink2.jsonl

then completions parsed back: each assistant turn the template renders after a user message, read back to its message
(reasoning, content, calls - their arguments as Python's json.dumps writes what the model wrote), and completions out
of the format (refused).

a line a case: {"kind": "render", "messages", "tools", "effort", "out" | "error"} - effort absent: the template's
default (0.9) - or {"kind": "parse", "text", "out" | "error"}."""
import json
import sys

from transformers import AutoTokenizer

ABSENT = object()


def render(tok, messages, tools, effort):
    kw = {"tokenize": False}
    if tools is not None:
        kw["tools"] = tools
    if effort is not ABSENT:
        kw["reasoning_effort"] = effort
    return tok.apply_chat_template(messages, **kw)


def call(name, args, cid=None):
    return {"id": cid or "c_" + name, "type": "function", "function": {"name": name, "arguments": args}}


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
                                            "parameters": {"type": "object", "required": ["city"], "properties": {
                                                "days": {"type": "integer", "minimum": 1, "maximum": 7},
                                                "city": {"type": "string", "description": "the city"}}}}}
RUN = {"type": "function", "function": {"name": "run", "description": "Run a command — carefully.",
                                        "parameters": {"type": "object", "properties": {
                                            "cmd": {"type": "string"}, "env": {"type": "object"},
                                            "Args": {"type": "array"}, "dry": {"type": "boolean"},
                                            "rate": {"type": "number"}, "note": {"type": ["string", "null"]}}}}}
TOOLS = [WEATHER, RUN]
U = {"role": "user", "content": "Hello"}
S = {"role": "system", "content": "Be brief."}
A_CALLS = {"role": "assistant", "content": "", "reasoning_content": "two cities",
           "tool_calls": [call("get_weather", {"city": "Paris", "days": 3}, "c1"),
                          call("get_weather", {"city": "Tōkyō"}, "c2")]}
TOOL_TURNS = [{"role": "user", "content": "Weather in Paris and Tōkyō?"}, A_CALLS,
              {"role": "tool", "tool_call_id": "c1", "content": "sun, 21°C"},
              {"role": "tool", "tool_call_id": "c2", "content": "rain"}]

# (messages, tools, effort)
RENDER = [
    ([U], None, ABSENT),
    ([U], None, None),
    ([U], None, "none"), ([U], None, "minimal"), ([U], None, "low"), ([U], None, "medium"), ([U], None, "high"),
    ([U], None, "xhigh"), ([U], None, "max"), ([U], None, " low "),
    ([U], None, 0), ([U], None, 0.0), ([U], None, 0.5), ([U], None, 0.25), ([U], None, 0.99), ([U], None, 1e-05),
    ([U], None, 0.123456789), ([U], None, 1.0), ([U], None, -0.1), ([U], None, "extreme"), ([U], None, "0.5"),
    ([U], None, False), ([U], None, True), ([U], None, 1),
    ([], None, ABSENT),
    ([S], None, ABSENT),
    ([S, U], None, ABSENT),
    ([S, S, U], None, "low"),
    ([U, S, U], None, ABSENT),
    ([{"role": "developer", "content": "Use metric."}, U], None, ABSENT),
    ([{"role": "critic", "content": "x"}], None, ABSENT),
    ([{"role": "system", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "two"}]}, U],
     None, ABSENT),
    ([{"role": "user", "content": [{"type": "text", "text": "a "}, {"text": "b"}, "c", {"type": "input_text",
                                                                                        "text": "d"}]}], None, ABSENT),
    ([{"role": "user", "content": [{"type": "text"}, {"type": "text", "text": 5}]}], None, ABSENT),
    ([{"role": "user", "content": [{"type": "audio"}]}], None, ABSENT),
    ([{"role": "user", "content": [{"type": "video"}]}], None, ABSENT),
    ([{"role": "user", "content": ""}], None, ABSENT),
    ([{"role": "user", "content": None}], None, ABSENT),
    ([{"role": "user", "content": []}], None, ABSENT),
    ([{"role": "user"}], None, ABSENT),
    ([U, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}], None, ABSENT),
    ([U, {"role": "assistant", "content": "4", "reasoning_content": " add \n"}, U], None, ABSENT),
    ([U, {"role": "assistant", "content": "", "reasoning_content": "only thought"}], None, ABSENT),
    ([U, {"role": "assistant", "content": ""}], None, ABSENT),
    ([U, {"role": "assistant", "content": None}], None, ABSENT),
    ([U, {"role": "assistant", "content": "x", "reasoning_content": ""}], None, ABSENT),
    ([U, {"role": "assistant", "content": "named", "name": "helper"}], None, ABSENT),
    ([U], TOOLS, ABSENT),
    ([U], TOOLS, "low"),
    ([S, U], TOOLS, ABSENT),
    ([U], [], ABSENT),
    ([U], [{"type": "function", "function": {"name": "bare"}}], ABSENT),
    ([U], [{"name": "unwrapped", "description": "no function key", "parameters": {"b": 1, "a": {"z": 0, "y": 1}}}],
     ABSENT),
    ([U], [{"type": "", "function": {"name": "t", "description": "", "parameters": {}}}], ABSENT),
    ([S] + TOOL_TURNS, TOOLS, ABSENT),
    (TOOL_TURNS + [{"role": "assistant", "content": "Paris sun, Tōkyō rain.", "reasoning_content": "both in"},
                   {"role": "user", "content": "thanks"}], TOOLS, "high"),
    ([U, {"role": "tool", "name": "get_weather", "content": "named result"}], TOOLS, ABSENT),
    ([U, {"role": "tool", "content": "no id"}], TOOLS, ABSENT),
    ([U, {"role": "tool", "tool_call_id": "nobody", "content": "orphan"}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": "", "tool_calls": [call("run", {"cmd": "a"}, "same"),
                                                               call("get_weather", {"city": "x"}, "same")]},
      {"role": "tool", "tool_call_id": "same", "content": "two names"}], TOOLS, ABSENT),
    ([{"role": "user", "content": "go"},
      {"role": "assistant", "content": " Running it. ", "tool_calls": [
          call("run", {"cmd": "ls -la\n/tmp", "env": {"B": [1, 2.5, None], "A": "1"}, "Args": ["x", 2], "dry": True,
                       "rate": 0.25, "note": None, "big": 12345678901234567890, "e": 1e20, "u": "日本 \"q\" \\"}),
          call("run", {"cmd": "\"quoted\"", "rate": 2})]},
      {"role": "tool", "tool_call_id": "c_run", "content": "ok"}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": None, "tool_calls": [call("run", {})]}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": None, "tool_calls": [call("run", None)]}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": "x", "tool_calls": [call("run", "{\"cmd\": \"ls\"}")]}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": "x", "tool_calls": [call("run", [1])]}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": "x", "tool_calls": [{"type": "function", "function": {"arguments": {}}}]}],
     TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": "x", "name": "other", "tool_calls": [call("run", {})]}], TOOLS, ABSENT),
    ([U, {"role": "assistant", "content": "x", "name": "run", "tool_calls": [call("run", {"z": 1, "a": 2})]}], TOOLS,
     ABSENT),
    ([{"role": "user", "content": "Unicode: 日本語 — ü 😀 \"quotes\" \\ back <|end_message|>"}], TOOLS, ABSENT),
]


EOS, END, MODEL = "<|content_model_end_sampling|>", "<|end_message|>", "<|message_model|>"
KINDS = {"<|content_thinking|>": "thinking", "<|content_text|>": "text", "<|content_invoke_tool_json|>": "call"}
SPECIALS = [EOS, END, MODEL, "<|message_user|>", "<|message_system|>", "<|message_tool|>", "<|content_xml|>",
            "<|content_tool_error|>", "<|content_image|>"] + list(KINDS)


def no_constant(x):
    raise ValueError("a non-finite number")


def parse(text):
    """a completion (the model's turn, through its end) as the assistant message: its blocks in turn - a thinking
    block's body the reasoning, a text block's the content (each kind's joined), a call block's {"name", "args"} a
    tool call (its header the name; args an object) - and nothing after the end of turn"""
    if not text.endswith(EOS):
        raise ValueError("no end of turn")
    rest, reasoning, content, calls = text[:-len(EOS)], "", "", []
    while rest:
        if not rest.startswith(MODEL):
            raise ValueError("a block without its header")
        rest = rest[len(MODEL):]
        k = rest.find("<|content_")
        if k < 0:
            raise ValueError("a block without its kind")
        name, rest = rest[:k], rest[k:]
        kind = next((v for t, v in KINDS.items() if rest.startswith(t)), None)
        if kind is None:
            raise ValueError("a block of a kind the model does not write")
        rest = rest[len(next(t for t, v in KINDS.items() if v == kind)):]
        e = rest.find(END)
        if e < 0:
            raise ValueError("a block without its end")
        body, rest = rest[:e], rest[e + len(END):]
        if any(sp in body or sp in name for sp in SPECIALS):
            raise ValueError("a special token in the text")
        if (kind == "call") != bool(name):
            raise ValueError("a call's header names it; no other block's does")
        if kind == "thinking":
            reasoning += body
        elif kind == "text":
            content += body
        else:
            v = json.loads(body, parse_constant=no_constant)
            if not isinstance(v, dict) or set(v) != {"name", "args"} or v["name"] != name or not isinstance(v["args"], dict):
                raise ValueError("a call's JSON not {name, args} of its header's name")
            calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(v["args"], ensure_ascii=False)}})
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}


C = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}' + END
PARSE = [
    MODEL + "<|content_thinking|>think<|end_message|><|message_model|><|content_text|>The answer.<|end_message|>" + EOS,
    MODEL + "<|content_text|>Just text.<|end_message|>" + EOS,
    MODEL + "<|content_thinking|>t<|end_message|>" + C + C.replace("Paris", "Tōkyō") + EOS,
    MODEL + "<|content_text|>a<|end_message|>" + MODEL + "<|content_text|> b<|end_message|>" + EOS,
    MODEL + "<|content_text|><|end_message|>" + EOS,
    EOS,
    MODEL + "<|content_text|>no end of turn<|end_message|>",
    MODEL + "<|content_text|>no block end" + EOS,
    "plain text" + EOS,
    MODEL + "<|content_text|>a<|message_user|>b<|end_message|>" + EOS,
    MODEL + "<|content_xml|>x<|end_message|>" + EOS,
    MODEL + "named<|content_text|>x<|end_message|>" + EOS,
    MODEL + "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>" + EOS,
    MODEL + "f<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{}}<|end_message|>" + EOS,
    MODEL + "f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":[1]}<|end_message|>" + EOS,
    MODEL + "f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{},\"x\":1}<|end_message|>" + EOS,
    MODEL + "f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"n\":NaN}}<|end_message|>" + EOS,
    MODEL + "f<|content_invoke_tool_json|>not json<|end_message|>" + EOS,
    MODEL + "f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"b\":{\"z\":1,\"a\":2.50,\"u\":\"\\u00e9\"}}}<|end_message|>"
    + EOS,
    MODEL + "<|content_text|>x<|end_message|>" + EOS + "after",
]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    for messages, tools, effort in RENDER:
        case = {"kind": "render", "messages": messages, "tools": tools}
        if effort is not ABSENT:
            case["effort"] = effort
        try:
            case["out"] = render(tok, messages, tools, effort)
        except Exception as e:  # the template's refusal (raise_exception, or jinja's own on a missing field)
            case["error"] = type(e).__name__ + ": " + str(e)
        sys.stdout.write(json.dumps(case, ensure_ascii=False) + "\n")
        # an assistant turn after a user message: the render's bytes past the conversation before it, parsed back
        for i, m in enumerate(messages):
            if "out" not in case or i == 0 or m.get("role") != "assistant" or messages[0].get("role") != "user":
                continue
            before, upto = render(tok, messages[:i], tools, effort), render(tok, messages[:i + 1], tools, effort)
            assert upto.startswith(before)
            PARSE.append(upto[len(before):])
    seen = set()
    for text in PARSE:
        if text in seen:
            continue
        seen.add(text)
        case = {"kind": "parse", "text": text}
        try:
            case["out"] = parse(text)
        except ValueError as e:
            case["error"] = str(e)
        sys.stdout.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
