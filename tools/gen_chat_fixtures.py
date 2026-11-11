"""tests/fixtures/chat.jsonl: DeepSeek-V4.1's own chat encoding (its encoding/encoding.py, MIT, as maya vendors it
unchanged: serve/ds41_encoding.py) over chats chosen to cross its branches - roles, thinking on and off, the
reasoning effort, earlier reasoning dropped, tools and their schemas, tool calls (string and JSON arguments, a
namespace, double-encoded), tool results merged and put in call order, a system message mid-conversation, Unicode
and JSON's corners - and its parse of completions back (reasoning, content, tool calls). the C port's test reads it
(tests/test_chat.c). run: python3 tools/gen_chat_fixtures.py <maya>/serve/ds41_encoding.py > tests/fixtures/chat.jsonl

a line a case: {"kind": "render", "messages", "thinking", "effort", "out"}, {"kind": "parse", "text", "thinking",
"out"} - out the reference's string, or its parsed message - or {"kind": "refuse", "text", "thinking"}: a completion
the reference's parse raises on (each PARSE case is one or the other, as the reference finds it)."""
import importlib.util
import json
import sys

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
                                            "parameters": {"type": "object", "properties": {
                                                "city": {"type": "string", "description": "the city"},
                                                "days": {"type": "integer", "minimum": 1, "maximum": 7}},
                                                "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "web::search", "description": "Search the web — fast.",
                                           "parameters": {"type": "object", "properties": {
                                               "q": {"type": "string"}, "top": {"type": "number", "default": 2.5}}}}}


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


SYS_TOOLS = {"role": "system", "content": "You are helpful.", "tools": [WEATHER, SEARCH]}
RF = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
RENDER = [
    ([{"role": "user", "content": "Hello"}], "chat", None),
    ([{"role": "system", "content": "Be brief.", "response_format": RF}, {"role": "user", "content": "Hi"}], "chat",
     None),
    ([dict(SYS_TOOLS, response_format=RF), {"role": "user", "content": "Hi"}], "thinking", None),
    ([{"role": "user", "content": "Hello"}], "thinking", None),
    ([{"role": "user", "content": "Hello"}], "thinking", "low"),
    ([{"role": "user", "content": "Hello"}], "thinking", 30),
    ([{"role": "user", "content": "Hello"}], "thinking", "max"),
    ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "chat", None),
    ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "thinking", None),
    ([{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "add"},
      {"role": "user", "content": "and 3+3?"}], "thinking", None),
    ([{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "add"},
      {"role": "user", "content": "and 3+3?"}], "chat", None),
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "system", "content": "mid"},
      {"role": "user", "content": "c"}], "thinking", None),
    ([SYS_TOOLS, {"role": "user", "content": "Weather in Paris and Tōkyō?"}], "thinking", None),
    ([SYS_TOOLS, {"role": "user", "content": "Weather in Paris and Tōkyō?"},
      {"role": "assistant", "content": "", "reasoning_content": "two cities",
       "tool_calls": [call("c1", "get_weather", "{\"city\": \"Paris\", \"days\": 3}"),
                      call("c2", "get_weather", {"city": "Tōkyō"})]},
      {"role": "tool", "tool_call_id": "c2", "content": "rain"},
      {"role": "tool", "tool_call_id": "c1", "content": "sun, 21°C"}], "thinking", None),
    ([SYS_TOOLS, {"role": "user", "content": "search it"},
      {"role": "assistant", "content": "Searching.", "tool_calls": [
          call("s1", "web::search", {"q": "say \"hi\"\n", "top": 1.0, "flags": [True, False, None],
                                     "nested": {"a": [1, 2.25, -3e-7, 1e21], "ü": "√"}}),
          call("s2", "get_weather", json.dumps(json.dumps({"city": "Oslo"})))]},
      {"role": "tool", "tool_call_id": "s1", "content": "results <b>bold</b>"},
      {"role": "tool", "tool_call_id": "s2", "content": "snow"},
      {"role": "user", "content": "thanks"}], "chat", None),
    ([SYS_TOOLS, {"role": "user", "content": "x"}, {"role": "assistant", "content": "y",
                                                     "tool_calls": [call("z", "get_weather", "not json")]},
      {"role": "tool", "tool_call_id": "z", "content": [{"type": "text", "text": "part 1"},
                                                         {"type": "text", "text": "part 2"}]}], "thinking", 90),
    ([{"role": "user", "content": "日本語 😀 \t tabs and\nnewlines \\ backslash"}], "chat", None),
]

PARSE = [
    ("Just text.<｜end▁of▁sentence｜>", "chat"),
    ("let me think</think>The answer is 4.<｜end▁of▁sentence｜>", "thinking"),
    ("hm</think>Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
     "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
     "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
     "<｜DSML｜ invoke name=\"web::search\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">a \"b\"\nc"
     "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>", "thinking"),
]
# the tags' whitespace (issue #25): the reference's checks are exact ('>\n' between tags), but its regexes end in `$`,
# which also matches before one trailing newline - an invoke's name line, a parameter's '<', a newline more
D, END = "｜DSML｜", "<｜end▁of▁sentence｜>"
INV = f'<{D} invoke name="mcp">\n'
PAR = f'<{D} parameter name="search" string="true">fl</{D} parameter>\n'
BLOCK = lambda body: f"\n\n<{D} calls>\n{body}</{D} calls>{END}"
PARSE += [
    (BLOCK(f'{INV}\n</{D} invoke>\n{INV}{PAR}</{D} invoke>\n'), "chat"),  # the issue's: an empty invoke, a blank line
    (BLOCK(f'{INV}\n{PAR}</{D} invoke>\n'), "chat"),                       # a blank line before a parameter
    (BLOCK(f'{INV}<{D} parameter name="q" string="true">fl<\n/{D} parameter>\n</{D} invoke>\n'), "chat"),
    (BLOCK(f'{INV}\n\n</{D} invoke>\n'), "chat"),                          # two blank lines: refused
    (BLOCK(f'{INV}</{D} invoke>\n\n{INV}</{D} invoke>\n'), "chat"),         # a blank line between invokes
    (BLOCK(f'{INV}{PAR}\n</{D} invoke>\n'), "chat"),                       # a blank line after a parameter
    (BLOCK(f'{INV}</{D} invoke>\n\n'), "chat"),                             # a blank line before the block's end
    (BLOCK(f'{INV}</{D} invoke> \n'), "chat"),                              # a trailing space
    (BLOCK(f'{INV} </{D} invoke>\n'), "chat"),                              # a space after the name's line
    ("x</think>\n\n" + BLOCK(f'{INV}</{D} invoke>\n')[2:].replace(END, "after" + END), "thinking"),  # text after
]


def main():
    spec = importlib.util.spec_from_file_location("ds41_encoding", sys.argv[1])
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    out = []
    for messages, thinking, effort in RENDER:
        out.append({"kind": "render", "messages": messages, "thinking": thinking, "effort": effort,
                    "out": enc.encode_messages(messages, thinking_mode=thinking, reasoning_effort=effort)})
    for text, thinking in PARSE:
        try:
            out.append({"kind": "parse", "text": text, "thinking": thinking,
                        "out": enc.parse_message_from_completion_text(text, thinking_mode=thinking)})
        except (ValueError, AssertionError):
            out.append({"kind": "refuse", "text": text, "thinking": thinking})
    sys.stdout.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out))


if __name__ == "__main__":
    main()
