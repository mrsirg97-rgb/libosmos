#!/usr/bin/env python3
"""osmos-api against a fake engine (fake_engine.py: the protocol, scripted): the wire as rig and OpenAI clients read
it - health while loading, streamed reasoning / content / tool calls, a whole answer, refusals by name, a client gone
(its run STOPped), the engine gone (the server with it); and Qwen's dialect: the prompt the ids of its own template
(jinja2 over the GGUF's, tools/gen_chat_fixtures_qwen.py), its reasoning, content and tool calls read back, its
refusals. the tokenizer is the model's: each skipped without its first shard"""
import http.client
import json
import re
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = os.path.join(ROOT, "bin", "osmos-api")
FAKE = os.path.join(ROOT, "tests", "api", "fake_engine.py")
MODEL = os.environ.get("OSMOS_TEST_SHARD") or os.path.expanduser(
    "~/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf")
EOS = "<｜end▁of▁sentence｜>"
CALLS = ("hm</think>Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
         "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
         "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
         "</｜DSML｜ calls>" + EOS)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "the weather",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}, "days": {"type": "integer"}}}}}]


QWEN = os.environ.get("OSMOS_TEST_QWEN") or os.path.expanduser(
    "~/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf")
GGUF_PY = os.environ.get("OSMOS_GGUF_PY") or os.path.expanduser("~/Projects/llama.cpp/gguf-py")
IM_END = "<|im_end|>"


def ids(text, model=MODEL):
    r = subprocess.run([API, "--tokenize", "--model", model], input=text.encode(), capture_output=True, check=True)
    return r.stdout.decode().strip()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Version(unittest.TestCase):
    def test_both_binaries_say_the_header_s(self):          # the release's tag is checked against this
        with open(os.path.join(ROOT, "include", "osmos", "version.h")) as f:
            want = re.search(r'#define OSMOS_VERSION "([0-9]+\.[0-9]+\.[0-9]+)"', f.read()).group(1)
        for exe in (API, os.path.join(ROOT, "bin", "osmos-serve")):
            if os.path.exists(exe):
                out = subprocess.run([exe, "--version"], capture_output=True, text=True)
                self.assertEqual((out.returncode, out.stdout), (0, f"osmos {want}\n"), exe)


class Served:
    """osmos-api over the fake engine, the tokenizer cls.model's"""
    model = MODEL
    settings = {}  # a class's own pack settings beside these

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.cfg = os.path.join(cls.dir, "osmos.json")
        with open(cls.cfg, "w") as f:
            json.dump(dict({"pack": "test", "model": cls.model, "max_context": 4096, "no_guess": True}, **cls.settings), f)
        cls.port = free_port()
        env = dict(os.environ, FAKE_DIR=cls.dir)
        cls.proc = subprocess.Popen([API, "--config", cls.cfg, "--port", str(cls.port), "--engine", FAKE], env=env,
                                    stderr=subprocess.DEVNULL)
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", cls.port)).close()
                break
            except OSError:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.proc.kill()
        cls.proc.wait()

    def reply(self, text, finish="stop", pace=0.005):
        with open(os.path.join(self.dir, "reply"), "w") as f:
            f.write(ids(text, self.model) + "\n" + finish)
        with open(os.path.join(self.dir, "pace"), "w") as f:
            f.write(str(pace))

    def ready(self):
        open(os.path.join(self.dir, "ready"), "w").close()
        for _ in range(200):
            if self.get("/health")[0] == 200:
                return
            time.sleep(0.02)
        self.fail("never ready")

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()

    def post(self, body, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions", body=raw if raw is not None else json.dumps(body),
                  headers={"Content-Type": "application/json"})
        return c.getresponse()

    def stream(self, body):
        r = self.post(dict(body, stream=True, stream_options={"include_usage": True}))
        self.assertEqual(r.status, 200)
        events = [e[len("data: "):] for e in r.read().decode("utf-8").split("\n\n") if e]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(e) for e in events[:-1]]

    def log(self):
        with open(os.path.join(self.dir, "log")) as f:
            return f.read().splitlines()

    def last_gen(self):
        return [l for l in self.log() if l.startswith("GEN ")][-1].split()


@unittest.skipUnless(os.path.exists(MODEL), "no UD-IQ1_M first shard (the tokenizer's)")
class Api(Served, unittest.TestCase):
    settings = {"max_tokens": 8192}  # a DeepSeek pack's prompt step
    def test_0_health_while_loading(self):
        self.assertEqual(self.get("/health")[0], 503)
        self.assertEqual(self.post({"messages": [{"role": "user", "content": "hi"}]}).status, 503)
        self.ready()
        argv = self.log()[0].split()
        self.assertEqual(argv[argv.index("--max-context") + 1], "4096")
        self.assertIn("--no-guess", argv)
        self.assertEqual(argv[argv.index("--budget") + 1], "auto")      # no budget given: sized to the GPU
        self.assertEqual(argv[argv.index("--max-tokens") + 1], "8192")  # the pack's step, osmos-serve's
        self.assertEqual(argv[argv.index("--eos") + 1], ids(EOS))
        models = json.loads(self.get("/v1/models")[1])
        self.assertEqual(models["data"][0]["id"], "osmos-test")

    def test_1_thinking_stream(self):
        self.ready()
        self.reply("let me think about it \n</think>The answer is 4, 🦀 é.\n" + EOS)
        ev = self.stream({"messages": [{"role": "user", "content": "2+2?"}], "reasoning_effort": "high",
                          "max_tokens": 100})
        deltas = [e["choices"][0]["delta"] for e in ev if e["choices"]]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(reasoning, "let me think about it")
        self.assertEqual(content, "The answer is 4, 🦀 é.")
        self.assertNotIn("</think>", reasoning + content)
        fin = [e for e in ev if e["choices"] and e["choices"][0]["finish_reason"]]
        self.assertEqual(fin[0]["choices"][0]["finish_reason"], "stop")
        self.assertGreater(fin[0]["usage"]["prompt_tokens"], 0)
        self.assertIn("predicted_per_second", fin[0]["timings"])
        gen = [l for l in self.log() if l.startswith("GEN ")][-1].split()
        self.assertEqual(gen[1], "100")
        self.assertTrue(gen[-1].startswith(ids("<｜begin▁of▁sentence｜>")))
        self.assertTrue(gen[-1].endswith(ids("<｜Assistant｜><think>")))

    def test_2_tool_calls_stream(self):
        self.ready()
        self.reply(CALLS)
        ev = self.stream({"messages": [{"role": "user", "content": "weather in Paris?"}], "tools": TOOLS})
        deltas = [e["choices"][0]["delta"] for e in ev if e["choices"]]
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(content, "Checking.")
        calls = [c for d in deltas for c in d.get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris", "days": 3})
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(calls[0]["index"], 0)
        fin = [e for e in ev if e["choices"] and e["choices"][0]["finish_reason"]]
        self.assertEqual(fin[0]["choices"][0]["finish_reason"], "tool_calls")
        gen = [l for l in self.log() if l.startswith("GEN ")][-1].split()
        self.assertIn(ids("get_weather"), gen[-1])  # the tools rendered into the prompt

    def test_3_whole_plain(self):
        self.ready()
        self.reply("Hello there." + EOS)
        r = self.post({"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}],
                       "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0})
        self.assertEqual(r.status, 200)
        b = json.loads(r.read())
        self.assertEqual(b["object"], "chat.completion")
        self.assertEqual(b["choices"][0]["message"], {"role": "assistant", "content": "Hello there."})
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        gen = [l for l in self.log() if l.startswith("GEN ")][-1].split()
        self.assertTrue(gen[-1].endswith(ids("<｜Assistant｜></think>")))

    def test_4_length(self):
        self.ready()
        self.reply("no end in sight   ", finish="length")
        r = self.post({"messages": [{"role": "user", "content": "go"}], "reasoning_effort": "none",
                       "max_tokens": 5})
        b = json.loads(r.read())
        self.assertEqual(b["choices"][0]["finish_reason"], "length")
        self.assertEqual(b["choices"][0]["message"]["content"], "no end in sight   ")

    def test_5_refusals(self):
        self.ready()
        for body, word in [({"messages": [{"role": "user", "content": "x"}], "temperature": 0.7}, "temperature"),
                           ({"messages": []}, "messages"),
                           ({"messages": [{"role": "user", "content": "x"}], "stop": ["\n"]}, "stop"),
                           ({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"},
                            "reasoning_effort"),
                           ({"messages": [{"role": "wizard", "content": "x"}]}, "")]:
            r = self.post(body)
            self.assertEqual(r.status, 400, body)
            self.assertIn(word, json.loads(r.read())["error"]["message"])
        r = self.post(None, raw="{not json")
        self.assertEqual(r.status, 400)
        self.assertEqual(self.get("/nope")[0], 404)

    def test_6_client_gone(self):
        self.ready()
        self.reply("a " * 50, finish="hold", pace=0.02)
        s = socket.create_connection(("127.0.0.1", self.port))
        body = json.dumps({"messages": [{"role": "user", "content": "x"}], "stream": True}).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        s.recv(4096)
        s.close()
        for _ in range(100):
            if any(l.startswith("STOP ") for l in self.log()):
                break
            time.sleep(0.05)
        self.assertTrue(any(l.startswith("STOP ") for l in self.log()))
        self.reply("ok" + EOS)  # the server still answers
        r = self.post({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"})
        self.assertEqual(json.loads(r.read())["choices"][0]["message"]["content"], "ok")

    def test_7_engine_gone(self):
        self.ready()
        self.reply("a b c d e f g h", finish="exit")
        r = self.post({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"})
        self.assertEqual(r.status, 500)
        self.assertIn("engine", json.loads(r.read())["error"]["message"])
        self.assertNotEqual(self.proc.wait(timeout=5), 0)


def qwen_template():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gen_chat_fixtures_qwen
    return gen_chat_fixtures_qwen.template(QWEN, GGUF_PY)


QCALL = ("Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n"
         "3\n</parameter>\n</function>\n</tool_call>" + IM_END)


@unittest.skipUnless(os.path.exists(QWEN) and os.path.isdir(GGUF_PY), "no Qwen3.8 first shard, or no gguf-py")
class QwenApi(Served, unittest.TestCase):
    model = QWEN

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tpl = qwen_template()

    def prompt(self, messages, tools=None, thinking=True, effort=None, preserve=None):
        kw = {"messages": messages, "add_generation_prompt": True, "enable_thinking": thinking}
        if preserve is not None:
            kw["preserve_thinking"] = preserve
        if tools:
            kw["tools"] = tools
        if effort:
            kw["reasoning_effort"] = effort
        return ids(self.tpl.render(**kw), QWEN)

    def test_0_eos(self):
        self.ready()
        argv = self.log()[0].split()
        self.assertEqual(argv[argv.index("--eos") + 1], ids(IM_END, QWEN))
        self.assertNotIn("--max-tokens", argv)  # no step set: osmos-serve's own

    def test_1_thinking_stream(self):
        self.ready()
        self.reply("let me think \n</think>\n\nThe answer is 4, 🦀 é." + IM_END)
        msgs = [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "2+2?"}]
        ev = self.stream({"messages": msgs, "reasoning_effort": "low", "max_tokens": 100})
        deltas = [e["choices"][0]["delta"] for e in ev if e["choices"]]
        self.assertEqual("".join(d.get("reasoning_content", "") for d in deltas), "let me think")
        self.assertEqual("".join(d.get("content", "") for d in deltas), "The answer is 4, 🦀 é.")
        fin = [e for e in ev if e["choices"] and e["choices"][0]["finish_reason"]]
        self.assertEqual(fin[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.last_gen()[-1], self.prompt(msgs, effort="low"))

    def test_2_tool_calls_stream(self):
        self.ready()
        self.reply(QCALL)
        msgs = [{"role": "user", "content": "weather in Paris?"}]
        ev = self.stream({"messages": msgs, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}})
        deltas = [e["choices"][0]["delta"] for e in ev if e["choices"]]
        self.assertEqual("".join(d.get("content", "") for d in deltas), "Checking.")
        calls = [c for d in deltas for c in d.get("tool_calls", [])]
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris", "days": 3})
        fin = [e for e in ev if e["choices"] and e["choices"][0]["finish_reason"]]
        self.assertEqual(fin[0]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(self.last_gen()[-1], self.prompt(msgs, TOOLS, thinking=False))

    def test_3_history_whole(self):
        self.ready()
        self.reply("Sunny." + IM_END)
        sent = [{"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": "", "reasoning": "look it up",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sun"}]
        r = self.post({"messages": sent, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["choices"][0]["message"], {"role": "assistant", "content": "Sunny."})
        want = [sent[0], {"role": "assistant", "content": "", "reasoning_content": "look it up",
                          "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                sent[2]]
        self.assertEqual(self.last_gen()[-1], self.prompt(want, TOOLS, thinking=False))

    def test_4_earlier_reasoning_dropped(self):
        self.ready()
        self.reply("x\n</think>\n\n6" + IM_END)
        msgs = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "add"},
                {"role": "user", "content": "3+3?"}]
        r = self.post({"messages": msgs, "reasoning_effort": "medium", "chat_template_kwargs": {"preserve_thinking": False}})
        self.assertEqual(json.loads(r.read())["choices"][0]["message"]["content"], "6")
        self.assertEqual(self.last_gen()[-1], self.prompt(msgs, effort="medium", preserve=False))
        self.assertNotEqual(self.last_gen()[-1], self.prompt(msgs, effort="medium"))

    def test_5_refusals(self):
        self.ready()
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": "[1]"}}]}
        for body, word in [({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"},
                            "reasoning effort"),
                           ({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": 50},
                            "reasoning_effort"),
                           ({"messages": [{"role": "user", "content": [
                               {"type": "image_url", "image_url": {"url": "x"}}]}]}, "image"),
                           ({"messages": [{"role": "user", "content": "x"}, call]}, "arguments")]:
            r = self.post(body)
            self.assertEqual(r.status, 400, body)
            self.assertIn(word, json.loads(r.read())["error"]["message"])


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False, verbosity=2).result.wasSuccessful())
