#ifndef OSMOS_TEXT_CHAT_QWEN_H
#define OSMOS_TEXT_CHAT_QWEN_H

#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"

#define QCHAT_EOS "<|im_end|>"
#define QCHAT_THINK_END "</think>"
#define QCHAT_CALL_OPEN "<tool_call>"

bool qchat_render(const Json *messages, const Json *tools, ChatMode mode, const char *effort, bool preserve,
                  JsonBuf *out, char err[CHAT_ERR]);
bool qchat_parse(const char *text, size_t len, ChatMode mode, const Json *tools, JsonBuf *out, char err[CHAT_ERR]);

#endif

/*
Qwen3.8-Flash-Next's chat template (the tokenizer.chat_template of its GGUF: Qwen's, with unsloth's fixes): the port
held to it byte for byte as transformers renders it (tests/test_chat_qwen.c, tools/gen_chat_fixtures_qwen.py). text
only: an image or a video part refused by name (the template would put a vision placeholder in).

qchat_render
  messages (a JSON array: role system / developer / user / assistant / tool; content a string, text parts or null;
  an assistant's reasoning_content and tool_calls - OpenAI's, or {name, arguments} bare - their arguments an object)
  and tools (OpenAI function tools, or NULL) as the prompt the model reads, the generation prompt last: the leading
  system and developer messages merged, the tools section, the reasoning effort's line when thinking (effort NULL:
  the template's default, xhigh; "high" read as xhigh; medium no line), every assistant turn's reasoning kept unless
  preserve is false (then only after the last real user query), tool results grouped into one user turn, and
  <think>\n - or an empty think block when not thinking - after the assistant header. every string trimmed as
  Python's str.strip trims. false with why: the template's refusals (no messages, a system message after the start,
  an unknown role or content, an effort it does not name, a call without a name, arguments a non-empty string or
  not an object).

qchat_parse
  a completion - the model's text after the rendered prompt, through its end of turn (QCHAT_EOS) - as the assistant
  message, JSON into out: {"role": "assistant", "content", "reasoning_content", "tool_calls": [{"type": "function",
  "function": {"name", "arguments": a JSON string}}]} - the reasoning before </think> when thinking, both trimmed;
  each <tool_call> block's parameters: the text the template wrote (the newline it puts on each side dropped), read
  as JSON where that is a value other than a string of a type the tool's schema allows for it (any, without one) -
  else the text itself, as the template writes a string. false with why for a completion out of the format: no end
  of turn, no </think> when thinking, a special token in the text, a malformed or repeated parameter, text after the
  calls.
*/
