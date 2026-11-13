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
bool qchat_render_with(const Json *messages, const Json *branch, const Json *tools, ChatMode mode, const char *effort,
                       bool preserve, JsonBuf *out, char err[CHAT_ERR]);
bool qchat_parse(const char *text, size_t len, ChatMode mode, const Json *tools, JsonBuf *out, char err[CHAT_ERR]);
bool qchat_parse_shaped(const char *text, const char *shape, size_t len, ChatMode mode, const Json *tools,
                        JsonBuf *out, char err[CHAT_ERR]);
size_t qchat_space_lead(const char *p, size_t n);
size_t qchat_space_trail(const char *p, size_t n);
extern const char *const qchat_refused[]; /* special tokens' text a completion can't hold (NULL-terminated) */

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
  not an object) - and a call's name or argument key holding a newline or '<' (osmos's: the template's '>' after it
  could complete a special token's text, a turn forged).

qchat_render_with
  qchat_render, but the template's choices that read a message's text (the last real user query, for preserve false)
  made on branch's messages - the same count - not on messages': a render of the request's strings defused (osmos-api's
  shape) takes the text's branches, its bytes then the text's but for those defused (0.7.1: a user turn of tool
  responses as text, defused, stopped looking like one - the renders differed, a 400).

qchat_parse
  a completion - the model's text after the rendered prompt, through its end of turn (QCHAT_EOS) - as the assistant
  message, JSON into out: {"role": "assistant", "content", "reasoning_content", "tool_calls": [{"type": "function",
  "function": {"name", "arguments": a JSON string}}]} - the reasoning before </think> when thinking, both trimmed;
  each <tool_call> block's parameters: the text the template wrote (the newline it puts on each side dropped), read
  as JSON where that is a value other than a string of a type the tool's schema allows for it (its "type", or an
  anyOf / oneOf's) - else the text itself, as the template writes a string (no tool, parameter or type: a string,
  as vLLM's qwen3coder parser reads it). false with why for a completion out of the format: no end
  of turn, no </think> when thinking, a special token in the text, a malformed or repeated parameter, text after the
  calls.

qchat_space_lead / qchat_space_trail
  the bytes of whitespace Python's str.strip takes off p's start / end (str.isspace's code points): the parse's trim,
  and what osmos-api's stream holds or passes over so its text is the parse's.

qchat_parse_shaped
  qchat_parse with its markers read from shape and what it gives the text's, as chat_parse_shaped (#34).
*/
