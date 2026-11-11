#ifndef OSMOS_TEXT_CHAT_H
#define OSMOS_TEXT_CHAT_H

#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/json.h"

#define CHAT_ERR 256

typedef enum ChatMode
{
  CHAT_PLAIN,    /* DeepSeek's "chat": no reasoning */
  CHAT_THINKING, /* "thinking": the answer after its reasoning, in <think> .. </think> */
} ChatMode;

int chat_effort(const Json *effort, char err[CHAT_ERR]);
bool chat_render(const Json *messages, ChatMode mode, int effort, JsonBuf *out, char err[CHAT_ERR]);
bool chat_parse(const char *text, size_t len, ChatMode mode, JsonBuf *out, char err[CHAT_ERR]);
bool chat_parse_shaped(const char *text, const char *shape, size_t len, ChatMode mode, JsonBuf *out,
                       char err[CHAT_ERR]);
extern const char *const chat_refused[]; /* special tokens' text a completion's reasoning or content can't hold */

#endif

/*
DeepSeek-V4.1-Flash-1017's chat encoding (its encoding/encoding.py, MIT - maya vendors it unchanged as
serve/ds41_encoding.py): tests/test_chat.c holds the port to it byte for byte (tools/gen_chat_fixtures.py). text
only: an image block, or the image placeholder in text, refused by name (as an internal `task`).

chat_effort
  the reasoning effort's budget: null "high" (75), "low" 50, "high" 75, "max" 100, an integer 1 .. 100 itself; -1
  with why for anything else.

chat_render
  messages (a JSON array of the encoding's messages: role system / user / assistant / tool / latest_reminder; content
  a string or text parts; a system message's `tools` (OpenAI function tools) and `response_format`; an assistant's
  `reasoning_content` and `tool_calls` (OpenAI's: arguments a JSON string or an object); a tool message's
  `tool_call_id`) rendered as the prompt the model reads, BOS first, into out: tool results folded into the user turn
  in their calls' order, earlier reasoning dropped (unless tools are given), the reasoning effort's line in thinking
  mode, the assistant header and its <think> or </think> after the last turn. false with why: a role, a field or a
  tool name the encoding refuses.

chat_parse
  a completion - the model's text after the rendered prompt, through its EOS - as the assistant message, JSON into
  out: {"role": "assistant", "content", "reasoning_content", "tool_calls": [OpenAI's, each its "namespace" if
  qualified]} - the reasoning before </think> in thinking mode, DSML tool calls decoded (a string="true" parameter
  as a JSON string, a "false" one as written). false with why for a completion out of the format (no </think>, no
  EOS, a malformed call, a special token in the text) - the caller decides what to answer then.

chat_parse_shaped
  chat_parse with its markers read from shape - the text with every special token's text the model wrote in plain
  pieces defused (tokenizer_defuse over the pieces between its special tokens), the same length - and what it gives
  the text's own bytes: a special token's text the model said is text, the token itself still the format's (#34).
  chat_parse is shape the text.
*/
