#ifndef OSMOS_TEXT_CHAT_LFM3_H
#define OSMOS_TEXT_CHAT_LFM3_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"

#define LCHAT_BOS "<|startoftext|>"
#define LCHAT_EOS "<|im_end|>"

bool lchat_render(const Json *messages, const Json *tools, ChatMode mode, bool preserve, JsonBuf *out,
                  char err[OSMOS_ERR]);
bool lchat_parse(const char *text, size_t len, ChatMode mode, const Json *tools, JsonBuf *out, char err[OSMOS_ERR]);
bool lchat_parse_shaped(const char *text, const char *shape, size_t len, ChatMode mode, const Json *tools,
                        JsonBuf *out, char err[OSMOS_ERR]);
extern const char *const lchat_refused[]; /* special tokens' text a completion can't hold (NULL-terminated) */

#endif

/*
LFM3's chat template (LiquidAI's, the checkpoint's own chat_template.jinja), ported byte for byte as transformers
renders it. text only.

lchat_render
  messages (a JSON array: a role and content - a string, text parts or null; an assistant's thinking / reasoning /
  reasoning_content and tool_calls - OpenAI's, or {name, arguments} bare, their arguments an object) and tools
  (OpenAI function tools, or NULL) as the prompt the model reads: <|startoftext|>; a leading system message's content
  and the tools (each its function's JSON, a line each, in <tools>) one system turn; each message a ChatML turn
  (<|im_start|>role ... <|im_end|>, its role as given) - an assistant's reasoning kept after the last user message
  (or always: preserve) as <think>..</think>, else <think></think> before its content (its text to its last </think>
  dropped); its calls Qwen3.5's XML after its content (no separator); a tool result <tool_response> - consecutive
  ones of one tool_call_id one turn; then the generation prompt, <think> (or <think></think>: not thinking). nothing
  trimmed (the template trims nothing but a dropped reasoning's rest). false with why: the template's refusals (a
  call's arguments a non-empty JSON string) and osmos's: an image or a video part, additional_tools or a tool
  reference (the template's tool-search forms: not served), a call's name or argument key holding a newline or '<'
  (its markup), no messages.

lchat_parse
  a completion - the model's text after the generation prompt through <|im_end|> - as the assistant message:
  qchat_parse's (the calls the same XML; the reasoning before </think>), but LFM3's special tokens refused in it.

lchat_parse_shaped
  lchat_parse with its markers read from shape and what it gives the text's, as qchat_parse_shaped.
*/
