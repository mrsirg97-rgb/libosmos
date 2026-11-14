#ifndef OSMOS_TEXT_CHAT_INK2_H
#define OSMOS_TEXT_CHAT_INK2_H

#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"

#define ICHAT_EOS "<|content_model_end_sampling|>"
#define ICHAT_END "<|end_message|>"
#define ICHAT_MODEL "<|message_model|>"
#define ICHAT_THINKING "<|content_thinking|>"
#define ICHAT_TEXT "<|content_text|>"
#define ICHAT_CALL "<|content_invoke_tool_json|>"

bool ichat_render(const Json *messages, const Json *tools, const Json *effort, JsonBuf *out, char err[CHAT_ERR]);

#endif

/*
Inkling-2-Small's chat template (the checkpoint's chat_template.jinja - its maker's; unsloth's GGUF carries a longer
reconstruction of a reference server's renderer, not this one): the port held to it byte for byte as transformers
renders it (tests/test_chat_ink2.c, tools/gen_chat_fixtures_ink2.py). text only: an image part refused by name.

ichat_render
  messages (a JSON array: role system / user / assistant / tool; content a string, parts or null; an assistant's
  reasoning_content, name and tool_calls - OpenAI's, their arguments an object) and tools (OpenAI function tools, or
  NULL) as the prompt the model reads - no generation prompt: the model opens its own turn. the tools' declarations
  first (each one's description, name, parameters and type, the template's defaults, sorted at every depth, compact),
  the effort line before the first message that is not a system one (else last: "Thinking effort level: " and the
  number as Python prints it, 0 for none), each message's blocks under its header (a tool result's name, or the name
  of every call its tool_call_id answers; an assistant's name), an assistant's reasoning block, its content's text
  blocks (an empty string not rendered beside calls or reasoning), its calls ({"name", "args"}: the arguments' keys
  sorted, their values compact), and <|content_model_end_sampling|> closing every assistant turn. effort: the
  request's reasoning_effort (a name - trimmed - or a number; NULL or JSON null the template's 0.9). Python's
  truthiness throughout. false with why: the template's refusals in its words (no messages, an unknown role, effort
  or part type, effort out of [0, 0.99], a call without a string name, arguments a string or not an object, an
  assistant's name not its call's, audio) - and where Python would print a value's str() into the prompt (a name,
  reasoning or text that is not a string; a part or a tool that is not an object; content an object), and an image.
*/
