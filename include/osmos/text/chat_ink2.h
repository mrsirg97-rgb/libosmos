#ifndef OSMOS_TEXT_CHAT_INK2_H
#define OSMOS_TEXT_CHAT_INK2_H

#include "osmos/err.h"
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

bool ichat_render(const Json *messages, const Json *tools, const Json *effort, JsonBuf *out, char err[OSMOS_ERR]);
bool ichat_parse(const char *text, size_t len, JsonBuf *out, char err[OSMOS_ERR]);
bool ichat_parse_shaped(const char *text, const char *shape, size_t len, JsonBuf *out, char err[OSMOS_ERR]);
void ichat_loose(const char *text, const char *shape, size_t len, JsonBuf *out);

/* a block's header (a stream's boundary): what the model's next block is */
typedef enum IchatBlock
{
  ICHAT_BLOCK_THINKING, /* <|message_model|><|content_thinking|> */
  ICHAT_BLOCK_TEXT,     /* <|message_model|><|content_text|> */
  ICHAT_BLOCK_CALL,     /* <|message_model|> and a name: a tool call's */
  ICHAT_BLOCK_END,      /* the end of turn */
  ICHAT_BLOCK_OTHER,    /* none the model writes */
} IchatBlock;
#define ICHAT_HOLD (-1L)
long ichat_header(const char *p, size_t n, IchatBlock *kind);
extern const char *const ichat_refused[]; /* special tokens' text a block's body can't hold (NULL-terminated) */

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

ichat_parse
  a completion - the model's turn, its blocks through <|content_model_end_sampling|> - as the assistant message, JSON
  into out: {"role": "assistant", "content", "reasoning_content", "tool_calls": [{"type": "function", "function":
  {"name", "arguments": a JSON string}}]} - the thinking blocks' bodies the reasoning, the text blocks' the content
  (each kind's joined in order), each call block's {"name", "args"} a call (its header's name; the arguments as
  json.dumps writes them). false with why for a completion out of the format: no end of turn, a block without its
  header, kind or end, a kind the model does not write (content_xml, tool_error, image), a special token in a body,
  a name on a block that is not a call (a named assistant's render: not a turn the model writes) or none on a call,
  a call's JSON not {name, args} of its header's name.

ichat_parse_shaped
  ichat_parse with its markers read from shape and what it gives the text's, as chat_parse_shaped (#34).

ichat_loose
  a completion out of the format (cut by max_tokens) as its message: the thinking and text blocks up to the first
  that is neither, the last one's body to the end - the stream's fields' text; no calls.

ichat_header
  the block starting at p (n bytes there): its kind and its header's length (ICHAT_BLOCK_CALL, _END and _OTHER: 0 -
  the stream stops there, the parse reads the rest), or ICHAT_HOLD while n bytes could still begin either.
*/
