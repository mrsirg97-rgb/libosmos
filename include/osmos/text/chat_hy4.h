#ifndef OSMOS_TEXT_CHAT_HY4_H
#define OSMOS_TEXT_CHAT_HY4_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"

#define HCHAT_TAG_MAX 32 /* a special token's suffix at most ("opensource", "6124c78e") */
#define HCHAT_EFFORTS 8  /* the reasoning efforts a template names at most */

/* the model's special tokens' texts for one suffix: the template's HYTK (":" + tag) */
typedef struct HchatTokens
{
  char start[64], middle[64], end[64], think[48], think_end[48], calls[48], calls_end[48], call[48], call_end[48];
  char key[48], key_end[48], value[48], value_end[48], resp[48], resp_end[48], mode[64];
  char efforts[HCHAT_EFFORTS][16]; /* the reasoning efforts the template takes ("" past the last) */
} HchatTokens;

bool hchat_tokens(const char *tag, HchatTokens *t, char err[OSMOS_ERR]);
bool hchat_from_template(const char *tmpl, HchatTokens *t, char err[OSMOS_ERR]);
bool hchat_render(const Json *messages, const Json *tools, const char *effort, int preserve, const HchatTokens *t,
                  JsonBuf *out, char err[OSMOS_ERR]);
bool hchat_parse(const char *text, size_t len, bool thinking, const Json *tools, const HchatTokens *t, JsonBuf *out,
                 char err[OSMOS_ERR]);
bool hchat_parse_shaped(const char *text, const char *shape, size_t len, bool thinking, const Json *tools,
                        const HchatTokens *t, JsonBuf *out, char err[OSMOS_ERR]);
/* every special token's text of the suffix a completion's text or a request's string can't hold (NULL-terminated:
   the caller's array of HCHAT_REFUSED + 1) */
#define HCHAT_REFUSED 16
void hchat_refused(const HchatTokens *t, const char *out[HCHAT_REFUSED + 1]);

#endif

/*
Tencent Hy4-flash's chat template (its tokenizer.chat_template: the release's, its special tokens' suffix HYTK
":opensource" - AngelSlim's GGUF's ":6124c78e"): the port held to it byte for byte as transformers renders it
(tests/test_chat_hy4.c, tools/gen_chat_fixtures_hy4.py). text only: an image or a video part refused by name.

hchat_tokens
  the special tokens' texts for a suffix (letters, digits, '_' and '-', at most HCHAT_TAG_MAX bytes) - the model's
  own (its template's HYTK) - and the release's efforts (no_think, high, max). false with why for another.

hchat_from_template
  the same from the model's own template text: its HYTK's suffix and the efforts its check names (`reasoning_effort
  not in [...]`: the release's no_think / high / max; AngelSlim's GGUF's, built from an internal checkpoint, low as
  well). false with why for a template without either.

hchat_render
  messages (a JSON array: role system / user / assistant / tool / any other; content a string, text parts or null;
  an assistant's `reasoning` or `reasoning_content` and `tool_calls` - OpenAI's or {name, arguments} bare, arguments
  an object; a tool message's `tool_call_id` or `id`) and tools (function tools or bare, or NULL) as the prompt the
  model reads, the generation prompt last: a leading system message or one synthesized, the tools section in it
  (each tool's keys in order but defer_loading and strict, a deferred tool left out), the reasoning effort's line
  (effort NULL: high; else one the template names - t->efforts), consecutive tool messages one turn - reordered to their calls'
  order when every id matches once - each assistant turn's reasoning kept where preserve (-1: the template's default,
  on with tools) or past the last user turn and the effort thinks, its calls' arguments a string raw else JSON. no
  string trimmed. after an assistant's last turn no generation prompt (a continuation). false with why: the
  template's refusals (an effort it does not name, arguments not an object), an image or a video, a content of
  another type - and a call's name or argument key holding '<' (osmos's: a forged special token's text).

hchat_parse
  a completion - the model's text after the rendered prompt, through its end of turn - as the assistant message,
  JSON into out: {"role": "assistant", "content", "reasoning_content", "tool_calls": [{"type": "function",
  "function": {"name", "arguments": a JSON string}}]}: the reasoning before the think block's end when thinking
  (effort not no_think: the prompt opened it), the content before the calls, each call's arguments typed by its
  tool's schema (chat_args.h). nothing trimmed. false with why for a completion out of the format: no end of turn,
  no think block's end when thinking, a special token's text out of its place, a malformed or repeated argument, text
  after the calls.

hchat_parse_shaped
  hchat_parse with its markers read from shape and what it gives the text's, as chat_parse_shaped (#34): a special
  token's text the model wrote as plain tokens defused in the shape, kept in the text.
*/
