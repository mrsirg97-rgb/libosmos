#pragma once
#include <stdbool.h>
#include <stddef.h>

#include "osmos/text/chat.h"
#include "osmos/text/json.h"

/* DeepSeek-V4.1's tokens and DSML tags (encoding.py's) */
#define CHAT_BOS "<｜begin▁of▁sentence｜>"
#define CHAT_EOS "<｜end▁of▁sentence｜>"
#define CHAT_THINK "<think>"
#define CHAT_THINK_END "</think>"
#define CHAT_DSML "｜DSML｜"
#define CHAT_USER "<｜User｜>"
#define CHAT_ASSISTANT "<｜Assistant｜>"
#define CHAT_SYSTEM "<｜System｜>"
#define CHAT_REMINDER "<｜latest_reminder｜>"
#define CHAT_IMAGE "<｜deepseek_image｜>"
#define CHAT_CALLS_OPEN "<" CHAT_DSML " calls>"
#define CHAT_CALLS_CLOSE "</" CHAT_DSML " calls>"
#define CHAT_INVOKE_OPEN "<" CHAT_DSML " invoke"
#define CHAT_INVOKE_CLOSE "</" CHAT_DSML " invoke"
#define CHAT_PARAM_OPEN "<" CHAT_DSML " parameter"
#define CHAT_PARAM_CLOSE "</" CHAT_DSML " parameter>"

bool chat_fail(char *err, const char *what, const char *why);

/* a message as the encoding renders it: a user turn's blocks (its text, tool results merged into it), another's
   content as text */
enum { BLK_TEXT, BLK_TOOL, BLK_OTHER };

typedef struct Block
{
  int kind;
  const Json *v; /* a text's string, a tool result's content (a string or parts), another's block */
  const char *id;
} Block;

typedef struct Msg
{
  const char *role;
  const Json *src;
  const char *content; /* NULL: none */
  size_t clen;
  char *joined; /* (owned: a list content's text parts joined) */
  const char *reasoning;
  Block *blocks;
  size_t nb, cap;
} Msg;

/* the messages read and merged (chat_read.c): *n_out of them; err set (and the array still returned) on a refusal.
   free each one's blocks and joined, then the array */
Msg *chat_read(const Json *src, size_t *n_out, char *err);

/* a system message's tools: the encoding's "## Tools" section (render_tools over tools_from_openai_format) */
bool chat_tools(JsonBuf *b, const Json *tools, char *err);

/* an assistant message's tool calls: "\n\n" and its DSML calls block */
bool chat_calls(JsonBuf *b, const Json *calls, char *err);

/* a tool name and its namespace (encoding.py's _split_tool_name): the qualified "ns::name" into out, or why not */
bool chat_tool_name(const char *name, const char *ns, JsonBuf *out, char *err);
