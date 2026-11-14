#include <string.h>

#include "chat_impl.h"
#include "dialect.h"
#include "osmos/text/chat_qwen.h"

/* each family's chat: DeepSeek's encoding (osmos/text/chat.h), Qwen's template (osmos/text/chat_qwen.h) */

static bool ds_render(const Request *r, JsonBuf *out, char err[CHAT_ERR])
{
  return chat_render(r->messages, r->mode, r->effort, out, err);
}

static bool ds_parse(const Request *r, const char *text, const char *shape, size_t n, JsonBuf *out, char err[CHAT_ERR])
{
  return chat_parse_shaped(text, shape, n, r->mode, out, err);
}

static bool qwen_render(const Request *r, JsonBuf *out, char err[CHAT_ERR])
{
  return qchat_render_with(r->messages, r->branch ? r->branch : r->messages, r->tools, r->mode,
                           r->level[0] ? r->level : NULL, r->preserve, out, err);
}

static bool qwen_parse(const Request *r, const char *text, const char *shape, size_t n, JsonBuf *out,
                       char err[CHAT_ERR])
{
  return qchat_parse_shaped(text, shape, n, r->mode, r->tools, out, err);
}

/* the last message an answer or a tool's result: reasoning since the last user message, which a new one ends */
static bool past_user(const Request *r)
{
  const Json *ms = r->messages;
  const Json *last = ms && ms->n ? ms->item[ms->n - 1] : NULL;
  const char *role = last ? json_str(last, "role") : NULL;
  return role && strcmp(role, "user") != 0;
}

/* DeepSeek's encoding drops an earlier answer's reasoning unless tools are given (chat.c's drop) */
static bool ds_drops(const Request *r)
{
  for (size_t i = 0; r->messages && i < r->messages->n; i++)
  {
    const Json *t = json_get(r->messages->item[i], "tools");
    if (t && t->type == JSON_ARRAY && t->n)
      return false;
  }
  return r->mode == CHAT_THINKING && past_user(r);
}

/* Qwen's template drops it unless preserve_thinking (the request's, on unless false) */
static bool qwen_drops(const Request *r) { return r->mode == CHAT_THINKING && !r->preserve && past_user(r); }

static const Dialect kDialects[] = {
    {"deepseek41", CHAT_EOS, CHAT_THINK_END, CHAT_CALLS_START, false, chat_refused, request_ds, ds_render, ds_parse,
     ds_drops},
    {"qwen4exp", QCHAT_EOS, QCHAT_THINK_END, QCHAT_CALL_OPEN, true, qchat_refused, request_qwen, qwen_render, qwen_parse,
     qwen_drops},
};

const Dialect *dialect_of(const char *arch)
{
  for (size_t i = 0; arch && i < sizeof kDialects / sizeof *kDialects; i++)
    if (!strcmp(arch, kDialects[i].arch))
    {
      size_t n = 0;
      while (kDialects[i].refused[n])
        n++;
      return n <= DIALECT_REFUSED_MAX ? &kDialects[i] : NULL;
    }
  return NULL;
}
