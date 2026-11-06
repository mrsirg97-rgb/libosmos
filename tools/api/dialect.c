#include <string.h>

#include "chat_impl.h"
#include "dialect.h"
#include "osmos/text/chat_qwen.h"

/* each family's chat: DeepSeek's encoding (osmos/text/chat.h), Qwen's template (osmos/text/chat_qwen.h) */

static bool ds_render(const Request *r, JsonBuf *out, char err[CHAT_ERR])
{
  return chat_render(r->messages, r->mode, r->effort, out, err);
}

static bool ds_parse(const Request *r, const char *text, size_t n, JsonBuf *out, char err[CHAT_ERR])
{
  return chat_parse(text, n, r->mode, out, err);
}

static bool qwen_render(const Request *r, JsonBuf *out, char err[CHAT_ERR])
{
  return qchat_render(r->messages, r->tools, r->mode, r->level[0] ? r->level : NULL, r->preserve, out, err);
}

static bool qwen_parse(const Request *r, const char *text, size_t n, JsonBuf *out, char err[CHAT_ERR])
{
  return qchat_parse(text, n, r->mode, r->tools, out, err);
}

static const Dialect kDialects[] = {
    {"deepseek41", CHAT_EOS, CHAT_THINK_END, CHAT_CALLS_OPEN, false, request_ds, ds_render, ds_parse},
    {"qwen4exp", QCHAT_EOS, QCHAT_THINK_END, QCHAT_CALL_OPEN, true, request_qwen, qwen_render, qwen_parse},
};

const Dialect *dialect_of(const char *arch)
{
  for (size_t i = 0; arch && i < sizeof kDialects / sizeof *kDialects; i++)
    if (!strcmp(arch, kDialects[i].arch))
      return &kDialects[i];
  return NULL;
}
