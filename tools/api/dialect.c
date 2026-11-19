#include <string.h>

#include "chat_impl.h"
#include "dialect.h"
#include "osmos/text/chat_ink2.h"
#include "osmos/text/chat_lfm3.h"
#include "osmos/text/chat_qwen.h"

/* each family's chat: DeepSeek's encoding (osmos/text/chat.h), Qwen's template (osmos/text/chat_qwen.h), Inkling's
   (osmos/text/chat_ink2.h) */

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

static bool lfm3_render(const Request *r, JsonBuf *out, char err[CHAT_ERR])
{
  return lchat_render(r->messages, r->tools, r->mode, r->preserve, out, err);
}

static bool lfm3_parse(const Request *r, const char *text, const char *shape, size_t n, JsonBuf *out,
                       char err[CHAT_ERR])
{
  return lchat_parse_shaped(text, shape, n, r->mode, r->tools, out, err);
}

static bool ink2_render(const Request *r, JsonBuf *out, char err[CHAT_ERR])
{
  return ichat_render(r->messages, r->tools, r->effort_value, out, err);
}

static bool ink2_parse(const Request *r, const char *text, const char *shape, size_t n, JsonBuf *out,
                       char err[CHAT_ERR])
{
  (void)r;
  return ichat_parse_shaped(text, shape, n, out, err);
}

/* Inkling's template keeps every turn's reasoning */
static bool ink2_drops(const Request *r) { return (void)r, false; }

static long ink2_header(const char *p, size_t n, DialectField *field)
{
  IchatBlock kind;
  const long h = ichat_header(p, n, &kind);
  *field = kind == ICHAT_BLOCK_THINKING ? DIALECT_REASONING : kind == ICHAT_BLOCK_TEXT ? DIALECT_CONTENT : DIALECT_DONE;
  return h;
}

static const Dialect kDialects[] = {
    {"deepseek41", CHAT_EOS, CHAT_THINK_END, CHAT_CALLS_START, false, chat_refused, request_ds, ds_render, ds_parse,
     ds_drops, CHAT_THINK_END, NULL, NULL, NULL},
    {"qwen4exp", QCHAT_EOS, QCHAT_THINK_END, QCHAT_CALL_OPEN, true, qchat_refused, request_qwen, qwen_render, qwen_parse,
     qwen_drops, QCHAT_THINK_END, NULL, NULL, NULL},
    /* Qwen3.5 / 3.8's dense models (vibes27b): Qwen-Next's turn - their template the same but for several system
       messages (the first one alone, not merged) and "high" effort (not raised to "xhigh") */
    {"qwen35", QCHAT_EOS, QCHAT_THINK_END, QCHAT_CALL_OPEN, true, qchat_refused, request_qwen, qwen_render, qwen_parse,
     qwen_drops, QCHAT_THINK_END, NULL, NULL, NULL},
    /* LFM3's (vibes1.7b): ChatML with <think>, Qwen3.5's calls - its template drops reasoning before the last user
       message unless preserve_thinking (the request's, off unless true: the template's default) */
    {"lfm3", LCHAT_EOS, QCHAT_THINK_END, QCHAT_CALL_OPEN, true, lchat_refused, request_lfm3, lfm3_render, lfm3_parse,
     qwen_drops, QCHAT_THINK_END, NULL, NULL, NULL},
    /* Inkling's turn: blocks, the thinking optional (none at effort 0) - no answer-after (a block's end is every
       block's: the rule would refuse a text block's end of turn) */
    {"inkling2", ICHAT_EOS, ICHAT_END, ICHAT_CALL, false, ichat_refused, request_ink2, ink2_render, ink2_parse,
     ink2_drops, NULL, ink2_header, ICHAT_END, ichat_loose},
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
