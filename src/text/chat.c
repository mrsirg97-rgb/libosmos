#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_impl.h"

/* encoding.py's encode_messages for text: the messages read (chat_read.c), tool results put in their calls' order
   (sort_tool_results_by_call_order), earlier reasoning dropped (_drop_thinking_messages), each rendered
   (render_message) */

bool chat_fail(char *err, const char *what, const char *why)
{
  snprintf(err, CHAT_ERR, "chat: %s: %s", what, why);
  return false;
}

int chat_effort(const Json *effort, char err[CHAT_ERR])
{
  if (!effort || effort->type == JSON_NULL)
    return 75; /* encoding.py's DEFAULT_REASONING_EFFORT, "high" */
  if (effort->type == JSON_STRING)
  {
    const char *names[] = {"low", "high", "max"};
    const int budget[] = {50, 75, 100};
    for (int i = 0; i < 3; i++)
      if (strcmp(effort->s, names[i]) == 0)
        return budget[i];
  }
  if (effort->type == JSON_INT && effort->len <= 3 && atoi(effort->s) >= 1 && atoi(effort->s) <= 100)
    return atoi(effort->s);
  return chat_fail(err, "reasoning effort", "not an integer 1-100 nor low / high / max"), -1;
}

/* tool results in their calls' order: the last assistant's calls' ids before them (sort_tool_results_by_call_order) */
static bool sort_results(Msg *ms, size_t n)
{
  const Json *calls = NULL;
  for (size_t i = 0; i < n; i++)
  {
    const Json *tc = json_get(ms[i].src, "tool_calls");
    if (strcmp(ms[i].role, "assistant") == 0 && tc && tc->type == JSON_ARRAY && tc->n)
      calls = tc;
    if (strcmp(ms[i].role, "user") || !calls)
      continue;
    size_t nt = 0;
    for (size_t k = 0; k < ms[i].nb; k++)
      nt += ms[i].blocks[k].kind == BLK_TOOL;
    if (nt < 2)
      continue;
    size_t *at = malloc(nt * sizeof *at);
    long *key = malloc(nt * sizeof *key);
    Block *sorted = malloc(nt * sizeof *sorted);
    if (!at || !key || !sorted)
      return free(at), free(key), free(sorted), false;
    for (size_t k = 0, t = 0; k < ms[i].nb; k++)
      if (ms[i].blocks[k].kind == BLK_TOOL)
        at[t++] = k;
    for (size_t t = 0; t < nt; t++)
    {
      key[t] = 0; /* an id no call has: 0 (Python's .get(id, 0)); a repeated id: its last call */
      for (size_t c = 0; c < calls->n; c++)
      {
        const char *id = json_str(calls->item[c], "id");
        if (!id || !id[0])
          id = json_str(json_get(calls->item[c], "function"), "id");
        if (id && id[0] && strcmp(id, ms[i].blocks[at[t]].id) == 0)
          key[t] = (long)c;
      }
      sorted[t] = ms[i].blocks[at[t]];
    }
    for (size_t t = 1; t < nt; t++) /* stable: an insertion sort by the call's place */
      for (size_t u = t; u > 0 && key[u - 1] > key[u]; u--)
      {
        const Block x = sorted[u];
        const long kx = key[u];
        sorted[u] = sorted[u - 1], key[u] = key[u - 1], sorted[u - 1] = x, key[u - 1] = kx;
      }
    for (size_t t = 0; t < nt; t++)
      ms[i].blocks[at[t]] = sorted[t];
    free(at), free(key), free(sorted);
  }
  return true;
}

static bool has_text(const Msg *m)
{
  for (size_t k = 0; k < m->nb; k++)
    if (m->blocks[k].kind == BLK_TEXT)
      return true;
  return !m->nb;
}

/* find_last_user_index: a mid-conversation system counts; a turn of tool results alone does not */
static long last_user(const Msg *ms, size_t n)
{
  for (size_t i = n; i-- > 0;)
    if ((strcmp(ms[i].role, "user") == 0 || (strcmp(ms[i].role, "system") == 0 && i > 0)) && has_text(&ms[i]))
      return (long)i;
  return -1;
}

static void put_blocks(JsonBuf *b, const Msg *m)
{
  for (size_t k = 0; k < m->nb; k++)
  {
    const Block *x = &m->blocks[k];
    json_puts(b, k ? "\n\n" : "");
    if (x->kind == BLK_TEXT)
      json_puts(b, x->v && x->v->type == JSON_STRING ? x->v->s : "");
    else if (x->kind == BLK_OTHER)
      json_puts(b, "[Unsupported "), json_puts(b, json_str(x->v, "type") ? json_str(x->v, "type") : "None"),
          json_puts(b, "]");
    else
    {
      json_puts(b, "<tool_result>");
      if (x->v && x->v->type == JSON_ARRAY)
        for (size_t i = 0; i < x->v->n; i++)
        {
          const char *t = json_str(x->v->item[i], "type"), *s = json_str(x->v->item[i], "text");
          json_puts(b, i ? "\n\n" : "");
          if (t && strcmp(t, "text") == 0)
            json_puts(b, s ? s : "");
          else
            json_puts(b, "[Unsupported "), json_puts(b, t ? t : "None"), json_puts(b, "]");
        }
      else if (x->v && x->v->type == JSON_STRING)
        json_put(b, x->v->s, x->v->len);
      json_puts(b, "</tool_result>");
    }
  }
}

static bool render(JsonBuf *b, const Msg *ms, size_t n, size_t i, ChatMode mode, int effort, bool drop, char *err)
{
  const Msg *m = &ms[i];
  const long lu = last_user(ms, n);
  const bool think = mode == CHAT_THINKING, sys = strcmp(m->role, "system") == 0;
  if (i == 0 && (think || sys))
    json_puts(b, CHAT_SYSTEM);
  if (i == 0 && think)
  {
    char e[160];
    snprintf(e, sizeof e,
             "Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n", effort);
    json_puts(b, e);
  }
  if (sys)
  {
    const Json *tools = json_get(m->src, "tools"), *rf = json_get(m->src, "response_format");
    json_puts(b, i > 0 ? CHAT_SYSTEM : "");
    json_put(b, m->content ? m->content : "", m->content ? m->clen : 0);
    if (tools && tools->type == JSON_ARRAY && tools->n && (json_puts(b, "\n\n"), !chat_tools(b, tools, err)))
      return false;
    if (rf && rf->type != JSON_NULL && !(rf->type == JSON_OBJECT && !rf->n))
      json_puts(b, "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"),
          json_dump(b, rf);
  }
  else if (strcmp(m->role, "user") == 0)
    json_puts(b, CHAT_USER), put_blocks(b, m);
  else if (strcmp(m->role, "latest_reminder") == 0)
    json_puts(b, CHAT_REMINDER), json_put(b, m->content, m->clen);
  else
  {
    const Json *tc = json_get(m->src, "tool_calls"), *wo = json_get(m->src, "wo_eos");
    if (think && (!drop || (long)i > lu))
      json_puts(b, m->reasoning ? m->reasoning : ""), json_puts(b, CHAT_THINK_END);
    json_put(b, m->content ? m->content : "", m->content ? m->clen : 0);
    if (tc && tc->type == JSON_ARRAY && tc->n && !chat_calls(b, tc, err))
      return false;
    json_puts(b, wo && wo->type == JSON_TRUE ? "" : CHAT_EOS);
  }
  if (i + 1 < n && strcmp(ms[i + 1].role, "assistant") && strcmp(ms[i + 1].role, "latest_reminder"))
    return true;
  if (strcmp(m->role, "user") == 0 || (sys && i > 0))
    json_puts(b, CHAT_ASSISTANT), json_puts(b, think && (!drop || (long)i >= lu) ? CHAT_THINK : CHAT_THINK_END);
  return true;
}

bool chat_render(const Json *messages, ChatMode mode, int effort, JsonBuf *out, char err[CHAT_ERR])
{
  err[0] = 0;
  if (!messages || messages->type != JSON_ARRAY || !messages->n)
    return chat_fail(err, "messages", "not a non-empty list");
  size_t n = 0;
  Msg *ms = chat_read(messages, &n, err);
  bool ok = ms && !err[0], drop = true;
  for (size_t i = 0; i < messages->n; i++)
  {
    const Json *t = json_get(messages->item[i], "tools");
    if (t && t->type == JSON_ARRAY && t->n)
      drop = false; /* tools anywhere: reasoning kept */
  }
  if (ok && !sort_results(ms, n))
    ok = chat_fail(err, "tool results", "out of memory");
  /* earlier reasoning dropped (_drop_thinking_messages): an assistant before the last user loses its reasoning; every
     role this reads is one it keeps, so no message goes */
  const long lu = ok ? last_user(ms, n) : -1;
  for (size_t i = 0; ok && mode == CHAT_THINKING && drop && i < n; i++)
    if (strcmp(ms[i].role, "assistant") == 0 && (long)i < lu)
      ms[i].reasoning = NULL;
  if (ok)
    json_puts(out, CHAT_BOS);
  for (size_t i = 0; ok && i < n; i++)
    ok = render(out, ms, n, i, mode, effort, drop, err);
  if (ok && out->oom)
    ok = chat_fail(err, "the prompt", "out of memory");
  for (size_t i = 0; ms && i < n; i++)
    free(ms[i].blocks), free(ms[i].joined);
  free(ms);
  return ok;
}
