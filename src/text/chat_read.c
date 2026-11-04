#include <stdlib.h>
#include <string.h>

#include "chat_impl.h"

/* encoding.py's messages read for text: process_image_messages' text parts joined (an image refused), and
   merge_tool_messages - a tool message a user turn's tool result, consecutive user turns one */

static bool add_block(Msg *m, int kind, const Json *v, const char *id)
{
  if (m->nb == m->cap)
  {
    Block *b = realloc(m->blocks, (m->cap = m->cap ? 2 * m->cap : 4) * sizeof *b);
    if (!b)
      return false;
    m->blocks = b;
  }
  m->blocks[m->nb++] = (Block){kind, v, id};
  return true;
}

static bool image(const Json *part) /* an image part: text-only here */
{
  const char *t = json_str(part, "type");
  return t && (strcmp(t, "image") == 0 || strcmp(t, "image_url") == 0);
}

/* a message's content as text: a string, or text parts joined by blank lines (none for null) */
static bool text_of(Msg *m, const Json *c, char *err)
{
  if (!c || c->type == JSON_NULL)
    return true;
  if (c->type == JSON_STRING)
  {
    if (strstr(c->s, CHAT_IMAGE))
      return chat_fail(err, m->role, "the image placeholder in its text (images: not taken yet)");
    m->content = c->s, m->clen = c->len;
    return true;
  }
  if (c->type != JSON_ARRAY)
    return chat_fail(err, m->role, "content not a string nor parts");
  JsonBuf b = {0};
  for (size_t i = 0, k = 0; i < c->n; i++)
  {
    if (image(c->item[i]))
      return free(b.p), chat_fail(err, m->role, "an image part (images: not taken yet)");
    const char *t = json_str(c->item[i], "type"), *x = json_str(c->item[i], "text");
    if (t && strcmp(t, "text") == 0)
      json_puts(&b, k++ ? "\n\n" : ""), json_puts(&b, x ? x : "");
  }
  json_puts(&b, "");
  if (b.oom)
    return free(b.p), chat_fail(err, m->role, "out of memory");
  m->joined = b.p, m->content = b.p, m->clen = b.n;
  return true;
}

static Msg *last_user_turn(Msg *ms, size_t n)
{
  return n && strcmp(ms[n - 1].role, "user") == 0 && ms[n - 1].nb ? &ms[n - 1] : NULL;
}

Msg *chat_read(const Json *src, size_t *n_out, char *err)
{
  Msg *ms = calloc(src->n + 1, sizeof *ms);
  size_t n = 0;
  if (!ms)
    return chat_fail(err, "messages", "out of memory"), NULL;
  for (size_t i = 0; i < src->n; i++)
  {
    const Json *s = src->item[i], *c = json_get(s, "content");
    const char *role = json_str(s, "role");
    if (!role)
      return chat_fail(err, "a message", "no role"), (void)(*n_out = n), ms;
    if (json_get(s, "task") || json_get(s, "content_blocks"))
      return chat_fail(err, role, "an internal field (task, content_blocks)"), (void)(*n_out = n), ms;
    const char *rc = json_str(s, "reasoning_content");
    if (rc && strstr(rc, CHAT_IMAGE))
      return chat_fail(err, role, "the image placeholder in its reasoning"), (void)(*n_out = n), ms;
    if (strcmp(role, "tool") == 0 || strcmp(role, "user") == 0)
    {
      if (c && c->type == JSON_NULL)
        return chat_fail(err, role, "content null"), (void)(*n_out = n), ms;
      Msg *u = last_user_turn(ms, n), *m = u ? u : &ms[n];
      if (!u)
        *m = (Msg){.role = "user", .src = s}, n++;
      bool ok = true;
      if (strcmp(role, "tool") == 0)
      {
        if (c && c->type == JSON_ARRAY)
          for (size_t k = 0; k < c->n; k++)
            if (image(c->item[k]))
              return chat_fail(err, role, "an image part (images: not taken yet)"), (void)(*n_out = n), ms;
        const char *id = json_str(s, "tool_call_id");
        ok = add_block(m, BLK_TOOL, c, id ? id : "");
      }
      else if (c && c->type == JSON_ARRAY)
        for (size_t k = 0; ok && k < c->n; k++)
        {
          const char *t = json_str(c->item[k], "type");
          if (image(c->item[k]))
            return chat_fail(err, role, "an image part (images: not taken yet)"), (void)(*n_out = n), ms;
          ok = add_block(m, t && strcmp(t, "text") == 0 ? BLK_TEXT : BLK_OTHER,
                         t && strcmp(t, "text") == 0 ? json_get(c->item[k], "text") : c->item[k], NULL);
        }
      else
      {
        if (c && c->type == JSON_STRING && strstr(c->s, CHAT_IMAGE))
          return chat_fail(err, role, "the image placeholder in its text"), (void)(*n_out = n), ms;
        ok = add_block(m, BLK_TEXT, c, NULL);
      }
      if (!ok)
        return chat_fail(err, role, "out of memory"), (void)(*n_out = n), ms;
      continue;
    }
    if (strcmp(role, "system") && strcmp(role, "assistant") && strcmp(role, "latest_reminder"))
      return chat_fail(err, role, "a role the encoding does not take"), (void)(*n_out = n), ms;
    ms[n] = (Msg){.role = role, .src = s, .reasoning = rc};
    if (!text_of(&ms[n++], c, err))
      return (void)(*n_out = n), ms;
    if (strcmp(role, "latest_reminder") == 0 && !ms[n - 1].content)
      return chat_fail(err, role, "no content"), (void)(*n_out = n), ms;
  }
  *n_out = n;
  return ms;
}
