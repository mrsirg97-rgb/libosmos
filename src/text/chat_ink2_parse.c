#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/text/chat_ink2.h"

/* a completion's blocks read back (osmos/text/chat_ink2.h): the markers read in shape, the bytes taken from text */

const char *const ichat_refused[] = {"<|message_user|>", "<|message_system|>", "<|message_tool|>", "<|content_xml|>",
                                     "<|content_tool_error|>", "<|content_image|>", ICHAT_MODEL, ICHAT_THINKING,
                                     ICHAT_TEXT, ICHAT_CALL, NULL};

static bool fail(char *err, const char *why)
{
  snprintf(err, CHAT_ERR, "%s", why);
  return false;
}

static bool starts(const char *p, size_t n, const char *lit)
{
  const size_t k = strlen(lit);
  return n >= k && !memcmp(p, lit, k);
}

/* a prefix of lit (more bytes could make it) */
static bool begins(const char *p, size_t n, const char *lit)
{
  const size_t k = strlen(lit);
  return n < k && !memcmp(p, lit, n);
}

long ichat_header(const char *p, size_t n, IchatBlock *kind)
{
  static const struct
  {
    const char *lit;
    IchatBlock kind;
  } heads[] = {{ICHAT_MODEL ICHAT_THINKING, ICHAT_BLOCK_THINKING}, {ICHAT_MODEL ICHAT_TEXT, ICHAT_BLOCK_TEXT}};
  for (size_t i = 0; i < 2; i++)
  {
    if (starts(p, n, heads[i].lit))
      return *kind = heads[i].kind, (long)strlen(heads[i].lit);
    if (begins(p, n, heads[i].lit))
      return ICHAT_HOLD;
  }
  if (starts(p, n, ICHAT_EOS) || begins(p, n, ICHAT_EOS))
    return starts(p, n, ICHAT_EOS) ? (*kind = ICHAT_BLOCK_END, 0) : ICHAT_HOLD;
  if (starts(p, n, ICHAT_MODEL) && n > strlen(ICHAT_MODEL)) /* a name: a call's header (or none the model writes) */
    return *kind = p[strlen(ICHAT_MODEL)] == '<' ? ICHAT_BLOCK_OTHER : ICHAT_BLOCK_CALL, 0;
  if (starts(p, n, ICHAT_MODEL) || begins(p, n, ICHAT_MODEL))
    return ICHAT_HOLD;
  return *kind = ICHAT_BLOCK_OTHER, 0;
}

/* any special token's text in [p, p + n) of shape: a real one (the plain pieces' spellings defused) */
static bool special_in(const char *p, size_t n)
{
  static const char *const structural[] = {ICHAT_EOS, ICHAT_END};
  for (int k = 0; ichat_refused[k]; k++)
    if (memmem(p, n, ichat_refused[k], strlen(ichat_refused[k])))
      return true;
  for (int k = 0; k < 2; k++)
    if (memmem(p, n, structural[k], strlen(structural[k])))
      return true;
  return false;
}

bool ichat_parse_shaped(const char *text, const char *shape, size_t len, JsonBuf *out, char err[CHAT_ERR])
{
  const size_t ne = strlen(ICHAT_EOS), nm = strlen(ICHAT_MODEL);
  if (len < ne || memcmp(shape + len - ne, ICHAT_EOS, ne))
    return fail(err, "no end of turn");
  const size_t end = len - ne;
  JsonBuf reasoning = {0}, content = {0}, calls = {0};
  size_t at = 0, n_calls = 0;
  bool ok = true;
  while (ok && at < end)
  {
    if (!starts(shape + at, end - at, ICHAT_MODEL))
    {
      ok = fail(err, "a block without its header");
      break;
    }
    at += nm;
    const char *k = memmem(shape + at, end - at, "<|content_", 10);
    if (!k)
    {
      ok = fail(err, "a block without its kind");
      break;
    }
    const size_t name0 = at, name_n = (size_t)(k - (shape + at));
    at = (size_t)(k - shape);
    const char *kinds[3] = {ICHAT_THINKING, ICHAT_TEXT, ICHAT_CALL};
    int kind = -1;
    for (int i = 0; i < 3; i++)
      if (starts(shape + at, end - at, kinds[i]))
        kind = i;
    if (kind < 0)
    {
      ok = fail(err, "a block of a kind the model does not write");
      break;
    }
    at += strlen(kinds[kind]);
    const char *e = memmem(shape + at, end - at, ICHAT_END, strlen(ICHAT_END));
    if (!e)
    {
      ok = fail(err, "a block without its end");
      break;
    }
    const size_t body0 = at, body_n = (size_t)(e - (shape + at));
    at = body0 + body_n + strlen(ICHAT_END);
    if (special_in(shape + body0, body_n) || special_in(shape + name0, name_n))
      ok = fail(err, "a special token in the text");
    else if ((kind == 2) != (name_n > 0))
      ok = fail(err, "a call's header names it; no other block's does");
    else if (kind == 0)
      json_put(&reasoning, text + body0, body_n);
    else if (kind == 1)
      json_put(&content, text + body0, body_n);
    else
    {
      char jerr[JSON_ERR];
      Json *v = json_parse(text + body0, body_n, jerr);
      const Json *nm_v = v ? json_get(v, "name") : NULL, *args = v ? json_get(v, "args") : NULL;
      if (!v || v->type != JSON_OBJECT || v->n != 2 || !nm_v || nm_v->type != JSON_STRING || !args ||
          args->type != JSON_OBJECT || nm_v->len != name_n || memcmp(nm_v->s, text + name0, name_n))
        ok = fail(err, v ? "a call's JSON not {name, args} of its header's name" : "a call's JSON does not parse");
      else
      {
        json_puts(&calls, n_calls++ ? ", " : "");
        json_puts(&calls, "{\"type\": \"function\", \"function\": {\"name\": ");
        json_dump_string(&calls, text + name0, name_n);
        JsonBuf a = {0};
        json_dump(&a, args);
        json_puts(&calls, ", \"arguments\": "), json_dump_string(&calls, a.p ? a.p : "", a.n), json_puts(&calls, "}}");
        calls.oom |= a.oom;
        free(a.p);
      }
      json_free(v);
    }
  }
  if (ok)
  {
    json_puts(out, "{\"role\": \"assistant\", \"content\": "), json_dump_string(out, content.p ? content.p : "", content.n);
    json_puts(out, ", \"reasoning_content\": ");
    json_dump_string(out, reasoning.p ? reasoning.p : "", reasoning.n);
    json_puts(out, ", \"tool_calls\": ["), json_put(out, calls.p ? calls.p : "", calls.n), json_puts(out, "]}");
    ok = !(out->oom || reasoning.oom || content.oom || calls.oom) || fail(err, "out of memory");
  }
  free(reasoning.p), free(content.p), free(calls.p);
  return ok;
}

bool ichat_parse(const char *text, size_t len, JsonBuf *out, char err[CHAT_ERR])
{
  return ichat_parse_shaped(text, text, len, out, err);
}

void ichat_loose(const char *text, const char *shape, size_t len, JsonBuf *out)
{
  JsonBuf reasoning = {0}, content = {0};
  const size_t ne = strlen(ICHAT_EOS);
  if (len >= ne && !memcmp(shape + len - ne, ICHAT_EOS, ne))
    len -= ne;
  for (size_t at = 0; at < len;)
  {
    IchatBlock kind;
    const long h = ichat_header(shape + at, len - at, &kind);
    if (h <= 0 || (kind != ICHAT_BLOCK_THINKING && kind != ICHAT_BLOCK_TEXT))
      break;
    at += (size_t)h;
    const char *e = memmem(shape + at, len - at, ICHAT_END, strlen(ICHAT_END));
    const size_t n = e ? (size_t)(e - (shape + at)) : len - at;
    json_put(kind == ICHAT_BLOCK_THINKING ? &reasoning : &content, text + at, n);
    at += n + (e ? strlen(ICHAT_END) : 0);
  }
  json_puts(out, "{\"role\": \"assistant\", \"content\": "), json_dump_string(out, content.p ? content.p : "", content.n);
  json_puts(out, ", \"reasoning_content\": "), json_dump_string(out, reasoning.p ? reasoning.p : "", reasoning.n);
  json_puts(out, ", \"tool_calls\": []}");
  free(reasoning.p), free(content.p);
}
