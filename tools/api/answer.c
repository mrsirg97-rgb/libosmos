#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "answer.h"

/* the streamed answer (DESIGN.md, "osmos-api: the answer"): reasoning_content deltas until </think>, content after,
   nothing from a calls block, the EOS or a special token's text the parse refuses on (the parse at DONE says the
   rest: an error, never that text). what could still turn out to be one of those markers - a marker's start,
   whitespace before it (the parse trims it), a UTF-8 character's first bytes - is held until it cannot */

static bool space(char c)
{
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

/* a chunk: its delta (a JSON object's text), its finish_reason (NULL: null), more top-level fields (", ..." or "") */
bool answer_chunk(Answer *s, const char *delta, const char *finish, const char *extra)
{
  if (s->gone || !s->stream)
    return !s->gone;
  JsonBuf b = {0};
  json_puts(&b, "{\"id\": \""), json_puts(&b, s->id), json_puts(&b, "\", \"object\": \"chat.completion.chunk\"");
  char t[64];
  snprintf(t, sizeof t, ", \"created\": %lld, \"model\": ", s->created);
  json_puts(&b, t), json_dump_string(&b, s->model, strlen(s->model));
  json_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": "), json_puts(&b, delta);
  json_puts(&b, ", \"finish_reason\": ");
  if (finish)
    json_dump_string(&b, finish, strlen(finish));
  else
    json_puts(&b, "null");
  json_puts(&b, "}]"), json_puts(&b, extra), json_puts(&b, "}");
  s->gone = b.oom || !http_stream_event(s->fd, b.p, b.n);
  free(b.p);
  return !s->gone;
}

static void delta(Answer *s, const char *p, size_t n)
{
  if (!n)
    return;
  s->begun = true;
  JsonBuf b = {0};
  json_puts(&b, s->phase == PHASE_REASONING ? "{\"reasoning_content\": " : "{\"content\": ");
  json_dump_string(&b, p, n), json_puts(&b, "}");
  if (!b.oom)
    answer_chunk(s, b.p, NULL, "");
  free(b.p);
}

/* the bytes at p's end that could still begin a marker, or end inside a character, and the whitespace before them */
static size_t held(const char *p, size_t n, const char *const *marks, int nm)
{
  size_t keep = 0;
  for (int m = 0; m < nm; m++)
    for (size_t k = strlen(marks[m]) - 1; k; k--)
      if (k <= n && !memcmp(p + n - k, marks[m], k))
      {
        keep = k > keep ? k : keep;
        break;
      }
  size_t j = n;
  while (j && n - j < 3 && ((unsigned char)p[j - 1] & 0xC0) == 0x80)
    j--;
  if (j)
  {
    const unsigned char lead = (unsigned char)p[j - 1];
    const size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    if (n - (j - 1) < need && n - (j - 1) > keep)
      keep = n - (j - 1);
  }
  while (keep < n && space(p[n - keep - 1]))
    keep++;
  return keep;
}

void answer_advance(Answer *s)
{
  while (s->phase != PHASE_DONE)
  {
    while (s->d->trims && !s->begun && s->sent < s->text.n && space(s->text.p[s->sent]))
      s->sent++;
    const char *p = s->text.p + s->sent;
    const size_t n = s->text.n - s->sent;
    const char *marks[2 + DIALECT_REFUSED_MAX] = {s->phase == PHASE_REASONING ? s->d->think_end : s->d->calls_open,
                                                  s->d->eos}; /* the phase's own end first: it wins a tie */
    int nm = 2;
    for (int k = 0; s->d->refused[k]; k++)
      if (strcmp(s->d->refused[k], marks[0]) && strcmp(s->d->refused[k], marks[1]))
        marks[nm++] = s->d->refused[k];
    const char *at = NULL;
    int which = 0;
    for (int m = 0; m < nm; m++)
    {
      const char *q = memmem(p, n, marks[m], strlen(marks[m]));
      if (q && (!at || q < at))
        at = q, which = m;
    }
    if (!at)
    {
      const size_t out = n - held(p, n, marks, nm);
      delta(s, p, out);
      s->sent += out;
      return;
    }
    size_t e = (size_t)(at - p);
    if (which >= 2) /* a refused text: what before it could still begin a marker never streams either */
      e -= held(p, e, marks, nm);
    while (e && space(p[e - 1]))
      e--;
    delta(s, p, e);
    if (which == 0 && s->phase == PHASE_REASONING)
      s->phase = PHASE_CONTENT, s->begun = false, s->sent += (size_t)(at - p) + strlen(s->d->think_end);
    else
      s->phase = PHASE_DONE, s->sent += e; /* what was held stays unsent: a flush (length) sends it */
  }
}

/* a completion out of the format: what was held back goes out as it is (but the EOS) */
void answer_flush(Answer *s)
{
  size_t n = s->text.n - s->sent;
  const size_t eos = strlen(s->d->eos);
  if (n >= eos && !memcmp(s->text.p + s->text.n - eos, s->d->eos, eos))
    n -= eos;
  if (s->phase == PHASE_DONE)
    s->phase = PHASE_CONTENT;
  delta(s, s->text.p + s->sent, n);
  s->sent += n;
}
