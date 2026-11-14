#define _GNU_SOURCE
#include "loop.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool is_role(const Json *m, const char *role)
{
  const char *r = json_str(m, "role");
  return r && !strcmp(r, role);
}

static const Json *calls_of(const Json *m)
{
  const Json *c = is_role(m, "assistant") ? json_get(m, "tool_calls") : NULL;
  return c && c->type == JSON_ARRAY && c->n ? c : NULL;
}

/* a call's function (OpenAI's {"function": {...}} or bare {name, arguments}) */
static const Json *fn_of(const Json *call)
{
  const Json *f = json_get(call, "function");
  return f ? f : call;
}

/* a call's identity: its name and its arguments as their JSON (a string of JSON parsed and written again, so two
   spellings of one object match), into b */
static void identity(const Json *call, JsonBuf *b)
{
  const Json *f = fn_of(call), *a = json_get(f, "arguments");
  const char *name = json_str(f, "name");
  json_puts(b, name ? name : ""), json_puts(b, "\n");
  if (a && a->type == JSON_STRING)
  {
    char jerr[JSON_ERR];
    Json *o = json_parse(a->s, a->len, jerr);
    if (o)
      json_dump(b, o), json_free(o);
    else
      json_put(b, a->s, a->len);
  }
  else if (a)
    json_dump(b, a);
}

static void keep(char (*list)[LOOP_ID_MAX], unsigned *n, const char *id)
{
  if (*n < LOOP_IDS)
    snprintf(list[*n], LOOP_ID_MAX, "%s", id ? id : "");
  (*n)++;
}

void loop_read(const Json *ms, Loop *l)
{
  memset(l, 0, sizeof *l);
  if (!ms || ms->type != JSON_ARRAY)
    return;
  /* turns since the last call */
  for (size_t i = ms->n; i-- > 0;)
  {
    if (!is_role(ms->item[i], "assistant"))
      continue;
    if (calls_of(ms->item[i]))
      break;
    l->turns_without_call++;
  }
  /* the recent calls' identities, their most repeated (the latest among equals) */
  JsonBuf id[LOOP_WINDOW] = {{0}};
  const char *names[LOOP_WINDOW] = {0};
  unsigned k = 0;
  for (size_t i = ms->n; i-- > 0 && k < LOOP_WINDOW;)
  {
    const Json *c = calls_of(ms->item[i]);
    for (size_t j = c ? c->n : 0; j-- > 0 && k < LOOP_WINDOW;)
      identity(c->item[j], &id[k]), names[k] = json_str(fn_of(c->item[j]), "name"), k++;
  }
  for (unsigned a = 0; a < k; a++)
  {
    unsigned same = 0;
    for (unsigned b = 0; b < k; b++)
      same += id[a].n == id[b].n && !memcmp(id[a].p, id[b].p, id[a].n);
    if (same > l->repeat_count)
      l->repeat_count = same, snprintf(l->repeat_name, sizeof l->repeat_name, "%s", names[a] ? names[a] : "");
  }
  for (unsigned a = 0; a < k; a++)
    free(id[a].p);
  /* orphans: each assistant's calls against the tool messages after it, to the next assistant */
  for (size_t i = 0; i < ms->n; i++)
  {
    const Json *c = calls_of(ms->item[i]);
    if (is_role(ms->item[i], "tool") && (i == 0 || !is_role(ms->item[i - 1], "tool")))
    { /* a run of results with no assistant's calls before it (the message before the run not a call) */
      size_t j = i;
      while (j > 0 && is_role(ms->item[j - 1], "tool"))
        j--;
      if (j == 0 || !calls_of(ms->item[j - 1]))
        for (size_t t = i; t < ms->n && is_role(ms->item[t], "tool"); t++)
          keep(l->orphan_results, &l->n_orphan_results, json_str(ms->item[t], "tool_call_id"));
    }
    if (!c)
      continue;
    size_t end = i + 1;
    while (end < ms->n && is_role(ms->item[end], "tool"))
      end++;
    bool answered[256] = {false}; /* (a turn's calls: past 256 counted as answered - never a false orphan) */
    for (size_t t = i + 1; t < end; t++)
    {
      const char *rid = json_str(ms->item[t], "tool_call_id");
      bool found = false;
      for (size_t q = 0; q < c->n && q < 256; q++)
      {
        const char *cid = json_str(c->item[q], "id");
        if (!answered[q] && (!rid || !cid ? !rid : !strcmp(rid, cid))) /* no ids: results take calls in order */
        {
          answered[q] = found = true;
          break;
        }
      }
      if (!found)
        keep(l->orphan_results, &l->n_orphan_results, rid);
    }
    for (size_t q = 0; q < c->n && q < 256; q++)
      if (!answered[q])
        keep(l->orphan_calls, &l->n_orphan_calls, json_str(c->item[q], "id"));
  }
  l->stuck = l->turns_without_call >= LOOP_TURNS || l->repeat_count >= LOOP_REPEATS;
}

bool loop_fill(const Json *body, JsonBuf *out)
{
  const Json *ms = json_get(body, "messages");
  if (!ms || ms->type != JSON_ARRAY || body->type != JSON_OBJECT)
    return false;
  json_puts(out, "{");
  for (size_t k = 0; k < body->n; k++)
  {
    json_puts(out, k ? ", " : ""), json_dump_string(out, body->key[k], body->klen[k]), json_puts(out, ": ");
    if (body->item[k] != ms)
    {
      json_dump(out, body->item[k]);
      continue;
    }
    json_puts(out, "[");
    for (size_t i = 0; i < ms->n; i++)
    {
      json_puts(out, i ? ", " : ""), json_dump(out, ms->item[i]);
      const Json *c = calls_of(ms->item[i]);
      if (!c)
        continue;
      size_t end = i + 1; /* the call's results as they are, then a placeholder for each id none of them answers */
      while (end < ms->n && is_role(ms->item[end], "tool"))
        end++;
      for (size_t t = i + 1; t < end; t++)
        json_puts(out, ", "), json_dump(out, ms->item[t]);
      bool used[256] = {false};
      for (size_t t = i + 1; t < end; t++)
      {
        const char *rid = json_str(ms->item[t], "tool_call_id");
        for (size_t q = 0; q < c->n && q < 256; q++)
        {
          const char *cid = json_str(c->item[q], "id");
          if (!used[q] && (!rid || !cid ? !rid : !strcmp(rid, cid)))
          {
            used[q] = true;
            break;
          }
        }
      }
      for (size_t q = 0; q < c->n && q < 256; q++)
        if (!used[q])
        {
          const char *cid = json_str(c->item[q], "id");
          json_puts(out, ", {\"role\": \"tool\", \"tool_call_id\": ");
          json_dump_string(out, cid ? cid : "", cid ? strlen(cid) : 0);
          json_puts(out, ", \"content\": "), json_dump_string(out, LOOP_ORPHAN_TEXT, strlen(LOOP_ORPHAN_TEXT));
          json_puts(out, "}");
        }
      i = end - 1;
    }
    json_puts(out, "]");
  }
  json_puts(out, "}");
  return !out->oom;
}

static void ids(JsonBuf *b, char (*list)[LOOP_ID_MAX], unsigned n)
{
  json_puts(b, "[");
  for (unsigned i = 0; i < n && i < LOOP_IDS; i++)
    json_puts(b, i ? ", " : ""), json_dump_string(b, list[i], strlen(list[i]));
  json_puts(b, "]");
}

void loop_json(JsonBuf *b, const Loop *l)
{
  char t[128];
  snprintf(t, sizeof t, ", \"osmos\": {\"loop\": {\"turns_without_call\": %u, \"repeated_call\": {\"name\": ",
           l->turns_without_call);
  json_puts(b, t), json_dump_string(b, l->repeat_name, strlen(l->repeat_name));
  snprintf(t, sizeof t, ", \"count\": %u}, \"orphan_calls\": ", l->repeat_count);
  json_puts(b, t), ids(b, (char (*)[LOOP_ID_MAX])l->orphan_calls, l->n_orphan_calls);
  json_puts(b, ", \"orphan_results\": "), ids(b, (char (*)[LOOP_ID_MAX])l->orphan_results, l->n_orphan_results);
  snprintf(t, sizeof t, ", \"orphans_filled\": %s, \"stuck\": %s}}", l->filled ? "true" : "false",
           l->stuck ? "true" : "false");
  json_puts(b, t);
}

void loop_header(char *out, size_t cap, const Loop *l)
{
  char name[LOOP_NAME_MAX];
  size_t k = 0; /* the name's bytes a header can hold: printable ASCII, no separators */
  for (const char *p = l->repeat_name; *p && k + 1 < sizeof name; p++)
    if (*p > ' ' && *p < 127 && *p != ';' && *p != ',')
      name[k++] = *p;
  name[k] = 0;
  snprintf(out, cap, "turns-without-call=%u; repeated-call=%u%s%s; orphan-calls=%u; orphan-results=%u; "
           "orphans-filled=%d; stuck=%d", l->turns_without_call, l->repeat_count, k ? " " : "", name,
           l->n_orphan_calls, l->n_orphan_results, l->filled, l->stuck);
}
