/* a round of --parallel's answering lanes, drafting ones among them (multi.h) */
#include <stdio.h>
#include <string.h>

#include "multi.h"

#define MAX_LANES 64

typedef struct Entry
{
  Lane *ln;
  int off, k; /* its tokens in the step: off .. off + k (its token, then its k drafts) */
  int32_t pos;
} Entry;

/* a greedy lane's token printed and its end judged before it is fed: false when the lane is done */
static bool greedy_goes_on(Multi *m, Lane *ln)
{
  out("T %lld %d\n", ln->g.rid, ln->t), ln->produced++;
  if (server_eos(m->sv, ln->t))
    return lane_finish(m, ln, "stop"), false;
  if (ln->produced >= ln->limit || (uint32_t)oseq_position(ln->c.s) + 1 >= m->sv->max_ctx)
    return lane_finish(m, ln, "length"), false;
  return true;
}

/* a drafting lane's drafts, cut to the room it has (its limit, the context, the round's tokens less one a lane after
   it); -1 for the drafter's failure */
static int drafts(Multi *m, Lane *ln, int32_t *dr, int room)
{
  float cf[OSMOS_MAX_BLOCK];
  int k = oseq_draft(ln->c.s, ln->t, OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, dr, cf, ln->c.err);
  if (k < 0)
    return -1;
  const long long left = ln->limit - ln->produced - 1;
  if (k > left)
    k = left > 0 ? (int)left : 0;
  if ((uint32_t)oseq_position(ln->c.s) + (uint32_t)k + 2 >= m->sv->max_ctx)
    k = 0;
  return k < room ? k : room > 0 ? room : 0;
}

/* a drafting lane after the verify: its token and accepted drafts printed (to an end token, its limit), the state
   rolled back to all but the last printed, the drafter seeded from the verify's rows */
static void settle(Multi *m, const Entry *e, const int32_t *toks, const int32_t *picks)
{
  Lane *ln = e->ln;
  const int32_t *dr = toks + e->off + 1, *pk = picks + e->off;
  int acc = 0;
  while (acc < e->k && pk[acc] == dr[acc])
    acc++;
  ln->drafted += e->k, ln->accepted += acc;
  int printed = 0;
  const char *end = NULL;
  for (int i = 0; i <= acc && !end; i++, printed++)
  {
    const int32_t x = toks[e->off + i];
    out("T %lld %d\n", ln->g.rid, x), ln->produced++;
    end = server_eos(m->sv, x) ? "stop" : ln->produced >= ln->limit ? "length" : NULL;
  }
  const int keep = end ? printed - 1 : acc + 1;
  bool ok = oseq_rollback(ln->c.s, (uint32_t)(e->k + 1 - keep)) &&
            (!keep || oseq_seed(ln->c.s, e->pos, e->pos + keep, ln->c.err));
  for (int i = 0; ok && i < keep; i++)
    if (!(ok = ids_push(&ln->c.held, toks[e->off + i])))
      snprintf(ln->c.err, sizeof ln->c.err, "out of memory");
  if (!ok)
  {
    lane_fail(m, ln);
    return;
  }
  ln->t = pk[acc];
  if (end)
    lane_finish(m, ln, end);
  else if ((uint32_t)oseq_position(ln->c.s) + 1 >= m->sv->max_ctx)
    lane_finish(m, ln, "length");
}

void round_all(Multi *m)
{
  Server *sv = m->sv;
  OsmosSeq *seqs[MAX_LANES * (OSMOS_MAX_BLOCK + 1)];
  int32_t toks[MAX_LANES * (OSMOS_MAX_BLOCK + 1)], picks[MAX_LANES * (OSMOS_MAX_BLOCK + 1)];
  Entry in[MAX_LANES];
  int nin = 0, total = 0, after = 0;
  for (int i = 0; i < m->n; i++)
    after += m->lane[i].st == ANSWERING;
  for (int i = 0; i < m->n; i++)
  {
    Lane *ln = &m->lane[i];
    if (ln->st != ANSWERING)
      continue;
    after--;
    if (!ln->g.spec && !greedy_goes_on(m, ln))
      continue;
    int k = 0;
    if (ln->g.spec && (k = drafts(m, ln, toks + total + 1, (int)sv->round_tokens - total - 1 - after)) < 0)
    {
      lane_fail(m, ln);
      continue;
    }
    in[nin++] = (Entry){ln, total, k, oseq_position(ln->c.s)};
    toks[total] = ln->t;
    for (int j = 0; j <= k; j++)
      seqs[total + j] = ln->c.s;
    total += k + 1;
  }
  if (!total)
    return;
  char err[OSMOS_ERR];
  if (!osmos_batch(sv->e, seqs, toks, (uint32_t)total, err) || !osmos_greedy(sv->e, 0, (uint32_t)total, picks, err))
  {
    for (int j = 0; j < nin; j++)
      snprintf(in[j].ln->c.err, sizeof in[j].ln->c.err, "%s", err), lane_fail(m, in[j].ln);
    return;
  }
  for (int j = 0; j < nin; j++)
  {
    Lane *ln = in[j].ln;
    if (ln->g.spec)
      settle(m, &in[j], toks, picks);
    else if (!ids_push(&ln->c.held, toks[in[j].off]))
      snprintf(ln->c.err, sizeof ln->c.err, "out of memory"), lane_fail(m, ln);
    else
      ln->t = picks[in[j].off];
  }
}
