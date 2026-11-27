/* the hy4 forward's step (osmos/hy4/forward.h): the runs checked, their tokens into their histories, Engram's rows
   read, the embedding into the hc streams, the layers (hip_hy4_layer.c), the kept rows' head - the reference's
   order, one stream */
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"
#include "osmos/err.h"

/* the tokens' embedding rows (decoded, unrounded: the file's values) copied into each of the hc streams */
static bool embed(Hy4Forward *f, uint32_t n)
{
  const Hy4Geometry *g = &f->w->geo;
  const GgufTensor *t = f->w->embed;
  const uint64_t D = (uint64_t)g->n_embd, hc = (uint64_t)g->hc;
  bool ok = q4x_rows(t->type, hy4_dev(f, t), t->bytes / t->dims[1], (uint32_t)D, f->tok, n, f->y, f->st);
  for (uint64_t k = 0; ok && k < hc; k++)
    ok = hipMemcpy2DAsync(f->h + k * D, hc * D * F, f->y, D * F, D * F, n, hipMemcpyDeviceToDevice, f->st) == hipSuccess;
  return ok;
}

/* a step's launches: the embedding, the layers (a tap after each), the kept rows' streams gathered, the head's
   collapse and the head on them */
static bool launches(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, uint32_t kept, bool read)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, V = (uint32_t)g->n_vocab;
  double t0 = f->pf.on > 1 ? hy4_now() : 0;
  bool ok = embed(f, n);
  if (ok && f->pf.on > 1 && read == (f->pf.on > 2))
    ok = hipStreamSynchronize((hipStream_t)f->st) == hipSuccess, f->pf.part_s[6] += hy4_now() - t0;
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    ok = hy4_layer(f, run, runs, n, l, read);
    if (ok && f->tap && (ok = hipStreamSynchronize(f->st) == hipSuccess))
      f->tap(f->tap_arg, l, f->h, n, hc, D);
  }
  for (uint32_t row = 0; ok && row < n; row++)
    if (f->slot[row] >= 0)
      ok = hipMemcpyAsync(f->hx + (uint64_t)f->slot[row] * hc * D, f->h + (uint64_t)row * hc * D, (uint64_t)hc * D * F,
                          hipMemcpyDeviceToDevice, f->st) == hipSuccess;
  ok = ok && (!kept || hy4_hc_pre(f->hx, f->fn_out, hy4_fv(f, f->w->out_hc_scale), hy4_fv(f, f->w->out_hc_base), hc,
                                  hc, D, g->rms_eps, g->hc_eps, g->hc_magnitude, hy4_fv(f, f->w->output_norm), kept,
                                  f->xn, NULL, f->hcs, f->st));
  /* DFly2's snapshot: each run's last rows' normed head input (its fusion's target hidden), where the head ran */
  for (uint32_t r = 0; ok && f->dfly && r < runs; r++)
  {
    const uint32_t len = run[r].len, keep = len < HY4_DFLY_KEEP ? len : HY4_DFLY_KEEP;
    for (uint32_t i = 0; ok && i < keep; i++)
    {
      const int32_t sl = f->slot[run[r].off + len - keep + i];
      run[r].s->dhid_ok[i] = sl >= 0;
      if (sl >= 0)
        ok = hipMemcpyAsync(run[r].s->dhid + (uint64_t)i * D, f->xn + (uint64_t)sl * D, (uint64_t)D * F,
                            hipMemcpyDeviceToDevice, f->st) == hipSuccess;
    }
  }
  t0 = f->pf.on > 1 ? hy4_now() : 0;
  ok = ok && (!kept || (f->head8 ? hy4_mat_raw(f, QTYPE_Q8_0, f->head8, V, D, f->xn, kept, f->logits, read)
                                 : hy4_mat(f, f->w->output, V, D, f->xn, kept, f->logits, read)));
  if (ok && f->pf.on > 1 && read == (f->pf.on > 2))
    ok = hipStreamSynchronize((hipStream_t)f->st) == hipSuccess, f->pf.part_s[7] += hy4_now() - t0;
  return ok;
}

/* the step (hy4_step_runs); keep_from: rows below it not kept whatever the prompts say (hy4_step's: its last rows) */
static bool step(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                 bool read, uint32_t keep_from, char *err)
{
  if (!f || !seqs || !lens || !tokens || !runs || runs > HY4_MAX_RUNS)
    return osmos_fail(err, "no forward, sequences or tokens, or past HY4_MAX_RUNS runs");
  if (!overflow_settle(&f->ovf, err)) /* (a reset the last step owes: refused with its cause while it fails) */
    return false;
  f->rows = 0; /* no row kept until the step runs: a refused one leaves none, not the step before's through a
                  half-rewritten slot table */
  Hy4Run run[HY4_MAX_RUNS];
  uint32_t n = 0, kept = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    Hy4Seq *s = seqs[r];
    if (!s || s->f != f || !lens[r])
      return osmos_fail(err, "a run of no sequence, of another forward, or of no tokens");
    if (s->broken)
      return osmos_fail(err, "a sequence after a failed step: load it, or make another");
    for (uint32_t k = 0; k < r; k++)
      if (seqs[k] == s)
        return osmos_fail(err, "a sequence twice in one step");
    if ((uint64_t)n + lens[r] > f->max_tokens || (uint64_t)s->pos + lens[r] > f->max_ctx)
      return osmos_fail(err, "the tokens past the forward's step, or a sequence past its context");
    run[r] = (Hy4Run){s, n, lens[r]};
    for (uint32_t t = 0; t < lens[r]; t++)
      kept += n + t >= keep_from && (!s->prompt_end || s->pos + t + 1 >= s->prompt_end);
    n += lens[r];
  }
  if (kept > f->logit_rows)
    return osmos_fail(err, "a step's kept rows past the logits' (Hy4Options' logit_rows)");
  for (uint32_t i = 0; i < n; i++)
    if (tokens[i] < 0 || tokens[i] >= f->w->geo.n_vocab)
      return osmos_fail(err, "a token outside the vocabulary");
  for (uint32_t r = 0; r < runs; r++)
    for (uint32_t t = 0; t < run[r].len; t++)
    {
      run[r].s->tokens[run[r].s->pos + t] = tokens[run[r].off + t];
      f->pin_pos[run[r].off + t] = run[r].s->pos + (int32_t)t;
    }
  /* the kept rows (the head's): their slots in the logits */
  for (uint32_t r = 0, k = 0; r < runs; r++)
    for (uint32_t t = 0; t < run[r].len; t++)
    {
      const uint32_t row = run[r].off + t;
      const bool keep = row >= keep_from && (!run[r].s->prompt_end || run[r].s->pos + t + 1 >= run[r].s->prompt_end);
      f->slot[row] = keep ? (int32_t)k : -1;
      k += keep;
    }
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  const double pf0 = f->pf.on ? hy4_now() : 0;
  f->pf.read = read;
  bool ok = hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            hipMemcpyAsync(f->pos, f->pin_pos, (size_t)n * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            hy4_engram_offsets(f, run, runs, n); /* (each Engram layer's rows read as it is queued) */
  ok = ok && launches(f, run, runs, n, kept, read);
  ok = ok && overflow_read(&f->ovf, f->st);
  /* one sync, whatever failed: the flags read, and the kernels done before membrane_end lets their experts go */
  const bool drained = hipStreamSynchronize(f->st) == hipSuccess;
  ok = ok && drained;
  if (f->pf.on)
    f->pf.steps++, f->pf.dsteps += read == (f->pf.on > 2), f->pf.tokens += n, f->pf.step_s += hy4_now() - pf0;
  if (f->ex.mem)
    membrane_end(f->ex.mem); /* the step's holds ended (its experts' kernels done) */
  const int32_t over = ok ? f->pin_flags[0] : 0, fault = ok ? f->pin_flags[1] : 0;
  f->rows = ok && !over && !fault ? n : 0;
  if (!ok || over || fault)
  {
    (void)rowstream_wait(&f->eg_rs);
    for (uint32_t r = 0; r < runs; r++)
      run[r].s->broken = true;
    { char rerr_[OSMOS_ERR] = ""; overflow_reset(&f->ovf, rerr_); } /* (failing: owed, the next step says so) */
    return osmos_fail(err, "%s", over    ? "an activation past f16's range (the step refused)"
                                 : fault ? "the router picked an expert out of range or twice (the step refused)"
                                         : "a GPU call or an Engram read failed in the step");
  }
  for (uint32_t r = 0; r < runs; r++)
  {
    Hy4Seq *s = run[r].s;
    s->dkeep = run[r].len < HY4_DFLY_KEEP ? run[r].len : HY4_DFLY_KEEP;
    s->dpos0 = s->pos + (int32_t)(run[r].len - s->dkeep);
    s->pos += (int32_t)run[r].len;
    s->back = s->back + run[r].len < HY4_MAX_BACK ? s->back + run[r].len : HY4_MAX_BACK;
  }
  return true;
}

bool hy4_step_runs(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[OSMOS_ERR])
{
  return step(f, seqs, lens, runs, tokens, false, 0, err);
}

bool hy4_step_read(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[OSMOS_ERR])
{
  return step(f, seqs, lens, runs, tokens, true, 0, err);
}

bool hy4_step(Hy4Forward *f, Hy4Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[OSMOS_ERR])
{
  if (!f || !s || !n || !rows || rows > n || rows > f->logit_rows)
    return osmos_fail(err, "a step of no tokens, or more rows than tokens or the logits'");
  if (!step(f, &s, &n, 1, tokens, n > HY4_MV_TOKENS, n - rows, err))
    return false;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  return hipMemcpy(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost) == hipSuccess ||
         osmos_fail(err, "a GPU call failed (the logits' copy)");
}

const float *hy4_logits(const Hy4Forward *f) { return f ? f->pin_logits : NULL; }

void hy4_kept(const Hy4Forward *f, KeptView *k)
{
  *k = (KeptView){0};
  if (f)
    *k = (KeptView){f->slot, f->rows, f->logits, (uint32_t)f->w->geo.n_vocab, f->ids, f->argmax, f->st, NULL, NULL, NULL};
}

const float *hy4_row_logits(const Hy4Forward *f, uint32_t row)
{
  KeptView k;
  hy4_kept(f, &k);
  return kept_row(&k, row);
}

bool hy4_greedy(Hy4Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  KeptView k;
  hy4_kept(f, &k);
  const char *why = kept_greedy(&k, row0, rows, ids);
  return !why || osmos_fail(err, "%s", why);
}
