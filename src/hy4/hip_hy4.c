/* the hy4 forward's step (osmos/hy4/forward.h): the runs checked, their tokens into their histories, Engram's rows
   read, the embedding into the hc streams, the layers (hip_hy4_layer.c), the kept rows' head - the reference's
   order, one stream */
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "hy4_impl.h"

/* the tokens' embedding rows (decoded, unrounded: the file's values) copied into each of the hc streams */
static bool embed(Hy4Forward *f, uint32_t n)
{
  const Hy4Geometry *g = &f->w->geo;
  const GgufTensor *t = f->w->embed;
  const uint64_t D = (uint64_t)g->n_embd, hc = (uint64_t)g->hc;
  bool ok = q4x_rows(t->type, hy4_dev(f, t), t->bytes / t->dims[1], (uint32_t)D, f->tok, n, f->y, NULL);
  for (uint64_t k = 0; ok && k < hc; k++)
    ok = hipMemcpy2DAsync(f->h + k * D, hc * D * F, f->y, D * F, D * F, n, hipMemcpyDeviceToDevice, NULL) == hipSuccess;
  return ok;
}

/* the step (hy4_step_runs); keep_from: rows below it not kept whatever the prompts say (hy4_step's: its last rows) */
static bool step(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                 uint32_t keep_from, char *err)
{
  if (!f || !seqs || !lens || !tokens || !runs || runs > HY4_MAX_RUNS)
    return hy4_fail(err, "no forward, sequences or tokens, or past HY4_MAX_RUNS runs");
  Hy4Run run[HY4_MAX_RUNS];
  uint32_t n = 0, kept = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    Hy4Seq *s = seqs[r];
    if (!s || s->f != f || !lens[r])
      return hy4_fail(err, "a run of no sequence, of another forward, or of no tokens");
    if (s->broken)
      return hy4_fail(err, "a sequence after a failed step: load it, or make another");
    for (uint32_t k = 0; k < r; k++)
      if (seqs[k] == s)
        return hy4_fail(err, "a sequence twice in one step");
    if ((uint64_t)n + lens[r] > f->max_tokens || (uint64_t)s->pos + lens[r] > f->max_ctx)
      return hy4_fail(err, "the tokens past the forward's step, or a sequence past its context");
    run[r] = (Hy4Run){s, n, lens[r]};
    for (uint32_t t = 0; t < lens[r]; t++)
      kept += n + t >= keep_from && (!s->prompt_end || s->pos + t + 1 >= s->prompt_end);
    n += lens[r];
  }
  if (kept > f->logit_rows)
    return hy4_fail(err, "a step's kept rows past the logits' (Hy4Options' logit_rows)");
  for (uint32_t i = 0; i < n; i++)
    if (tokens[i] < 0 || tokens[i] >= f->w->geo.n_vocab)
      return hy4_fail(err, "a token outside the vocabulary");
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, V = (uint32_t)g->n_vocab;
  for (uint32_t r = 0; r < runs; r++)
    for (uint32_t t = 0; t < run[r].len; t++)
    {
      run[r].s->tokens[run[r].s->pos + t] = tokens[run[r].off + t];
      f->pin_pos[run[r].off + t] = run[r].s->pos + (int32_t)t;
    }
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  bool ok = hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
            hipMemcpyAsync(f->pos, f->pin_pos, (size_t)n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
            hy4_engram_read(f, run, runs, n) && embed(f, n);
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    ok = hy4_layer(f, run, runs, n, l);
    if (ok && f->tap && (ok = hipStreamSynchronize(NULL) == hipSuccess))
      f->tap(f->tap_arg, l, f->h, n, hc, D);
  }
  /* the kept rows' streams gathered, then the head's collapse and the head on them alone */
  uint32_t k = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    const Hy4Seq *s = run[r].s;
    for (uint32_t t = 0; t < run[r].len; t++)
    {
      const uint32_t row = run[r].off + t;
      const bool keep = row >= keep_from && (!s->prompt_end || s->pos + t + 1 >= s->prompt_end);
      f->slot[row] = keep ? (int32_t)k : -1;
      if (keep && ok)
        ok = hipMemcpyAsync(f->hx + (uint64_t)k * hc * D, f->h + (uint64_t)row * hc * D, (uint64_t)hc * D * F,
                            hipMemcpyDeviceToDevice, NULL) == hipSuccess;
      k += keep;
    }
  }
  if (ok && kept)
    ok = hy4_hc_pre(f->hx, f->fn_out, hy4_fv(f, f->w->out_hc_scale), hy4_fv(f, f->w->out_hc_base), hc, hc, D,
                    g->rms_eps, g->hc_eps, g->hc_magnitude, hy4_fv(f, f->w->output_norm), kept, f->xn, NULL, f->hcs, NULL) &&
         hy4_mat(f, f->w->output, V, D, f->xn, kept, f->logits);
  ok = ok && hipStreamSynchronize(NULL) == hipSuccess;
  f->rows = ok ? n : 0;
  if (!ok)
  {
    for (uint32_t r = 0; r < runs; r++)
      run[r].s->broken = true;
    return hy4_fail(err, "a GPU call or an Engram read failed in the step");
  }
  for (uint32_t r = 0; r < runs; r++)
  {
    Hy4Seq *s = run[r].s;
    s->pos += (int32_t)run[r].len;
    s->back = s->back + run[r].len < HY4_MAX_BACK ? s->back + run[r].len : HY4_MAX_BACK;
  }
  return true;
}

bool hy4_step_runs(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[HY4_ERR])
{
  return step(f, seqs, lens, runs, tokens, 0, err);
}

bool hy4_step(Hy4Forward *f, Hy4Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[HY4_ERR])
{
  if (!f || !s || !n || !rows || rows > n || rows > f->logit_rows)
    return hy4_fail(err, "a step of no tokens, or more rows than tokens or the logits'");
  if (!step(f, &s, &n, 1, tokens, n - rows, err))
    return false;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  return hipMemcpy(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost) == hipSuccess ||
         hy4_fail(err, "a GPU call failed (the logits' copy)");
}

const float *hy4_logits(const Hy4Forward *f) { return f ? f->pin_logits : NULL; }

const float *hy4_row_logits(const Hy4Forward *f, uint32_t row)
{
  return f && row < f->rows && f->slot[row] >= 0 ? f->logits + (uint64_t)f->slot[row] * f->w->geo.n_vocab : NULL;
}

bool hy4_greedy(Hy4Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[HY4_ERR])
{
  if (!f || !ids || !rows || row0 + rows > f->rows)
    return hy4_fail(err, "no forward or ids, or rows past the last step's");
  for (uint32_t r = 0; r < rows; r++)
    if (f->slot[row0 + r] < 0 || f->slot[row0 + r] != f->slot[row0] + (int32_t)r)
      return hy4_fail(err, "a row the last step kept no logits of");
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  if (!ops_argmax(f->logits + (uint64_t)f->slot[row0] * V, V, rows, V, f->ids, f->argmax, NULL) ||
      hipMemcpy(ids, f->ids, rows * 4, hipMemcpyDeviceToHost) != hipSuccess)
    return hy4_fail(err, "a GPU call failed (the greedy pick)");
  for (uint32_t r = 0; r < rows; r++)
    if (ids[r] < 0)
      return hy4_fail(err, "a NaN in a row's logits");
  return true;
}
