#include "forward_impl.h"
#include "osmos/gpu/mv.h"
#include "osmos/model/qtype.h"

/* the next layer's experts guessed a layer ahead, for a small step (a decode's, a verify's: picks fewer than a
   layer's experts - a prompt's step already reads the whole next layer ahead, hip_ffn.c's prefetch). the guess: the
   next layer's router on this layer's FFN input (the stream it will route moves a layer's residual on), the top
   FWD_GUESS_K a token, written into the host's mapped r_guess. queued before the routing event, so the host's one
   wait covers both; the guessed experts the next layer would stream are then wanted (their reads begun) under this
   layer's experts and the next one's attention. which experts are read ahead changes no bit */

static bool small(const Forward *f, int l)
{
  const uint32_t E = (uint32_t)f->g->n_experts;
  /* (so n < E: within r_guess's rows) */
  return !f->no_guess && l + 1 < f->n_layers && f->n * (uint32_t)f->g->n_used < E;
}

/* a prompt's step (its picks a layer's experts at least) with a layer after it: its read ahead the union of the next
   router's picks for its tokens (ahead.h) - the router's own matmul on this layer's input, whatever no_guess says
   (the decode's guess is that; this reads less than a read ahead of every expert) */
bool fwd_prompt_ahead(const Forward *f, int l)
{
  return l + 1 < f->n_layers && f->n * (uint32_t)f->g->n_used >= (uint32_t)f->g->n_experts;
}

bool fwd_guess_queue(Forward *f, int l)
{
  if (fwd_prompt_ahead(f, l))
  {
    const LayerWeights *L1 = &f->w->layer[l + 1];
    const uint32_t D = f->g->n_embd, E = f->g->n_experts;
    return fwd_mat(f, L1->router, fwd_dev(f, L1->router), E, D, f->b.xf, D, f->n, f->b.logits, E) &&
           ops_router(f->b.logits, (const float *)fwd_dev(f, L1->router_bias), E, FWD_GUESS_K, f->g->expert_scale,
                      f->n, f->r_guess_dev, f->b.gw, NULL);
  }
  if (!small(f, l))
    return true;
  const Geometry *g = f->g;
  const LayerWeights *L1 = &f->w->layer[l + 1];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, E = g->n_experts, n = f->n;
  /* b->logits is free: this layer's router has read it (the stream in order). an F32 router's BF16 copy where there
     is one (guess_routers: the guess's alone), its matvec a few tokens a call */
  bool ok = true;
  if (f->guess_router[l + 1])
    for (uint32_t t0 = 0; ok && t0 < n; t0 += MV_MAX_TOKENS)
      ok = mv_strided(QTYPE_BF16, f->guess_router[l + 1], E, D, b->xf + (uint64_t)t0 * D, D,
                      n - t0 < MV_MAX_TOKENS ? n - t0 : MV_MAX_TOKENS, b->logits + (uint64_t)t0 * E, E, NULL);
  else
    ok = fwd_mat(f, L1->router, fwd_dev(f, L1->router), E, D, b->xf, D, n, b->logits, E);
  return ok && ops_router(b->logits, (const float *)fwd_dev(f, L1->router_bias), E, FWD_GUESS_K, g->expert_scale, n,
                          f->r_guess_dev, b->gw, NULL); /* into the host's mapped r_guess */
}

/* after the routing event: the guessed experts of the next layer not resident, wanted at the next visit (the next
   layer's), each marked for the score (membrane_guess) */
void fwd_guess_want(Forward *f, int l)
{
  if (!small(f, l))
    return;
  const uint32_t E = (uint32_t)f->g->n_experts, next = (uint32_t)(l + 1) * E;
  for (uint32_t m = 0; m < f->n * FWD_GUESS_K; m++)
  {
    const int32_t e = f->r_guess[m];
    if (e >= 0 && (uint32_t)e < E)
      membrane_guess(f->mem, next + (uint32_t)e); /* wanted at the next layer's visit */
  }
}
