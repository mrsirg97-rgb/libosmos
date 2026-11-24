/* Inkling-2-Mini's weights (osmos/ink2/weights.h). on its reference GGUF (tools/convert/ink2_ref_gguf.py --name
   Inkling-2-Mini): the loop (its core, its passes' cache slots, each core layer's twelve controls and the two message
   tables), the experts NVFP4 with their scales then MXFP8; then the file edited in memory - a loop out of shape, a
   control missing, misshapen or past the core, a scale gone - each refused, its cause named. on unsloth's GGUF
   (upstream llama.cpp's keys): the same geometry, a value a slot (a slot unlike its layer refused), and the file
   refused at its BF16 experts (no expert path reads BF16). a fused gate and up's expert units, on weights built by
   hand. OSMOS_TEST_INK2_MINI (the reference's first shard) and OSMOS_TEST_INK2_MINI_UNSLOTH, else ~/Models/
   inkling2-mini's; each skipped without */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"

#define REF "%s/Models/inkling2-mini/osmos/ink2-mini-00001-of-00002.gguf"
#define UNSLOTH "%s/Models/inkling2-mini/gguf/Inkling-2-Mini-UD-Q8_K_XL.gguf"
#define FD 7 /* a descriptor the units carry, never read */

static Gguf *model;

static GgufTensor *tensor(const char *name)
{
  GgufTensor *t = (GgufTensor *)gguf_tensor(model, name);
  assert(t);
  return t;
}

static GgufKv *kv(const char *key)
{
  GgufKv *k = (GgufKv *)gguf_kv(model, key);
  assert(k);
  return k;
}

static bool accepted(void)
{
  char err[OSMOS_ERR] = "";
  Ink2Weights *w = ink2_weights_create(model, err);
  if (!w)
    printf("    refused: %s\n", err);
  ink2_weights_destroy(w);
  return w != NULL;
}

static void refused(const char *what, const char *needle)
{
  char err[OSMOS_ERR] = "";
  Ink2Weights *w = ink2_weights_create(model, err);
  if (w || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, w ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-52s -> %s\n", what, err);
}

static Gguf *open_model(const char *env, const char *fallback)
{
  char path[512], gerr[OSMOS_ERR] = "";
  if (getenv(env))
    snprintf(path, sizeof path, "%s", getenv(env));
  else
    snprintf(path, sizeof path, fallback, getenv("HOME") ? getenv("HOME") : "");
  Gguf *m = gguf_create(path, gerr);
  if (!m)
    printf("  %s: not on this machine, skipped (%s)\n", path, gerr);
  return m;
}

/* the release's text config (config.json), read back - whichever keys the file spells it in */
static void test_geometry(const Ink2Geometry *g)
{
  assert(g->n_layers == 24 && g->n_slots == 48 && g->n_embd == 2304);
  assert(g->loop.start == 6 && g->loop.end == 18 && g->loop.passes == 3 && g->loop.gate_rank == 64);
  assert(g->n_slots == g->n_layers + (g->loop.passes - 1) * (g->loop.end - g->loop.start));
  for (int s = 0; s < g->n_slots; s++)
    assert(ink2_slot_layer(g, s) == (s < 24 ? s : 6 + (s - 24) % 12));
  assert(g->n_heads == 24 && g->n_kv_heads == 8 && g->head_dim == 128 && g->window == 1024);
  assert(g->d_rel == 16 && g->rel_extent == 1024 && g->log_floor == 8192 && fabsf(g->log_alpha - 0.1f) < 1e-6f);
  assert(g->q_clip == g->k_clip && fabsf(g->q_clip - 22.627417f) < 1e-4f && g->rel_clip == 8.0f);
  assert(g->conv == 4 && g->res_block == 6 && fabsf(g->rms_eps - 1e-6f) < 1e-9f && g->native_ctx == 1048576);
  assert(g->n_experts == 128 && g->n_used == 8 && g->ff_exp == 1152 && g->route_scale == 8.0f);
  assert(g->gate_softcap == 4.0f && g->up_softcap == 25.0f && g->logit_softcap == 30.0f);
  assert(g->n_vocab == 200058);
  for (int l = 0; l < g->n_layers; l++)
    assert(g->sliding[l] == (l % 6 != 5)); /* full at 5, 11, 17, 23 */
  printf("  %d layers in %d slots (core %d-%d, %d passes, gate rank %d), %d heads (%d KV) of %d, window %d, "
         "relative %d x %d, %d experts of %d (%d a token)\n",
         g->n_layers, g->n_slots, g->loop.start, g->loop.end - 1, g->loop.passes, g->loop.gate_rank, g->n_heads,
         g->n_kv_heads, g->head_dim, g->window, g->d_rel, g->rel_extent, g->n_experts, g->ff_exp, g->n_used);
}

static void test_layers(const Ink2Weights *w)
{
  const Ink2Geometry *g = &w->geo;
  int nv = 0, mx = 0;
  for (int l = 0; l < g->n_layers; l++)
  {
    const Ink2Layer *L = &w->layer[l];
    assert(g->sliding[l] ? L->attn_norm && L->k_conv && !L->q_norm && !L->k_conv_proj
                         : L->q_norm && L->k_norm && L->v_norm && L->k_conv_proj && L->v_conv_static && !L->attn_norm);
    assert(l == 0 ? !L->attn_res_query : L->attn_res_query && L->attn_res_query->n_dims == (g->sliding[l] ? 1u : 2u));
    /* the release's: layers 0-19 NVFP4 with a scale an expert, 20-23 MXFP8 without; gate and up apart */
    assert(!L->exp_fused && L->exp_gate != L->exp_up && L->exp_up->type == L->exp_gate->type);
    const bool is_nv = L->exp_gate->type == QTYPE_NVFP4;
    assert(is_nv == (l < 20) && (is_nv || L->exp_gate->type == QTYPE_MXFP8));
    assert(is_nv == (L->exp_gate_scale && L->exp_up_scale && L->exp_down_scale));
    nv += is_nv, mx += !is_nv;
    const bool core = l >= g->loop.start && l < g->loop.end;
    for (int c = 0; c < INK2_CTRLS; c++)
    {
      assert(!L->ctrl_down[c] == !core && !L->ctrl_gain[c] == !core && !L->ctrl_forget[c] == !core);
      if (core)
        assert(L->ctrl_down[c]->dims[0] == 2u * (uint64_t)g->n_embd && L->ctrl_down[c]->dims[1] == 64u &&
               L->ctrl_gain[c]->dims[0] == 64u && L->ctrl_forget[c]->dims[1] == (uint64_t)g->n_embd);
    }
  }
  const uint64_t core2 = 2u * (uint64_t)(g->loop.end - g->loop.start);
  assert(w->loop_query && w->loop_kv_query && w->loop_query->dims[1] == core2 && w->loop_kv_query->dims[1] == core2);
  assert(ink2_weights_experts(w, NULL, NULL, 0) == (size_t)g->n_layers * (size_t)g->n_experts);
  printf("  experts: %d layers NVFP4 (a scale an expert), %d MXFP8; %d core layers' controls, two message tables of "
         "%llu\n",
         nv, mx, g->loop.end - g->loop.start, (unsigned long long)core2);
}

static void test_loop_refused(void)
{
  printf("  the loop refused:\n");
  const struct
  {
    const char *key, *what;
    uint64_t bad;
  } bad[] = {
      {"inkling2.num_loops", "one pass: no loop", 1},
      {"inkling2.loop.core_start", "a core not whole attn_res blocks", 5},
      {"inkling2.loop.core_end", "a core past the layers", 24},
      {"inkling2.loop.gate_rank", "no gate rank", 0},
      {"inkling2.block_count", "slots for the first pass alone", 24},
  };
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
  {
    GgufKv *k = kv(bad[i].key);
    const uint64_t was = k->v.u;
    k->v.u = bad[i].bad;
    refused(bad[i].what, bad[i].key + strlen("inkling2."));
    k->v.u = was;
  }
  assert(accepted());
}

static void test_tensors_refused(void)
{
  printf("  tensors refused:\n");
  GgufTensor *t = tensor("blk.6.loop_k_gain.weight");
  t->name[0] = 'X';
  refused("a core layer's k gain missing", "blk.6.loop_k_gain.weight: missing");
  t->name[0] = 'b';
  t = tensor("blk.17.loop_mlp_down.weight");
  t->dims[1]--;
  refused("a down of another rank", "blk.17.loop_mlp_down.weight: of another shape");
  t->dims[1]++;
  t = tensor("loop_kv_res_query.weight");
  t->name[0] = 'X';
  refused("the k and v message table missing", "loop_kv_res_query.weight: missing");
  t->name[0] = 'l';
  t = tensor("loop_res_query.weight");
  t->dims[1] -= 2;
  refused("a message table a core layer short", "loop_res_query.weight: of another shape");
  t->dims[1] += 2;
  GgufKv *end = kv("inkling2.loop.core_end"), *slots = kv("inkling2.block_count");
  GgufTensor *q = tensor("loop_res_query.weight"), *qkv = tensor("loop_kv_res_query.weight");
  end->v.u = 12, slots->v.u = 24 + 2 * 6; /* a core of layers 6-11 (its tables 12 wide): 12-17's controls past it */
  q->dims[1] = qkv->dims[1] = 12, q->bytes /= 2, qkv->bytes /= 2;
  refused("controls past the core", "blk.12.loop_attn_down.weight: a tensor no role");
  end->v.u = 18, slots->v.u = 48, q->dims[1] = qkv->dims[1] = 24, q->bytes *= 2, qkv->bytes *= 2;
  t = tensor("blk.3.ffn_gate_exps.scale");
  t->name[0] = 'X';
  refused("NVFP4's experts without their scale", "blk.3.ffn_gate_exps.scale: missing");
  t->name[0] = 'b';
  assert(accepted());
}

/* unsloth's: a value a slot, each slot's its layer's */
static void test_unsloth(void)
{
  char err[OSMOS_ERR] = "";
  Ink2Geometry g;
  assert(ink2_geometry(model, &g, err) && g.keys == INK2_KEYS_UPSTREAM);
  test_geometry(&g);
  printf("  unsloth's refused:\n");
  refused("its BF16 experts (no expert path reads BF16)", "blk.20.ffn_gate_up_exps.weight: a type its reader");
  GgufKv *k = kv("inkling2.attention.head_count");
  assert(k->type == GGUF_ARRAY && k->n == 48 && (k->item == GGUF_UINT32 || k->item == GGUF_INT32));
  int32_t *heads = k->v.items, was = heads[30];
  heads[30] = was + 1; /* slot 30: layer 12's second pass */
  refused("a slot of other heads than its layer's", "head_count: a slot's value not its layer's");
  heads[30] = was;
  heads[3] = was + 1; /* a prelude layer: no other slot */
  refused("a layer of other heads than the others'", "head_count: not the same in every layer");
  heads[3] = was;
  k = kv("inkling2.attention.sliding_window_pattern");
  assert(k->type == GGUF_ARRAY && k->item == GGUF_BOOL && k->n == 48);
  uint8_t *sw = k->v.items;
  sw[29] = !sw[29]; /* slot 29: layer 11's second pass */
  refused("a slot another kind than its layer", "sliding_window_pattern: a slot another kind");
  sw[29] = !sw[29];
  k = kv("inkling2.activation.situ_linear_beta");
  const double beta = k->v.f;
  k->v.f = 0;
  refused("no linear beta: up not softcapped", "situ_linear_beta");
  k->v.f = beta;
}

/* a fused gate and up (upstream's) by hand: an expert's gate rows the first half of its slice, its up rows the second */
static void test_fused_units(void)
{
  enum { E = 4, L = 2 };
  static Ink2Weights w;
  static GgufTensor gu[L], dn[L];
  memset(&w, 0, sizeof w);
  w.geo.n_layers = L, w.geo.n_experts = E;
  for (int l = 0; l < L; l++)
  {
    gu[l] = (GgufTensor){.file = l, .offset = 1000u * (l + 1), .bytes = E * 64};
    dn[l] = (GgufTensor){.file = l, .offset = 50000u * (l + 1), .bytes = E * 24};
    w.layer[l].exp_gate = w.layer[l].exp_up = &gu[l], w.layer[l].exp_down = &dn[l], w.layer[l].exp_fused = true;
  }
  const int fds[L] = {FD, FD + 1};
  PoolUnit u[L * E];
  assert(ink2_weights_experts(&w, fds, u, L * E) == L * E);
  for (int i = 0; i < L * E; i++)
  {
    const int l = i / E, e = i % E;
    assert(u[i].part[0].fd == fds[l] && u[i].part[0].offset == gu[l].offset + e * 64u && u[i].part[0].bytes == 32);
    assert(u[i].part[1].offset == gu[l].offset + e * 64u + 32 && u[i].part[1].bytes == 32);
    assert(u[i].part[2].offset == dn[l].offset + e * 24u && u[i].part[2].bytes == 24);
  }
  gu[1].bytes = E * 65; /* a slice of an odd number of bytes: no halves */
  assert(ink2_weights_experts(&w, fds, u, L * E) == 0);
  gu[1].bytes = E * 64 + 1; /* not E equal slices */
  assert(ink2_weights_experts(&w, fds, u, L * E) == 0);
  printf("  fused gate and up: each expert's units its slice's halves; a slice that does not halve refused\n");
}

int main(void)
{
  printf("[test_ink2_mini_weights] start\n");
  char err[OSMOS_ERR] = "";
  test_fused_units();
  if ((model = open_model("OSMOS_TEST_INK2_MINI", REF)))
  {
    printf("  the reference:\n");
    Ink2Weights *w = ink2_weights_create(model, err);
    if (!w)
      printf("  refused: %s\n", err);
    assert(w && w->geo.keys == INK2_KEYS_OURS);
    test_geometry(&w->geo);
    test_layers(w);
    ink2_weights_destroy(w);
    test_loop_refused();
    test_tensors_refused();
    gguf_destroy(model);
  }
  if ((model = open_model("OSMOS_TEST_INK2_MINI_UNSLOTH", UNSLOTH)))
  {
    printf("  unsloth's:\n");
    test_unsloth();
    gguf_destroy(model);
  }
  printf("[test_ink2_mini_weights] ok\n");
  return 0;
}
