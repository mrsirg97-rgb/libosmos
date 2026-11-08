/* Qwen3.8-Flash-Next's weights (osmos/q4x/weights.h) on unsloth's UD-IQ3_XXS: the geometry its metadata says, each
   layer's kind its tensors, then the file edited in memory - a tensor missing, of another shape or type, a key out of
   range or of another layout - each refused, its cause named. needs ~/Models/qwen38/UD-IQ3_XXS; skipped without */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/model/gguf.h"
#include "osmos/q4x/weights.h"

#define T_Q8_0 8
#define T_BF16 30
#define QWEN "%s/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf"
#define QWEN_MTP "%s/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"

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

/* an integer array's entry i, whatever its width: set to v, the old value back */
static int64_t set_item(GgufKv *k, size_t i, int64_t v)
{
  int64_t old = 0;
  switch (k->item)
  {
  case GGUF_UINT32:
  case GGUF_INT32: old = ((int32_t *)k->v.items)[i], ((int32_t *)k->v.items)[i] = (int32_t)v; break;
  case GGUF_UINT64:
  case GGUF_INT64: old = ((int64_t *)k->v.items)[i], ((int64_t *)k->v.items)[i] = v; break;
  default: assert(!"an integer array of another width");
  }
  return old;
}

static bool accepted(void)
{
  char err[Q4X_ERR] = "";
  Q4xWeights *w = q4x_weights_create(model, NULL, err);
  if (!w)
    printf("    refused: %s\n", err);
  q4x_weights_destroy(w);
  return w != NULL;
}

static void refused(const char *what, const char *needle)
{
  char err[Q4X_ERR] = "";
  Q4xWeights *w = q4x_weights_create(model, NULL, err);
  if (w || !strstr(err, needle))
  {
    printf("  %s: %s (err \"%s\", wanted \"%s\")\n", what, w ? "ACCEPTED" : "wrong reason", err, needle);
    exit(1);
  }
  printf("    %-46s -> %s\n", what, err);
}

static void test_real(void)
{
  char err[Q4X_ERR] = "";
  Q4xWeights *w = q4x_weights_create(model, NULL, err);
  if (!w)
    printf("  real: refused: %s\n", err), exit(1);
  const Q4xGeometry *g = &w->geo;
  assert(g->n_layers == 48 && g->n_embd == 2560 && g->n_vocab == 248320 && g->native_ctx == 262144);
  assert(g->hc == 4 && g->hc_rank == 320 && g->attn_interval == 4);
  int recurrent = 0;
  for (int l = 0; l < g->n_layers; l++)
    recurrent += g->recurrent[l], assert(g->recurrent[l] == ((l + 1) % 4 != 0));
  assert(recurrent == 36);
  assert(g->n_heads == 24 && g->n_kv_heads == 2 && g->head_dim == 256 && g->rope_dim == 64);
  assert(g->rope_sections[0] == 11 && g->rope_sections[1] == 11 && g->rope_sections[2] == 10 && g->rope_sections[3] == 0);
  assert(g->rope_base == 1e7f && g->rms_eps > 9.9e-7f && g->rms_eps < 1.01e-6f);
  assert(g->ix_heads == 4 && g->ix_dim == 128 && g->ix_top_k == 2048 && g->ix_ratio == 4);
  assert(g->ssm_conv == 4 && g->ssm_state == 128 && g->ssm_groups == 16 && g->ssm_v_heads == 48 &&
         g->ssm_inner == 6144);
  assert(g->n_experts == 512 && g->n_used == 10 && g->ff_exp == 640 && g->ff_shexp == 640);
  assert(g->ple_layer == 1 && g->ple_ngram == 3 && g->ple_heads_per_ngram == 8 && g->ple_heads == 16 &&
         g->ple_dim == 160 && g->ple_conv == 4 && g->ple_eos == 248044 && g->ple_image == 248056);
  assert(g->ple_mult[0] == 23703573157769 && g->ple_mult[2] == 8052911324071);
  assert(g->ple_offset[0] == 0 && g->ple_rows[0] == 20000003 && g->ple_offset[15] == 300001275 &&
         g->ple_rows[15] == 20000171);
  const Q4xLayer *d = &w->layer[0], *a = &w->layer[3], *p = &w->layer[1];
  assert(d->qkv && d->gate && d->conv && d->ssm_out && !d->q && !d->ix_q && !d->ple_key);
  assert(a->q && a->k && a->v && a->out && a->ix_q && a->ix_k && !a->qkv && !a->conv);
  assert(p->ple_key && p->ple_conv && w->ple_table && w->out_hc_up && w->embed && w->output);
  for (int l = 0; l < g->n_layers; l++)
    assert(w->layer[l].router && w->layer[l].exp_down && w->layer[l].sh_gate_inp && w->layer[l].hc_ffn_inject);
  printf("  real: 48 layers (36 Gated DeltaNet, 12 sparse attention), every tensor in its role\n");
  q4x_weights_destroy(w);
}

static void test_tensors(void)
{
  printf("  tensors refused:\n");
  GgufTensor *t = tensor("blk.5.attn_qkv.weight");
  t->name[0] = 'X';
  refused("a DeltaNet's qkv missing", "blk.5.attn_qkv");
  t->name[0] = 'b';
  t->dims[1]--;
  refused("qkv one row short", "blk.5.attn_qkv");
  t->dims[1]++;
  t->n_dims = 3;
  refused("qkv of 3 dims", "blk.5.attn_qkv");
  t->n_dims = 2;

  t = tensor("blk.7.attn_q.weight");
  t->name[0] = 'X';
  refused("an attention layer's q missing", "blk.7.attn_q");
  t->name[0] = 'b';

  t = tensor("blk.5.ssm_norm.weight");
  t->type = T_BF16;
  refused("a norm in BF16", "blk.5.ssm_norm");
  t->type = 0;

  t = tensor("blk.9.ffn_up_exps.weight");
  const uint32_t ty = t->type;
  t->type = T_Q8_0;
  refused("experts of a type their bytes are not", "blk.9.ffn_up_exps");
  t->type = ty;

  t = tensor("per_layer_token_embd.weight");
  {
    const uint32_t was = t->type;
    t->type = T_BF16; /* the decode path's, not q4x_rows' (IQ4_NL, Q8_0, Q6_K) */
    refused("the n-gram table in a type its rows' kernel does not decode", "per_layer_token_embd.weight: a type its reader");
    t->type = was;
  }
  const uint64_t rows = t->dims[1];
  t->dims[1] = 300000000; /* fewer than the heads' ranges reach */
  refused("an n-gram table too short", "per_layer_token_embd");
  t->dims[1] = rows;

  t = tensor("output_hc_up.weight");
  t->name[0] = 'X';
  refused("the head's hc_up missing", "output_hc_up");
  t->name[0] = 'o';
  assert(accepted());
}

static void test_kv(void)
{
  printf("  metadata refused:\n");
  GgufKv *k = kv("general.architecture");
  k->v.s[0] = 'X';
  refused("another architecture", "general.architecture");
  k->v.s[0] = 'q';

  k = kv("qwen4exp.block_count");
  k->v.u = 65;
  refused("65 layers", "block_count");
  k->v.u = 48;

  k = kv("qwen4exp.full_attention_interval");
  k->v.u = 2; /* layer 1 attention by the interval, a DeltaNet by the ratios: the metadata against itself */
  refused("another layer pattern", "compress_ratios");
  k->v.u = 4;

  k = kv("qwen4exp.hyper_connection.count");
  k->v.u = 1;
  refused("one stream: nothing to mix", "hyper_connection.count");
  k->v.u = 4;

  k = kv("qwen4exp.expert_used_count");
  k->v.u = 600;
  refused("more experts used than there are", "expert_used_count");
  k->v.u = 10;

  k = kv("qwen4exp.attention.compress_ratios");
  int64_t old = set_item(k, 0, 4);
  refused("a DeltaNet layer compressed", "compress_ratios");
  set_item(k, 0, old);
  old = set_item(k, 3, 2);
  refused("two pooling ratios", "compress_ratios");
  set_item(k, 3, old);

  k = kv("qwen4exp.ple.head_offsets");
  old = set_item(k, 1, 20000004);
  refused("a head's rows not after the one before", "head_offsets");
  set_item(k, 1, old);

  k = kv("qwen4exp.ple.layers");
  old = set_item(k, 0, 3);
  refused("the n-gram table on an attention layer", "ple.layers");
  set_item(k, 0, old);

  /* the kernels' fixed shapes (q4x_ops.h's), refused at load - not a step's "a GPU call failed" */
  const struct { const char *key, *what; uint64_t bad; } fixed[] = {
      {"qwen4exp.attention.key_length", "a head's dim not the attention's 256", 128},
      {"qwen4exp.ssm.state_size", "a DeltaNet state not the kernels' 128", 64},
      {"qwen4exp.attention.indexer.key_length", "an indexer head not its 128", 64},
      {"qwen4exp.attention.indexer.head_count", "indexer heads past the score tile's 4", 8},
      {"qwen4exp.attention.head_count_kv", "query heads a key head past 16", 1},
      {"qwen4exp.ssm.conv_kernel", "a conv's taps past 8", 9},
      {"qwen4exp.ple.conv_kernel", "the n-gram conv's taps past 8", 9},
      {"qwen4exp.expert_feed_forward_length", "an expert's width not whole tiles", 648},
      {"qwen4exp.embedding_length_per_layer_input", "the n-gram heads not the hidden width", 161},
  };
  for (size_t i = 0; i < sizeof fixed / sizeof fixed[0]; i++)
  {
    k = kv(fixed[i].key);
    const uint64_t was = k->v.u;
    k->v.u = fixed[i].bad;
    refused(fixed[i].what, fixed[i].key + strlen("qwen4exp."));
    k->v.u = was;
  }
  /* the rotation past the indexer's 128 (it rotates with the attention's): 160, its sections summing 80 */
  GgufKv *rd = kv("qwen4exp.rope.dimension_count"), *rs = kv("qwen4exp.rope.dimension_sections");
  const uint64_t rd0 = rd->v.u;
  int64_t s0 = 0;
  for (size_t i = 0; i < 4; i++)
    s0 += ((int32_t *)rs->v.items)[i];
  rd->v.u = 160;
  old = set_item(rs, 3, 80 - s0 + ((int32_t *)rs->v.items)[3]);
  refused("the rotation wider than the indexer's head", "rope.dimension_count");
  set_item(rs, 3, old), rd->v.u = rd0;
  assert(accepted());
}

/* the MTP head: taken whole against the model's geometry; the model's own file, a head of other widths or layers, a
   self-contained head (its own embedding), a tensor missing - each refused */
static void test_mtp(const Q4xWeights *w, const char *path)
{
  char err[Q4X_ERR] = "", gerr[GGUF_ERR] = "";
  Gguf *mtp = gguf_header(path, gerr);
  if (!mtp)
  {
    printf("  MTP: no head file, skipped\n");
    return;
  }
  Q4xMtp *m = q4x_mtp_create(mtp, w, err);
  if (!m)
    printf("  MTP refused: %s\n", err), exit(1);
  assert(m->layer.q && m->layer.exp_down && m->layer.hc_attn_inject && !m->layer.qkv && m->eh_proj && m->head_up);
  assert(m->eh_proj->dims[0] == 2u * (uint64_t)w->geo.n_embd);
  q4x_mtp_destroy(m);
  printf("  MTP: blk.%d and the nextn tensors taken\n", w->geo.n_layers);
  assert(!q4x_mtp_create(model, w, err) && strstr(err, "nextn"));
  printf("    the model's own file as a head -> %s\n", err);
  struct
  {
    const char *key, *what, *needle;
    uint64_t bad;
  } keys[] = {{"qwen4exp.embedding_length", "another width", "embedding_length", 2048},
              {"qwen4exp.block_count", "its layer not after the model's", "block_count", 50},
              {"qwen4exp.expert_count", "other experts", "expert_count", 384}};
  for (size_t i = 0; i < sizeof keys / sizeof *keys; i++)
  {
    GgufKv *k = (GgufKv *)gguf_kv(mtp, keys[i].key);
    assert(k);
    const uint64_t old = k->v.u;
    k->v.u = keys[i].bad;
    assert(!q4x_mtp_create(mtp, w, err) && strstr(err, keys[i].needle));
    printf("    %-36s -> %s\n", keys[i].what, err);
    k->v.u = old;
  }
  GgufKv *shared = (GgufKv *)gguf_kv(mtp, "qwen4exp.nextn_shared_target_tensors");
  assert(shared);
  shared->v.u = 0;
  assert(!q4x_mtp_create(mtp, w, err) && strstr(err, "shared"));
  printf("    %-36s -> %s\n", "a self-contained head", err);
  shared->v.u = 1;
  GgufTensor *t = (GgufTensor *)gguf_tensor(mtp, "blk.48.nextn.eh_proj.weight");
  assert(t);
  t->name[0] = 'X';
  assert(!q4x_mtp_create(mtp, w, err) && strstr(err, "eh_proj"));
  printf("    %-36s -> %s\n", "its eh_proj missing", err);
  t->name[0] = 'b';
  assert((m = q4x_mtp_create(mtp, w, err)));
  q4x_mtp_destroy(m);
  gguf_destroy(mtp);
}

/* an overlay (huihui's abliteration, tools/install/overlay_q4x.py's): its 96 tensors stand in for the model's of their
   names, the rest the model's; refused - an overlay tensor no role reads, one of another shape, one the model has no
   tensor of */
static void test_overlay(const char *path)
{
  char err[Q4X_ERR] = "", gerr[GGUF_ERR] = "";
  Gguf *ov = gguf_create(path, gerr);
  if (!ov)
  {
    printf("  overlay: not on this machine, skipped (%s)\n", gerr);
    return;
  }
  Q4xWeights *w = q4x_weights_create(model, ov, err);
  if (!w)
    printf("  overlay refused: %s\n", err);
  assert(w && w->overlay == ov);
  size_t from_ov = 0;
  for (int l = 0; l < w->geo.n_layers; l++)
  {
    const Q4xLayer *L = &w->layer[l];
    const GgufTensor *mix = w->geo.recurrent[l] ? L->ssm_out : L->out;
    const bool in_ov = mix >= ov->tensors && mix < ov->tensors + ov->n_tensors;
    const bool sh_ov = L->sh_down >= ov->tensors && L->sh_down < ov->tensors + ov->n_tensors;
    assert(in_ov && sh_ov), from_ov += 2;
    assert(L->sh_up >= model->tensors && L->sh_up < model->tensors + model->n_tensors); /* not patched */
  }
  assert(from_ov == ov->n_tensors);
  q4x_weights_destroy(w);
  printf("  overlay: %zu tensors the overlay's, the rest the model's\n", from_ov);
  GgufTensor *t = (GgufTensor *)gguf_tensor(ov, "blk.0.ssm_out.weight");
  assert(t);
  t->name[0] = 'X';
  w = q4x_weights_create(model, ov, err);
  assert(!w && strstr(err, "Xlk.0.ssm_out.weight")), printf("    an overlay tensor no role reads -> %s\n", err);
  t->name[0] = 'b';
  const uint64_t d0 = t->dims[0];
  t->dims[0] = d0 / 2;
  w = q4x_weights_create(model, ov, err);
  assert(!w && strstr(err, "blk.0.ssm_out.weight")), printf("    an overlay tensor of another shape -> %s\n", err);
  t->dims[0] = d0;
  GgufTensor *m = (GgufTensor *)gguf_tensor(model, "blk.0.ssm_out.weight");
  m->name[0] = 'X';
  w = q4x_weights_create(model, ov, err);
  assert(!w), printf("    an overlay tensor the model has none of -> %s\n", err);
  m->name[0] = 'b';
  assert((w = q4x_weights_create(model, ov, err)));
  q4x_weights_destroy(w);
  gguf_destroy(ov);
}

int main(void)
{
  printf("[test_q4x_weights] start\n");
  char err[Q4X_ERR] = "", path[512], gerr[GGUF_ERR] = "";
  assert(!q4x_weights_create(NULL, NULL, err) && err[0]);
  q4x_weights_destroy(NULL);
  snprintf(path, sizeof path, QWEN, getenv("HOME") ? getenv("HOME") : "");
  if (!(model = gguf_create(path, gerr)))
  {
    printf("  Qwen3.8-Flash-Next UD-IQ3_XXS: not on this machine, skipped (%s)\n", gerr);
    printf("[test_q4x_weights] ok\n");
    return 0;
  }
  test_real();
  test_tensors();
  test_kv();
  Q4xWeights *w = q4x_weights_create(model, NULL, err);
  assert(w);
  snprintf(path, sizeof path, QWEN_MTP, getenv("HOME") ? getenv("HOME") : "");
  test_mtp(w, path);
  q4x_weights_destroy(w);
  snprintf(path, sizeof path, "%s/Models/qwen38/huihui-patch/qwen38-ud-iq3_xxs-abliterated-overlay.gguf",
           getenv("HOME") ? getenv("HOME") : "");
  test_overlay(path);
  gguf_destroy(model);
  printf("[test_q4x_weights] ok\n");
  return 0;
}
