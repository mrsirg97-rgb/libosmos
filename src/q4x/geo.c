#include <string.h>

#include "gguf_meta.h"
#include "q4x_geo.h"
#include "osmos/gpu/q4x_ops.h"

#define ARCH "qwen4exp"
#define DIM_MAX (1 << 24)
#define CTX_MAX (1 << 30)
#define NGRAM_MIN 2
_Static_assert(Q4X_ERR == GGUF_META_ERR && Q4X_MAX_LAYERS <= GGUF_META_LIST_MAX, "the readers' buffers");

/* layer l a Gated DeltaNet unless every attn_interval-th; the sparse attention's keys pooled in blocks of one ratio, a
   DeltaNet's none, the indexer's budget whole blocks of it */
static bool kinds(GgufMeta *m, Q4xGeometry *g)
{
  if (!meta_uint_in(m, ARCH ".full_attention_interval", 1, g->n_layers, &g->attn_interval))
    return false;
  int64_t r[GGUF_META_LIST_MAX];
  size_t n = 0;
  if (!meta_ints(m, ARCH ".attention.compress_ratios", (size_t)g->n_layers, (size_t)g->n_layers, r, &n))
    return false;
  for (int l = 0; l < g->n_layers; l++)
  {
    g->recurrent[l] = (l + 1) % g->attn_interval != 0;
    if (g->recurrent[l] ? r[l] != 0 : r[l] < 2 || r[l] > DIM_MAX || (g->ix_ratio && r[l] != g->ix_ratio))
      return meta_fail(m, ARCH ".attention.compress_ratios",
                       g->recurrent[l] ? "a Gated DeltaNet layer pooled" : "an attention layer's ratio not the others'");
    if (!g->recurrent[l])
      g->ix_ratio = (int)r[l];
  }
  return g->ix_top_k % g->ix_ratio == 0 ||
         meta_fail(m, ARCH ".attention.indexer.top_k", "not whole blocks of the pooling ratio");
}

static bool rope(GgufMeta *m, Q4xGeometry *g)
{
  int64_t s[Q4X_ROPE_SECTIONS];
  size_t n = 0;
  if (!meta_ints(m, ARCH ".rope.dimension_sections", Q4X_ROPE_SECTIONS, Q4X_ROPE_SECTIONS, s, &n))
    return false;
  int64_t sum = 0;
  for (int i = 0; i < Q4X_ROPE_SECTIONS; i++)
  {
    if (s[i] < 0 || s[i] > DIM_MAX)
      return meta_fail(m, ARCH ".rope.dimension_sections", "out of range");
    g->rope_sections[i] = (int)s[i], sum += s[i];
  }
  return 2 * sum == g->rope_dim ||
         meta_fail(m, ARCH ".rope.dimension_sections", "not the rotated dims' pairs (rope.dimension_count)");
}

/* the n-gram table: one layer (a DeltaNet's: its conv's history is the recurrent state's), its heads' rows one after
   another from 0 */
static bool ple(GgufMeta *m, Q4xGeometry *g)
{
  int64_t v[GGUF_META_LIST_MAX];
  size_t n = 0;
  if (!meta_ints(m, ARCH ".ple.layers", 1, 1, v, &n))
    return false;
  if (v[0] < 0 || v[0] >= g->n_layers || !g->recurrent[v[0]])
    return meta_fail(m, ARCH ".ple.layers", "not a Gated DeltaNet layer of the model's");
  g->ple_layer = (int)v[0];
  if (!meta_uint_in(m, ARCH ".ple.ngram_size", NGRAM_MIN, Q4X_MAX_NGRAM, &g->ple_ngram) ||
      !meta_uint_in(m, ARCH ".ple.heads_per_ngram", 1, Q4X_MAX_PLE_HEADS, &g->ple_heads_per_ngram) ||
      !meta_uint_in(m, ARCH ".ple.conv_kernel", 1, Q4X_CONV_TAPS_MAX, &g->ple_conv) ||
      !meta_uint_in(m, ARCH ".embedding_length_per_layer_input", 1, DIM_MAX, &g->ple_dim) ||
      !meta_uint_in(m, ARCH ".ple.eos_token_id", 0, INT32_MAX, &g->ple_eos) ||
      !meta_uint_in(m, ARCH ".ple.image_token_id", 0, INT32_MAX, &g->ple_image))
    return false;
  g->ple_heads = (g->ple_ngram - 1) * g->ple_heads_per_ngram;
  if (g->ple_heads > Q4X_MAX_PLE_HEADS)
    return meta_fail(m, ARCH ".ple.heads_per_ngram", "more heads than this build holds");
  if ((uint64_t)g->ple_heads * (uint64_t)g->ple_dim != (uint64_t)g->n_embd) /* a token's rows fill its embedding */
    return meta_fail(m, ARCH ".embedding_length_per_layer_input", "its heads' rows not the hidden width");
  const size_t h = (size_t)g->ple_heads, ng = (size_t)g->ple_ngram;
  int64_t off[Q4X_MAX_PLE_HEADS], rows[Q4X_MAX_PLE_HEADS];
  if (!meta_ints(m, ARCH ".ple.layer_multipliers", ng, ng, g->ple_mult, &n) ||
      !meta_ints(m, ARCH ".ple.head_offsets", h, h, off, &n) || !meta_ints(m, ARCH ".ple.head_vocab_sizes", h, h, rows, &n))
    return false;
  for (size_t i = 0; i < h; i++)
  {
    if (rows[i] <= 0 || rows[i] > INT32_MAX)
      return meta_fail(m, ARCH ".ple.head_vocab_sizes", "a head's rows empty or past an int32");
    if (off[i] != (i ? off[i - 1] + rows[i - 1] : 0))
      return meta_fail(m, ARCH ".ple.head_offsets", "a head's rows not right after the one before");
    g->ple_offset[i] = (uint64_t)off[i], g->ple_rows[i] = (uint64_t)rows[i];
  }
  return true;
}

bool q4x_geometry(const Gguf *model, Q4xGeometry *g, char err[Q4X_ERR])
{
  GgufMeta m = {model, err, NULL, 0};
  memset(g, 0, sizeof *g);
  int ctx = 0;
  if (!meta_text_is(&m, "general.architecture", ARCH) ||
      !meta_uint_in(&m, ARCH ".block_count", 1, Q4X_MAX_LAYERS, &g->n_layers))
    return false;
  const bool ok =
      meta_uint_in(&m, ARCH ".embedding_length", 1, DIM_MAX, &g->n_embd) &&
      meta_uint_in(&m, ARCH ".context_length", 1, CTX_MAX, &ctx) &&
      meta_real(&m, ARCH ".attention.layer_norm_rms_epsilon", &g->rms_eps) &&
      meta_uint_in(&m, ARCH ".hyper_connection.count", 2, Q4X_MAX_HC, &g->hc) &&
      meta_uint_in(&m, ARCH ".hyper_connection.low_rank", 1, DIM_MAX, &g->hc_rank) &&
      meta_uint_in(&m, ARCH ".attention.head_count", 1, DIM_MAX, &g->n_heads) &&
      meta_uint_in(&m, ARCH ".attention.head_count_kv", 1, g->n_heads, &g->n_kv_heads) &&
      (g->n_heads % g->n_kv_heads == 0 || meta_fail(&m, ARCH ".attention.head_count_kv", "not a heads' divisor")) &&
      (g->n_heads / g->n_kv_heads <= Q4X_GQA_MAX ||
       meta_fail(&m, ARCH ".attention.head_count_kv", "more query heads a key head than the attention's tile")) &&
      meta_uint_is(&m, ARCH ".attention.key_length", Q4X_HEAD_DIM) && (g->head_dim = Q4X_HEAD_DIM) &&
      meta_uint_is(&m, ARCH ".attention.value_length", (uint64_t)g->head_dim) &&
      meta_uint_in(&m, ARCH ".rope.dimension_count", 2, g->head_dim, &g->rope_dim) && rope(&m, g) &&
      meta_real(&m, ARCH ".rope.freq_base", &g->rope_base) &&
      meta_uint_in(&m, ARCH ".attention.indexer.head_count", 1, Q4X_IX_HEADS_MAX, &g->ix_heads) &&
      meta_uint_is(&m, ARCH ".attention.indexer.key_length", Q4X_IX_DIM) && (g->ix_dim = Q4X_IX_DIM) &&
      (g->rope_dim <= g->ix_dim ||
       meta_fail(&m, ARCH ".rope.dimension_count", "wider than the indexer's head (it rotates with the attention)")) &&
      meta_uint_in(&m, ARCH ".attention.indexer.top_k", 1, DIM_MAX, &g->ix_top_k) && kinds(&m, g) &&
      meta_uint_in(&m, ARCH ".ssm.conv_kernel", 1, Q4X_CONV_TAPS_MAX, &g->ssm_conv) &&
      meta_uint_is(&m, ARCH ".ssm.state_size", Q4X_SSM_STATE) && (g->ssm_state = Q4X_SSM_STATE) &&
      meta_uint_in(&m, ARCH ".ssm.group_count", 1, DIM_MAX, &g->ssm_groups) &&
      meta_uint_in(&m, ARCH ".ssm.time_step_rank", 1, DIM_MAX, &g->ssm_v_heads) &&
      (g->ssm_v_heads % g->ssm_groups == 0 || meta_fail(&m, ARCH ".ssm.group_count", "not the value heads' divisor")) &&
      meta_uint_is(&m, ARCH ".ssm.inner_size", (uint64_t)g->ssm_v_heads * (uint64_t)g->ssm_state) &&
      (g->ssm_inner = g->ssm_v_heads * g->ssm_state) &&
      meta_uint_in(&m, ARCH ".expert_count", 1, DIM_MAX, &g->n_experts) &&
      meta_uint_in(&m, ARCH ".expert_used_count", 1, g->n_experts, &g->n_used) &&
      meta_uint_in(&m, ARCH ".expert_feed_forward_length", 1, DIM_MAX, &g->ff_exp) &&
      (g->ff_exp % Q4X_FF_TILE == 0 ||
       meta_fail(&m, ARCH ".expert_feed_forward_length", "not whole tiles of the experts' matmul")) &&
      meta_uint_in(&m, ARCH ".expert_shared_feed_forward_length", 1, DIM_MAX, &g->ff_shexp) && ple(&m, g);
  g->native_ctx = (uint32_t)ctx;
  return ok;
}
