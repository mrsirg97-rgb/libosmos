/* requant_ink2_mtp: Inkling-2-Small's drafter (ink2_mtp_gguf.py's BF16 GGUF) with its layers' matrices quantized by
   ggml's reference quantizers (llama.cpp, MIT: libggml-base; no imatrix), the rest copied as it is.

     requant_ink2_mtp <bf16 gguf> <out gguf> <mlp type> <attention type>

   types by ggml's names (q8_0, q6_K, q5_K, iq4_nl). osmos-evo-inkling's drafter: q5_K q5_K (1.76 -> 1.18 GiB against
   q8_0 q8_0, its drafts the same; q8_0 q8_0 is ink2_mtp_gguf.py's Q8_0 byte for byte). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "gguf.h"

static enum ggml_type by_name(const char *s) {
  for (int t = 0; t < GGML_TYPE_COUNT; t++)
    if (ggml_type_name(t) && !strcmp(ggml_type_name(t), s)) return t;
  fprintf(stderr, "no type %s\n", s), exit(2);
}

int main(int argc, char **argv) {
  if (argc != 5) return fprintf(stderr, "mtpq <in> <out> <mlp type> <attn type>\n"), 2;
  const enum ggml_type tm = by_name(argv[3]), ta = by_name(argv[4]);
  struct ggml_context *ctx = NULL;
  struct gguf_init_params ip = {false, &ctx};
  struct gguf_context *in = gguf_init_from_file(argv[1], ip);
  if (!in) return fprintf(stderr, "read %s\n", argv[1]), 1;
  struct gguf_context *out = gguf_init_empty();
  gguf_set_kv(out, in);
  struct ggml_init_params cp = {(size_t)8 << 30, NULL, false};
  struct ggml_context *oc = ggml_init(cp);
  for (int64_t i = 0; i < gguf_get_n_tensors(in); i++) {
    const char *name = gguf_get_tensor_name(in, i);
    struct ggml_tensor *t = ggml_get_tensor(ctx, name);
    const int mlp = strstr(name, ".mlp.") != NULL, attn = strstr(name, ".self_attn.") && strstr(name, "_proj");
    if (t->type == GGML_TYPE_BF16 && (mlp || attn)) {
      const enum ggml_type q = mlp ? tm : ta;
      const int64_t n = ggml_nelements(t), cols = t->ne[0], rows = n / cols;
      float *f = malloc(n * 4);
      ggml_bf16_to_fp32_row(t->data, f, n);
      struct ggml_tensor *o = ggml_new_tensor_2d(oc, q, t->ne[0], t->ne[1]);
      ggml_set_name(o, name);
      ggml_quantize_chunk(q, f, o->data, 0, rows, cols, NULL);
      free(f);
      gguf_add_tensor(out, o);
    } else {
      gguf_add_tensor(out, t);
    }
  }
  if (!gguf_write_to_file(out, argv[2], false)) return fprintf(stderr, "write %s\n", argv[2]), 1;
  return 0;
}
