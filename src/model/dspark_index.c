#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "substrate/model/dspark.h"

/* dspark.json's one key the index reads: "tensors": [[name, type, offset, bytes, [dims ...]], ...] - a reader of
   just that shape: strings without escapes, unsigned integers */

static bool fail(char *err, const char *why)
{
  snprintf(err, DSPARK_ERR, "dspark: %s", why);
  return false;
}

typedef struct Cur
{
  const char *p, *end;
} Cur;

static void ws(Cur *c)
{
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\n' || *c->p == '\r' || *c->p == '\t'))
    c->p++;
}

static bool eat(Cur *c, char ch)
{
  ws(c);
  if (c->p >= c->end || *c->p != ch)
    return false;
  c->p++;
  return true;
}

static bool string(Cur *c, char *out, size_t cap)
{
  if (!eat(c, '"'))
    return false;
  size_t n = 0;
  while (c->p < c->end && *c->p != '"')
  {
    if (*c->p == '\\' || n + 1 >= cap)
      return false;
    out[n++] = *c->p++;
  }
  out[n] = 0;
  return eat(c, '"');
}

static bool number(Cur *c, uint64_t *v)
{
  ws(c);
  if (c->p >= c->end || *c->p < '0' || *c->p > '9')
    return false;
  *v = 0;
  while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
    *v = *v * 10 + (uint64_t)(*c->p++ - '0');
  return true;
}

/* a type's id and its block: values a block, bytes a block */
static bool type_of(const char *s, uint32_t *id, uint64_t *values, uint64_t *bytes)
{
  static const struct
  {
    const char *name;
    uint32_t id;
    uint64_t values, bytes;
  } k[] = {{"F32", 0, 1, 4}, {"BF16", 30, 1, 2}, {"Q8_0", 8, 32, 34}, {"MXFP4", 39, 32, 17}};
  for (size_t i = 0; i < sizeof k / sizeof *k; i++)
    if (!strcmp(s, k[i].name))
      return *id = k[i].id, *values = k[i].values, *bytes = k[i].bytes, true;
  return false;
}

static bool entry(Cur *c, DsparkTensor *t, uint64_t file_bytes, char *err)
{
  char type[16];
  uint64_t values = 1, per_block, block_bytes;
  if (!eat(c, '[') || !string(c, t->name, sizeof t->name) || !eat(c, ',') || !string(c, type, sizeof type) ||
      !eat(c, ',') || !number(c, &t->offset) || !eat(c, ',') || !number(c, &t->bytes) || !eat(c, ',') ||
      !eat(c, '['))
    return fail(err, "dspark.json: a tensor entry it cannot read");
  if (!type_of(type, &t->type, &per_block, &block_bytes))
    return fail(err, "dspark.json: a tensor of a type it does not know");
  t->n_dims = 0;
  do
  {
    if (t->n_dims == DSPARK_MAX_DIMS || !number(c, &t->dims[t->n_dims]))
      return fail(err, "dspark.json: a tensor's dims it cannot read");
    values *= t->dims[t->n_dims++];
  } while (eat(c, ','));
  if (!eat(c, ']') || !eat(c, ']'))
    return fail(err, "dspark.json: a tensor entry it cannot read");
  if (values % per_block || values / per_block * block_bytes != t->bytes)
    return fail(err, "dspark.json: a tensor's bytes are not its type's for its dims");
  if (t->offset > file_bytes || t->bytes > file_bytes - t->offset)
    return fail(err, "dspark.json: a tensor past the end of dspark.bin");
  return true;
}

DsparkIndex *dspark_index(const char *dir, char err[DSPARK_ERR])
{
  char path[4096];
  struct stat st;
  DsparkIndex *x = calloc(1, sizeof *x);
  char *text = NULL;
  bool ok = x != NULL || fail(err, "out of memory");
  if (ok)
  {
    snprintf(path, sizeof path, "%s/dspark.bin", dir);
    ok = (stat(path, &st) == 0 && (x->bin = strdup(path))) || fail(err, "no dspark.bin");
  }
  FILE *f = NULL;
  long len = 0;
  if (ok)
  {
    snprintf(path, sizeof path, "%s/dspark.json", dir);
    f = fopen(path, "rb");
    ok = (f && fseek(f, 0, SEEK_END) == 0 && (len = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
          (text = malloc((size_t)len + 1)) && fread(text, 1, (size_t)len, f) == (size_t)len) ||
         fail(err, "dspark.json could not be read");
  }
  if (f)
    fclose(f);
  if (ok)
    text[len] = 0;
  const char *key = ok ? strstr(text, "\"tensors\"") : NULL;
  if (ok && !key)
    ok = fail(err, "dspark.json has no tensors");
  Cur c = {key ? key + strlen("\"tensors\"") : NULL, text + len};
  uint32_t cap = 0;
  if (ok && (!eat(&c, ':') || !eat(&c, '[')))
    ok = fail(err, "dspark.json: tensors is not a list");
  while (ok && !eat(&c, ']'))
  {
    if (x->n && !eat(&c, ','))
    {
      ok = fail(err, "dspark.json: tensors is not a list");
      break;
    }
    if (x->n == cap)
    {
      DsparkTensor *grown = realloc(x->t, (cap = cap ? cap * 2 : 256) * sizeof *grown);
      if (!grown)
      {
        ok = fail(err, "out of memory");
        break;
      }
      x->t = grown;
    }
    ok = entry(&c, &x->t[x->n], (uint64_t)st.st_size, err);
    x->n += ok;
  }
  free(text);
  if (!ok)
  {
    dspark_index_destroy(x);
    return NULL;
  }
  return x;
}

void dspark_index_destroy(DsparkIndex *x)
{
  if (!x)
    return;
  free(x->bin);
  free(x->t);
  free(x);
}

const DsparkTensor *dspark_tensor(const DsparkIndex *x, const char *name)
{
  for (uint32_t i = 0; x && i < x->n; i++)
    if (!strcmp(x->t[i].name, name))
      return &x->t[i];
  return NULL;
}
