/* osmos/gpu/state_parts.h: a state's parts moved between the device and a sink or source, its sum */
#include <hip/hip_runtime_api.h>
#include <stdlib.h>

#include "osmos/gpu/state_parts.h"
#include "osmos/model/fnv.h"

uint64_t state_parts_bytes(uint64_t head_bytes, const StatePart *p, int n)
{
  uint64_t b = head_bytes + sizeof(uint64_t);
  for (int i = 0; i < n; i++)
    b += p[i].bytes;
  return b;
}

/* a part between the device and a sink or source in pieces through bounce (STATE_CHUNK bytes), every byte moved
   into the sum in order */
static bool move(const StatePart *p, uint8_t *bounce, StateSink sink, StateSource src, void *ctx, uint64_t *sum)
{
  for (uint64_t done = 0; done < p->bytes;)
  {
    const uint64_t n = p->bytes - done < STATE_CHUNK ? p->bytes - done : STATE_CHUNK;
    uint8_t *dev = (uint8_t *)p->dev + done;
    if (sink)
    {
      if (hipMemcpy(bounce, dev, n, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
      *sum = fnv(bounce, n, *sum);
      if (!sink(ctx, bounce, n))
        return false;
    }
    else
    {
      if (!src(ctx, bounce, n))
        return false;
      *sum = fnv(bounce, n, *sum);
      if (hipMemcpy(dev, bounce, n, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    }
    done += n;
  }
  return true;
}

const char *state_parts_save(const void *head, uint64_t head_bytes, const StatePart *p, int n, StateSink sink,
                             void *ctx)
{
  if (hipDeviceSynchronize() != hipSuccess) /* every stream's writes of the state done */
    return "a GPU call failed (before the state's save)";
  uint8_t *bounce = malloc(STATE_CHUNK);
  if (!bounce)
    return "out of memory (the save)";
  uint64_t sum = fnv(head, head_bytes, FNV_SEED);
  bool ok = sink(ctx, head, head_bytes);
  for (int i = 0; ok && i < n; i++)
    ok = move(&p[i], bounce, sink, NULL, ctx, &sum);
  ok = ok && sink(ctx, &sum, sizeof sum);
  free(bounce);
  return ok ? NULL : "a GPU copy or the sink failed (the state's save)";
}

const char *state_parts_load(const void *head, uint64_t head_bytes, const StatePart *p, int n, StateSource src,
                             void *ctx)
{
  if (hipDeviceSynchronize() != hipSuccess) /* the sequence's state not still being read */
    return "a GPU call failed (before the state's load)";
  uint8_t *bounce = malloc(STATE_CHUNK);
  if (!bounce)
    return "out of memory (the state's load)";
  uint64_t got = fnv(head, head_bytes, FNV_SEED), sum = 0;
  bool ok = true;
  for (int i = 0; ok && i < n; i++)
    ok = move(&p[i], bounce, NULL, src, ctx, &got);
  free(bounce);
  if (!ok || !src(ctx, &sum, sizeof sum))
    return "a GPU copy or the source failed (the state's load)";
  return got == sum ? NULL : "a damaged state (its checksum)";
}

const char *state_head_check(uint64_t magic, uint64_t want_magic, uint32_t version, uint32_t want_version,
                             uint64_t identity, uint64_t want_identity)
{
  if (magic != want_magic || version != want_version)
    return "a state of another format, or another version's";
  return identity == want_identity ? NULL : "a state of another model or build (its identity)";
}

const char *state_pos_check(int64_t pos, uint32_t max_ctx)
{
  return pos >= 0 && pos <= (int64_t)max_ctx ? NULL : "a state past this context";
}

const char *state_tokens_check(const int32_t *t, int64_t n, uint32_t vocab)
{
  for (int64_t i = 0; i < n; i++)
    if (t[i] < 0 || (uint32_t)t[i] >= vocab)
      return "a state whose last tokens are outside the vocabulary";
  return NULL;
}

const char *state_parts_to_host(const StatePart *p, int n, uint8_t *out)
{
  for (int i = 0; i < n; out += p[i].bytes, i++)
    if (p[i].bytes && hipMemcpy(out, p[i].dev, p[i].bytes, hipMemcpyDeviceToHost) != hipSuccess)
      return "a GPU copy failed (the mark's save)";
  return NULL;
}

const char *state_parts_from_host(const StatePart *p, int n, const uint8_t *in)
{
  for (int i = 0; i < n; in += p[i].bytes, i++)
    if (p[i].bytes && hipMemcpy(p[i].dev, in, p[i].bytes, hipMemcpyHostToDevice) != hipSuccess)
      return "a GPU copy failed (the mark's load)";
  return NULL;
}
