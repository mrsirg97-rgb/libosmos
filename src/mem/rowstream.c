/* a step's table rows read while the GPU works (osmos/mem/rowstream.h) */
#include "osmos/mem/rowstream.h"

bool rowstream_issue(RowStream *s, uint64_t at, uint32_t n, uint64_t row_bytes)
{
  if (s->pending) /* a step that failed or stopped before its layer: its batch done with first */
    s->pending = false, (void)rowread_wait(s->read);
  s->at = at, s->n = n, s->row_bytes = row_bytes;
  if (!n)
    return true;
  if (!rowread_start(s->read, s->fd, s->off + at, n, (uint32_t)row_bytes, s->stage + at * row_bytes))
    return false;
  s->pending = true;
  return true;
}

bool rowstream_wait(RowStream *s)
{
  if (!s->pending)
    return true;
  s->pending = false;
  return rowread_wait(s->read);
}

const uint8_t *rowstream_rows(const RowStream *s, uint64_t at) { return s->stage + at * s->row_bytes; }
