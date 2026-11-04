#include <string.h>
#include <stddef.h>

#include "osmos/model/qtype.h"

#define QTYPE_ROW(name, id, values, bytes, paths) {id, values, bytes, paths, #name},
static const QTypeInfo rows[] = {QTYPES(QTYPE_ROW)};
#undef QTYPE_ROW

const QTypeInfo *qtype_info(uint32_t id)
{
  for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++)
    if (rows[i].id == id)
      return &rows[i];
  return NULL;
}

const QTypeInfo *qtype_named(const char *name)
{
  for (size_t i = 0; name && i < sizeof rows / sizeof rows[0]; i++)
    if (!strcmp(rows[i].name, name))
      return &rows[i];
  return NULL;
}

uint32_t qtype_paths(uint32_t id)
{
  const QTypeInfo *t = qtype_info(id);
  return t ? t->paths : 0;
}
