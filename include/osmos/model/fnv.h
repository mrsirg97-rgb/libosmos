#ifndef OSMOS_MODEL_FNV_H
#define OSMOS_MODEL_FNV_H

#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

/* 64-bit FNV-1a: a saved state's checksum, an identity's key (the families' seq states) */
#define FNV_SEED 1469598103934665603ull
#define FNV_PRIME 1099511628211ull

static inline uint64_t fnv(const void *p, uint64_t n, uint64_t h)
{
  const uint8_t *b = (const uint8_t *)p;
  for (uint64_t i = 0; i < n; i++)
    h = (h ^ b[i]) * FNV_PRIME;
  return h;
}

static inline uint64_t fnv_u64(uint64_t v, uint64_t h) { return fnv(&v, sizeof v, h); }

/* a weights file in an identity (a kept conversation's, slots.h): its name, size, and which file it is now - device,
   inode, modification time to the nanosecond (a link's target's) - so a file rewritten in place at the same size never
   passes for the one a state was read with. a file moved or copied (cp without -p) is another: its kept states are
   read again once, never another file's bits */
static inline uint64_t fnv_file(const char *path, uint64_t h)
{
  const char *base = strrchr(path, '/');
  base = base ? base + 1 : path;
  h = fnv(base, strlen(base), h);
  struct stat st;
  if (stat(path, &st) != 0)
    return fnv_u64(0, h);
  const uint64_t id[] = {(uint64_t)st.st_size, (uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_mtim.tv_sec,
                         (uint64_t)st.st_mtim.tv_nsec};
  return fnv(id, sizeof id, h);
}

#endif
