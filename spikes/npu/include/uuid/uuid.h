/* the spike's stand-in for uuid-dev's header (libuuid.so.1 is installed, its header is not): the six calls XRT's
   headers make, libuuid's own declarations */
#ifndef SPIKE_UUID_H
#define SPIKE_UUID_H
#ifdef __cplusplus
extern "C" {
#endif
typedef unsigned char uuid_t[16];
void uuid_clear(uuid_t uu);
int uuid_compare(const uuid_t uu1, const uuid_t uu2);
void uuid_copy(uuid_t dst, const uuid_t src);
int uuid_is_null(const uuid_t uu);
int uuid_parse(const char *in, uuid_t uu);
void uuid_unparse_lower(const uuid_t uu, char *out);
#ifdef __cplusplus
}
#endif
#endif
