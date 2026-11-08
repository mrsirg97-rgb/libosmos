// a state streamed in pieces (osmos/model/state_io.h) for the engine tests: a sink that refuses any piece past
// STATE_CHUNK and keeps the rest in order (or fails after `fail_after` bytes), a source that gives a buffer back in the
// sizes the loader asks for (refusing any past STATE_CHUNK), one byte of it flipped if asked
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/model/state_io.h"
}

struct StreamSink {
  std::vector<uint8_t> bytes;
  uint64_t pieces = 0, widest = 0, fail_after = UINT64_MAX;
  bool over = false;  // a piece past STATE_CHUNK seen
};

inline bool stream_sink(void* ctx, const void* p, uint64_t n) {
  StreamSink* s = (StreamSink*) ctx;
  if (n > STATE_CHUNK) s->over = true;
  if (n > STATE_CHUNK || s->bytes.size() + n > s->fail_after) return false;
  s->bytes.insert(s->bytes.end(), (const uint8_t*) p, (const uint8_t*) p + n);
  s->pieces++, s->widest = n > s->widest ? n : s->widest;
  return true;
}

struct StreamSource {
  const std::vector<uint8_t>* bytes;
  uint64_t at = 0, flip = UINT64_MAX, widest = 0;
  bool over = false;
};

inline bool stream_source(void* ctx, void* p, uint64_t n) {
  StreamSource* s = (StreamSource*) ctx;
  if (n > STATE_CHUNK) s->over = true;
  if (n > STATE_CHUNK || s->at + n > s->bytes->size()) return false;
  std::memcpy(p, s->bytes->data() + s->at, n);
  if (s->flip >= s->at && s->flip < s->at + n) ((uint8_t*) p)[s->flip - s->at] ^= 1;
  s->at += n, s->widest = n > s->widest ? n : s->widest;
  return true;
}

// a family's streamed save / load over one buffer (as the engine's oseq_save / oseq_load): a buffer not the state's
// size refused
struct StreamMem {
  uint8_t* p;
  uint64_t at, n;
};
inline bool mem_sink(void* ctx, const void* b, uint64_t n) {
  StreamMem* m = (StreamMem*) ctx;
  if (n > m->n - m->at) return false;
  std::memcpy(m->p + m->at, b, n), m->at += n;
  return true;
}
inline bool mem_source(void* ctx, void* b, uint64_t n) {
  StreamMem* m = (StreamMem*) ctx;
  if (n > m->n - m->at) return false;
  std::memcpy(b, m->p + m->at, n), m->at += n;
  return true;
}
template <class SaveTo, class Seq>
bool save_buf(SaveTo save_to, const Seq* s, uint8_t* out, uint64_t n, char* err) {
  StreamMem m{out, 0, n};
  return save_to(s, mem_sink, &m, err) && m.at == n;
}
template <class LoadFrom, class Seq>
bool load_buf(LoadFrom load_from, Seq* s, const uint8_t* in, uint64_t n, char* err) {
  StreamMem m{(uint8_t*) in, 0, n};
  return load_from(s, n, mem_source, &m, err);
}
