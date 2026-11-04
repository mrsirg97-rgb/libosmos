#pragma once
#include <stdbool.h>

#include "api.h"

/* a chat request as the encoding takes it (request.c) */
typedef struct Request
{
  Json *messages; /* the encoding's messages (owned) */
  ChatMode mode;
  int effort;
  unsigned max_tokens; /* 0: as many as the context leaves */
  bool stream, usage;
} Request;

bool request_read(const Json *body, Request *r, char err[API_ERR]);
