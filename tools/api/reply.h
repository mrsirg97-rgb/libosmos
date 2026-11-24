#pragma once
/* POST /v1/chat/completions, its two halves: chat.c the request - its messages copied with
   their special tokens defused, its prompt's ids, its mark; reply.c the answer - its tokens streamed as they land or
   answered whole, its finish, usage and timings */
#include <stdbool.h>
#include <stddef.h>

#include "answer.h"
#include "api.h"

/* reply.c's, for chat.c */
const char *reply_boot_tag(void);
void reply_fail(const Http *h, bool stream, const char *why, const JsonBuf *completion, const char *eos);
void reply_fail_at(const Http *h, int status, bool stream, const char *why, const JsonBuf *completion, const char *eos);
void reply_loop_head(char *out, size_t cap, const Answer *s);
Json *reply_message(const Answer *s, const Request *rq, long long rid, bool *parsed, char err[OSMOS_ERR]);
bool reply_more(Api *a, Run *r, size_t *seen, Answer *s);
void reply_stream_end(Answer *s, const Request *rq, const Run *r, const Json *m, size_t prompt_n);
void reply_whole(Answer *s, const Run *r, const Json *m, size_t prompt_n);

#define WAIT_MS 500 /* how often a waiting request looks whether its client went away */
/* a stream silent this long gets an SSE comment: a long prompt's read (minutes past 100K tokens) not taken by a
   client's idle timeout for a dead server (its retry would read the prompt from the start again) */
#define KEEPALIVE_MS 15000
