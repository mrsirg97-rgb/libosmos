// q8_experts (include/osmos/gpu/q8.h): a family's routed experts through the int8 lane at a decode - x rounded, gate
// and up and the activation, h rounded, down
#include "q8_launch.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q8.h"
}

extern "C" bool q8_experts(const Q8Experts* e, void* stream) {
  const hipStream_t s = (hipStream_t) stream;
  if (!e || !e->n || !e->max_cnt || e->max_cnt > MV_MAX_TOKENS) return false;
  const bool fused = e->gate_type == e->up_type;
  const int act = e->act == Q8_ACT_INK2 ? q8::kActInk2 : q8::kActLane;
  if (!q8_takes(e->gate_type) || !q8_takes(e->up_type) || !q8_takes(e->down_type) || !e->h || !e->hq || !e->hqd ||
      !e->h_rows || !e->y || (e->act != Q8_ACT_LANE && e->act != Q8_ACT_INK2) ||
      (!fused && (act != q8::kActLane || !e->ga || !e->up || !e->scale || e->win)))
    return false;
  const q8::ActArgs args = {e->scale, e->win, e->lim, e->sg, e->su, e->gcap, e->ucap};
  bool ok = !e->x || q8_round(e->x, e->x_rows, e->d, e->d, e->xq, e->xqd, stream);
  // each role over the table: a run of one host count a launch, or every count at once (exact)
  const auto each = [&](auto role) {
    if (!e->h_cnt) return role(0u, e->n, e->max_cnt, true);
    bool r = true;
    for (uint32_t a = 0, z; r && a < e->n; a = z) {
      for (z = a + 1; z < e->n && e->h_cnt[z] == e->h_cnt[a];) z++;
      if (e->h_cnt[a] > 0) r = role(a, z - a, (uint32_t) e->h_cnt[a], false);
    }
    return r;
  };
  if (fused)
    ok = ok && each([&](uint32_t a, uint32_t n, uint32_t c, bool exact) {
           const q8::GateUp g = {e->wg + a, e->wu + a, e->cnt + a, e->xrow + a, e->xin, n, e->ff, e->d,
                                 e->xq,     e->xqd,    act,         args,         e->h,   e->ff};
           return q8::gate_up(e->gate_type, g, c, exact, s);
         });
  else
    ok = ok &&
         each([&](uint32_t a, uint32_t n, uint32_t c, bool exact) {
           return q8::group(e->gate_type, e->wg + a, e->cnt + a, e->xrow + a, e->xin, n, c, e->ff, e->d, e->xq,
                            e->xqd, e->ga, e->ff, exact, s) &&
                  q8::group(e->up_type, e->wu + a, e->cnt + a, e->xrow + a, e->xin, n, c, e->ff, e->d, e->xq,
                            e->xqd, e->up, e->ff, exact, s);
         }) &&
         ops_swiglu_lane(e->ga, e->up, e->ff, e->lim, e->scale, e->h_rows, e->h, stream);
  ok = ok && q8_round(e->h, e->h_rows, e->ff, e->ff, e->hq, e->hqd, stream);
  return ok && each([&](uint32_t a, uint32_t n, uint32_t c, bool exact) {
           return q8::group(e->down_type, e->wd + a, e->cnt + a, e->xrow + a, NULL, n, c, e->d, e->ff, e->hq, e->hqd,
                            e->y, e->d, exact, s);
         });
}
