// the budget sized to this GPU (osmos/run/fit.h's fit_sized, osmos-serve --budget auto): a forward built at budget 0
// (its dense weights), then again at each build's room but the headroom, ends with the headroom free and no more than
// the slack past it - the experts took the rest. needs a gfx1151 and a model (test_model.h); skipped (77)
// without.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/fit.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr double kGiB = 1ull << 30;
constexpr uint64_t kSlack = 1ull << 26;  // 64 MiB: the runtime's own between the last read and this one
// the memory's steps: a budget 0.2 GiB larger took 0.94 GiB more of the room (UD-IQ1_M, 2026-11-08); fit_sized keeps
// the largest budget that left the headroom when the next is past it - so the room above the window by up to a step
constexpr uint64_t kStep = 1ull << 30;

int main() {
  std::printf("[test_fit_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_fit_gpu] skipped: no model or GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  struct Built {
    const Weights* w;
    ForwardOptions o;
    Forward* f;
    int builds;
  } b{w, {(int) w->geo.n_layers, 65536, 0, nullptr, 2048, true}, nullptr, 0};
  auto build = [](void* ctx, uint64_t budget, uint64_t* built, uint64_t* room, char err[FIT_ERR]) -> bool {
    Built* z = (Built*) ctx;
    char ferr[FORWARD_ERR] = "";
    z->o.budget = budget, z->builds++;
    if (!(z->f = forward_create_with(z->w, &z->o, ferr)) || !forward_room(z->f, room))
      return forward_destroy(z->f), z->f = nullptr, std::snprintf(err, FIT_ERR, "%s", ferr), false;
    *built = forward_budget(z->f);
    std::printf("  build %d at %.2f GiB: took %.2f, %.2f GiB room left\n", z->builds, budget / kGiB, *built / kGiB,
                *room / kGiB);
    return true;
  };
  auto unbuild = [](void* ctx) { forward_destroy(((Built*) ctx)->f), ((Built*) ctx)->f = nullptr; };
  uint64_t budget = 0, left = 0;
  char err[FIT_ERR] = "";
  if (!fit_sized(build, unbuild, &b, FIT_HEADROOM, &budget, err)) std::printf("  %s\n", err);
  CHECK(b.f && forward_budget(b.f) == budget && forward_room(b.f, &left));
  std::printf("  sized to %.2f GiB in %d builds: %.3f GiB left (the headroom %.2f, slack %.2f)\n", budget / kGiB,
              b.builds, left / kGiB, FIT_HEADROOM / kGiB, FIT_SLACK / kGiB);
  CHECK(left >= FIT_HEADROOM && left <= FIT_HEADROOM + FIT_SLACK + kSlack + kStep);
  CHECK(fit_ram_available() >= FIT_RAM_FREE); /* the operator's rule: the system's RAM left free */
  forward_destroy(b.f);

  /* --fill (ForwardOptions' fill): the room a build reports by the RAM it keeps free - the default's FIT_RAM_FREE, 0
     every byte past the headroom, more than there is none. at the floor (the dense weights alone), a build each */
  auto room_at = [&](bool set, uint64_t fill) {
    ForwardOptions o = b.o;
    o.budget = 0, o.fill_set = set, o.fill = fill;
    char ferr[FORWARD_ERR] = "";
    Forward* f = forward_create_with(w, &o, ferr);
    uint64_t r = 0;
    CHECK(f && forward_room(f, &r));
    forward_destroy(f);
    return r;
  };
  const uint64_t r_def = room_at(false, 0), r_zero = room_at(true, 0), r_huge = room_at(true, 1000ull << 30);
  std::printf("  the dense build's room: %.2f GiB (8 GiB kept free), %.2f (--fill 0), %.2f (--fill 1000)\n",
              r_def / kGiB, r_zero / kGiB, r_huge / kGiB);
  CHECK(r_huge == 0);
  CHECK(r_zero + FIT_SLACK >= r_def); /* (the RAM free moves a little between builds: the slack's room) */
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_fit_gpu] ok\n");
  return 0;
}
