// a GPU test's build sized as the engine sizes it (osmos/run/fit.h): at the test's own budget where this machine
// leaves the headroom after it (the room fit_room reads: the GPU's free memory, or the RAM past FIT_RAM_FREE), else
// the largest that does (fit_sized's search over the same build) - a test behaves as the engine would on the machine
// it runs on, and never takes the RAM a 128 GB box's fixed budget assumed. `want` keeps its reason (many experts
// held, a small pool to stream): the sizing only ever lowers it. build(budget) makes everything the test holds
// (its forward, a drafter, its sequences), took() what it took, room(&bytes) the room it left, unbuild() frees it
#pragma once
#include <cstdint>
#include <cstdio>
#include <functional>

extern "C" {
#include "osmos/run/engine.h"
#include "osmos/run/fit.h"
}

struct TestFit {
  std::function<bool(uint64_t)> build;
  std::function<uint64_t()> took;
  std::function<bool(uint64_t*)> room;
  std::function<void()> unbuild;
};

static bool test_fit_build(void* ctx, uint64_t budget, uint64_t* built, uint64_t* room, char err[OSMOS_ERR]) {
  TestFit* t = (TestFit*) ctx;
  if (!t->build(budget) || !t->room(room)) {
    t->unbuild();
    return std::snprintf(err, OSMOS_ERR, "the test's build refused at %.2f GiB", budget / 1073741824.0), false;
  }
  *built = t->took();
  return true;
}

static void test_fit_unbuild(void* ctx) { ((TestFit*) ctx)->unbuild(); }

// the build left built: first at 0 (everything but the experts - the engine's first build), its room telling how far
// this machine goes (fit_budget's next: the fill); the test's `want` built where it is within that and leaves the
// headroom, else the machine's fill (fit_sized) - never a build past the room a build has measured. *budget its
// budget. false (nothing built) when no build leaves the headroom: the test refuses, not the machine's memory
static bool test_fit(TestFit& t, uint64_t want, uint64_t* budget) {
  uint64_t room = 0, b0 = 0;
  char err[OSMOS_ERR] = "";
  if (!test_fit_build(&t, 0, &b0, &room, err))
    return std::fprintf(stderr, "  sized to this machine: %s\n", err), false;
  t.unbuild();
  if (want <= fit_budget(b0, room, FIT_HEADROOM)) {
    if (t.build(want) && t.room(&room) && room >= FIT_HEADROOM) return *budget = want, true;
    t.unbuild();
  }
  if (!fit_sized(test_fit_build, test_fit_unbuild, &t, FIT_HEADROOM, budget, err) || !t.room(&room) ||
      room < FIT_HEADROOM) {
    std::fprintf(stderr, "  sized to this machine: %s (room %.2f GiB)\n", err[0] ? err : "under the headroom",
                 room / 1073741824.0);
    return false;
  }
  std::printf("  the budget sized to this machine: %.2f GiB (the test's own %.2f)\n", *budget / 1073741824.0,
              want / 1073741824.0);
  return true;
}

// an engine (osmos_create) sized so: at o's budget where the machine leaves the headroom after it, else its fill -
// the build's room osmos_room's, what it took osmos_stats' budget. NULL (err) when no build leaves the headroom
static Osmos* test_osmos(const OsmosModel* m, OsmosOptions o, char err[OSMOS_ERR]) {
  Osmos* e = nullptr;
  TestFit fit = {[&](uint64_t b) {
                   o.budget = b;
                   return (e = osmos_create(m, &o, err)) != nullptr;
                 },
                 [&] {
                   OsmosStats st;
                   osmos_stats(e, &st);
                   return st.budget;
                 },
                 [&](uint64_t* r) { return osmos_room(e, r); }, [&] { osmos_destroy(e), e = nullptr; }};
  uint64_t budget = 0;
  return test_fit(fit, o.budget, &budget) ? e : nullptr;
}
