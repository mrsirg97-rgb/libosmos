# Contributing

osmos is built design-first: the decision and its facts in `DESIGN.md`, then the shapes (the headers in
`include/osmos/`, each with its contract in a comment at its end), then tests that fail, then the code. Read
`DESIGN.md`'s section on the part you touch, and the header's contract, before you change it.

## the process

1. **Design.** A change lands with its section in `DESIGN.md`: what it is for, the facts and numbers it rests on, the
   decision, what was tried and did not work, with its numbers. A change that needs no design note has not been
   thought about.
2. **Schema.** The header's types and its contract comment, written before the code; the contract is the API.
3. **Tests before code.** A test is seen to fail before the code that makes it pass - a test that never failed proves
   nothing. Properties over examples; a bit-exactness test compares bits, not tolerances. Kernel tests replay the
   arithmetic contract and carry mutations that must fail them.
4. **Implementation**, against the header. Then the docs move with it: `README.md`, `DESIGN.md`, `CHANGELOG.md`.

## house rules

- **Deterministic.** A change that moves a bit of any output says so and why; the default is that no bit moves. The
  fingerprint tests (`test_lane_gpu`, `test_drafter_gpu`, `test_model_gpu`, `test_batch_gpu`, `test_longctx_gpu`)
  are the guarantee.
- **Errors are loud and fail closed.** Every refusal names what was wrong; an untrusted file (every GGUF, every
  request) is checked before it is acted on. No silent fallback, no guessing.
- **No magic numbers.** A constant is named and says where it comes from - a measurement, a format, a hardware limit.
- **One shape per concept.** A second way to do one thing is collapsed into the first, not added beside it.
- **Small files.** Approaching 300 lines, a file is doing two things.
- **Sparse comments.** The code says what; a comment says why, when the why is not obvious. The long English belongs
  in `DESIGN.md`.

## measuring

Every speed claim is a measured number on this hardware with the command that reproduces it. Benchmarks run on a
quiet box (nothing else reading the SSD), each configuration in a fresh process, interleaved, two rounds at least;
the run-to-run spread is reported, not assumed. A comparison runs both engines on the same files the same day. Power
is the APU package's (amdgpu PPT, `tools/power.py`), named as such - it is not wall power; `tools/bench/energy.py`
turns it and a bench's windows into joules a token. A change is developed and A/B'd at the low-power profile
(`powerprofilesctl set power-saver`; check `/sys/firmware/acpi/platform_profile` before a run) and checked at
performance once a release. What was not measured is said.

## building and testing

```sh
git submodule update --init      # libevt: the forward's reads and osmos-serve's loop (built into libosmos.a)
make                             # lib/libosmos.a
make lib/libosmos_hip.a serve api
make tests && make asan && make tsan
make api-test
python3 tools/install/test_install.py
make gpu-tests                   # on a gfx1151 with a model: exit 77 is a skip
```

CI (`.github/workflows/ci.yml`) runs the host tests, the sanitizers, osmos-api against its scripted engine, the
installer's tests, and compiles every GPU test for gfx1151 in `rocm/dev-ubuntu-24.04`; it fetches the model's first
shard (its tokenizer, 5.8 MB, at the manifest's pinned commit) so the text tests never skip there. The GPU and
whole-model tests run on a Halo: run them before a change that touches `src/run`, `src/mem` or `kernels/`.

## versions and releases

- `include/osmos/version.h`'s `OSMOS_VERSION` is the one source of truth; `osmos-api --version` and
  `osmos-serve --version` print it, and the release workflow refuses a tag that is not `v` + it.
- `CHANGELOG.md` gets a section for every release, `## [X.Y.Z]: <what it is>`; the release's notes are that section,
  and a tag without one is refused.
- A pack's published files (`tools/install/packs.json`) are pinned by commit and SHA-256; a new upload is a new pin.
