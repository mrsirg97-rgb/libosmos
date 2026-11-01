# libsubstrate

A deterministic inference engine for **DeepSeek-V4.1-Flash** (~768B parameters: ~557B in routed experts, ~196B in
Engram tables) on a **single AMD Strix Halo** (Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB).

**The guarantee:** the same input gives the same output, bit for bit, however it is run:

- alone or batched with other tokens or sequences,
- at any memory budget, whichever experts happen to be resident in GPU memory,
- decoded one token at a time or drafted (DSpark speculative decoding) and verified,
- with an expert computed on the GPU or on the CPU (the CPU lane's AVX-512 kernels give the GPU's exact bits).

Each of these is a test that compares bits, not tolerances: fingerprints of every output stream must match across
the configurations (`tests/gpu/test_lane_gpu`, `test_drafter_gpu`, `test_model_gpu`, `test_forward_gpu`), and the
kernels are checked against reference replays with deliberate mutations that must fail them.

C, HIP and a little C++. No Python at runtime.

## Performance

Measured on one Strix Halo (128 GB, ~113 GiB GPU memory), DeepSeek-V4.1-Flash GSQ-RCO IQ2_XXS (experts mostly
IQ2_XXS / IQ1_M, Engram IQ4_NL), greedy decoding, 256 tokens an answer, each engine in a fresh process, runs
interleaved. The comparison is [Project Maya](https://github.com/mw00/project-maya) 1.1.46, the fastest engine we
know of for this model on this hardware. Details and every run: `DESIGN.md`.

| | libsubstrate | Maya 1.1.46 |
| --- | --- | --- |
| Decode, plain (3 bench questions, mean) | 16.1 tok/s | 17.2 tok/s (v1.1.48) |
| Decode, DSpark drafted (same questions) | 27.3 tok/s | 29.5 tok/s (v1.1.46) |
| Drafted, repetitive output (CSV to JSON) | 35.5 tok/s | 37.0 tok/s |
| Drafted, boilerplate code (getters and setters) | 38.8 tok/s | 29.5 tok/s |
| Prompt, 2,048 tokens | ~200-255 tok/s | ~240 tok/s |
| Prompt, 8,192 tokens (decode config: 2,048-token steps, 106 GiB) | ~400 tok/s | ~385 tok/s |
| Prompt, 8,192 tokens (one 8,192-token step, 92 GiB) | ~540 tok/s | ~385 tok/s |
| Output reproducible bit for bit | yes | no (greedy decode varies run to run) |

What bounds it: the GPU's own work per token is the same as Maya's, stage for stage (~50 ms). What remains is
waiting on the SSD for experts not in memory (~6 ms a token; RAM, not code, is the limit: GPU memory and the page
cache are the same 128 GB) and the GPU's ~2 us between dependent kernels.

## How it runs a 250 GB model in 128 GB

- **Experts by residency.** The dense weights and as many experts as the budget holds live in GPU memory, chosen by
  how often a calibration text picks them (a profile beside the model). The rest are read from the GGUF files when
  the router picks them, guessed a layer ahead, and the resident set follows the text (hot experts swapped in).
- **Int8 experts.** A routed expert's input is rounded to int8 a block of 32, and its 1-2-bit weights are summed with
  exact integer dot products (`v_dot4` on the GPU, `vpdpbusd` on the CPU), then scaled in one fixed order - the
  contract in `kernels/lane.h` that makes the CPU and the GPU agree bit for bit.
- **Engram** rows are read from their file a few at a time per token (~7 KB).
- **DSpark**, DeepSeek's own drafter, proposes up to 5 tokens a round; one batched forward verifies them.

## Requirements

- AMD Ryzen AI MAX+ 395 (gfx1151) with 128 GB. The CPU lane (optional) needs AVX-512 VNNI and VBMI.
- Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14).
- GPU memory raised to ~113 GiB: `options ttm pages_limit=29637703` in `/etc/modprobe.d/ttm.conf` (a system setting:
  yours to make).
- The model (~250 GB on disk, an NVMe SSD; the faster its random reads, the faster the decode):
  - the backbone GGUF, IQ2_XXS, 4 shards: [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
    (developed on a derivative with the same tensor layout),
  - the Engram GGUF (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`),
  - for drafting, DSpark's weights as Maya packs them (`dspark.bin`, `dspark.json`),
  - optionally, an expert pick profile (`substrate-picks.bin`, counted with `forward_picks` on calibration text).

## Build and test

```sh
make                        # lib/libsubstrate.a (the host library)
make lib/libsubstrate_hip.a # the GPU library
make tests                  # CPU tests (no GPU, no model)
make asan && make tsan      # the same under AddressSanitizer / ThreadSanitizer
make gpu-tests              # GPU and whole-model tests (skip without a gfx1151 or ~/models/ds1017)
```

The whole-model tests need the model in `~/models/ds1017` and, for `test_model_gpu`, a dump of the reference
forward (`oracle/`, see its README).

## Use

```c
#include "substrate/run/forward.h"

char err[FORWARD_ERR];
Gguf *g = gguf_create(".../DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf", err);
Gguf *eg = gguf_create(".../DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", err);
Weights *w = weights_create(g, eg, err);

ForwardOptions o = {.n_layers = w->geo.n_layers, .max_ctx = 4096, .budget = 106ull << 30,
                    .picks = picks /* or NULL */, .max_tokens = 2048};
Forward *f = forward_create_with(w, &o, err);
forward_lane(f, 0, err);               /* decode's experts in the CPU-repeatable order; 0: all on the GPU */

Seq *s = seq_create(f, err);
seq_prompt(s, n_prompt);
/* a prompt of at most max_tokens (longer: several forward_batch calls; forward_logits says which rows a step keeps) */
forward_batch(f, seqs /* s, n_prompt times */, prompt, n_prompt, err);
int32_t t;
forward_greedy(f, n_prompt - 1, 1, &t, err);   /* the next token, picked on the GPU */
for (;;) {
  forward_step(f, s, t, err);
  forward_greedy(f, 0, 1, &t, err);
}
```

Every call returns `false` (or `NULL`) with the reason in `err` rather than guess. Drafting: `substrate/run/drafter.h`
(`drafter_create`, `drafter_draft`, then verify with `forward_batch` and `seq_rollback`; `tools/bench/dbench.hip`
shows the loop).

## Reproduce the numbers

```sh
python tools/bench/prompts.py <maya checkout> ~/models/ds1017/maya-ds4/tokenizer bench   # token ids (Maya's tokenizer)
hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/dbench.hip -Llib -lsubstrate_hip -lsubstrate \
      -lhsa-runtime64 -o bin/dbench
PICKS=~/models/ds1017/substrate-picks.bin GREEDY=1 \
  bin/dbench 106 2048 256 bench/warmup.txt bench/q1.txt bench/q2.txt bench/q3.txt                    # plain
PICKS=... GREEDY=1 DRAFT=~/models/dspark bin/dbench 97 2048 256 bench/warmup.txt bench/q1.txt ...  # drafted
```

Run each configuration in a fresh process and interleave them; a first run after the machine has been idle is slower
(a cold page cache). Each answer prints a hash of its tokens: the same hash in every configuration is the guarantee.

## Scope

One model (DeepSeek-V4.1-Flash), one machine (Strix Halo, 128 GB), greedy decoding, one sequence at a time in the
benchmarks. The quant kernels (Q8_0, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ4_NL, IQ1_M, BF16, MXFP4; the int8 expert path
and the CPU lane: IQ2_XXS, IQ2_XS, IQ1_M) and the expert streaming are model-independent; the attention,
hyper-connections, Engram and DSpark are DeepSeek's.

## Layout

| | |
| --- | --- |
| `include/substrate/` | the public headers, each with its contract in a comment at the end |
| `src/model/` | GGUF, weights, Engram, DSpark index |
| `src/mem/` | the expert pool, the resident set, the file readers |
| `src/run/` | the forward: steps, layers, the lane, drafting |
| `src/cpu/` | the CPU lane: AVX-512 int8 experts, its thread pool |
| `kernels/` | the HIP kernels and the shared arithmetic contracts (`lane.h`, `quant.h`) |
| `tests/`, `tests/gpu/` | CPU tests; GPU and whole-model tests |
| `oracle/` | the reference forward in PyTorch the numerics are held to (from Maya, MIT) |
| `tools/` | grid generation, the bench |
| `DESIGN.md` | every decision and measurement, including what was tried and did not help |

## Credits

- [Project Maya](https://github.com/mw00/project-maya) (MIT): the reference this engine is measured against and
  learned from - the int8 expert approach (`ds4_iq.cu`), the DeepSeek reference tools in `oracle/` (copied, with
  `oracle/LICENSE.maya`), DSpark's packing. Findings from this work go back to Maya: `MAYA-UPSTREAM-PATCHES.md`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the IQ grid tables in `kernels/quant_grids.h` are
  generated from its `gguf-py` (`tools/gen_grids.py`), and the quant formats are ggml's.
- [DeepSeek](https://github.com/deepseek-ai) (MIT): the model, its inference reference, and the Engram hashing that
  `oracle/engram_hash.py` follows.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): the GSQ-RCO quantization this was developed on.

## License

MIT - see `LICENSE`. Third-party material keeps its own license (`oracle/LICENSE.maya`).
