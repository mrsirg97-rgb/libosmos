# libsubstrate

A deterministic inference engine for **DeepSeek-V4.1-Flash** (~768B parameters: ~557B in routed experts, ~196B in
Engram tables) on a **single AMD Strix Halo** (Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB).

It runs two quantizations of the model:

- **unsloth's UD-IQ1_M** ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF),
  5 shards, ~234 GB with its Engram table): experts IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S, attention Q6_K. The one most
  people run; the numbers below are on it.
- **ISTA-DASLab's GSQ-RCO IQ2_XXS** ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)):
  experts IQ2_XXS / IQ2_XS / IQ1_M, its Engram in a separate GGUF. What the engine was first built on.

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

Measured on one Strix Halo (128 GB, ~113 GiB GPU memory), **UD-IQ1_M**, 64K context, greedy decoding, 256 tokens an
answer to [Project Maya](https://github.com/mw00/project-maya)'s bench questions and its 2K / 8K prompts, each engine
in a fresh process, two rounds each. The comparison is Maya **1.1.52** (2026-11-03), the fastest engine we know of
for this model on this hardware, on the same model, its pack of the same files. Details and every run: `DESIGN.md`.

| | libsubstrate | Maya 1.1.52 |
| --- | --- | --- |
| Decode, plain (3 questions; mean, range) | 16.3 tok/s (16.1-16.4) | 16.9 tok/s (16.5-17.2) |
| Decode, DSpark drafted (same questions) | 29.5 tok/s (23.8-33.5) | 29.6 tok/s (25.9-32.2) |
| Prompt, 2,048 tokens | 332-333 tok/s | 232-269 tok/s |
| Prompt, 8,192 tokens (2,048-token steps, 106 GiB) | 472-484 tok/s | 407-412 tok/s |
| Prompt, 8,192 tokens (one 8,192-token step, 100 GiB, 16K context) | ~578 tok/s | - |
| Next-token distribution against the reference forward (mean KL, 32 positions; argmax agrees) | 0.209 (28 of 32) | 0.272 (29 of 32; an earlier version's logits) |
| Attention state (KV) a token | ~0.9 KB (the model's FP8 / FP4 formats) | |
| Output reproducible bit for bit | yes | no (greedy decode varied run to run, measured on 1.1.46) |

Budgets: plain decode 110 GiB, drafted 100 GiB, prompts 106 GiB (`bin/dbench`'s first argument; the rest of GPU
memory is the step's buffers). A larger prompt step trades decode budget for long-prompt speed (DESIGN.md, "a
prompt's step size").

What bounds it: the GPU's own work per decode token is below Maya's (~54 ms against ~58 in kernel traces); what
remains is waiting on the SSD for experts not in memory (~4.9 ms a token at the 110 GiB budget, a layer's burst of reads filling the SSD:
RAM, not code, is the limit - GPU memory and the page cache are the same 128 GB) and the GPU's ~2 us between
dependent kernels.

Earlier, on **GSQ-RCO IQ2_XXS** against Maya 1.1.46 (before the UD work: the KV's formats, the decode and prompt
changes since; not re-measured - its files are not on the development box now):

| | libsubstrate | Maya 1.1.46 |
| --- | --- | --- |
| Decode, plain (3 bench questions, mean) | 16.1 tok/s | 17.2 tok/s (v1.1.48) |
| Decode, DSpark drafted (same questions) | 27.3 tok/s | 29.5 tok/s |
| Drafted, repetitive output (CSV to JSON) | 35.5 tok/s | 37.0 tok/s |
| Drafted, boilerplate code (getters and setters) | 38.8 tok/s | 29.5 tok/s |
| Prompt, 2,048 tokens | ~200-255 tok/s | ~240 tok/s |
| Prompt, 8,192 tokens (one 8,192-token step, 92 GiB) | ~540 tok/s | ~385 tok/s |

## How it runs a 250 GB model in 128 GB

- **Experts by residency.** The dense weights and as many experts as the budget holds live in GPU memory, chosen by
  how often a calibration text picks them (a profile beside the model). The rest are read from the GGUF files when
  the router picks them, guessed a layer ahead, and the resident set follows the text (hot experts swapped in).
- **Int8 experts.** A routed expert's input is rounded to int8 a block of 32, and its 1-2-bit weights are summed with
  exact integer dot products (`v_dot4` on the GPU, `vpdpbusd` on the CPU), then scaled in one fixed order - the
  contract in `kernels/lane.h` that makes the CPU and the GPU agree bit for bit.
- **Engram** rows are read from their file at each step's start, every row a reader of its own (~7 KB a token).
- **The attention's state in the model's own formats**: the window's keys FP8, the compressed rows and the indexer's
  keys FP4, as DeepSeek's design keeps them - ~0.9 KB a token, so a 64K context fits beside a 110 GiB expert budget.
  A packed row read back is exactly the F32 it was rounded to; a prompt's step unpacks the rows once for all its
  tokens.
- **DSpark**, DeepSeek's own drafter, proposes up to 5 tokens a round; one batched forward verifies them.

## Requirements

- AMD Ryzen AI MAX+ 395 (gfx1151) with 128 GB. The CPU lane (optional) needs AVX-512 VNNI and VBMI.
- Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14).
- GPU memory raised to ~113 GiB: `options ttm pages_limit=29637703` in `/etc/modprobe.d/ttm.conf` (a system setting:
  yours to make).
- The model, on an NVMe SSD (the faster its random reads, the faster the decode), either:
  - UD-IQ1_M's 5 shards (`DeepSeek-V4.1-Flash-1017-UD-IQ1_M-0000N-of-00005.gguf`, ~234 GB; its Engram table is in
    them), or
  - GSQ-RCO's 4 IQ2_XXS shards and its Engram GGUF (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`),
- for drafting, DSpark's weights as Maya packs them (`dspark.bin`, `dspark.json`),
- optionally (UD-IQ1_M's decode ~12-13 -> ~16 tok/s with one), an expert pick profile beside the model (`substrate-picks.bin`: every
  pick on calibration text, `bin/dbench` with `PICKS_OUT` writes one).

## Build and test

```sh
make                        # lib/libsubstrate.a (the host library)
make lib/libsubstrate_hip.a # the GPU library
make tests                  # CPU tests (no GPU, no model)
make asan && make tsan      # the same under AddressSanitizer / ThreadSanitizer
make gpu-tests              # GPU and whole-model tests (skip without a gfx1151 or a model)
```

The whole-model tests read GSQ-RCO from `~/models/ds1017` where it is, else UD-IQ1_M from `~/Models/ds1017/UD-IQ1_M`
(`tests/gpu/test_model.h`); a few still need GSQ-RCO and skip without it. `test_model_gpu` also needs a dump of the
reference forward (`oracle/`, see its README).

## Use

```c
#include "substrate/run/forward.h"

char err[FORWARD_ERR];
Gguf *g = gguf_create(".../DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf", err);
Weights *w = weights_create(g, NULL, err);   /* GSQ-RCO: its Engram GGUF as the second argument */

ForwardOptions o = {.n_layers = w->geo.n_layers, .max_ctx = 65536, .budget = 110ull << 30,
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
python tools/bench/prompts.py <maya checkout> <maya pack>/tokenizer bench   # token ids (Maya's tokenizer)
hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/dbench.hip -Llib -lsubstrate_hip -lsubstrate \
      -lhsa-runtime64 -o bin/dbench
export MODEL=~/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf
export PICKS=~/Models/ds1017/UD-IQ1_M/substrate-picks.bin MAX_CTX=65536
bin/dbench 110 512 256 bench/warmup.txt bench/q1.txt bench/q2.txt bench/q3.txt                       # plain
GREEDY=1 DRAFT=~/Models/ds1017/dspark bin/dbench 100 512 256 bench/warmup.txt bench/q1.txt ...    # drafted
bin/dbench 106 2048 8 bench/warmup.txt bench/p2048.txt bench/p8192.txt                              # prompts
```

`dbench budget_gib max_tokens n_new warmup q...`: `MODEL` a model's first shard (GSQ-RCO: `MODEL_DIR` and `ENGRAM`),
`PICKS` the profile, `DRAFT` DSpark's directory, `PICKS_OUT` writes a profile from the run. Run each configuration in
a fresh process and interleave them; a first run after the machine has been idle is slower (a cold page cache). Each
answer prints a hash of its tokens: the same hash in every configuration is the guarantee.

## Scope

One model (DeepSeek-V4.1-Flash), one machine (Strix Halo, 128 GB), greedy decoding, one sequence at a time in the
benchmarks. The quant kernels (Q8_0, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ4_NL, BF16, MXFP4, from one
type table, `include/substrate/model/qtype.h`; the int8 expert path and the CPU lane: the IQ1 and IQ2 types) and
the expert streaming are model-independent; the attention, hyper-connections, Engram and DSpark are DeepSeek's.

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
- [unsloth](https://huggingface.co/unsloth): the UD-IQ1_M quantization.

## License

MIT - see `LICENSE`. Third-party material keeps its own license (`oracle/LICENSE.maya`).
