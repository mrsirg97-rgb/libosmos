# libosmos

An inference engine for two big mixture-of-experts models on one **AMD Strix Halo** (Ryzen AI MAX+ 395, Radeon
8060S / gfx1151, 128 GB):

- **DeepSeek-V4.1-Flash** (~768B parameters, ~250 GB quantized), more than fits in memory: the experts it needs are
  streamed from the SSD as it runs.
- **Qwen3.8-Flash-Next** (~177B, 512 experts, the Qwen4 architecture's preview), fully resident in GPU memory.

**The same input gives the same output, bit for bit**, however it runs: alone or batched with other conversations,
at any memory budget, decoded one token at a time or drafted and verified, with an expert computed on the GPU or the
CPU. Each of those is a test that compares bits, not tolerances (`tests/gpu/`), and the kernels are checked against
reference replays with deliberate mutations that must fail them.

C, HIP and a little C++. No Python at runtime.

## Quick start

```sh
python3 tools/install/osmos_install.py ud-iq1_s --serve            # DeepSeek; or ud-iq1_m, ud-iq2_xxs
python3 tools/install/osmos_install.py qwen38-ud-iq3_xxs --serve   # Qwen; or qwen38-ud-iq3_xxs-1m
```

The installer builds the engine, downloads the pack's files at pinned commits, checks every SHA-256, writes the
pack's `osmos.json` and starts **osmos-api**, an OpenAI-compatible server on port 8091 (`/v1/chat/completions`
streamed or whole, with tools and reasoning; `/v1/models`; `/health`). Options:

- `--abliterated` adds huihui-ai's abliteration as an overlay (DeepSeek and Qwen alike, not yet UD-Q4_K_XL; the stock files stay as they are).
- `--have <dir>` uses files you already have.
- `--rig` registers the pack in [rig](https://github.com/mrsirg97-rgb/rig) instead of serving it, as
  `deepseek-1017-ud-s` / `-m` / `-l` (UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS) or `qwen3.8-flash-next` (`-1m`, `-l`).

## The packs

| Pack | Files | Context | Pick it for |
| --- | --- | --- | --- |
| `ud-iq1_s` | 122 GB + 203 GB Engram table | 1M | speed |
| `ud-iq1_m` | 140 GB + 203 GB Engram table | 1M | quality (unsloth's benchmarks 85.55 against 83.53) |
| `ud-iq2_xxs` | 165 GB + 203 GB Engram table | 1M | the best perplexity, the slowest decode |
| `qwen38-ud-iq3_xxs` | 78 GB + 3 GB MTP head | 262K | Qwen, fully in GPU memory |
| `qwen38-ud-iq3_xxs-1m` | the same files | 1M (YaRN) | long documents: recalls a fact 355K tokens back, 0.4% worse on short text |
| `qwen38-ud-q4_k_xl` | 111 GB (82 GB in GPU memory, its 29 GB n-gram table read from the SSD) + 3 GB MTP head | 262K | the larger quant, if you want it: 10-16% slower than UD-IQ3_XXS on our agent replay, no better perplexity on our text |

The DeepSeek packs are [unsloth's quants](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) with the
lossless MXFP8 Engram table, plus our 4.7 GB DSpark drafter and expert pick profile
([osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)). Qwen is
[unsloth's UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (or UD-Q4_K_XL) and its MTP head. The abliteration
costs 0.2-0.5% perplexity (UD-IQ1_M 2.458 → 2.470, Qwen 2.714 → 2.723) at the same speed.

## Performance

Everything below was measured in one session (2026-11-09) on one machine at AMD's `low-power` profile (~70 W cap),
with nothing else running and greedy decoding. Each engine runs in a fresh process, in interleaved rounds with the
engines' order swapped from round to round, and each is set up the way its own setup does it. The scripts and logs
are listed in `DESIGN.md` ("0.5.11").

**Low-power is the profile for work that runs a while.** Against 0.5.9's run at `performance`, the agents bench below
took 20% longer and used half the energy per answer token (3.42 J against 7.03 J; mean package power 56 W against
104 W). Qwen's decode is the same at both (plain 26.5 against 27.1 tok/s, drafted 45.5 against 45.6): it is bound
by memory bandwidth, which the power cap doesn't lower.

### DeepSeek against Maya

[Project Maya](https://github.com/mw00/project-maya) 1.1.57 (build 0065548):
- rebuilt
- run with its own pack, its own 8 GB DSpark, and its CPU lane and expert profile calibrated on this machine
- reading the same model files

Both engines run at 64K context and answer Maya's three bench questions, 256 tokens each, plus its 2K and 8K prompts,
in two rounds.

| | UD-IQ1_M osmos | Maya | UD-IQ1_S osmos | Maya | UD-IQ2_XXS osmos | Maya |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt, 2,048 tokens | **318** | 206 | **322** | 228 | **281** | 216 |
| Prompt, 8,192 tokens | **438** | 352 | **451** | 368 | **383** | 361 |
| Decode, drafted | **32.7** | 24.4 | **39.5** | 33.6 | **24.8** | 9.9 |
| Decode, plain | **16.4** | 15.6 | 18.3 | **18.9** | **13.5** | 8.9 |
| Perplexity, 8,191 tokens | **2.4579** | 2.4606 | 2.5006 | **2.4983** | **2.4492** | 2.4505 |
| SSD read a token, plain / drafted | | | | | **0.064 / 0.082 GB** | ~0.17 / ~0.18 GB |

The bigger the quant, the wider the gap: plain decode goes from Maya 3% ahead on UD-IQ1_S to osmos 52% ahead on
UD-IQ2_XXS, drafted to 2.5x. That's IO: osmos's one cache keeps the experts decode comes back to, so it reads 2-3x
fewer SSD bytes for each token. (The device's counter, once a second: over osmos's exact generation windows; for
Maya, estimated from its steady decode, since it doesn't print its windows.) osmos's answers hash the same in every
run, and the same as 0.5.9's.

**Agents.** Maya's agent bench: 4 agents at once, 8 turns each, about 99K new prompt tokens against 8K answer tokens.
It ran on UD-IQ1_M at 131K context with drafting on. Power is the APU package's (CPU + GPU + NPU, not the wall).

| | osmos 0.5.11 | Maya 1.1.57 | |
| --- | ---: | ---: | ---: |
| Wall time, all 32 turns | **485 s** | 653 s | 26% less |
| First token, p50 / p95 | **7.0 / 73.0 s** | 7.4 / 82.6 s | |
| Turn done, p50 / p95 | **48.3 / 105.9 s** | 69.8 / 147.3 s | 31% / 28% less |
| Package power, mean | **56.4 W** | 58.2 W | |
| Energy per answer token | **3.42 J** | 4.75 J | 28% less |
| RAM left free, lowest | **8.2 GiB** | 2.2 GiB | |

osmos sizes itself to leave 8 GiB of RAM to the rest of the system: room for an editor, a browser, language servers
and the agent harness beside the model. A box with less beside it can give the cache more: `"fill_gib": 4` in the
pack's `osmos.json` (osmos-serve `--fill 4`) keeps 4 GiB free instead, 0 keeps none. The output is the same at every
value; only how much of the model stays in memory changes.

### Qwen3.8 against llama.cpp

llama.cpp 7d7f028, with HIP, all layers on the GPU and flash attention, at its faster batch for each length (2,048
or 4,096). osmos reads prompts in 4,096-token steps, the way its pack does. Three rounds.

| | osmos 0.5.11 | llama.cpp | |
| --- | ---: | ---: | ---: |
| Prompt, 2,048 tokens | **756** | 541 | +40% |
| Prompt, 8,192 tokens | **714** | 526 | +36% |
| Prompt, 16,384 tokens | **677** | 484 | +40% |
| Prompt, 65,536 tokens | **602** | 330 | 1.8x |
| Decode, plain | **26.5** | 26.0 | +2% |
| Decode at 16K context | **24.9** | 20.9 | +19% |
| Decode, drafted by the MTP head | **45.5** | not measured | |
| Perplexity, 8,192 / 16,384 tokens | 2.7143 / 2.2732 | 2.7169 / 2.2718 (f16 cache), 2.7110 / 2.2693 (q8_0) | |

osmos's decode holds up better at depth: going from a short context to 16K costs it 6%, llama.cpp 20%. It is also
steadier: its decode varied 0.2-0.4% from round to round, llama.cpp's 1.7% (plain) and 19% (at 16K); prompts
varied up to 7% for osmos, up to 18% for llama.cpp. Both run Qwen's sparse attention and pick the same key blocks
(`test_q4x_qsa_gpu`). osmos keeps its cache in Q8_0; its perplexity is within 0.2% of llama.cpp's with either cache.

## How it works

**DeepSeek in 128 GB.** The dense weights live in GPU memory, and the routed experts share one cache (the membrane):
- **At load** the cache fills with the experts a calibration text picks most.
- **As it runs**, the least recently used experts are evicted first, and a new expert takes the memory of an old one
  the same size.
- **On a miss**, the expert is read from the model's files into registered host memory the GPU reads directly
  (`O_DIRECT`, so no page cache and no CPU copy). [libevt](https://github.com/mrsirg97-rgb/libevt)'s priority queue
  orders the reads by when they're needed.
- **Prompts** go through a pool of their own and read ahead only the experts the next layer's router will pick, so a
  long prompt doesn't push out the experts decode wants back.
- **Bit-exact experts.** A routed expert's input is rounded to int8 in blocks of 32 and summed with exact integer dot
  products, then scaled in one fixed order (`kernels/lane.h`). That's why the CPU lane (AVX-512) and the GPU give the
  same bits.
- **Attention state** stays in DeepSeek's own formats (FP8 and FP4, about 0.9 KB a token), so a 1M context takes only
  ~0.85 GiB.
- **Engram** rows (about 7 KB a token) are read from their file at each step.
- **DSpark** drafts up to 5 tokens a round, and one batched forward checks them.

**Qwen** is fully resident:
- **Matmuls** run on the matrix cores, with f16 activations and F32 sums at every batch size, so a token's bits don't
  depend on its batch.
- **The cache** is Q8_0: 3.4 GB for 262K tokens.
- **QSA**, Qwen's sparse attention, attends to a token's top 512 blocks of 4 keys.
- **The MTP head** drafts up to 5 tokens.
- **Rollback.** The 36 Gated DeltaNet layers keep 8 steps of state, so a rejected draft rolls back.
- **The abliteration** is an overlay: 96 tensors loaded in place of the model's own.

`DESIGN.md` has every decision and measurement, including what was tried and didn't help.

## Requirements

- A Ryzen AI MAX+ 395 (gfx1151) with 128 GB, and Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14).
  The optional CPU lane needs AVX-512 VNNI and VBMI.
- GPU memory raised to ~113 GiB: `options ttm pages_limit=29637703` in `/etc/modprobe.d/ttm.conf`. This is a system
  setting, so it's yours to make.
- The model on an NVMe SSD, on a filesystem that takes `O_DIRECT` reads (ext4, btrfs and xfs do; osmos refuses one that
  doesn't, and names it). The faster the SSD's random reads, the faster DeepSeek decodes.

## Build and test

```sh
make && make lib/libosmos_hip.a   # the host and GPU libraries
make tests                        # CPU tests (no GPU, no model); make asan, make tsan
make gpu-tests                    # GPU and whole-model tests (skipped without a gfx1151 or the model)
make serve api                    # bin/osmos-serve (stdin / stdout) and bin/osmos-api (HTTP)
python3 tools/install/test_install.py
```

## Use as a library

The engine interface over both models is `include/osmos/run/engine.h`; DeepSeek's forward is `osmos/run/forward.h`,
and its drafter is `osmos/run/drafter.h`. Each header's contract is in a comment at its end, and every call returns
`false` or `NULL` with the reason in `err` rather than guess. `tools/bench/dbench.hip` is a complete example loop.

## Reproduce the numbers

```sh
# DeepSeek (token ids from Maya's tokenizer: tools/bench/prompts.py)
export MODEL=~/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf
export PICKS=~/Models/ds1017/UD-IQ1_M/osmos-picks.bin MAX_CTX=65536 NO_GUESS=1
GREEDY=1 bin/dbench auto 512 256 warmup.txt q1.txt q2.txt q3.txt                                # plain
GREEDY=1 DRAFT=~/Models/ds1017/dspark-iq2xs bin/dbench auto 512 256 warmup.txt q1.txt q2.txt q3.txt  # drafted
bin/dbench auto 2048 8 warmup.txt p2048.txt p8192.txt                                         # prompts
python tools/power.py --csv agents.csv -- python tools/serve/agent_bench.py --only osmos --model $MODEL \
  --picks $PICKS --dspark ~/Models/ds1017/dspark-iq2xs --budget auto --osmos-flag=--no-guess   # agents

# Qwen (token ids from its tokenizer: oracle/qwen_dump.cpp; llama.cpp's perplexity: qwen_dump -ppl)
export Q=~/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf Q4X_BENCH_STEP=4096
bin/q4x_bench $Q ids.txt 16384 64                                                             # prompt, then decode
bin/q4x_bench $Q ids.txt 512 128 ~/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf   # drafted
bin/q4x_ppl $Q ids.txt 8192 2048                                                              # perplexity
```

The bench tools are built with `hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/<tool>.hip -Llib -losmos_hip
-losmos -lhsa-runtime64`. Run each configuration in a fresh process, and interleave them. Every answer prints a hash
of its tokens: the same hash in every configuration is the guarantee.

## Layout

| | |
| --- | --- |
| `include/osmos/` | public headers, each with its contract in a comment at the end |
| `src/model/`, `src/mem/` | GGUF, weights, Engram; the expert cache and the file readers |
| `src/run/`, `src/q4x/` | DeepSeek's forward and the engine interface; Qwen's forward |
| `src/text/`, `src/cpu/` | tokenizers and chat encodings; the CPU lane |
| `kernels/` | HIP kernels and the shared arithmetic contracts (`lane.h`, `quant.h`) |
| `tests/`, `tests/gpu/` | CPU tests; GPU and whole-model tests |
| `oracle/` | reference forwards the numerics are held to (DeepSeek's from Maya; Qwen's against llama.cpp) |
| `tools/` | osmos-api, osmos-serve, the installer and packs manifest, the benches |

## Credits

- [libevt](https://github.com/mrsirg97-rgb/libevt) (MIT, the same author): the event scheduler under the expert reads
  and `osmos-serve --parallel`.
- [Project Maya](https://github.com/mw00/project-maya) (MIT): the reference this is measured against and learned from.
  The DeepSeek reference tools in `oracle/` are copied from it (`oracle/LICENSE.maya`). Several of Maya's AMD paths
  are this author's contributions there: the int8 experts on gfx1151
  ([#643](https://github.com/mw00/project-maya/pull/643)), conversation slots and prompt caching
  ([#356](https://github.com/mw00/project-maya/pull/356)), and libevt as its scheduler
  ([#587](https://github.com/mw00/project-maya/pull/587)). Findings go back to Maya: `MAYA-UPSTREAM-PATCHES.md`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT):
  - the quant formats
  - the IQ grid tables (generated from its `gguf-py`)
  - `qwen4exp`, the reference Qwen's forward is held to
- [DeepSeek](https://github.com/deepseek-ai) and [Qwen](https://huggingface.co/Qwen): the models.
- [unsloth](https://huggingface.co/unsloth): the quantizations, the MXFP8 Engram table and Qwen's MTP head.
- [huihui-ai](https://huggingface.co/huihui-ai): the abliteration patches.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): GSQ-RCO, the quant this was first built on (still supported).

## Contributing and license

`CONTRIBUTING.md` covers the process and how numbers are measured. `CHANGELOG.md` covers each release. MIT, see
`LICENSE`; third-party material keeps its own license.
