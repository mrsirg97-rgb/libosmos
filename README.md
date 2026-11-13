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
python3 tools/install/osmos_install.py osmos-evo-a --serve         # DeepSeek; or ud-iq1_s, ud-iq1_m, ud-iq2_xxs
python3 tools/install/osmos_install.py qwen38-ud-iq3_xxs --serve   # Qwen; or qwen38-ud-q4_k_xl
```

The installer builds the engine, downloads the pack's files at pinned commits, checks every SHA-256, writes the
pack's `osmos.json` and starts **osmos-api**, an OpenAI-compatible server on port 8091 (`/v1/chat/completions`
streamed or whole, with tools and reasoning; `/v1/models`; `/health`). Options:

- `--abliterated` adds huihui-ai's abliteration as an overlay (every pack; the stock files stay as they are).
- `--npu` puts the NPU to work beside the GPU on prompts (1-5% faster; decode the same): it checks for the NPU and
  XRT, builds with them, and fetches the pack's NPU kernels (the four DeepSeek packs and `qwen38-ud-iq3_xxs`).
  Opt-in: its output is its own, the same every run but not the GPU-only bits.
- `--have <dir>` uses files you already have.
- `--doctor` checks the machine as osmos will run on it, and says what to change: the GPU and the memory it may
  map, the driver and ROCm, the power profile (balanced is slower than the profiles osmos is measured at), the RAM,
  `/tmp` as RAM, direct reads on the models' folder, an SSD under it, and a wireless download.
- `--rig` registers the pack in [rig](https://github.com/mrsirg97-rgb/rig) instead of serving it, as
  `deepseek-1017-evo-a`, `deepseek-1017-ud-s` / `-m` / `-l` (UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS) or `qwen3.8-flash-next`
  (`-l`: UD-Q4_K_XL).

**Loop signals for agent harnesses** (#47). Every answer reports what its request's history says about the agent
sending it, computed from the messages alone: an `osmos.loop` object beside `usage` (whole answers and a stream's
last chunk) and an `x-osmos-loop` header - assistant turns since its last tool call, its most repeated recent call,
tool calls with no result and results with no call (a client's history compaction can drop one side), and `stuck`
past three turns without a call or three of one call. Nothing the model reads changes. An orphaned call's result is
left out as the client sent it unless asked: `"osmos": {"fill_orphans": true}` in a request (or `"fill_orphans": true`
in the pack's `osmos.json`) puts a placeholder saying the result was dropped in its place.

## The packs

| Pack | Files | Context | Pick it for |
| --- | --- | --- | --- |
| `osmos-evo-a` | 141 GB (UD-IQ1_M's files + a 34 GB delta, made where you install it) + 203 GB Engram table | 1M | **the default**: UD-IQ1_M's quality at about UD-IQ1_S's speed |
| `ud-iq1_s` | 122 GB + 203 GB Engram table | 1M | speed |
| `ud-iq1_m` | 140 GB + 203 GB Engram table | 1M | quality (unsloth's benchmarks 85.55 against 83.53) |
| `ud-iq2_xxs` | 165 GB + 203 GB Engram table | 1M | the closest to the model, the slowest decode |
| `qwen38-ud-iq3_xxs` | 84 GB + 3 GB MTP head | 512K | Qwen, fully in GPU memory |
| `qwen38-ud-q4_k_xl` | 111 GB (82 GB in GPU memory, its 29 GB n-gram table read from the SSD) + 3 GB MTP head | 512K | the larger quant, if you want it: 10-18% slower than UD-IQ3_XXS on our agent replay, no better perplexity on our text |

**How close each DeepSeek pack is to the model itself.** The KL divergence from DeepSeek's own weights (the experts
as deepseek-ai stores them, MXFP4; the dense layers BF16), over the whole vocabulary, a token at a time: lower is
closer, 0 is the model. Measured on this box against a reference that runs the original 296 GB of experts from the
SSD (DESIGN.md, "0.6.6"):

| Pack | Wiki text (8,191 tokens) | Agent traces (27,459 assistant tokens) | Top-1 agreement (Wiki) |
| --- | --- | --- | --- |
| `ud-iq2_xxs` | 0.126 ± 0.005 | 0.036 ± 0.001 | 92.4% |
| `osmos-evo-a` | 0.175 ± 0.006 | 0.048 ± 0.001 | 90.5% |
| `ud-iq1_m` | 0.190 ± 0.007 | 0.048 ± 0.001 | 90.3% |
| `ud-iq1_s` | 0.259 ± 0.008 | 0.066 ± 0.001 | 88.1% |

The agent traces are rig sessions of our own (local, never published), the assistant's tokens alone. Perplexity
does not rank these packs reliably: on those traces UD-IQ1_M's (2.477) is below UD-IQ2_XXS's (2.503), though it is
further from the model. `osmos_install.py --list` shows each pack's numbers.

**osmos-evo-A, the default DeepSeek pack.** osmos's own, built on UD-IQ1_M: the routed experts' down projections of
layers 20-39 requantized to Q2_K from DeepSeek's MXFP4 originals (with unsloth's imatrix), and the attention and the
head requantized to Q5_K by GPTQ (from UD-IQ1_M's own values, its Hessians from our agent sessions); every other
tensor UD-IQ1_M's as it is. Against UD-IQ1_M it is level on agent traces (0.0485 against 0.0479,
+0.0005 ± 0.0003), closer on Wiki text (0.175 against 0.190, -0.014 ± 0.004), and it decodes 8% faster plain and 15%
faster drafted, within 2-6% of UD-IQ1_S (`low-power`, interleaved: 17.9 / 29.7 tok/s against UD-IQ1_S's 18.3 / 31.7
and UD-IQ1_M's 16.6 / 25.8). At 64K of context it decodes 16.4 tok/s, the fastest of the three. Its prompts are 5-9%
slower than the UD packs' (the larger dense). It ships as a 34 GB delta on unsloth's UD-IQ1_M
([osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)): the installer makes the pack from UD-IQ1_M's
files and the delta (`tools/convert/gguf_assemble.py`), every byte they share cloned on btrfs or XFS - so it adds 34
GB to the download and almost nothing to the disk beside UD-IQ1_M - and checks each shard's SHA-256. The abliteration
is huihui-ai's UD-IQ1_M patch, which replaces the same 18 attention tensors in evo-A.

The UD packs are [unsloth's quants](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) with the
lossless MXFP8 Engram table, plus our 4.7 GB DSpark drafter and expert pick profile
([osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)). Qwen is
[unsloth's UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (or UD-Q4_K_XL) and its MTP head. The abliteration
costs 0.2-0.5% perplexity (UD-IQ1_M 2.458 → 2.470, Qwen 2.714 → 2.723) at the same speed.

## Performance

### The machine

Every number here comes from one box:

| | |
| --- | --- |
| System | AMD Ryzen AI Halo (RAH-001), BIOS 04.01 |
| APU | Ryzen AI MAX+ 395: 16 Zen 5 cores, Radeon 8060S (gfx1151, 40 CUs) |
| Memory | 128 GB LPDDR5X, shared by CPU and GPU; GPU memory raised to ~113 GiB (`ttm pages_limit=29637704`) |
| SSD | Micron 4600 2 TB (MTFDLBA2T0THJ, firmware PBMA002): a PCIe 5.0 drive, linked at PCIe 4.0 x4 here; btrfs |
| Software | Linux 7.2.7, ROCm with HIP 7.15, gcc 14.2 |

DeepSeek streams its experts from the SSD, so the drive is part of DeepSeek's numbers. A prompt reads at about
7.1 GB/s here, near the ceiling of the PCIe 4.0 x4 link, so a good Gen4 NVMe should come close on prompts. Decode
reads experts as the router picks them, a few MB at a time (`O_DIRECT`): there the drive's random-read latency
counts, and a slower or DRAM-less drive shows first. The "SSD read a token" rows are the device's own counter over
each run's decode.

### How these were measured

0.7.0 at AMD's `performance` profile, 2026-11-13, nothing else running, every run a fresh process, greedy decoding.
DeepSeek runs at 64K context, Maya's three questions at 256 tokens each (the mean of the three) and its 2K and 8K
prompts, in the packs' 4,096-token steps. Prompts were run twice, GPU alone and with the NPU interleaved; the ranges
are the two runs'. Tokens a second. The scripts and logs are listed in `DESIGN.md` ("0.7.0").

**The NPU column** is osmos with `--npu` (0.7.0, opt-in): Strix Halo's NPU takes a share of each prompt step's expert
work beside the GPU. It only touches prompts - decode is the GPU's either way - and its output is its own (its sums
round differently from the GPU's; the same bits every run).

### osmos-evo-A, the default DeepSeek pack

0.7.2's numbers (its faster IQ1_S prompt decode), the same way, 2026-11-13; the other tables are 0.7.0's.

| | GPU | with the NPU |
| --- | ---: | ---: |
| Prompt, 2,048 tokens | 384-385 | **386-389** |
| Prompt, 8,192 tokens | 620 | **644-647** |
| Decode, drafted | 35.6 | |
| Decode, plain | 18.9 | |

### DeepSeek against Maya

[Project Maya](https://github.com/mw00/project-maya) 1.1.61, rebuilt, run with its own pack, its own 8 GB DSpark,
and its CPU lane and expert profile calibrated on this machine, reading the same model files. Both engines at 64K
context, Maya's own `--bench`.

| | UD-IQ1_M osmos | with the NPU | Maya | UD-IQ2_XXS osmos | with the NPU | Maya |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt, 2,048 tokens | **381-389** | 385-388 | 256-266 | **326-331** | 324-328 | 284-286 |
| Prompt, 8,192 tokens | 626-627 | **640-641** | 459-460 | 568-570 | **577-578** | 477-484 |
| Decode, drafted | **35.1** | | 27.9 | **26.0** | | 13.4 |
| Decode, plain | **17.6** | | 16.5 | **14.2** | | 11.1 |

osmos reads 8K prompts 36% (UD-IQ1_M) and 18% (UD-IQ2_XXS) faster than Maya - the NPU adds 1-2% more at 8K, nothing
at 2K - and decodes 7% and 28% faster plain, 26% and 1.9x drafted. The bigger the quant, the wider the gap: that's
IO - osmos's one cache keeps the experts decode comes back to, so it reads fewer SSD bytes a token.

**Agents.** Maya's agent bench: 4 agents at once, 8 turns each, about 99K new prompt tokens against 8K answer tokens,
UD-IQ1_M at 131K context with drafting on. Power is the APU package's (CPU + GPU + NPU, not the wall).

| | osmos | with the NPU | Maya 1.1.61 |
| --- | ---: | ---: | ---: |
| Wall time, all 32 turns | **384.8 s** | 399.3 s | 504.9 s |
| First token, p50 / p95 | 5.20 / 52.2 s | **5.17 / 51.3 s** | 5.65 / 71.6 s |
| Turn done, p50 / p95 | **37.3 / 82.0 s** | 42.2 / 81.6 s | 52.7 / 115.0 s |
| Package power, mean | **118.8 W** | 120.9 W | 122.4 W |
| Energy per answer token | 6.27 J | **6.25 J** | 8.26 J |

osmos finishes the same 32 turns in 24% less time than Maya and 24% less energy per answer token. The NPU doesn't help
here: an agent's turns are mostly short additions to a long conversation, where its share is small (one run; its
answers came out 1.6% longer, its rounding its own).

osmos sizes itself to leave 8 GiB of RAM to the rest of the system: room for an editor, a browser, language servers
and the agent harness beside the model. A box with less beside it can give the cache more: `"fill_gib": 4` in the
pack's `osmos.json` (osmos-serve `--fill 4`) keeps 4 GiB free instead, 0 keeps none. The output is the same at every
value; only how much of the model stays in memory changes.

### Qwen3.8 against llama.cpp

llama.cpp 7d7f028, with HIP, all layers on the GPU and flash attention, at its faster batch for each length (2,048
or 4,096). osmos reads prompts in 4,096-token steps, the way its pack does (`q4x_bench`). UD-Q4_K_XL has no NPU kernels.

| | UD-IQ3_XXS osmos | with the NPU | llama.cpp | UD-Q4_K_XL osmos | llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prompt, 2,048 tokens | 1,002-1,011 | **1,017-1,018** | 578 | **911-973** | 575 |
| Prompt, 8,192 tokens | 940-943 | **964-972** | 557 | **908** | 560 |
| Prompt, 16,384 tokens | 907-908 | **935-942** | 520 | **874-878** | 533 |
| Prompt, 65,536 tokens | 802-803 | **839-841** | 377 | **776-779** | 378 |
| Decode, plain | **29.0** | | 26.4 | **25.4** | 22.6 |
| Decode at 16K context | **27.9** | | 24.2 | **24.1** | 20.2 |
| Decode, drafted by the MTP head | **50.3** | | not measured | **41.4** | not measured |

osmos reads Qwen's prompts 1.6-2.1x as fast as llama.cpp, and the NPU adds 1-5% to that, more the longer the
prompt. Its decode is 10-12% faster plain and 15-19% at 16K of context.

### Qwen3.8 against Strata

[Strata](https://github.com/Niko1221/Strata) 0.2.13.1 with its signed engine 0.1.59 for this chip, launched with its
own defaults (`strata launch ... --context 128k --vision off`), against osmos-api - both on the same files, Unsloth's.
The same requests go to each server's OpenAI API: greedy, thinking off, prompts of ~2K / 4K / 16K tokens asked for a
summary, and three questions answered in 256 tokens, one round. Energy is the APU package's for the whole run.

| UD-IQ3_XXS | Prompt 2K | 4K | 16K | Code answer | Explanation | Web page | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| osmos, no drafter | 888 | 884 | 866 | 29.1 | 29.1 | 29.1 | 2.12 Wh |
| osmos, no drafter, with the NPU | **894** | **910** | **894** | 29.0 | 29.0 | 29.1 | 2.16 Wh |
| Strata, no drafter | 496 | 651 | 747 | **29.2** | **29.5** | **31.0** | 2.45 Wh |
| osmos, DFlash2 | 860 | 869 | 858 | **83.5** | **63.4** | **75.7** | **1.44 Wh** |
| Strata, DFlash2 | 517 | 669 | 743 | 69.1 | 54.5 | 59.7 | 1.85 Wh |
| osmos, MTP head (its default) | 868 | 889 | 876 | 70.2 | 60.1 | 68.4 | 1.46 Wh |

| UD-Q4_K_XL | Prompt 2K | 4K | 16K | Code answer | Explanation | Web page | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| osmos, no drafter | **862** | **871** | **860** | **25.2** | **25.3** | **25.3** | **2.23 Wh** |
| Strata, no drafter | 218 | 405 | 459 | 20.8 | 20.3 | 20.2 | 3.52 Wh |
| osmos, DFlash2 | **830** | **858** | **846** | **71.7** | **53.2** | **57.7** | **1.52 Wh** |
| Strata, DFlash2 | 255 | 279 | 439 | 42.8 | 40.9 | 40.9 | 2.88 Wh |
| osmos, MTP head (its default) | 807 | 850 | 848 | 56.1 | 49.6 | 54.2 | 1.60 Wh |

On UD-IQ3_XXS osmos reads prompts 16-80% faster (the gap widest on short prompts); without a drafter the answers are
level, Strata 0-7% ahead; with DFlash2 on both, osmos's answers are 16-27% faster, and the same work takes about 20%
less energy. On UD-Q4_K_XL osmos reads prompts 1.9-4x as fast and answers 21-25% faster without a drafter, 30-68%
with DFlash2. The two set up differently: Strata sizes its expert cache from the RAM free when it starts and streams
the rest, osmos holds UD-IQ3_XXS whole in memory (UD-Q4_K_XL all but its 29 GB n-gram table, read from the SSD);
Strata packs the DFlash2 drafter as Q4_0, osmos as Q8_0. Strata's opt-in fast set, documented as changing the
output's bits, was not used. Strata's `--drafting mtp` ran at its no-drafter speed in this version, so its MTP rows
are left out. Strata's DeepSeek engine is not shipped for Strix Halo on Linux, so there is no same-file DeepSeek
comparison.

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
- **The cache** is Q8_0: 6.8 GB for 512K tokens.
- **QSA**, Qwen's sparse attention, attends to a token's top 512 blocks of 4 keys.
- **The MTP head** drafts up to 5 tokens. A pack can name DFlash2 instead (a block of 7 drafts from one pass of a
  small drafter fed the model's own layers): ahead on chat, behind on real agent work, so an option.
- **The n-gram table** (29 GB) can stay on the SSD, a step's rows read as it runs: the same bits, and how
  UD-Q4_K_XL fits.
- **Rollback.** The 36 Gated DeltaNet layers keep 8 steps of state, so a rejected draft rolls back.
- **The abliteration** is an overlay: 96 tensors loaded in place of the model's own.

`DESIGN.md` has every decision and measurement, including what was tried and didn't help.

## Requirements

- A Ryzen AI MAX+ 395 (gfx1151) with 128 GB, and Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14).
  The optional CPU lane needs AVX-512 VNNI and VBMI.
- GPU memory raised to ~113 GiB: `options ttm pages_limit=29637704` in `/etc/modprobe.d/ttm.conf`. This is a system
  setting, so it's yours to make.
- The model on an NVMe SSD, on a filesystem that takes `O_DIRECT` reads (ext4, btrfs and xfs do; osmos refuses one that
  doesn't, and names it). The faster the SSD's random reads, the faster DeepSeek decodes. The numbers here are from a
  Micron 4600 on a PCIe 4.0 x4 link ("The machine", above).
- For `--npu`: the XDNA driver (`amdxdna`, `/dev/accel/accel0`) and XRT with its headers (`libxrt-dev`). Building the
  kernels yourself (`tools/npu/build.sh`) needs mlir-aie's IRON; the installer fetches them prebuilt.

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
powerprofilesctl set performance
# DeepSeek (token ids from Maya's tokenizer: tools/bench/prompts.py)
export MODEL=~/Models/osmos/ud-iq1_m/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf
export PICKS=~/Models/osmos/ud-iq1_m/osmos-picks.bin MAX_CTX=65536 NO_GUESS=1 DSPARK=~/Models/osmos/ud-iq1_m/dspark
GREEDY=1 bin/dbench auto 4096 256 warmup.txt q1.txt q2.txt q3.txt                               # plain
GREEDY=1 DRAFT=$DSPARK bin/dbench auto 4096 256 warmup.txt q1.txt q2.txt q3.txt                 # drafted
bin/dbench auto 4096 8 warmup.txt p2048.txt p8192.txt                                         # prompts
NPU=<pack>/npu bin/dbench auto 4096 8 warmup.txt p2048.txt p8192.txt                          # prompts, the NPU beside
python tools/power.py --csv agents.csv -- python tools/serve/agent_bench.py --only osmos --model $MODEL \
  --picks $PICKS --dspark $DSPARK --budget auto --max-tokens 4096 --osmos-flag=--no-guess    # agents (+ --osmos-flag=--npu
                                                                                              #  --osmos-flag=<pack>/npu)
# Qwen (token ids from its tokenizer: oracle/qwen_dump.cpp; llama.cpp's perplexity: qwen_dump -ppl)
export Q=~/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf Q4X_BENCH_STEP=4096
Q4X_BENCH_CTX=17408 bin/q4x_bench $Q ids.txt 16384 64                                         # prompt, then decode
NPU=<pack>/npu Q4X_BENCH_CTX=17408 bin/q4x_bench $Q ids.txt 16384 64                          # the NPU beside
bin/q4x_bench $Q ids.txt 512 128 ~/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf   # drafted (or a DFlash2 GGUF)
python3 tools/bench/rig_replay.py ~/.rig out/ --shard $Q --gguf-py <llama.cpp>/gguf-py \
  --tokenize <llama.cpp>/build-hip/bin/llama-tokenize                                          # prompts from your rig sessions
bin/q4x_ppl $Q ids.txt 8192 2048                                                              # perplexity
```

The bench tools are built with `hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/<tool>.hip -Llib -losmos_hip
-losmos -lhsa-runtime64` (with the NPU: `make NPU=1` first, and `-lxrt_coreutil`). Run each configuration in a fresh
process, and interleave them. Every answer prints a hash of its tokens: the same hash in every run of a configuration
is the guarantee (the NPU's is its own).

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
