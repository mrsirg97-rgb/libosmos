# libosmos

An inference engine for three big mixture-of-experts models on one **AMD Strix Halo** (Ryzen AI MAX+ 395, Radeon
8060S / gfx1151, 128 GB):

- **DeepSeek-V4.1-Flash** (~768B parameters, ~250 GB quantized), more than fits in memory: the experts it needs are
  streamed from the SSD as it runs.
- **Qwen3.8-Flash-Next** (~177B, 512 experts, the Qwen4 architecture's preview), fully resident in GPU memory.
- **Inkling-2-Small** (Thinking Machines; 384 experts a layer, a 158 GB pack), its experts streamed from the SSD beside
  the ~84 GB osmos keeps in memory, drafted by its own multi-token-prediction head.

**The same input gives the same output, bit for bit**, however it runs: alone or batched with other conversations,
at any memory budget, decoded one token at a time or drafted and verified, with an expert computed on the GPU or the
CPU. Each of those is a test that compares bits, not tolerances (`tests/gpu/`), and the kernels are checked against
reference replays with deliberate mutations that must fail them.

C, HIP and a little C++. No Python at runtime.

## Quick start

```sh
python3 tools/install/osmos_install.py osmos-evo-1017 --serve      # DeepSeek; or ud-iq1_s, ud-iq1_m, ud-iq2_xxs
python3 tools/install/osmos_install.py qwen38-ud-iq3_xxs --serve   # Qwen; or qwen38-ud-q4_k_xl
python3 tools/install/osmos_install.py osmos-evo-inkling --serve   # Inkling-2-Small
```

The installer builds the engine, downloads the pack's files at pinned commits, checks every SHA-256, writes the
pack's `osmos.json` and starts **osmos-api**, an OpenAI-compatible server on the first free port from 8091, or
`--port` (`/v1/chat/completions` streamed or whole, with tools and reasoning; `/v1/models`; `/health`). Options:

- `--abliterated` adds huihui-ai's abliteration as an overlay (the DeepSeek and Qwen packs; the stock files stay as
  they are).
- `--npu` puts the NPU to work beside the GPU on prompts (decode the same): it checks for the NPU and
  XRT, builds with them, and fetches the pack's NPU kernels (the four DeepSeek packs and `qwen38-ud-iq3_xxs`).
  Opt-in: its output is its own, the same every run but not the GPU-only bits.
- `--have <dir> ...` uses files you already have (several folders, or the flag again).
- `--doctor` checks the machine as osmos will run on it, and says what to change: the GPU and the memory it may
  map, the driver and ROCm, the power profile (balanced is slower than the profiles osmos is measured at), the RAM,
  `/tmp` as RAM, direct reads on the models' folder, an SSD under it, and a wireless download.
- `--rig` registers the pack in [rig](https://github.com/mrsirg97-rgb/rig) instead of serving it, as
  `deepseek-1017-evo`, `deepseek-1017-ud-s` / `-m` / `-l` (UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS), `qwen3.8-flash-next`
  (`-l`: UD-Q4_K_XL) or `inkling-2-small-evo`.

**Upgrading** (0.9.2): each osmos version is built from its tag beside the others, in `~/.local/share/osmos/<version>`,
and rig's rows move between them by their command alone - their ports and settings stay yours.

```sh
python3 tools/install/osmos_install.py --upgrade            # the newest release, for the rows on the newest version here
python3 tools/install/osmos_install.py --upgrade 0.9.2 --rows deepseek-1017-evo,inkling-2-small-evo   # or --rows all
python3 tools/install/osmos_install.py --rollback           # the last upgrade's rows back where they were
python3 tools/install/osmos_install.py --versions           # the versions built here, the rows on each
python3 tools/install/osmos_install.py --prune [--yes]      # the versions no row runs (removed with --yes)
```

Each version reads its own frozen `osmos.<version>.json`, so a newer installer's settings never break an older row.
A running row takes the new version at its next start.

**Loop signals for agent harnesses** (#47): every answer carries an `osmos.loop` object and an `x-osmos-loop`
header - turns since the last tool call, the most repeated recent call, orphaned calls and results (a client's
history compaction can drop one side), and `stuck`. Nothing the model reads changes; `"osmos": {"fill_orphans":
true}` puts a placeholder after an orphaned call. Context length: the pack's own (DeepSeek 1M, Qwen 512K, Inkling 256K), or
`--ctx` (installer or osmos-api: `512k`, `1m`, ...).

## The packs

| Pack | Files | Context | Pick it for |
| --- | --- | --- | --- |
| `osmos-evo-1017` | 147 GB (UD-IQ1_M's files + a 34 GB delta + a 23 GB patch, made where you install it) + 203 GB Engram table | 1M | **the default**: about halfway from UD-IQ1_M's quality to UD-IQ2_XXS's, faster than either (`osmos-evo` and `osmos-evo-a` still install it) |
| `ud-iq1_s` | 122 GB + 203 GB Engram table | 1M | speed |
| `ud-iq1_m` | 140 GB + 203 GB Engram table | 1M | quality (unsloth's benchmarks 85.55 against 83.53) |
| `ud-iq2_xxs` | 165 GB + 203 GB Engram table | 1M | the closest to the model, the slowest decode |
| `qwen38-ud-iq3_xxs` | 84 GB + 3 GB MTP head | 512K | Qwen, fully in GPU memory |
| `qwen38-ud-q4_k_xl` | 111 GB (82 GB in GPU memory, its 29 GB n-gram table read from the SSD) + 3 GB MTP head | 512K | the larger quant: slower than UD-IQ3_XXS, no better perplexity on our text |
| `osmos-evo-inkling` | 158 GB (unsloth's UD-IQ1_S files + a 55.5 GB delta in two parts, made where you install it) + 1.3 GB drafter | 256K | Inkling-2-Small: UD-IQ1_S with its experts' down projections IQ2_S / IQ3_XXS and its attention and head Q5_K |

## osmos-evo-1017 in numbers

The default DeepSeek pack, on one Strix Halo (osmos 0.9.0, the `performance` profile, nothing else running; greedy,
the pack's 4,096-token steps; a warm room - see "Let your Strix breathe" below):

| | |
| --- | ---: |
| Decode, drafted / plain (64K context, three questions) | **35.0** / 18.9, **30.5** / 18.6, **40.4** / 18.3 tok/s |
| Decode after a 65K / 131K prompt, drafted / plain | **29.4** / 18.2, 14.9 / 13.7 tok/s |
| Prompt, 2,048 / 8,192 / 65,536 / 131,072 tokens | 388 / **630** / 711 / 659 tok/s |
| Facts recalled from up to 985K tokens back | **11 of 11**, every token the model's first choice |

**Recall to the window's edge.** An agent conversation built from our own rig sessions (1,038,695 tokens) with 11 facts
planted in user turns ("the access code for the red locker is RW-6315-KD"), each asked for at the end, plus a locker
never mentioned. Each answer scored teacher-forced - the probability of the whole code:

| Distance back | 985K | 881K | 787K | 687K | 587K | 487K | 388K | 267K | 188K | 78K | 12K | never planted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| osmos-evo-1017 | 0.986 | 0.992 | 0.995 | 0.993 | 0.998 | 0.996 | 0.998 | 0.994 | 0.955 | 0.995 | 0.996 | 0.0000 |
| UD-IQ2_XXS | 0.998 | 0.998 | 0.999 | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.983 | 0.999 | 0.998 | 0.0000 |

A session rarely reads 1M at once - a cached prompt reads only what a turn adds; a full read follows a client's
compaction.

**Qwen3.8 UD-IQ3_XXS at its 512K** (YaRN past its native 262K; the same sessions in its own template, 457,506 tokens),
recalled the same way: the facts 10K-418K back (all 11, every token its first choice; never planted 0.0000):

| Distance back | 418K | 375K | 344K | 298K | 255K | 210K | 159K | 110K | 60K | 20K | 10K | never planted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 UD-IQ3_XXS | 0.988 | 0.991 | 0.998 | 0.995 | 0.995 | 0.994 | 0.964 | 0.994 | 0.982 | 0.997 | 0.993 | 0.0000 |

**How close each DeepSeek pack is to the model itself** - the KL divergence from DeepSeek's own weights (MXFP4
experts, BF16 dense) over the whole vocabulary, against a reference that runs the original 296 GB of experts
(DESIGN.md, "0.6.6"); lower is closer:

| Pack | Wiki text (8,191 tokens) | Agent traces (27,459 assistant tokens) | Top-1 agreement (Wiki) |
| --- | --- | --- | --- |
| `ud-iq2_xxs` | 0.126 ± 0.005 | 0.036 ± 0.001 | 92.4% |
| `osmos-evo-1017` | 0.158 ± 0.006 | 0.042 ± 0.001 | 91.2% |
| `ud-iq1_m` | 0.190 ± 0.007 | 0.048 ± 0.001 | 90.3% |
| `ud-iq1_s` | 0.259 ± 0.008 | 0.066 ± 0.001 | 88.1% |

**What osmos-evo-1017 is.** UD-IQ1_M with the routed experts' down projections of layers 20-39 requantized to Q2_K from
DeepSeek's MXFP4 originals (unsloth's imatrix), the attention and the head requantized to Q5_K by GPTQ (Hessians
from our agent sessions), and the routed experts' gate and up of layers 10-19 at IQ2_XXS (UD-IQ2_XXS's tensors) -
found by probing the one fact the layers-20-39 pack wavered on and bisecting its layers by overlay (DESIGN.md, "0.7.6";
`tools/evo`). It ships as a delta and a patch on UD-IQ1_M
([osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)), assembled at install with every shared byte
cloned on btrfs or XFS, each shard SHA-256 checked.

The UD packs are [unsloth's quants](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) with the lossless
MXFP8 Engram table, our 4.7 GB DSpark drafter and expert pick profile. Qwen is
[unsloth's UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (or UD-Q4_K_XL) and its MTP head. The
abliteration (huihui-ai's) costs 0.2-0.5% perplexity at the same speed.

## osmos-evo-inkling in numbers

Inkling-2-Small on one Strix Halo (osmos 0.9.0, the `performance` profile, nothing else running; osmos-serve at the
pack's 256K and 4,096-token steps, its Q5_K drafter):

| | |
| --- | ---: |
| Decode, drafted / plain (three short questions) | **16.6** / 10.6, 15.1 / 13.3, 18.3 / 17.9 tok/s |
| Decode after a 64K / 248K prompt, drafted / plain | **18.1** / 9.2, **22.0** / 7.3 tok/s (the 248K text one its drafter predicts: 94% accepted) |
| Prompt, 8,192 / 65,536 / 253,952 tokens | **355** / 302 / 185 tok/s |
| KL against Inkling-2-Small in BF16 (Wiki, 10,239 tokens) | 0.197, top-1 85.8% |

Every token is the bits of the same tokens read as a prompt, drafted or not, at any steps. 87 GB of the 143 GB of
experts (61%) stay in memory and the rest stream from the SSD; a full-attention layer's keys are cached Q8 and its
values Q5 (KL +0.001 against f16), so a 256K context's cache is 3.5 GB. Drafted runs vary by about ±8% from run to
run at the same acceptance (which experts are resident when); plain decode and prompts within ~2%.

## Performance

### The machine

| | |
| --- | --- |
| System | AMD Ryzen AI Halo (RAH-001), BIOS 04.01 |
| APU | Ryzen AI MAX+ 395: 16 Zen 5 cores, Radeon 8060S (gfx1151, 40 CUs) |
| Memory | 128 GB LPDDR5X, shared by CPU and GPU; GPU memory raised to ~113 GiB (`ttm pages_limit=29637704`) |
| SSD | Micron 4600 2 TB (MTFDLBA2T0THJ, firmware PBMA002): a PCIe 5.0 drive, linked at PCIe 4.0 x4 here; btrfs |
| Software | Linux 7.2.7, ROCm with HIP 7.15, gcc 14.2 |
| Placement | raised off its surface, open air beneath it |

DeepSeek streams its experts from the SSD, so the drive is part of its numbers: a prompt reads at ~7.1 GB/s (near
the PCIe 4.0 x4 ceiling); decode reads experts a few MB at a time (`O_DIRECT`), where random-read latency counts.

**NOTE - let your Strix breathe.** It cools through its base. On a warm table beside another machine, a sustained
DeepSeek run took the GPU to 102 °C and its clock down to 925 MHz, and drafted decode after a 64K prompt read 17 tok/s
against 29 raised in open air (the same code, the same tokens). These numbers are from a warm room: under sustained
load the GPU still runs 95-100 °C with its clock near 2.6 GHz.

The comparisons below are 0.9.0's, at the `performance` profile, every run a fresh process, greedy, one run each;
DeepSeek at 64K context in the packs' 4,096-token steps.

### DeepSeek against Maya

[Project Maya](https://github.com/mw00/project-maya) v1.1.65, built and calibrated on this machine (`--calibrate`: its
own tensor-parallel and CPU-thread settings), run with its own pack and its own DSpark, reading the same model files.
Both engines at 64K context; Maya's own `--bench`, osmos's `dbench` on the same three questions and prompts.

| UD-IQ2_XXS | osmos | Maya |
| --- | ---: | ---: |
| Prompt, 2,048 tokens | **330** | 220 |
| Prompt, 8,192 tokens | **580** | 440 |
| Decode, drafted (three questions) | **29.0 / 22.1 / 27.4** | 17.6 / 10.9 / 11.0 |
| Decode, plain | **14.7 / 14.4 / 13.7** | 12.6 / 10.8 / 9.8 |

osmos reads prompts 1.3-1.5x as fast as Maya and decodes 30% faster plain and 2.0x drafted: osmos's one cache keeps
the experts decode comes back to.

**Agents.** Maya's agent bench: 4 agents at once, 8 turns each, about 99K new prompt tokens against 6-8K answer
tokens, at 131K context with drafting on (DeepSeek its DSpark, Qwen its MTP head). Power is the APU package's (CPU +
GPU + NPU, not the wall).

| | osmos-evo-1017 | Qwen UD-IQ3_XXS |
| --- | ---: | ---: |
| Wall time, all 32 turns | 462.8 s | **185.2 s** |
| First token, p50 / p95 | 8.46 / 53.6 s | **2.06 / 25.8 s** |
| Turn done, p50 / p95 | 49.2 / 92.5 s | **15.3 / 45.0 s** |
| Package power, mean | **114.6 W** | 123.7 W |
| Energy per answer token | 7.10 J | **4.26 J** |

Every turn answered, every later turn continued its conversation in place. osmos sizes itself to leave 8 GiB of RAM
free beside the model (`"fill_gib"` in `osmos.json`, osmos-serve `--fill`: the output the same at any value).

### Qwen3.8 against llama.cpp

llama.cpp 7d7f028, with HIP, all layers on the GPU and flash attention, at its faster batch for each length (2,048
or 4,096). osmos reads prompts in 4,096-token steps, the way its pack does (`q4x_bench`).

| UD-IQ3_XXS | osmos | llama.cpp |
| --- | ---: | ---: |
| Prompt, 2,048 tokens | **1,012** | 574 |
| Prompt, 8,192 tokens | **952** | 549 |
| Prompt, 16,384 tokens | **925** | 515 |
| Prompt, 65,536 tokens | **865** | 374 |
| Decode, plain | **29.1** | 26.5 |
| Decode at 16K context | **28.5** | 24.0 |
| Decode, drafted by the MTP head | **50.7** | not measured |

osmos reads Qwen's prompts 1.7-2.3x as fast as llama.cpp, the more the longer the prompt. Its decode is 10% faster
plain and 19% at 16K of context.

### Qwen3.8 against Strata

[Strata](https://github.com/Niko1221/Strata) 0.2.13.1 with its signed engine 0.1.59 for this chip, launched with its
own defaults (`strata launch ... --context 128k --vision off`), against osmos-api - both on the same files, Unsloth's.
The same requests go to each server's OpenAI API: greedy, thinking off, prompts of ~2K / 4K / 16K tokens asked for a
summary, and three questions answered in 256 tokens, one round. Energy is the APU package's for the whole run.

| UD-IQ3_XXS | Prompt 2K | 4K | 16K | Code answer | Explanation | Web page | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| osmos, no drafter | **893** | **882** | **894** | 29.1 | 29.1 | 29.2 | **2.14 Wh** |
| Strata, no drafter | 518 | 667 | 744 | **29.4** | **29.5** | **30.7** | 2.45 Wh |
| osmos, DFlash2 | **898** | **884** | **891** | **83.5** | **63.2** | **75.4** | **1.44 Wh** |
| Strata, DFlash2 | 517 | 666 | 742 | 71.4 | 54.3 | 59.9 | 1.85 Wh |
| osmos, MTP head (its default) | 895 | 880 | 890 | 69.7 | 60.0 | 68.6 | 1.46 Wh |

osmos reads prompts 20-72% faster; without a drafter the answers are level (Strata 1-5% ahead) on 13% less energy,
with DFlash2 on both osmos answers 16-26% faster on 22% less energy. Setups differ: Strata packs DFlash2 as Q4_0
(osmos Q8_0); its opt-in fast set (documented as changing the output's bits) was not used, its `--drafting mtp` ran at
no-drafter speed here, and its DeepSeek engine is not shipped for Strix Halo on Linux.

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
cfg() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" ~/Models/osmos/osmos-evo-1017/osmos.json $1; }
export MODEL=$(cfg model) PICKS=$(cfg picks) DSPARK=$(cfg drafter) MAX_CTX=65536 NO_GUESS=1   # the installed pack's own paths
GREEDY=1 bin/dbench auto 4096 256 warmup.txt q1.txt q2.txt q3.txt                               # plain
GREEDY=1 DRAFT=$DSPARK bin/dbench auto 4096 256 warmup.txt q1.txt q2.txt q3.txt                 # drafted
bin/dbench auto 4096 8 warmup.txt p2048.txt p8192.txt                                         # prompts
MAX_CTX=139264 GREEDY=1 bin/dbench auto 4096 256 warmup.txt p65536.txt p131072.txt           # decode after a long prompt
NPU=<pack>/npu bin/dbench auto 4096 8 warmup.txt p2048.txt p8192.txt                          # prompts, the NPU beside
python tools/power.py --csv agents.csv -- python tools/serve/agent_bench.py --only osmos --model $MODEL \
  --picks $PICKS --dspark $DSPARK --budget auto --max-tokens 4096 --osmos-flag=--no-guess    # agents (+ --osmos-flag=--npu
                                                                                              #  --osmos-flag=<pack>/npu)
# Qwen (token ids from its tokenizer: oracle/qwen_dump.cpp; llama.cpp's perplexity: qwen_dump -ppl)
export Q=~/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf Q4X_BENCH_STEP=4096
Q4X_BENCH_CTX=17408 bin/q4x_bench $Q ids.txt 16384 64                                         # prompt, then decode
NPU=<pack>/npu Q4X_BENCH_CTX=17408 bin/q4x_bench $Q ids.txt 16384 64                          # the NPU beside
bin/q4x_bench $Q ids.txt 512 128 ~/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf   # drafted (or a DFlash2 GGUF)
python tools/power.py --csv qagents.csv -- python tools/serve/agent_bench.py --only osmos --model $Q \
  --dspark <the MTP GGUF> --budget auto --max-tokens 4096 --prompts <Qwen-tokenized prompts>      # agents
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
- [DeepSeek](https://github.com/deepseek-ai), [Qwen](https://huggingface.co/Qwen) and
  [Thinking Machines](https://huggingface.co/thinkingmachines) (Inkling-2-Small, Apache 2.0): the models.
- [unsloth](https://huggingface.co/unsloth): the quantizations (Inkling's UD-IQ1_S and its importance matrix among
  them), the MXFP8 Engram table and Qwen's MTP head.
- [eaddario/imatrix-calibration](https://huggingface.co/datasets/eaddario/imatrix-calibration) (MIT): the text
  Inkling's picks are counted on.
- [huihui-ai](https://huggingface.co/huihui-ai): the abliteration patches.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): GSQ-RCO, the quant this was first built on (still supported).

## Contributing and license

`CONTRIBUTING.md` covers the process and how numbers are measured. `CHANGELOG.md` covers each release. MIT, see
`LICENSE`; third-party material keeps its own license.
