# Changelog

## [0.5.2]: Qwen at a million tokens

Qwen3.8-Flash-Next's context to **1,048,576 tokens** with YaRN - the model card's static YaRN, factor 4 over the
native 262,144 - as a pack of its own, `qwen38-ud-iq3_xxs-1m` (the same files; its rig row's alias `qwen-1m`). The
262K pack is unchanged: its rotation, and every token, 0.5.1's bits.

- **One rotation** (`q4x_rope`, `kernels/rope.h`): plain to the native context; past it llama.cpp's YaRN (ext 1,
  beta 32 / 1, the pair's magnitude 1 + 0.1 ln(factor)) for every rotation - the attention's, the indexer's, the MTP
  head's. Held to llama.cpp's own at positions 0 and 700,000 (`qwen_dump`'s `QWEN_DUMP_YARN`, `QWEN_DUMP_POS0`).
- **The context's own forward**: its rotation in its identity (a 262K state is not a 1M one's); 1M's cache ~13.6 GB,
  ~93 GB of GPU memory in all.
- `q4x_ppl` and `q4x_bench` take a forward's context (`Q4X_PPL_CTX`, `Q4X_BENCH_CTX`).

Measured (2026-11-06; DESIGN.md, "0.5.2: YaRN to a million positions"): a 394,045-token document's fact recalled
355K tokens back, word for word (the prompt read at 414 tok/s). On short text YaRN costs what the card says it may:
perplexity 8K / 16K 2.7256 / 2.2839 (plain 2.7143 / 2.2732; llama.cpp's YaRN on its Q8_0 cache 2.7196 / 2.2911), a 2K
prompt 745 tok/s (810), drafting 34.9 (40.4: the MTP head's acceptance 59 -> 50%) - the 262K pack the default.

## [0.5.1]: Qwen's prompts, faster

Qwen3.8-Flash-Next's prompts **13% (2K) to 28% (64K) faster**, its decode 2.6%, every token the same bits as 0.5.0
(each benchmark's hash, every length). Profiled with rocprofv3, each change A/B'd at low-power against a v0.5.0 build
(DESIGN.md, "0.5.1: decode and prompts, profiled"):

- **The attention's shared-memory bank conflicts.** Its V^T stores sent a wave's 32 lanes to two banks - a 16-way
  conflict on every store; V's lanes now run along the keys (conflict cycles 31x fewer, the kernel's cycles -45%). Its
  cache read 8 values a load and a block ahead in registers.
- **The DeltaNet's**: a lane's q / k half and its pair's on one bank; one float of padding apart.
- A block's output added to the residual in the next mixer's norm launch (`q4x_hc_combine_norm`): ~100 launches a
  token fewer.
- `test_mm_gpu` holds and times Qwen's mixer matrices.

Measured and not kept: the attention's scores once a 64-key round (slower: half the blocks a CU), its dims split over
the waves (moved the bits for 1.4%), the Q8_0 mixers' K split across blocks (slower than unsplit). DeepSeek
unchanged - its code and its outputs.

At `performance` (2026-11-06; README):

| Qwen3.8 UD-IQ3_XXS | 0.5.0 | 0.5.1 | llama.cpp |
| --- | ---: | ---: | ---: |
| Prompt 2K / 8K / 16K | 700 / 591 / 561 | **790 / 714 / 696** | 520 / 489 / 463 |
| Prompt 64K | 501 | **640** | not measured |
| Decode plain / drafted | 23.1 / 39.5 | **23.7 / 40.5** | 26.2 / not measured |

## [0.5.0]: Qwen3.8-Flash-Next

A second model family behind the engine interface: **Qwen3.8-Flash-Next** (`qwen4exp`, ~177B, 512 experts), unsloth's
UD-IQ3_XXS with the model's MTP head drafting, through osmos-serve, osmos-api and rig. Held as DeepSeek is: each block
against llama.cpp's own tensors, perplexity at or below llama.cpp's at 2K and 8K (16K: +0.05%, within the spread of
its own two caches), and a token's bits the same however it is batched, drafted or shared with other conversations.

- **The forward** (`src/q4x/`, `include/osmos/q4x/`): the Gated DeltaNet (36 layers, their states in rings 8 steps
  back), the gated residual streams, the n-gram embedding, the MoE (10 of 512, a step's experts read once for up to
  128 of its tokens), every matmul on the matrix cores with f16 activations at every token count.
- **QSA**, Qwen's sparse attention: a token's top 512 blocks of 4 keys by its indexer's score - llama.cpp's picks
  exactly; the dense attention's bits below 2,052 positions.
- **The cache in Q8_0** (~13 KB a token): the native **262,144-token** context. YaRN to 1M: 0.5.1.
- **Shared steps and drafts**: a batch's conversations one step, each expert read once for all of them; the MTP head
  drafts every lane at once.
- **Text**: Qwen's tokenizer (llama.cpp's qwen35 pre-tokenizer) and chat template (tools, reasoning), osmos-api's
  dialect by family.
- The engine: `oseq_draft_all` (a family's drafter on several sequences at once; DeepSeek's one at a time, as before).
- **The installer's pack** `qwen38-ud-iq3_xxs`: unsloth's 4 shards and the MTP head pinned at a commit with their
  SHA-256s, laid out with the head as its one GGUF (`drafter_kind: mtp`); each pack's rig row its own (packs.json's
  `rig`: Qwen's 262K window with 64K replies and its efforts low / medium / high, alias `qwen`; DeepSeek's rows as
  before).

Measured (2026-11-06, `performance`; README): prompts 2K / 8K / 16K **672 / 588 / 567 tok/s** (llama.cpp 520 / 489 /
463), 64K 500; decode MTP drafted **~40 tok/s**, plain 23.2 (llama.cpp 26.2); the agents bench (4 x 8) **292 s**
(DeepSeek UD-IQ1_M: 446 s).

DeepSeek unchanged: osmos-serve's token streams the v0.4.0 binary's on 16 requests (the five questions greedy and
drafted, a 2K prompt, a continued turn, four at once - `~/bench/v050-streams.log`); its GPU tests pass; UD-IQ1_M
plain 17.1 tok/s, drafted 33.6 (0.3.2's).

## [0.4.0]: one engine interface

osmos-serve now runs a model through one interface (`osmos/run/engine.h`: `Osmos`, `OsmosSeq`, `osmos_*`, `oseq_*`)
instead of DeepSeek's forward by name - the ground for a second model family (Qwen3.8-Flash-Next, designed in
DESIGN.md). `osmos_open` picks the family by the file's `general.architecture` and refuses any other by name.
DeepSeek-V4.1-Flash is the one family behind it, unchanged: osmos-serve's token streams are 0.3.2's on every request
checked (single and `--parallel`, greedy and drafted, conversation reuse), the agents bench level (447.6 s, 7.07 J an
answer token against 442.5 / 448.7 s, 7.07 J). No model, output or speed changes.

## [0.3.2]: the low-power round

Work cut at AMD's low-power profile, each change A/B'd there (interleaved builds, the answers' hashes compared) and
kept in its own commit only if no worse; then the whole release measured at `performance` against 0.3.0.

- The expert kernels (`kernels/q8.h`) ask for a SIMD's 16 waves (`amdgpu_waves_per_eu`): at 5-6 tokens an expert they
  took 98-107 registers and ran at 12, their memory's latency less hidden. 19 experts at 5 tokens, gate+up: 1,143 ->
  798 us; at 6: 1,543 -> 1,004; down at 6: 909 -> 563. 1-4 tokens unchanged; no bit moves. `tools/bench/q8bench` times
  the kernels on a layer's own experts.
- The drafter's greedy pick runs on the GPU (`ops_argmax`): a draft row's 129,280 logits (517 KB) are no longer copied
  to the host and scanned; the same drafts, ties to the lowest token as before. A row with a NaN now ends the drafts.

Measured and not kept (DESIGN.md): the resident set's swaps off (-39% drafted: they keep the text's experts in GPU
memory), `hc_post_dot` reading its weights once a block of tokens (slower: the re-reads hit the cache), the drafter's
head through `mm` (slower at 5 rows).

At `performance`, against 0.3.0 (0.3.1's drafting floors included), UD-IQ1_M / UD-IQ1_S, 256 tokens an answer to the
three bench questions:

| | 0.3.0 | 0.3.2 |
| --- | ---: | ---: |
| UD-IQ1_M drafted | 32.0 tok/s, 4.10 J | **33.6 tok/s, 3.87 J** |
| UD-IQ1_S drafted | 37.8 tok/s | **40.8 tok/s**, 3.35 J |
| plain (M / S) | 17.1 / 19.3 | 17.1 / 19.3 |
| prompts 2K / 8K (M) | 296 / 443 | 293 / 441 |
| agents (4 x 8, M) | 444.6 s, 7.01 J | 442.5 / 448.7 s, 7.07 J |

0.3.0's agents rerun beside 0.3.2's two, the same evening: level. 0.3.0's 433.2 s that afternoon was the box's state
then, ~2.5% faster (the profile switched twice between).

At low-power, UD-IQ1_M: drafted 29.3 -> 30.7 tok/s, 2.50 -> 2.35 J a token; plain 15.9 / 4.45 J -> 15.8 / 4.49 J;
agents 479.0 s / 3.78 J -> 481.3 s / 3.79 J an answer token. The answers hash as 0.3.0's throughout.

## [0.3.1]: drafting tuned for energy

DSpark's drafts are kept while each confidence is at least 0.1 and their product too (`DRAFTER_CONF_MIN` /
`DRAFTER_CUM_MIN` in `osmos/run/drafter.h`, one setting for osmos-serve and the bench), down from Maya's 0.5 / 0.5.
Every draft is verified by the trunk, so the setting changes only the cost: the answers' hashes are the same at every
value swept. Swept at the low-power profile (UD-IQ1_M, drafted, 0.0 to 0.8): 0.1 the least energy a token - lower
floors let more drafts reach a verify pass that runs anyway, and a floor above 0 still lets the drafter stop a hopeless
row early (0.0, no floor, costs as much as 0.5). Confirmed over two interleaved rounds against 0.5: 29.4 -> 30.65
tok/s, 2.51 -> 2.345 J a token (-6.6%). The agents bench, mostly reading, measured level (3.78 -> 3.84 J an answer
token, 479 -> 483 s: within a run's spread). `dbench` takes `DRAFT_CONF`, `DRAFT_CUM` and `DRAFT_MAX` for sweeps.

## [0.3.0]: less energy a token - the experts read straight into memory the GPU reads

The streamed experts are now read with `O_DIRECT` straight into registered host memory the GPU reads (the SSD's DMA
lands them; no CPU copy), instead of a buffered read into the page cache and a CPU copy into GPU memory. The readers
used ~48% of a core each, twelve of them, on an APU whose CPU and GPU share one package power limit: at that limit the
watts they no longer spend go to the GPU. The pool's memory is one registered arena (`mem/host_slots`, `mem/arena`),
each expert part read as its 4 KiB-aligned range with the kernels reading the slice inside it; a filesystem that
refuses `O_DIRECT` refuses loudly. The host's waits for the GPU spin in decode (short, 40 a step; a wake costs ~0.2 ms)
and sleep in a prompt's steps (long), by the step's size: the device is set to blocking sync at a process's start
(an event's own flag is ignored on ROCm 7.15).

No output changes: the bench's answer hashes are 0.2.0's. Measured on UD-IQ1_M, budget auto, at the box's low-power
profile (power-profiles-daemon's power-saver: a stock AMD profile) and at performance, against 0.2.0 the same day;
energy is the APU package's (amdgpu PPT), not the wall's:

| | 0.2.0 low-power | 0.3.0 low-power | 0.2.0 performance | 0.3.0 performance |
| --- | ---: | ---: | ---: | ---: |
| plain decode | 14.8 tok/s, 4.80 J | **15.9, 4.45 J** | 16.4 | **17.1**, 7.71 J |
| drafted decode | 25.8, 2.83 J | **29.3, 2.50 J** | 29.8 | **32.0**, 4.10 J |
| agents (4 x 8) | 527.2 s, 4.12 J | **479.0 s, 3.78 J** | 450.6 s, 7.25 J | **433.2 s, 6.98 J** |

an answer token's energy for the agents. Prompts at performance about level within the bench's spread (2K 281-300 ->
296, 8K 458-466 -> 443 tok/s), 0.31 -> 0.29 J a prompt token. At low-power osmos now drafts as fast as 0.2.0 did at
performance, at ~71 W against ~125. Direct reads take no page-cache hits: the forward waits on more reads (the
agents' 44,058 against 37,598) - a RAM tier for streamed experts is next.

`tools/bench/energy.py` turns `tools/power.py`'s samples and a bench's windows (`dbench` now prints them) into joules a
token; `OSMOS_STATS=1` prints osmos-serve's read waits at exit.

## [0.2.0]: the forward's reads on libevt

The forward's expert reads now run on [libevt](https://github.com/mrsirg97-rgb/libevt), the scheduler osmos-serve's
loop already used. The readers take the read needed soonest from libevt's priority queue - a miss the forward waits on
first, then reads ahead by when the forward will reach them - and a read ahead that becomes waited on moves to the top
in place (libevt's new `queue_update`). A finished read is an event the forward's thread runs. A miss is split across
the readers unless another miss is out; reads side by side are read whole, keeping the file's readahead. This replaces
a hand-built inbox (a mutex, a condition, an array and a "broken" flag) and two FIFO queues with a `hurry` that moved a
read between them - one scheduler for the project, not two. libevt's sources are built into `libosmos.a`, so every
test and sanitizer run covers them; libevt itself gained `cap_submit` and `queue_push` reporting refusals and an
in-place `queue_update` (c70c58c, 32988ad).

No output changes: which experts are read when is outside the arithmetic; the fingerprint tests and the bench's answer
hashes are unchanged. Measured against 0.1.3 on the same box and sizing (UD-IQ1_M, `performance`, budget auto):
decode level (16.4 plain, 29.8 drafted tok/s), prompts faster (292 -> 300 tok/s at 2K, 444 -> 466 at 8K); the agents bench (4 agents x 8 turns) 450.6 s against 0.1.3's 443.8 and
451.8 s, the forward waiting on reads 11.5% less (85.4 s against 96.5 s), 7.25 J an answer token against 7.30.

The first build held reads back in libevt's `cap` until the forward thread pumped - level alone, but 4% slower with
four agents (461 s), and neither a deeper cap nor letting misses past it fixed that: ordering the reads where the
readers take them did. `OSMOS_STATS=1` makes osmos-serve print the forward's read counters at exit, which found it.

Also measured, not built: a forward-chaining rule on routing transitions (expert e at one layer predicting f at the
next), from a recorded trace (`dbench`'s new `ROUTES_OUT`). It catches ~8% of the next layer's misses against the
router guess's 56%, at 20x the reads: 82% of misses are an expert's first pick in the run, which no learned count can
predict. `test_pool_gpu` runs on UD-IQ1_M again (it had skipped since the original files left the test box).

## [0.1.3]: the budget sizing keeps the headroom it promises

The first rig session on 0.1.2 left 0.61 GiB of GPU memory free, under the 1 GiB the sizing promises. Two things
the sizing did not expect: a resident expert takes about 6% more memory than its bytes, so a step could overshoot;
and the memory here moves in steps of about 0.93 GiB, so no budget lands inside a narrow window. The sizing now
measures what a GiB of budget really costs between builds, aims each step with that, keeps a bracket of the largest
budget that left the headroom and the smallest that did not, and never keeps a build short of the headroom when one
that left it is known. On this box the UD-IQ1_S pack sizes to 100.85 GiB with 1.54 GiB free; finding it takes more
builds, so the engine is ready in about 90 s instead of 40. Each sizing build is a line in the log now.

## [0.1.2]: rig thinks at max by default

The rig row the installer writes now carries `"effort": "max"` beside its efforts (`low`, `high`, `max`): rig sends
it as `reasoning_effort`, and osmos-api maps it to DeepSeek's highest reasoning effort (100). rig's `-effort` (or its
session's effort) picks another; osmos-api's own default for a client that names none stays `high` (75).

## [0.1.1]: the installer sets up rig when it is not there

`osmos_install.py --rig` used to write the pack's row into rig's config and assume rig was installed. Now, when no
rig is on PATH or in `~/.local/bin`, it installs it first: `tools/install/packs.json` pins a rig release (its tag and
its `install.sh`'s SHA-256), the installer fetches that exact script, checks the hash before running it - a script
that is not the pinned one is refused and never run - and runs it at the pinned version (rig's script verifies the
binary it downloads against the release's checksums). Then it registers the pack as before. Checked from a fresh
clone on a machine with rig removed: rig v3.5.0 installed, the pack registered as its default, and rig started
osmos-api, sized to the GPU, and answered - a plain prompt and a tool call.

## [0.1.0]: DeepSeek-V4.1-Flash on one Strix Halo, installed in one command, served over an OpenAI API

The first release. osmos runs DeepSeek-V4.1-Flash-1017 (~768B parameters) on a single AMD Strix Halo (Ryzen AI MAX+
395, gfx1151, 128 GB), deterministically: the same input gives the same output, bit for bit, whether it runs alone or
batched with other sequences, at any memory budget, decoded a token at a time or drafted, with an expert on the GPU or
the CPU. Each of those is a test that compares bits.

Two packs, both on unsloth's quants and their shared MXFP8 Engram table, both at a 1M-token context:
`ud-iq1_s` (speed: 18.8 tok/s plain, 37.5 drafted) and `ud-iq1_m` (quality: 16.0 / 28.6). huihui-ai's abliteration
is an opt-in overlay (`--abliterated`), applied exactly at load time. The drafter is DeepSeek's own DSpark, its experts
requantized to IQ2_XS (4.7 GB, down from 8): it and the packs' expert pick profiles are published at
huggingface.co/mrsirg97-rgb/osmos-evo-1017 (MIT) and pinned by commit and SHA-256 in `tools/install/packs.json`.

`tools/install/osmos_install.py ud-iq1_s --serve` checks the machine (the GPU, the GPU memory the kernel lets it map,
the tools, the disk), builds the engine, fetches every file at its pinned commit or uses the ones already on disk
(`--have`), checks every hash, lays out the pack and starts **osmos-api**: an OpenAI-compatible server in C
(`/v1/chat/completions`, streamed or whole, with tool calls and reasoning; `/v1/models`; `/health`), with the model's
own tokenizer and DeepSeek-V4.1's chat encoding byte for byte against their references. `--rig` registers the pack in
[rig](https://github.com/mrsirg97-rgb/rig) and asks before making it the default model.

The engine sizes its expert budget to the GPU it finds (`--budget auto`): it builds itself at its dense weights, then
again with the room left, but a 1 GiB headroom, given to experts, until it fills the machine - on any Halo, context,
number of conversations or drafter. A 1M context costs ~0.85 GiB of attention state: the indexer scores a prompt's
step in chunks, so no buffer grows with the step times the context.

Measured on one Halo against Project Maya 1.1.53, on the same files (README, DESIGN.md): prompts read 26% faster at 2K, 11-12% at 8K, 6% at 128K; drafted decode 6-8%
faster; plain decode a tie; Maya's 4-agent bench in 20% less time on 13% less energy an answer token.

`osmos-serve` keeps several conversations at once on one engine through [libevt](https://github.com/mrsirg97-rgb/libevt),
a submodule. CI builds and tests what runs without a Halo - the host library under AddressSanitizer, UBSan and
ThreadSanitizer, the tokenizer and chat encoding against their fixtures, osmos-api against a scripted engine, the
installer - and compiles the GPU code for gfx1151 in ROCm's image; the GPU and whole-model tests run on the box.
