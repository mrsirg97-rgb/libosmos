# Changelog

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
