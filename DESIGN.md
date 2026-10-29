# libsubstrate: design

a runtime for one machine, the AMD Ryzen AI Halo (Ryzen AI MAX+ 395, Radeon 8060S, 128 GB unified memory), built
first for DeepSeek-V4.1-Flash-1017. status: **design** (AGENTS.md workflow step 1). nothing here is implemented.

every number below is measured on this box (`blade`) on 2026-10-28 unless it says otherwise. a claim without a number
is marked **unmeasured**.

## the bar

milestone 1 is done when, on this Halo:

- one command serves DeepSeek-V4.1-Flash-1017 (the GSQ-RCO IQ2_XXS GGUF, `~/models/ds1017`) on the OpenAI API
  (`/v1/chat/completions`, streaming, tools), the way rig already launches models (`~/.rig/models.json`: `command`,
  `port`, `places`);
- libevt schedules prompt reading against decode;
- it **beats Maya on the same box**, measured with Maya's own harness: `tools/ds4/agent_bench.py` (#557/#587), its
  seeded agent workload (4 agents x 8 turns, seed 1). the `--trace` replay of a rig `.served` log is a second check
  when one is on this box, not part of the bar: on #587 the seeded workload moved p95 first token 2.2x and the trace
  was a tie (most real turns fit in one piece) - they answer different questions. **the reference is Maya v1.1.36 + #643 + #587**, which is what Maya will
  be by the time this ships, not v1.1.35.
- token for token the same greedy answers as the reference where the arithmetic is the same, and perplexity within
  the reference's run-to-run spread where it is not (the #356 / #643 protocol).

## the claim this is for

the Halo (one APU, **$4,000**, against **~$15,000** for the operator's 3-GPU workstation - 27% of its price) running a ~140 GB-weight model for local agents,
with no CUDA - open drivers, ROCm, one binary - and a scheduler derived from first principles (libevt). the 3-GPU
box is ~3x this one on raw decode and prompt speed, so the claim is not raw speed. what it can honestly be, each
measured:

- **agent latency**: first token and turn done on `agent_bench`, where scheduling moves the number most.
- **per dollar**: agent turns an hour per dollar of hardware (prices: the operator's, $4,000 and ~$15,000).
  a first estimate, **provisional**: the workstation's 190.5 s wall on agent_bench 4x8 (#587, another build and day)
  against this box's 514.8 s is 2.7x the speed at 3.75x the price, so ~1.4x the agent work per dollar here, before
  any of this runtime. the claim needs both boxes on the same harness, seed and build, power measured on both.
- **per watt**: agent turns per Wh, package power (`tools/power.py`) and a wall meter. the workstation: ~700-800 W
  at the wall under heavy inference (the operator's estimate, not a log), so its 190.5 s agent_bench run is ~40 Wh at
  ~750 W; this box's: the powered bar run, **unmeasured** so far.
- **fit**: the model on one 128 GB APU, where a single consumer NVIDIA card cannot hold it.

### against the DGX Spark (the same class: unified memory, ~128 GB)

maya's docs measure the Spark (GB10, 121 GB) on **GLM-5.3** only - no DeepSeek number exists there. GLM to GLM,
`./maya.sh --bench`, different quants (Spark: Maya-S 96.5 GB; this box: unsloth UD-IQ2_XXS 101 GB):

| | DGX Spark (`docs/DGX_SPARK.md`) | this Halo (2026-10-28) |
| --- | ---: | ---: |
| decode, DFlash2 | 21.1 tok/s | **29.8** (+41%) |
| prompt, 2K / 8K tokens | **597-614 / 730-732** | 294-326 / 306-332 |
| price (the operator's) | $7,000 | $4,000 (57%) |
| memory bandwidth (rated) | 273 GB/s | 256 GB/s |
| dense FP16 / int8 compute | ~125-250 TOPS (**an estimate** from NVIDIA's 1 PFLOP sparse FP4) | ~59 |

decode wins with less bandwidth; the prompt trails about as far as the compute does. the prompt is the gap this
runtime's kernels go after first (GLM's experts at ~40% of int8 peak; the attention's kernels barely measured).

### it carries its own tuning

this box's numbers came from tuning most owners will never do: `ttm`'s pages_limit, the platform profile, hipBLASLt
tables retuned for the installed build (the shipped one was rejected for every shape), the grid sizes, the SSD's link,
a bad core taken offline. so the binary carries what it can: it measures its own GEMM picks on the first start, sizes
memory from what the machine reports (a 128 GB or a bigger unit, the same binary), and for what it cannot change -
the `ttm` limit, `RLIMIT_MEMLOCK`, the platform profile - it says so at start, with the measured cost and the exact
command. one command means the tuned numbers by default.

## the facts the design rests on

### the machine

| | measured | source |
| --- | --- | --- |
| GPU memory (GTT) | 107.6 GiB, set by `ttm pages_limit=28212259`; the operator keeps ~16-20 GB for the desktop and docker | `/etc/modprobe.d/ttm.conf` |
| RAM seen by Linux | 125 GB | `free -g` |
| GPU memory bandwidth, decode | ~200-215 GB/s effective (DeepSeek decode reads ~7.5 GB a token) | maya-ds4 decode, this session |
| SSD | Micron 4600 (Gen5), **linked at Gen4 x4**: 6.7 GB/s direct 16 MiB reads, 4.9 at 1 MiB, 3.3 at 128 KiB | `dd iflag=direct` |
| CPU | 16 cores / 32 threads, **core 1 offline** (corrected MCE errors at idle; under investigation) | `strix-perf.service` |
| NPU | XDNA 2, `amdxdna` loaded, `/dev/accel/accel0`, XRT 2.21.75; **unusable as the user**: `RLIMIT_MEMLOCK` is 8 MB, XRT maps 64 MB | `xrt-smi examine` |

### the model on this box

- 15,744 routed experts, 11,705 resident in GPU memory (104.6 GB), 4,039 (35.1 GB) read in place from the GGUF files.
  the model is ~140 GB of experts against 125 GB of RAM: **the non-resident experts cannot all stay in the page cache.**
- expert formats: gate/up IQ1_M (16 layers), IQ2_XXS (17), IQ2_XS (7); down IQ1_M (28), IQ2_XXS (8), IQ2_XS (4).
- a ~9.5K-token prompt reads **~29 GB from the SSD** (the 1.7% of rows routed to non-resident experts), at ~1.4 GB/s:
  paced by the GPU, not the SSD (readers 8 -> 32, ring 400 -> 1600 MB, read-ahead 2 -> 16 GB: no change).

### maya on this box (the reference's current numbers)

| | v1.1.35 / .36 | + #643 (fused int8 experts) |
| --- | --- | --- |
| prompt, 2K tokens (`--bench`) | 234 tok/s | 257 |
| prompt, 8K tokens | 430 | 468 |
| prompt, ~9.5K tokens (server) | 454 | 495 |
| decode (`--bench`, greedy, DSpark) | 25.5-27.9 tok/s | unchanged |
| decode, DSpark vs none (speed calibration) | 22.3 vs 19.6 tok/s | |

a ~9.5K-token prompt's GPU time (`DS4_PROF`, + #643): attention 9.7 s (sparse attention 3.0, output projections 3.1,
q 1.2, Engram + mixing 1.8, indexer 0.9), routing + shared expert 1.7 s, experts 7.4 s. the dense GEMMs are already
on hipBLASLt picks measured on this box (`lt-picks.txt`).

### the bar, measured (maya v1.1.36 + #643 + #587, `~/maya-bar`, branch `bar-643-587`)

`tools/ds4/agent_bench.py --pack ~/models/ds1017/maya-ds4 --exe build-hip/maya-ds4 --config
~/project-maya/maya-ds4-hip.json --agents 4 --turns 8 --seed 1` (`--parallel 4`, DSpark loaded, 131072 context):
new tokens a turn 150-18,693 (median 868), 98,757 in all.

| run | first token mean / p50 / p95 / max (s) | turn done mean / p50 / p95 / max (s) | wall (s) |
| --- | ---: | ---: | ---: |
| 1 | 13.02 / 5.80 / 40.25 / 71.57 | 60.37 / 53.32 / 102.60 / 152.57 | 514.8 |
| 2 | 12.84 / 5.75 / 37.43 / 71.96 | 60.11 / 52.40 / 102.96 / 153.75 | 512.7 |

32 of 32 turns answered, 28 of 28 later turns continued their conversation in place, both runs. **the spread**:
wall 0.4%, first token mean 1.4%, p95 7%, turn done mean 0.4% - a p95 claim must clear ~7%. for scale, #587's 3-GPU split measured 5.23 / 2.37 / 13.5 first token and 190.5 s wall on the same
workload.

## the model

the full inventory, with maya's file:line for every part: `.research/ds4-forward-inventory.md`. in short:

- **43 layers**: 40 of the trunk, 3 of DSpark (`mtp.0..2`). a token's residual is 4 hyper-connection streams of 5120
  (Sinkhorn-mixed, the pre-mix lagged a sublayer). attention is sparse: a 128-row window of FP8 rows plus up to 512
  compressed FP4 rows picked by an indexer, compressed KV and index keys computed on 4 source layers (2, 8, 14, 20)
  and shared by the layers after them. 384 routed experts top-6 + a shared one. Engram n-gram tables on layers 1 and
  14. DSpark drafts 4 tokens a round from the last three layers' states.
- **the quant mix**: routed experts IQ1_M / IQ2_XXS / IQ2_XS; dense Q5_K / Q6_K / Q8_0; router, indexer and
  compressors BF16 (the router held in F32); Engram IQ4_NL; DSpark's experts MXFP4.
- **numerics that carry the answers**: FP8 E4M3 and FP4 (E4M3/16, UE8M0/32) roundings exact on their grids, the
  router in F32 with one summation order, top-k ties to the lowest index. maya emulates them bit for bit on HIP.
- **a conversation's state** is ~2.9 MB fixed + **~0.92 KB a token** (30 MB at 32K, 0.96 GB at 1M) - small beside
  the weights, so many conversations fit (#356's slots save exactly this).
- **the memory picture**: ~139 GB of trunk + DSpark weights (132 GB experts) and **110.6 GB of Engram tables**,
  against 107.6 GiB of GPU memory and 125 GB of RAM. Engram is 48 random 144 B rows a token (~48 IOPS, ~200 KB of
  4 KiB pages from the SSD); a prompt multiplies that by its length (28K tokens = 1.36M rows). maya streams it the
  same way, so it costs both sides of the bar alike. it is a lookup, not compute: the prompt profile above bounds it -
  "Engram + mixing + projections" 1.8 s of a ~9.5K-token prompt's ~21 s, mixing and projections included; in decode
  ~1,350 random reads a second at 28 tok/s, and `engram.wkv` ~220 MB a token (the inventory's estimate) of ~7.5 GB. the
  one case to watch: long cold prompts (maya's source: 13-17 s for a 28K-token prompt's 1.36M rows, on Windows).
- **the size**: maya's math for this model is ~3,300 of `ds4_model.cu`'s 18,463 lines; the rest is multi-GPU (~4,000),
  expert tiering and streaming for many machines (~5,500), serving, calibration, vision and setup. the reader's
  estimate for one APU: ~3-4K lines of host code, ~4-5K of kernels (**an estimate**, to be replaced by the count).
- **the oracle exists**: maya's `--ppl`, `--dump-logits`, per-layer dumps (`DS4_DUMP_H/R/Y`), `tools/ds4/ref_gguf.py`
  (a GGUF reference sharing no engine code; whether it decodes IQ1_M / IQ2_XS / Q5_K / Q6_K / IQ4_NL is unverified)
  and `compare.py` (cosine, KL, top-1 by layer).

## the memory: one pool, read in place (spike, 2026-10-28)

`spikes/inplace_read/inplace_read.hip`, raw output in `.research/spike-inplace_read.txt`. a streaming kernel sums 2-24
GiB of a ds1017 GGUF shard; every GPU sum is checked against the CPU's.

| where the bytes are | GPU read |
| --- | ---: |
| A device memory (hipMalloc) | 239.7 GB/s |
| B pinned host memory (hipHostMalloc) | 238.2 GB/s |
| C the GGUF's own pages, mmap + `hipHostRegister` (read-only, mapped), in the page cache | 237.5 GB/s |
| C cold (evicted before mapping), 2 GiB: register 0.48 s (4.5 GB/s from the SSD), then | 226.5 -> 239.5 GB/s |
| C cold, 24 GiB: register 4.17 s (**6.2 GB/s**, the SSD's 6.7 GB/s direct-read ceiling), then | 235.9 -> 236.6 GB/s |
| D the mapping as plain pageable memory | not supported (the device reports no pageable access) |

and, held 4 GiB of device memory and 12 GiB registered: the driver's `mem_info_gtt_used` reads **4.17 GiB** - a
registration does not count against `ttm`'s pages_limit. `RLIMIT_MEMLOCK` (8 MB) did not refuse it either.

what it means for the design:

- **the GPU reads the model's files where they lie, at its own memory's speed.** no staging copy, no ring: maya's
  flow copies the non-resident experts through a ring at ~1.4 GB/s (GPU-paced); here they can be registered and read
  in place.
- **a cold registration is a read at the SSD's speed** (6.2 of 6.7 GB/s, one call): the planner's only I/O.
- **the limit is RAM, not GTT.** registered pages are pinned RAM outside `ttm`'s 107.6 GiB, so the budget is physical
  RAM (~116 GiB) less the operator's reserve (~16-20 GB for the desktop and docker), whatever mix of allocations and
  registrations fills it. the planner budgets RAM; GTT is one of its terms.
- **the overflow rotates.** ds1017's ~140 GB of weights exceed RAM, so a set of registered regions turns over: pinned
  where the router needs them, unpinned behind - the read ahead of the GPU at the SSD's speed.

at an expert's size (`spikes/inplace_read/regs.hip`, `.research/spike-regs.txt`; 3 MiB regions, one expert slice, of a
36.7 GiB shard):

| | measured |
| --- | --- |
| register, cold, one at a time (400 calls) | p50 2.19 ms, p95 3.02 ms, max 159 ms - **1.25 GB/s** (the pages read one by one) |
| register, warm (in the page cache) | **0.029 ms** |
| unregister | ~0 ms (p95 0.003) |
| two registrations sharing a 4 KiB page | **refused** ("part or all of the requested memory range is already mapped") |
| held at once | 4,000 of 4,000 (11.7 GiB), the GPU's sum ok, all released in ~0 s |

- **the I/O and the pinning are separate steps.** a cold registration is a read at queue depth 1 (1.25 GB/s); a warm
  one is 29 us. so the planner reads ahead into the page cache with many reads in flight (6.2 GB/s measured above)
  and pins once the pages are there.
- **registrations cannot share a page**, and an expert's slice is not page-aligned (an IQ2_XXS gate is 3,041,280 B),
  so neighbouring experts share a boundary page. the planner registers merged, page-aligned runs of experts; an expert
  leaving a run unregisters it and registers the rest again (warm: ~0.03 ms a piece).
- **the count is no constraint** at this model's scale (4,000 held).
- **still unmeasured**: a pinned set near the RAM ceiling (the desktop's behaviour); the page cache's eviction of
  unpinned file pages under the GPU's allocations.

## splash's bets, and what each is worth here

splash (Apache-2.0, Apple silicon) - read from its README and docs.

| splash | on the Halo | evidence | milestone 1? |
| --- | --- | --- | --- |
| DFlash2 drafts every model | DeepSeek's own DSpark drafter (and DFlash2 for GLM, #628) | DSpark 22.3 vs 19.6 tok/s here | **yes**: keep |
| precompiled fused GPU kernels | Maya's HIP kernels (MIT) incl. #643's fused int8 experts | +9-10% prompt here | **yes**: reuse |
| the Neural Engine helps the prompt | XDNA 2 NPU | no NPU runtime loads 1-2-bit weights; measured NPU prefill ~3-4x below the 8060S on gpt-oss-20b; running both costs 15-19% on shared memory bandwidth (third-party measurements) | **no**: park. later candidates: the drafter, or the sparse indexer, if measured to pay |
| automatic memory planning | a planner for the one pool: expert residency vs the page cache vs KV vs kept conversations | the 140 GB > 125 GB fact above | **yes**: the core |
| batches concurrent requests | libevt: a piece each, answers in the gaps (#557 / #587) | 4.8x first token on rig traffic (3 GPUs); **unmeasured here** | **yes**: the core |
| SSD KV cache | #356's conversation slots (RAM, then the folder) | 7x first token (3 GPUs); 37 s vs 64 s here (a tester) | **yes**: reuse |
| MoE experts stream from the SSD | the flow (in place, GPU-paced) | 29 GB a 9.5K prompt here | **yes**: measure first, then plan |
| one command, OpenAI API | rig's `command` / `port` / `places` | | **yes** |

## decisions (operator, 2026-10-28)

- **written fresh**, as compact as it can be: one model, one machine (gfx1151, one APU, unified memory), one quant
  family (the GGUF's IQ formats), no CUDA, no multi-GPU, no tensor parallel, no Windows. maya's DeepSeek engine is
  ~21K lines (`ds4_model.cu` 18,463, `ds4_main.cpp` 2,499) for all of those; the size this one needs is estimated
  in "the model" below once the forward pass is inventoried.
- **one binary.** it serves the OpenAI API itself (no python server in front).
- **the bar: maya v1.1.36 + #643 + #587**, on this box.
- **the harness is maya's own**: `tools/ds4/agent_bench.py` and `tools/ds4/parallel_test.py` (#557 / #587), unchanged.

what is ported and what is new (operator, 2026-10-28):

- **the math is ported exactly**, from the spec - maya's torch oracle `tools/ds4/ref.py` (~250 lines, the whole
  forward), not maya's CUDA. the FP8 / FP4 roundings, the router's F32 order, Sinkhorn, the indexer's ties, the lagged
  pre-mix are the model; "inspired" there is quietly worse answers.
- **maya is the oracle**: a layer is done when its outputs match maya's on the same inputs (`DS4_DUMP_H/R/Y`,
  `--dump-logits`, `compare.py`), not when the perplexity looks right.
- **the kernels are new**, for gfx1151 only, built on what was measured here: int8 WMMA with the decode fused beats the
  FP16 tiles; those tiles are occupancy-bound, not barrier-bound; the grid size depends on the quant type; hipBLASLt
  solution ids belong to a build and are tuned on this box.
- **the runtime is new**: one memory pool, not maya's VRAM / RAM / PCIe tiers (~5,500 lines for discrete cards); the
  scheduling is #557 / #587's on libevt.

## what the bar says about this machine

the operator's 3-GPU split runs this model at ~80 tok/s decode and ~1,500 tok/s prompt; this box at ~28 / ~470
(above): ~3x on both, and agent_bench's wall is 2.7x (514.8 / 190.5 s). maya's scheduling behaves here as there; the
gap is the hardware. so:

- **decode** is bandwidth-bound here (~7.5 GB a token at ~200 GB/s, measured above): the levers are DSpark's
  acceptance and fewer bytes a token, not a faster loop.
- **where a runtime for this box can win**: the memory tiers (the non-resident experts read in place instead of
  copied through a staging ring at 1.4 GB/s - **a hypothesis, unmeasured**), prompt kernels for this GPU, and
  scheduling tuned to one APU. those show in first token and turn done - what agent_bench measures.

## the scheduler

libevt (`~/Projects/libevt`, C, MIT, the operator's): one consumer, a total `(priority, id)` order, `own` (a resource
alone once the shared work out on it is back) and `cap` (n slots). its line: libtrdr (C) -> rig's `evt` (Go) ->
libevt (C); maya's DeepSeek serve loop runs a C++ port of it (#557 / #587).

| | ops/s | source |
| --- | ---: | --- |
| libtrdr, C, one core, empty closures | 10M | the operator |
| rig's `evt`, Go | > 300K | the operator |
| libevt | **unmeasured** | to measure at the scheduler step: empty contexts one core; 8 producers; idle wake-up |

the budget it has to fit: a token is ~35 ms here, so 10 us an event would be ~0.03% of one.

## the harness interface

`agent_bench` and `parallel_test` drive an engine over its stdin / stdout, token ids only - no tokenizer, no HTTP -
so they time the engine alone, the same way for both. the binary speaks that protocol (`--serve-lines`) beside its
HTTP API, and the reference is measured through the same harness:

- started as `<exe> <args> --serve`; ready when it prints `READY ... multi=<N>`.
- in: `GEN <max_new> rid=<r> spec=<0|1> <id,id,...>`, `STOP <rid>`, `QUIT`.
- out: `T <rid> <token>` per token, `PP <rid> ...` (prompt progress), `DONE <rid> <generated> <prompt> <ms> <ms>
  <finish> ... <reused>` (field 9), `ERR <rid> <reason>`; `STAT` / `NOTE` / `INFO` / `VRAM` lines.
- the exact field list is taken from maya's `ds4_main.cpp` at schema time.

## open questions

1. the NPU's memlock limit (8 MB): only when an NPU experiment is on the plan.

## the protocol (carried from #356 / #557 / #587)

- every claim is a measured number on this box, with the command that reproduces it, and the box's state with it:
  today **core 1 (cpus 1, 17) offline** (idle MCE errors), 15 of 16 cores. its cost is expected small (decode and
  prompt are GPU- and memory-bound; maya's CPU lane takes 0.2-1.7% of the experts) and **unmeasured**: a fixed unit is
  re-measured on the same commands.
- same binary, one switch off and on, alternating runs, at least two each; the run-to-run spread is measured and
  reported, not assumed.
- latency: first token and turn done at mean / p50 / p95, and wall. throughput: prompt tok/s at 2K / 8K / 32K, decode
  tok/s.
- correctness: greedy token-for-token against the reference where the arithmetic matches; perplexity (`--ppl`)
  within the measured spread where it does not; the engine's own run-to-run determinism measured first.
- power: every bar run under `tools/power.py` (amdgpu PPT, the APU package: CPU + GPU + NPU, 2 samples a second),
  reported as mean / peak watts and the energy of the run. PPT is not wall power (DRAM, the SSD, fans and the supply
  are outside it); a headline claim per watt uses a wall meter, with PPT beside it.
- what was tried and did not work is written down, with its numbers.
- what was not tested is said.
