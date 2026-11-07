# libosmos: design

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

**superseded 2026-10-29** (below, "the memory, revised"): registered file pages stall a long prompt for minutes under
the page cache's pressure. the measurements here stand; the design they led to does not.

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
- **under load, an unregister drains the GPU** (2026-10-29, measured: 64 unregisters with a 34 ms kernel in flight
  took 34.9 ms, the first waiting the kernel out): hipHostUnregister syncs every stream before HSA's unlock, so each
  of a layer's unpins stalled the forward. hip_pinner calls HSA's `hsa_amd_memory_lock_to_pool` / `hsa_amd_memory_unlock`
  itself - what hipHostRegister / hipHostUnregister are made of - and the unlock returns at once; the GPU reads a
  lock's pages at the same 241 GB/s, the device's pointer the host's. the pool is the CPU's coarse-grained one: a
  plain `hsa_amd_memory_lock` maps fine-grained pages, which stream at the same speed but made a decode step 4%
  slower (75.0 -> 78.0 ms). the pool's contract carries the safety: a unit
  is unpinned only with no hold on it, and the forward releases one only once the stream is past it.
- **the overflow is a cycle, and least recently used is its worst case.** a prompt's step touches ~277 of a layer's
  384 experts, ~115 GB a step against a 72 GB budget, in layer order every step: least recently released unpinned
  exactly the experts wanted soonest - **0 hits** on a warm real-text pass. `pool_release` takes the caller's "wanted
  again at" (the forward's: the layer's next visit) and room goes to the unit wanted furthest ahead - Belady's
  choice for a cycle: **62% hits**, and a 4096-token prompt's pins 1.6 s -> 0.18 s a pass, 271 -> 290 tok/s warm
  (2048 a step). the GPU is then the bound: the host waits on it for all but ~0.2 s of a 14.3 s pass.
- **still unmeasured**: a pinned set near the RAM ceiling (the desktop's behaviour); the page cache's eviction of
  unpinned file pages under the GPU's allocations.

## the fetcher (spike, 2026-10-28)

`spikes/fetch/fetch.c`, `.research/spike-fetch.txt`: 600 cold 3 MiB ranges spread over a 36.7 GiB shard, the file
evicted first, one process a run.

| | throughput | a range p50 / p95 |
| --- | ---: | --- |
| pread, 1 worker | 3.05 GB/s | 1.0 / 1.3 ms |
| **pread, 4 workers** | **6.85 GB/s** | 1.8 / 2.4 ms |
| pread, 8 / 16 / 32 | 6.98 / 7.02 / 6.95 | 3.5 / 6.9 / 13.7 ms p50 (queueing) |
| MADV_POPULATE_READ, 1-32 workers | <= 1.85 GB/s | the fault path does not scale on one file |
| posix_fadvise WILLNEED | 6.33 GB/s | ~280 ms each: no end is reported, only polling finds them |

so the Fetcher is **4 workers that pread** (`src/mem/fetchq.c`): the SSD's ceiling at 4, more only queue. the read
copies into a 1 MiB scratch a worker, thrown away (~7 GB/s of memory traffic beside the GPU's ~240). (since
2026-10-29 it reads straight into the GPU memory a unit was given: "the memory, revised".)

measured through the whole path (`tests/gpu/test_pool_gpu.hip`): ds1017 blk.3's 384 experts (2.97 GB) evicted, all
asked at once, read by fetchq and pinned by the pool through hip_pinner in **420 ms = 7.08 GB/s**, no read failed,
the GPU's sums the CPU's. (maya's ~1.4 GB/s on a prompt is paced by its GPU work, not its reads: not the same
measurement.)

## the memory, revised: copies in GPU memory (measured, 2026-10-29)

a long prompt stalled: at 8,192 tokens (`max_ctx` 16,384, a 72 GB budget) a 2,048-token step took **279 s**, the KFD's
`evicted_ms` jumping by 79 s and then 187 s, the page cache full (117 GB, 0 free), even a logger thread stalled 20-36 s
at a time. the mechanism: registered pages (`hsa_amd_memory_lock*`, `hipHostRegister`; on this kernel SVM ranges,
`svm_range_set_attr` in the stacks) are not pinned. reclaim takes them - the GPU's reads mark nothing accessed in the
CPU's tables, so they look idle - each take evicts every queue of the process, and the restore must fault every
registered page back in while reclaim keeps taking them. no `mlock` here: the operator keeps `RLIMIT_MEMLOCK` (8 MB)
and system settings as they are.

| test (scratch `evict3.hip`, `cat` of the Engram file the pressure) | evicted_ms | worst pass |
| --- | ---: | ---: |
| 48 GiB HSA-locked in place, a GPU sum over and over, 60 s | 1,376 | 0.83 s |
| 48 GiB copied to hipMalloc (GTT), the same | 4,024 | 2.13 s |

so short evictions happen with any GPU memory under pressure (the runtime's own registered buffers, likely); the
minutes-long restore is the registrations' alone - GTT has nothing to fault back. nothing big is registered now:

- **the resident set** (`resident.h`): every tensor but the routed experts, then experts while the budget lasts, an
  equal share of each layer's (expert 0 of every layer, then 1, ..). copies in GPU memory (`hip_slots.h`: HSA's GPU
  pool, the CPU allowed to write), read once by fetchq's workers straight in, in 1 GiB arenas (few buffers for the
  driver to keep), their file pages then dropped (`POSIX_FADV_DONTNEED`): RAM holds them once. cold, 6.56 GB/s.
- **the pool streams the rest** (`pool.h`): copies too, read straight into memory of their own (reused when its size
  is the next unit's), its room two layers' streamed experts; furthest-ahead as before. a prefill's step prefetches
  the next layer's streamed experts while this one runs (`pool_want`: room only from units wanted later), when the
  step's picks are at least a layer's experts.

the building blocks (scratch `gtt.hip`, `coh.hip`): HSA allocate 0.18 ms, free 0.11 ms without waiting for the GPU
(hipFree waits: 64 took 24 ms with a kernel in flight); pread from the page cache into GPU memory 10 GB/s on 1 thread,
25 GB/s on 4; the GPU reads it at 221-239 GB/s; a buffer the GPU had cached, rewritten by the CPU between kernels,
read stale 0 times in 4,800 (64 KiB to 16 MiB, memcpy and pread).

what each step bought (4,096 real-text tokens, 2,048 a step, the warm pass; 51156cd in place: **335 tok/s**):

| design | budget | tok/s |
| --- | ---: | ---: |
| every expert a copy, furthest-ahead over all of them | 72 / 85 GB | 160 / 192 |
| resident set + the rest registered while held | 72 / 85 GB | hangs (the same restore, on a few GB) |
| resident set + the rest streamed by copy | 72 / 85 GB | 172 / 136 |
| .. + the next layer's prefetch | 60 / 72 GB | 211 / 197 |
| .. + Engram's rows read at once (below) | 72 / 90 GB | **241 / 243** |

and 8,192 tokens (`max_ctx` 16,384, 90 GB): **241 tok/s**, every step 8.0-9.0 s - where in place never finished.
(a 100 GB budget does not fit: GTT's 107.6 GiB less the KV and the step's buffers.)

the bound now is the SSD: the streamed experts (~49 GB at a 90 GB budget) are read every step at ~5-8 GB/s. the page
cache cannot help - the step reads them in a fixed cycle bigger than it, and least recently used gets ~0 hits on a
cycle - and copies cap RAM for experts at GTT's 107.6 GiB, where in place the page cache held ~117 GB. so a short
prompt is slower than in place (241 vs 335): the price of not stalling. the lever is the tokens a step: the streamed
bytes are the same for 2,048 tokens or 8,192 (a step picks nearly all of a layer's experts), so a layer run for 8,192
tokens at once (maya's chunk) reads them a quarter as often a token.

## a layer for 8,192 tokens (measured, 2026-10-29)

a batch past MM_MAX_TOKENS runs as sub-batches of 2,048 (`FORWARD_SUBS` 4: maya's 8,192-token chunk), each layer
for every one of them before the next layer. a sub-batch carries its own state from layer to layer - its streams,
the next layer's pre, its positions, the indexer's selected rows, the decoder skip's rows, a pinned block, its Engram
rows (`Sub`, swapped into the Forward to run a layer) - and everything else is the layer's scratch, shared. each
layer's KV sees the sub-batches in the order the steps would give them, so the bits are a step a sub-batch's
(`test_batch_gpu`: 2 sequences of 1,500 interleaved in one batch of 3,000, each alone in steps of 512, with and without
the decoder skip). the experts a layer streams are held for all its sub-batches: read once for 8,192 tokens, not 4
times.

8,192 real-text tokens (`max_ctx` 16,384, an 88 GB budget, the warm pass), the same binary, interleaved runs:

| a batch | tok/s |
| --- | ---: |
| 2,048 tokens (4 steps) | 252, 246 |
| **8,192 tokens (one batch, 4 sub-batches)** | **306, 303** |

the reads fall from ~5-8 GB/s (the SSD's ceiling, the bound) to ~2 GB/s: the GPU is the bound again, its attention
growing with the context (a 2,048-token step's ~6 s at the prompt's start, ~8-10 s at 8K). runs of one binary vary by
up to ~40% from process to process (one 8,192 run gave 219, the next 313): only interleaved runs are compared.

## the GPU's time at 8,192 tokens, and the BF16 matrices (measured, 2026-10-30)

- a pass's kernels (`rocprofv3 --kernel-trace`, `promptl 8192 8192 2`, the second pass): the GPU busy 24.9 s of ~26,
  so the step is the kernels'. maya 1.1.41's `DS4_PROF` on the same prompt: 16.6 s. by stage, ours / maya's: routed
  experts 7.7 / ~5.0 s (its 6.75 less 1.7 of SSD waits); elementwise (hyper-connections, rope, x16, fp8) 4.5 / ~1.3;
  q, shared expert, router 4.7 / 2.5; sparse attention 4.6 / 2.7; indexer 1.6 / 0.8; output projections 1.8 / 2.7.
- the BF16 matrices (router, indexer, compressors) had no mm and went through mv in chunks of 8 tokens: every 8
  tokens read W again, every row x again - the router alone 15.2 ms a call at 8,192 tokens. mv_strided past
  MV_MAX_TOKENS now runs tiles of 4 rows a wave and 8 tokens a block, x shared through LDS: each lane's chunks in the
  same order, the same tree, so every token's bits are those it takes alone (test_mv_gpu: 77 tokens at once against
  each alone, every shape; reversing the tree fails it). the router 15.2 -> 5.5 ms (R 1-8, NW 4-32 tried: R 3 NW 12
  5.1, 16+ tokens 6.3-16); 8,192 tokens ~310 -> ~312 tok/s (re-measured; see the round-robin below).
- the elementwise kernels are bound by memory, and rope by its pattern: a head's last 64 of 512 values, 256 B of
  every 2 KB - 2.45 ms for q's 2,048 tokens with no trig at all, 0.10 ms for the same bytes side by side (scratch
  `ropeb.hip`; a thread computing an angle once for 2-64 heads was slower, 2.8-3.8 ms). so the attention's output
  no longer takes ops_rope, ops_fp8_round and mm_x16 apart: mm_x16_attn reads it once - a wave an FP8 block, a
  pair's lanes trading values for the rotation - and writes the f16 (test_mm_gpu: the bits of the three calls on
  37 tokens x 32,768; a flipped rotation fails it). hc_post a thread a (token, d), the hc streams read once for
  its 4 outputs, each sum in its order. 8,192 tokens: rope 0.82 -> 0.42 s (q's, kv's, the indexer's left),
  fp8 + x16 1.04 -> 0.31 + 0.39 (the fused), hc_post 0.97 -> 0.56; the GPU busy 24.9 -> 23.2 s, ~312 -> ~328 tok/s.
- sparse attention's two products (scratch `sab.hip` / `sav.hip`, 2,048 tokens x 640 keys): the 4 x 4 a thread
  read 2 B of LDS a fma, the bound of LDS's bandwidth, though every fma was already dual-issued; q's traffic was
  not it (every block on one token's q: 10.85 -> 10.57 ms). larger register tiles, each thread's row pointers once
  (the selection's index was read again every stage) and the next stage's loads issued before this one's sums:
  scores 16 heads x 8 keys a thread, 64 x 128 a block, 13.0 -> 8.9 ms; values 8 x 16 of 64 x 256, 13.9 -> 9.8
  (the keys still 16 a stage: the padding past n where it was). the same bits (the scratch benches against the old
  kernels; test_attn_gpu, new: 640 keys against a double attention, each token alone and among 24, and a tile's
  stores skipped fails it). 8,192 tokens: scores 1.96 -> 1.24 s, values 2.00 -> 1.44; ~328 -> ~348 tok/s.
- the indexer's scores were a wave a row, a lane a head, each lane reading the row's 128 values one broadcast load
  at a time: 47 ms for 2,048 tokens x ~4,096 rows (~0.6 T multiply-adds a second). a run's tokens share their
  keys, so it is a product: tiles of 2 tokens x 32 head slots by 64 rows, 8 x 4 a thread, d 16 a stage, then relu
  * w and the heads summed as the 32-lane tree sums them - 10.1 ms (scratch `ixb.hip`: 1-4 tokens, 64-128 rows,
  4-16 x 4-8 all 10-12 ms but 16 x 8's 38). the same bits: test_attn_gpu replays the order on the host in F32 (a
  multiply then an add, -ffp-contract=off), every score bit for bit, ~3,000 rows, 32 and 24 heads; a row tile's
  stores skipped fails it, and so does summing the heads in a line instead of the tree - which the model's
  fingerprint does not see (its 32 positions: every row selected, the scores' order unread). 8,192 tokens ~348 ->
  ~370 tok/s.
- a layer's experts once for its every sub-batch: each sub-batch up to the FFN's input (its rows one after another
  in a step-wide xf), the router, the shared expert and the routed ones for them all, then each sub-batch's end of
  the layer (a sub-batch's attention reads the earlier ones' keys of the layer, which the experts do not touch). an
  expert's 6 of 384 picks of 2,048 tokens are ~32 tokens, a slab's IQ decode serving 2 tiles; of 8,192, ~128 and 8
  (scratch `exg.hip`, blk.3's IQ1_M experts through mm_group: 1.14 us a token-expert at 32, 0.69 at 128; the down
  1.23 / 0.74). an expert past MM_MAX_TOKENS of its tokens runs in parts; fwd_mat a chunk of MM_MAX_TOKENS at a
  time. the cost: the experts' scratch for 8,192 tokens, ~3 GB more GPU memory. Engram's pinned stage is now a
  sub-batch's own: with no wait between the sub-batches' first halves, the next one's rows were read into it while
  the last one's copy was still queued - test_batch_gpu's 2 x 1,500 tokens caught it (rows differing), now bit for
  bit again. 8,192 tokens ~370 -> ~409 tok/s.
- **the round-robin** (each of the day's commits built clean, 2 rounds, the second pass of each run; the first
  A/Bs of this list compared against a baseline that `make lib` - a directory, always up to date - had not rebuilt,
  so their "before" was the day's start, not the commit before): 96945ea 310, a9eca19 312, 88fef1d 328, 955db80
  348, b1065b0 370, the experts at once 409 tok/s. the commits' messages for 955db80 and b1065b0 state the gain
  since the day's start as the commit's own. a baseline is built by its library files (`make lib/libosmos.a
  lib/libosmos_hip.a`), never `make lib`.
- the matmuls after it (scratch `mmd.hip`, `exg.hip`; 2,048 tokens): the dense ones 26-32 TFLOP/s but q_b's 19.6
  (K 1,280 under 8 waves' split, 2,048 row tiles each re-reading the x16, 10.7 GB at ~1.2 TB/s); the IQ experts at
  128 tokens 34 TFLOP/s, and not bound by their x16 (every expert on one x: 33.7 -> 32.3 ms). tried and given up:
  16 column tiles a block for Q5_K (the decode amortized twice as far) - the 16 accumulators spill: 3-4x slower,
  and the epilogue's unroll dropped to build it took Q6_K's out_b 5.4 -> 11.5 ms; the waves' 8 outputs a tile
  summed by 8 waves, not wave 0 alone (the same bits) - q_b 8.77 -> 8.33 ms but out_b 5.41 -> 5.72 and the experts
  33.7 -> 36.3: a loss. what is left in them wants another shape of kernel, not a knob.

## the decode's experts: resident by the router's picks (measured, 2026-10-30)

a decode token on the whole model (40 layers: the GGUF's 43 compress ratios include the 3 a drafter adds), budget
88 GB, a 512-token prompt then 64 greedy tokens (scratch `decl.hip`): p50 172-231 ms, ~630 MB read from the files a
token - its ~30% of 240 picks (40 layers x 6) the resident set did not hold, read at the SSD's and the page cache's
speed, a layer at a time behind its router. the resident experts were the lowest ids of each layer, which hold their
share of the picks and no more. the router's picks are skewed (scratch `routes.hip`: an 8,192-token calibration
text's counts against another 4,096 tokens' picks; the most-picked against the lowest ids, at each share of a
layer's experts resident: 50% 69.0 / 51.5% of the picks, 69% 84.3 / 70.2, 75% 88.5 / 75.3, 90% 96.1 / 90.7; the
evaluation text's own most-picked would hold 98-100% - a text uses far fewer experts than all text does).

so the forward counts each expert's picks (forward_picks), and forward_create_ranked takes such counts: each layer's
experts resident in their order, a rank at a time across the layers. which are resident changes no bit
(test_model_gpu: the least-picked first, the logits bit for bit; and the counts are every run's every pick). a
profile of 16,384 tokens of the real-text corpus from its start, decode on text from token 60,000 (2 rounds
alternating): by id p50 230.7 / 172.5 ms, ~630 MB a token; ranked **p50 91.6 / 90.9 ms, 64 MB a token**. the
profile is the caller's to keep (a file beside the model); forward_create stays by id. not done: following the text
as it goes (a set that adapts holds more than any fixed one) - next.

and then the set follows the text (hip_adapt.c). each expert's heat is its picks, halved every FWD_HEAT_HALF_LIFE
steps; when a layer's experts are queued, its hottest streamed one (held in the pool: its bytes already in GPU
memory) takes the place of its coldest resident one not picked in the step, if twice as hot - a device-to-device
copy on the stream behind the layer's kernels, FWD_SWAPS_A_LAYER a layer a step. a first try read the swapped-in
expert from the files at the step's end, two a step: p50 95 -> 88 ms, its reads the cost. the copy in GPU memory:
decode on held-out text after the 16K profile (128 steps, 2 rounds; all within noise of one another): off p50
94.3 / 91.0 ms, ~70 MB read a step; half-lives 8 / 32 / 128, margins 1.25 / 2, 2 or 6 a layer: p50 83.6 - 87.9 ms,
23 - 31 MB a step - the fixed constants half-life 128, margin 2, 2 a layer (its runs the steadiest: max 156 / 120
ms). what is left is first picks: a new text keeps reaching experts the set has not seen (~3 a token, ~30 MB),
read behind their router a layer at a time; 20 layers all resident decode at 35.9 ms a token, so ~72 ms is the
floor of 40. the KFD's evictions (0 - 900 ms in 128 steps, with the swaps or without) are most of the rest of the
spread. the model test's second run on text the set has seen: p50 157 -> 71.7 ms. 8,192-token prompts unchanged
(~410 tok/s warm). which experts are resident changes no bit (test_model_gpu: 1,983 swaps in its last forward,
the logits bit for bit).

the swaps found a latent fault: a read begun ahead (pool_want) lands in the forward's inbox, and the inbox was
handed to the pool only while the forward waited on an expert of its own - so reads never waited on stayed READING,
and a reading unit's memory cannot be made room of. with the set following the text the waits all but stop:
test_batch_gpu's 4,600 tokens in steps of 512 ran out of room (438 units reading, none pinned). every layer's
routing now hands the finished reads over first (fwd_drain, no wait); the skip's 4,600 tokens 454 -> 517 tok/s
with it. the final build: decode p50 87.9 / 85.7 ms (its means 98.6 / 86.9: the first run's evictions 1.2 s),
~28 MB read a step; 8,192 tokens 417 / 418 tok/s warm.

the decode's last costs, tried (128 steps on held-out text, 3 rounds alternating each):
- the KFD's evictions, as page-cache pressure: each streamed read's pages dropped after it (POSIX_FADV_DONTNEED in
  the fetcher) - the evictions 444 / 388 / 52 ms -> 0 / 0 / 216, the page cache 14-18 GB -> 1-4, but p50 87.2-87.9
  -> 89.7-90.9 ms (an expert read twice no longer from the cache) and the means the same. not taken; the evictions'
  cause is not only the page cache.
- a miss's latency: a worker read a unit's 3 parts in turn (a cold 3 MiB part p50 1.8 ms: ~5 ms an expert, the
  layer waiting on it). a unit's parts a 1 MiB chunk a job, many workers on one unit: decode p50 85-87 -> 82-84 ms,
  but the load 16 -> 22 s and a 512-token prompt 7.0-7.4 -> 9.6-10.1 s (many units' chunks side by side defeat the
  file's readahead). so chunks only when the queue is idle (fewer jobs than workers: a lone unit, a decode's
  miss), each part one job otherwise: decode p50 83.9-86.5 -> 80.7-84.1 ms, means 87.7-95.3 -> 82.1-87.3, the load
  (15.4-15.9 s), the prompt and 8,192 tokens (warm 406-411 / 402-406 tok/s) unchanged.

the decode's floor (20 layers all resident, 35.4 ms a token, `rocprofv3` over 32 of its steps): the GPU busy 33.8 ms
of it, ~4.1 GB of weights read (the layers' ~130 MB and their 6 experts' ~54 MB each, the head's 455 MB) - ~120
GB/s, where the matvec reads ~190-210 (test_mv_gpu). the matmuls are most of it: the dense mm (Q5_K 7.4 ms, Q6_K 3.3,
Q8_0 2.4, wo_a's groups 3.6) and the experts' groups (10.4 ms), at one token: the WMMA kernel's (test_mm_gpu: an IQ
expert's slice 62 GB/s by mm, 111 by mv; q_b 161 / 192). the decode takes mm for its one token so that a token's
bits are the same alone or in a prompt's batch - the matvec's F32 sums are not WMMA's. could a matvec emulate WMMA's
sums? scratch `wmma.hip`: v_wmma_f32_16x16x16_f16 on random tiles (exponents +-6, C +-10) against the exact sum of
its 16 products and C: 37% of the outputs that sum rounded (to nearest or to zero), the rest up to ~2,000 ulps from
it (149 distinct offsets) - not an F32 order of any kind (in sequence, a tree, exact pairs, 4s or 8s: 30-40% each):
a narrower accumulator of its own. emulating it bit for bit is a research project; the floor's lever is mm's own
speed at one token (the same WMMAs in each output's order: a kernel's shape, not its sums).

mm at one token, a kernel's shape (scratch `m1.hip`: one tile, the weights cycled through 64 copies so the cache
holds none, against the library's tile, every output bit for bit): each lane decoding its own row of WMMA's A
operand (lanes 0-15 a step's even chunk, 16-31 its odd, the halves traded by a shuffle) with no LDS between. Q5_K,
of short slabs (32 values, 8 waves): q_b 181 -> 143 us (159 -> 202 GB/s), wo_a's 8 groups 185 -> 147, the shared
expert's gate and up 62 -> 60. the IQ experts 172 -> 182, 168 -> 192, 166 -> 179 (slower: their 128-value slabs'
LDS layout coalesces their loads); Q6_K 167 -> 172, Q8_0 63 -> 65 / 62 by permlanex16 (the same). so Q5_K's one
tile alone (kRegA). decode on held-out text p50 83.9 / 86.9 -> 79.0 / 81.2 ms, its fastest step 73.2 -> 69.8; the
model test's p50 71.3 -> 67.5.
the IQ experts at one token (the same scratch, the library's layout with parts left out, without its chunks in flight:
362 us all of it): no WMMA 327 / 317 us (IQ1_M / IQ2_XXS), no decode 70 / 67 - the decode is the cost, the WMMAs ~10%,
not the floor (as was said here first). the library's 8 chunks in flight bring the whole to ~170 us (90-110 GB/s), the
matvec's own IQ decode ~111: one token's experts are as fast as their decode. their grids are __constant__, read a lane
its own index (a vector load through the cache, a dependent one a chunk). in LDS (scratch `glds.hip`, 2026-10-30:
the grids copied into each block's LDS first, the same bits): slower everywhere - IQ1_M's gate 107 -> 126 / 165 us
(2 / 1 rows a lane), its down 124 -> 131 / 190, IQ2_XXS 123 -> 127 / 135. a block's 16 KB copy costs more than the
lookups it saves; the 8 chunks in flight hide them already. not taken.

launches (2026-10-30; scratch `lo.hip`, after tubo's probes - gfx1151, its numbers clock-pinned, ours not): a pure read
of B bytes a launch, 64 back to back, plain against hipExtAnyOrderLaunch: 33.5 MB 225 / 235 GB/s, 8.4 MB 222 / 240,
2.1 MB 187 / 214, 0.5 MB 113 / 185, 64 KB 24 / 33 - a dependent launch costs ~2.4 us here (tubo's 11 us is not this
box's), its ramp and tail a few more on a large one: ~1,040 a token, ~3 ms of the ~65 ms floor at most. what is worth
more is a matmul whose rows fill too few blocks at one token: the shared expert's gate and up (2,304 x 5,120, 144
blocks: 130 GB/s where its down, 320, reads 200) and q_a / kv (1,024 and 512 rows: 64 and 32). mm_run2 runs two
matrices of one type on one x16 in one launch, each tile as alone (test_mm_gpu: a pair of row ranges, each mm_run's
bits; the matrices swapped fails it): gate + up 125 -> 112 us, q_a + kv 64.5 -> 54.7 (scratch `mbpair.hip`, weights
cycled through 512 MiB) - ~24 us a layer, ~1 ms a token; the x16 of their input once, not twice. 20 resident
layers, the lane, interleaved fresh runs: p50 29.6 / 29.6 / 29.7 -> 29.1 / 29.1 / 29.1 ms (both fingerprints the
same). what is left in a small Q5_K launch: by size at one token (pairs, `mbpair.hip`) its time is a fixed part
plus ~190 GB/s - the fixed part ~19 us at 4,096 columns, ~6.5 at 1,280 (2 x 512 rows 85 / 81 GB/s, 2 x 8,192 177 /
180): a block's length, so the last round's idle (a wave's 32-column steps are 128 in a 4,096 row, 16 a wave). kRegA
with 2 / 4 / 8 steps decoded ahead of their WMMAs (the same sums): slower at every size (gate + up 112 -> 124 / 141
/ 283 us) - the registers cost the waves that hid the loads already. not taken; the lever left is fewer, larger
launches (a layer's matrices on one input together) or a layout of the weights (tubo's Q5_K repack: 177 -> 210).

Q5_K repacked (2026-10-31; kernels/quant.h kQ5_KR, quant_repack_q5k): a block's decode loaded 32 bytes a lane for 8
values (d and dmin, the 12 packed scale bytes, 8 of qh, 8 of qs), 16 lanes 16 rows apart. now a 16-row tile's block is
its rows' d and dmin, then a segment a 32-column step - the 32 lanes' 16 values each side by side (8 B of nibbles, 2
of fifth bits: the wave's loads one 320-byte run) and the rows' 6-bit (s, m) unpacked: 180 bytes a row's block
(+2.3%). the same (d * s) * q - dmin * m, so the same f16 values and WMMAs: mm at 1 and 16 tokens and mv at 3 give
Q5_K's bits (test_mm_gpu, every Q5_K shape), and both fingerprints are the same. at one token (test_mm_gpu, weights
cycled): q_b 195 -> 222 GB/s, the shared gate 134 -> 199, the head 196 -> 225 (scratch `q5r.hip` first: wo_a 157 ->
209). the forward repacks every resident Q5_K matrix but the embedding's rows at load, in place: its unit the file's
bytes and room after them (resident.h: a part of no file), its tiles from the last back through a 64 MiB stage (a
tile's repacked bytes start at or past its file bytes, so no write reaches a tile still to be read); the matmuls
read fwd_type's type. 20 resident layers, the lane, interleaved fresh runs: p50 29.2 / 29.1 / 29.2 -> 27.7 / 27.6 /
27.6 ms; the load 9.8-10.9 -> 11.1-11.3 s; 8K prompts the same (warm 402.8 / 396.3 -> 402.2 / 410.1 tok/s).

the hyper-connections at one token (2026-10-31, `rocprofv3` over 20 resident layers): a layer's 48 launches, ~2 us
idle before each (~94 us a layer); beside them the mixes' chain twice a layer, ~47 us each - hc_dot_few 24 us (2 MB
at ~87 GB/s: 1,600 threads, a thread 320 FMAs of its part one load at a time) and hc_mix 14.6 (the Sinkhorn on one
lane). hc_dot_few now loads 16 ahead of their FMAs (in order: the same sums) - 10.5 us; hc_mix a lane an entry of
the hc x hc matrix (hc^2 <= 32), each row's and column's sum the serial one's in its order through shuffles - 4.3 us.
test_hc_gpu: the dots alone against a host replay and the tiled path, the mixes against the one-lane kernel kept as
the reference (pre, post, comb; hc 4 and 6) - a reversed sum fails either. 20 resident layers 27.6 / 27.6 / 27.6 ->
26.7 / 26.6 / 26.7 ms. what is left between launches: ~0.12 ms a layer, the fusions in reach (the kv's rms, rope,
FP8 and ring write; the MoE's sum, add and hc_post; a norm and its x16) ~9 launches a layer, ~2%. built (2026-10-31,
test_fuse_gpu: each against the kernels it stands for, every bit - the x16s to a whole tile with garbage in both
buffers first, the overflow flag): ops_kv_finish (the key's norm, rope, FP8 and the window's row: 4 launches 1),
ops_hc_pre_rms_x16 (the attention's mix writes q_a and kv's x16), ops_rms_x16 (q_a's norm, q_b's), ops_swiglu_x16
(the shared expert's, its down's), ops_gather_sum_add (the experts' sum and the shared one's add) - 7 launches a
layer fewer. 20 resident layers 26.3 / 26.3 / 26.3 -> 26.0 / 26.0 / 25.9 ms: ~1.3%, less than the launches' ~3.5 us
each would give (the fused kernels do the same work; the x16s' pad rows a block each). the FFN's mix (xf: its x16
the shared pair's) not fused: under the merged experts xf spans the step's sub-batches, and a sub-batch's x16 is
not theirs. and a race the speed exposed (test_drafter_gpu, its 256-token prompt: "an expert the pool cannot hold" at
layer 19, 101 held): a prompt reads the next layer's streamed experts ahead, and a reading unit's memory cannot be
made room of - with the GPU sooner at a layer's pins, its reads ahead were still under way and the pin was refused
FULL. now a pin refused while units read waits for a read's end (fwd_wait_read) and pins again; full with none
reading is still the error (pool_reading: the count, test_pool's invariants). with a probe installed (the oracle tests')
the attention's two fusions step apart - a probe may stand the oracle's x or keys in after the fused kernel would
have used them; test_forward_gpu's probed run (unfused) and unprobed one (fused) the same fingerprint.

## DSpark: the drafter (built, 2026-10-30)

DeepSeek's own drafter, as maya's tools/ds4/dspark_ref.py writes it from DeepSeek's inference/model.py: three
blocks (`mtp.0..2` in ~/models/dspark/dspark.bin, Maya's pack: its dense matrices Q8_0, its 3 x 128 experts MXFP4),
each the trunk's layer in small - hyper-connections, window attention, 128 routed experts of which a row takes 3, a
shared one - fed by the trunk: the streams' mean entering its target layers 37-39 (the GGUF's
dspark_target_layer_ids), through main_proj and main_norm into each block's ring of 128 keys. a round runs the block
[y, noise x 4] at the sequence's next positions (every row attends the ring before them and the whole block: no
causal mask in it), collapses the last streams, and the trunk's head plus a rank-256 markov bias of each row's
previous token gives a greedy draft a row; a confidence head (sigmoid of [x ; E[prev]]) cuts the run (each >= 0.5,
their product >= 0.5: maya's defaults). the trunk verifies [y, drafts] in one batch: a batch's rows are the bits of
stepping them one at a time, so the longest prefix whose argmax is the next draft is kept, the rest undone
(seq_rollback: the position, Engram's last tokens - everything else a forward writes is by position, rewritten before
it is read), and the drafter's rings seeded from the accepted rows only (a rejected row's would overwrite a slot
still in the window).

- the pieces: MXFP4 (quant.h, bit for bit gguf-py's); dspark.json's index (src/model/dspark_index.c); the forward's
  taps (fwd_taps_on, the last sub-batch's rows); seq_rollback; the drafter (hip_drafter.c, hip_drafter_round.c) - the
  ring through ops_sparse_attn as its window (every row's position p0 - 1: the same 128 ring rows) and the block's own
  keys as its selected rows (each row all 5), the experts' tables gathered on the device from the router's picks.
  its products the matvec's (F32 sums, as maya's GEMV). its weights in GPU memory of their own (~8 GB), so the
  trunk's budget gives them room.
- the first run (scratch `spec.hip`, a 512-token prompt from the corpus at 60,000, 64 greedy tokens, budget 80 GB
  with the profile): the same tokens as stepping one at a time; 54 drafted, 40 accepted (74%); 11.5 -> 18.7 tok/s
  (1.63x, the drafted run after the plain one: see the correction below); drafting 0.37 s of 3.42, verifying 3.05. test_drafter_gpu: 48 tokens, the same as plain, 42 drafted, 35
  accepted (83%); a drafter with no window (every row's ring empty) fails its floor of 30% accepted.
- not built: maya's lazy stages (the head runs once for all five rows when the first passes), sampled drafts, the
  multi-sequence round. the drafter's numerics against an oracle (maya's needs DeepSeek's checkpoint shards, not here):
  the acceptance is the check so far.
- **against maya, on the same prompts** (maya's --bench: its warm-up and three questions, its own token ids -
  `<|begin|><|User|>q<|Assistant|></think>`, 10 / 28 / 19 / 21 tokens - greedy, up to 256 tokens; scratch
  `mbench.hip`, budget 80 GB with the profile, each mode in a fresh process, 2 rounds): maya 1.1.41 DSpark 27.5 / 23.3
  / 28.3 tok/s (mean 26.4), no drafting 16.7 / 16.6 / 16.8 (16.7); osmos DSpark 16.2-16.3 / 14.4-14.9 / 18.6-19.1
  (16.6), plain 9.4-9.6 / 10.1-10.3 / 10.0-10.2 (9.9) - the drafted answers the plain ones token for token. ~60% of
  maya both ways; DSpark 1.7x here (maya 1.6x).
- **a correction**: the first numbers here (256 tokens after a 512-token prompt: 21.8 / 26.0 tok/s, "1.9-2.2x the best
  plain") ran the drafted decode after the plain one on the same prompt in one process - the resident set had
  already followed that very answer (hip_adapt.c), its misses gone. the same on maya's prompts: 26.3 / 20.4 / 28.4
  tok/s, within ~5% of maya's DSpark. so what parts osmos from maya on a fresh prompt is the experts it reads from
  the files (maya holds ~104 GB of experts in GPU memory and computes the rest on the CPU from RAM; osmos's GPU
  memory is at the ~107 GB cap: the trunk's 80-88, the drafter's 8, the scratch's ~9), not the kernels or the drafter.
  a round's time is the verify's (the trunk's batch of 3-6 rows: more distinct experts a step, so more misses).
- what the misses are, on maya's prompts (code, prose, HTML; fresh processes, DSpark, budget 80 GiB):
  - **the profile is not it**: calibrated on 15,360 tokens of code, markup and JSON from this box's files and the
    corpus' prose (scratch `mix.txt`; its top 69% a layer 88% the corpus profile's), or both summed: plain 9.8 / 9.9 /
    10.1, drafted 16.6 / 16.6 / 16.8 tok/s - the same. what streams is each answer's own first picks.
  - **a layer's next experts guessed and read ahead** (hip_guess.c: the next layer's router on this layer's FFN
    input, its top 6 a token, queued beside the routing's copy - no wait of its own): 59-64% of the streamed picks
    guessed a layer ahead, but plain 10.1 -> 10.4, drafted 16.8 -> 16.9. these answers stream ~15 experts a token
    (~135 MB; the corpus' prose ~3): the SSD's bandwidth, not its latency - read earlier, the bytes are the same.
    kept (it costs nothing and changes no bit).
  - **more of them resident**: GPU memory and the page cache are one 125 GB of RAM (the GTT's cap 107.6 GiB). at 80
    GiB the process peaks at 99 GiB (the drafter 7.4, the scratch ~9 - ~4 of it the experts' buffers for 8,192 tokens
    at once); 86 fits (104) and is no faster (the page cache it takes was holding experts too); 90 is refused. steps
    of 2,048 tokens (FORWARD_SUBS 1: that scratch gone) fit 92 GiB (106 peak): drafted 18.6 / 16.5 / 21.7 tok/s
    (18.9; 84 GiB 17.8, 88 18.1), streamed picks 16,938 -> 10,122 - +12%, at the 8K prompt's merged experts' cost.
  - **the attention at one token**: the 8K tiles (64 heads x 128 keys, 64 x 256 dims) gave a decode's token ~5 blocks
    of scores and 2 of values. fewer than 16 tokens now take 16 heads x 64 keys / 64 dims, 4 x 4 a thread (the same
    sums: test_attn_gpu's tokens alone take them, its 24 the large ones, bit for bit): scores 57 -> 24 us, values 39 ->
    7; 20 resident layers' token 34.9 -> 33.4 ms.
  - **and the floor**: 40 layers all resident would decode at ~68 ms a token, slower than maya's whole plain decode
    (~60 ms, its misses in): the matmuls at one token (~120 GB/s of ~200) and the small kernels between are the other
    part. maya holds ~104 GB of experts in GPU memory and computes a few percent on the CPU from the page cache (no
    copy of them twice); osmos has no CPU lane (determinism's rule 1).

## the lane: a decode's experts in the matvec's order (operator, 2026-10-30)

the operator's call: a CPU lane as maya has, at the cost of rule 2 at one boundary. forward_lane: in a step that picks
fewer experts than a layer has (a decode's, a verify's) the routed experts go in the matvec's order - F32 sums, each
lane's chunks in turn, a fixed tree; kernels/lane.h's activation (its own exp, IEEE operations only) - the order the
CPU can repeat bit for bit, so an expert gives the same bits on either and which runs it (or whether it is resident)
changes none: residency's history, the drafts, the lane's split - none tips a token. a prompt's steps keep WMMA; a
token's bits then depend on whether its step was small (rule 2 relaxed there, rule 1 kept: test_model_gpu, two runs
of the lane bit for bit).

- step 1, the GPU's half (every expert on the GPU, hip_lane.c): the matvec beats the WMMA tile at one token - 20
  resident layers' token 32.6 -> 29.2 ms; maya's first question, the whole model, p50 86.8-92.7 -> 73.0-79.0 ms, mean
  89-102 -> 73-80 (the KFD's evictions 368-1,304 -> 0-12 ms in those 128 steps: not understood). the drafted tokens
  the plain ones (test_drafter_gpu, the lane's pass: 85% accepted). against the oracle the lane's KL is 1.27e-2 (the
  default's 2.47e-3, maya's - F32 experts too - 1.22e-2): its streams nearer the oracle's at layers 3 and 6 (6.2e-2 /
  1.7e-1 against 9.9e-2 / 2.6e-1), then one token's route at layer 18 tips (position 6's drift 1.0, the others the
  default's) - a near-tie two orders round apart, not a fault (scratch lanecmp.hip: layer 0's input and weights the
  same bits, its MoE out 3e-4 apart - f16's rounding of the activations, which the matvec does not do).
- an expert once for its tokens: a pair a matrix-vector product read a verify's shared experts again for each token -
  drafted on maya's prompts 16.9 -> 15.3 tok/s. the pairs in rows by expert, each expert an entry of up to 8 tokens
  (mv_group_tok: a token's row the sums of mv alone - test_mv_gpu, entries of 1 .. 8 tokens on 9 shapes, an offset row
  fails it); one launch at the widest entry's width ran an expert of one token at six: 14.1. a launch a count (as mm's
  column tiles): 16.3 (the default 16.9; plain 10.4 either way, budget 80 with the drafter). the lane on the GPU alone
  is even at this budget: what it is for is the CPU's half.
- **the memory first** (the operator's order: the CPU kernel on top of it). a step's most tokens now the forward's own
  (ForwardOptions' max_tokens; the step-wide experts' buffers ~1 GB a sub-batch of 2,048): at 2,048 with the drafter,
  budget 88 GiB peaks at 103 GiB, 92 at 106 (the GTT's cap 107.6), 94 refused (the pool's room), 96 refused. maya's
  prompts drafted at 92: 18.9 / 16.4 / 21.0 tok/s (18.8; 8K steps at 80: 16.9), streamed picks 10,122 (16,938). what
  is left beside the budget at 92, ~14 GiB: the drafter 7.4 (its experts, resident: a round takes up to 45), the
  scratch ~4 (the head's 2,048 rows of logits ~1 of it), the pool ~2 (a prompt's two layers). the price: an 8K
  prompt's experts in 4 steps, not one (the merged experts' ~10%).
- **the misses** (2026-10-30, counters in scratch, maya's prompts plain, the lane, 92 GiB): a 256-token answer
  ~21 s, of it 2.8-3.7 s the host waiting on a read - ~13 ms of a token's ~83, the GPU idle under it (the floor: the
  20 layers' 30.5 ms a token is ~65 for all of them). ~6 waits a token, ~2 ms each: a lone expert's read (~6.5 MB),
  which on this SSD is 1.79 ms p50 by 4 workers in 1 MiB chunks, 1.51 at best (16 workers, 512 KiB; scratch
  `lat.c`, 60 cold ranges a setting) - the wait is the drive's latency, not the queue's. the next layer's guess buys
  nothing here: FWD_GUESS_K 0 / 3 / 6 answer at 11.9-12.3 / 11.8-12.1 / 11.7-12.0 tok/s - at 6, 58% of its reads
  never used, and the misses it saves come back as reads queued behind its others. so the fetcher's two queues
  (fetchq.h): a pin's read urgent, a worker taking those first, and a read begun ahead and then pinned hurried to
  them (test_fetchq: one worker, 39 ahead queued - the urgent next, then the hurried). waits 3.7 / 3.1 / 3.2 ->
  3.1 / 2.75 / 2.9 s an answer; plain 11.9 / 12.5 / 12.2 tok/s (11.7 / 12.0 / 11.9), drafted 18.9 / 16.1 / 22.1 and
  18.9 / 16.2 / 22.1 (19.0 / 16.7 / 21.8 and 18.6 / 15.9 / 21.2, interleaved) - within the runs' spread. what is
  left of a miss is a read the router names only when it is needed: the lever past it is fewer misses (the
  resident set) or the floor, not the reads.

- **the CPU's half, its kernel** (2026-10-31; src/cpu/expert.c, cpu_mv_rows): the experts' three types (IQ1_M,
  IQ2_XXS, IQ2_XS) decoded as kernels/quant.h decodes them (the same products, built -ffp-contract=off; the grids
  quant_grids.h's, now C's too), lane l of 32 summing its chunks l, l + 32, ... with fmaf in order, then the GPU's
  xor tree: test_cpu_expert_gpu - an expert of each type's gate and down, 1 to 8 tokens - the GPU matvec's bits
  (a chunk's FMAs reversed fails it). AVX2, 8 lanes at once. its speed: an IQ1_M expert (gate, up, down) for one
  token 24 ms on one thread, 2.3 ms on 15 (3.3 GB/s of weights) - the decode with a constant in its place 8.8 ms:
  each lane's FMAs one chain, 16 registers. against the GPU's ~0.13 ms and a cold read's ~1.8 ms.
  - **what it is worth here** (the misses above): a layer's output needs all its experts, so a miss from the drive
    waits for the drive whichever side computes it - the GPU has only the resident experts (~0.3 ms) to do
    meanwhile. the CPU saves the copy into GPU memory only for an expert already in the page cache: ~10% of the
    misses' bytes, ~0.6 experts a token at ~0.7 ms - at most ~0.4 ms a token (~0.5%), and only with a kernel ~5x
    this one's speed. and the CPU and the GPU share the memory's bandwidth: the CPU on resident experts slows the
    GPU. maya's lane pays where its GPU holds the experts it can and the page cache the rest; here the resident set
    is GPU memory of its own and the misses are cold. the kernel kept (the bits shown); the wiring not built.

- **the IQ experts' matvec at one token** (2026-10-31; scratch `ivr.hip`, `grp.hip`, weights cycled): what binds
  it. mv.h's kernel on an IQ2_XXS gate (2,304 x 5,120): 24.2 us as is, 17.9 with the grid and signs left out, 16.6
  with one byte a chunk and no decode - the decode a third, the rest the loads' latency and the launch's tail. the
  counters: the real decode keeps the VALU ~46% busy, the byte-only one ~17% at 70% of the time. tried, the same
  bits, not faster: the grids packed 2 bits a value (4x smaller tables: 24 -> 31 us, the unpacking dearer than the
  lookups); x held in registers a lane, the wave walking the rows (the down's 9 chunks a lane even, the gate's 20
  spill: 24 -> 130). what moved it: the lane's launches (mv_group_tok, ~6 experts a token) at 2 rows a wave for the
  IQ2s, and IQ1_M's short rows at 2 rows and 2 chunks in flight - 6 experts' IQ2_XXS gate 122.5 -> 109.4 us, its
  down 149.8 -> 125.1, IQ2_XS's gate 124 -> 115, IQ1_M's down 124.7 -> 117.7 (the order unchanged: test_mv_gpu's
  tokens alone). 20 resident layers 26.7 -> 26.3 ms. a repack of IQ2_XXS to one aligned word a lane would save a
  load or two a chunk but not the index-then-grid chain the experiment above shows is the cost: not built.

- **the misses, again** (2026-10-31; maya's prompts plain, the lane, 92 GiB, interleaved fresh runs a pair at a
  time - their spread ~3%, a first run after a while ~30% slow, its page cache cold): two levers tried, neither
  outside the spread. 16 readers at 512 KiB a chunk (a lone 9 MB expert 2.10 -> 1.76 ms p50, scratch `lat.c`):
  12.7 / 13.3 / 13.2 and 13.2 / 13.4 / 13.4, 13.7 / 14.1 / 13.6 tok/s at 4 against 13.8 / 14.2 / 13.1 and 13.2 /
  13.7 / 13.7, 13.6 / 13.7 / 13.4 at 16 - a layer's misses are seldom a lone read. the experts in GPU memory first
  (their rows first, launched before a read is waited for; the reading ones after - the same bits): 12.5 / 12.9 /
  12.5, 12.7 / 13.4 / 13.1 against 12.7 / 13.1 / 12.4, 12.8 / 13.0 / 12.6 - the ready experts' ~0.3 ms beside a
  read's ~2 is not seen. neither kept. kept: test_lane_gpu - the lane at 30 and 12 GiB (997 and 1,614 streamed
  picks over 24 steps) the same streams' bits (a token's sum ordered by residency fails it). what is left of a
  miss is the drive's: fewer of them is the lever - the memory's, not the code's.

- **what maya does with the same files** (2026-10-31, its v1.1.41 bench log, `maya-bench.txt`): its plain bench
  ran `maya-ds4-hip-nodraft.json` - no drafter in memory: dense 6.96 GB and 11,686 of 15,744 experts (104.5 GB) in
  GPU memory, 1.0 of 107.6 GiB free (prompt buffers borrow the pool's tail: no scratch of their own); the other
  4,058 (35.1 GB) on a CPU lane of 15 threads (AVX-512), read in place from the model's files - an expert 0.13 ms
  alone, 0.53-0.80 ms while the GPU decodes; streaming some to the GPU measured no faster (61.38 against 61.44 ms a
  token) and turned off. ours against it like for like (mbench4 with no drafter, maya's prompts, plain, the lane):
  92 GiB 12.9 / 13.5 / 13.3 tok/s, 99 13.8 / 14.3 / 14.2, 101 14.6 / 14.9 / 14.8 (maya 16.7); 102 and 103 refused
  at a prompt's pin - the scratch (~4 GiB at 2,048 tokens) and the pool's two layers (~2) beside the budget. the
  drafter's 7.4 GB are the gap between our plain numbers before (92 GiB, drafter loaded) and these.

- **a smaller step for the memory** (2026-10-31): the scratch is 3.92 GB at 2,048 tokens a step - the head's
  logits for every row 1.06, q and o 0.27 each, the attention's scores 0.34, the experts' ~0.95, Engram's 0.21 -
  every buffer a step's tokens'. max_tokens below 2,048 now sizes them all (f->step: to 16s; a sequence's window
  and pending rows too): test_forward_gpu's max_tokens 500 takes 500, refuses 501, and steps the ids to the bits of
  the default forward. maya's prompts, plain, no drafter: 2,048 at 101 GiB 13.9 / 13.9 / 13.9 tok/s, 1,024 at 103
  14.4 / 14.4 / 14.3, 512 at 104 15.0 / 14.8 / 14.7 (105 refused). but a prompt reads every expert a step: an 8K
  prompt at 512 a step 121.9 / 118.6 tok/s against 307.0 / 290.8 at 2,048 - a decode's +6% for a prompt's -60%.
  kept as the option it is (a server that reads little); the default unchanged. the way past it is maya's: the
  prompt's buffers in the resident set's tail (its last arenas, the least-ranked experts: re-read after) - ~1.2 s
  of reads a prompt, not built.

- **the lane's experts over int8 inputs** (2026-11-01; kernels/q8.h, include/osmos/gpu/q8.h): a kernel-by-kernel
  profile of both decodes (rocprofv3, maya's first question, 256 tokens, 101 GiB, no drafter) - the dense matrices
  ours ~26 ms a token against maya's ~31, but the routed experts ours 19.5 ms (~81 us an expert, ~105 GB/s) against
  maya's ~12 for the ~85-90% it runs on the GPU (~57 us, ~150 GB/s). maya rounds an expert's input to int8 a block of
  32 and sums integer dots (ds4_iq.cu, MIT); ours decoded each value to F32 and fmaf'd it. so the lane's contract
  (kernels/lane.h) is now that: x rounded q = rint(x * (127 / amax)), d = amax / 127; a weight's half-block (16
  values, one scale) an exact v_dot4 sum - IQ1_M's grid + delta as codes 8 grid +- 1 (8 (grid . x) +- the chunk's x
  summed) - then lane_q8_part's products, the halves added, the block's scale fmaf'd onto the lane's sum; a row's
  blocks to 32 lanes (5,120 columns) or 16 (2,304), the xor tree. the integer sums are exact in any order, so the
  CPU's half (next) repeats the contract with VNNI rather than an FMA chain. test_q8_gpu: an IQ1_M, IQ2_XXS and
  IQ2_XS gate and down (ds1017's, two experts a launch, 1 to 8 tokens), the GPU the host's replay of the contract bit
  for bit, the rounding's bytes and scales too (a block of zeros, a lone value); IQ1_M's delta flipped, the scale
  step unfused, the rounding by division - each fails it. against the F32 matvec 3.8e-3 rms (int8's rounding).
  scratch `q8x.hip` (random weights, 6 experts, cold): gate 110 -> 90 us (IQ2_XXS), 131 -> 98 (IQ2_XS), 118 -> 86
  (IQ1_M); down 142 -> 102, 154 -> 110, 135 -> 90, at 173-209 GB/s. test_model_gpu: the lane's KL against the oracle
  5.61e-3 (the F32 lane's 1.27e-2, maya's 1.22e-2; 32 positions, the old number one near-tie route at layer 18 -
  read it as no worse), argmax the oracle's 32 of 32. maya's three questions plain at 101 GiB, interleaved fresh
  runs: 15.2 / 14.8 / 15.6, 14.8 / 15.0 / 15.1, 15.1 / 14.9 / 15.2 tok/s against 14.9 / 14.3 / 14.8, 14.0 / 14.2 /
  14.6, 14.2 / 13.9 / 14.2 (+5%); a first int8 run before them 8.4 / 6.8 / 5.5, not repeated in three, not
  understood. the third question's tokens are the F32 lane's; the others' differ (the lane's numbers changed, by
  design: rule 1 kept, the prompt's WMMA unchanged).

- **the CPU's half of the int8 contract** (2026-11-01; src/cpu/q8.c, include/osmos/cpu/q8.h): cpu_q8_round
  and cpu_q8_rows, q8_round's and q8_group_tok's bits on AVX-512 (F, BW, VL, DQ, VNNI, VBMI, BMI2; cpu_q8_ready,
  else both refuse). 16 blocks of a row at a time: their codes decoded once (64 values a vector - magnitudes as
  bytes, the negatives a 64-bit mask), then for each token x negated under the mask and vpdpbusd'd (exact), a
  half-block's 4 sums added by an in-lane transpose, lane.h's float steps a block a float lane (the lanes in the
  transpose's order, put back before the xor tree). test_cpu_q8 (CPU only, so asan / tsan see it): a scalar replay
  of the contract on random blocks of each type at 16 and 32 lanes and rows ending half a vector in, 11 tokens,
  rows split - the same bits; IQ1_M's delta flipped, the fmadd unfused, two tree lanes swapped, the rounding
  truncated, the scale's product taken in another order - each fails it (d (s / 8) for (d s) / 8 does not: an
  eighth is exact). test_cpu_q8_gpu: the GPU's bytes and bits on ds1017's IQ1_M, IQ2_XXS and IQ2_XS gates and downs
  (experts 7 and 300, 1 to 8 tokens), rows split in three the same. an expert (gate, up, down, 64 of a layer cycled
  from memory, 15 threads one a core, a token): IQ2_XXS 0.19 ms, IQ2_XS / IQ1_M 0.195, IQ1_M 0.21 - the F32 lane's
  0.87; maya's log 0.127-0.167 on 16 (~20% less a core). a second token +0.24 ms on one thread (of 2.1). what moved
  it, one thread, IQ2_XXS / IQ1_M: 6.4 / 21.3 ms -> IQ1_M's delta without a branch (a coin) and the chunk loops
  unrolled (the grid words kept in registers, not stored and reloaded wide) -> each type's row loop its own (no
  call a pair) 3.4 / 4.4 -> kKsigns computed, not looked up (the index by pdep, its parity by shifts in the byte)
  2.8 -> the scales from registers (the 2s + 1 bytes by one vpermb, d per 256 by a blend) 2.5 / 3.7 -> a group's
  block count a constant (its 8 vectors in registers) 2.1 / 2.6. tried, slower: the grid words by vpgatherqq
  (decode 13 -> 10 values/ns alone); the grids as 16-bit codes (each grid's bytes 3 levels: 2 bits a value; 4 a
  word, expanded by vpmultishiftqb and vpshufb - the same bytes) 11.3 against 11.7 IQ2_XXS, 7.6 against 8.1 IQ1_M:
  the lookups were not the cost. at a token the decode is ~3/4 of it. 15 threads reach 4.5-4.6 GHz against one's 5.1;
  a run under ~0.1 s is clocks still rising (0.21 then 0.165 ms on one cached expert).
  - **what wiring it would buy** (2026-11-01, measured before building; the GTT's cap now 113 GiB, 15 cores):
    the budget first - maya's prompts plain, interleaved fresh runs: 101 GiB 14.5 / 14.4 / 14.8 tok/s, 104 14.5 /
    14.2 / 14.6, 106 14.5 / 14.6 / 14.7 and 15.2 / 15.1 / 15.5 (a first run, cold, 12.2 / 12.0 / 12.4); the streams
    the same bits at every budget. then the CPU and the GPU at once: decode at 106 alone 15.7 / 15.5; with this
    kernel running 12 threads through 64 experts from memory the whole answer 13.1 (-16%), the kernel 0.30 ms an
    expert against 0.22 alone (+36%) - at 15 threads 9.2 (the decode's own host thread and readers starved too). the
    two share one memory and both are bound by it. a layer's six routed experts are ~0.35 ms of the GPU's; the CPU
    at 0.30 a contended expert balances at one of them - ~0.06 ms a layer, ~2.4 a token (~4%) at most, before what
    its bandwidth costs the GPU and a host-to-device wait each layer (maya's wait_seq_kernel: 8.6 ms a token). and
    the misses are the drive's whichever side computes them: the CPU has no memory of its own to hold more experts
    in (the GTT is the same RAM). the wiring not built; the kernel kept (the GPU's bits, 4.6x the F32 lane's).

- **a decode's token, profiled after the int8 lane** (2026-11-01; rocprofv3 kernel trace, maya's first question
  plain at 106 GiB, 235 tokens; scratch `ksum.py`, `gaps.py`, and counters around fwd_pinned's wait and the swaps -
  not kept): 67.1 ms a token, the GPU busy 50.4 - the dense matrices ~30.3 (mm::kernel 23.1, mm::group_kernel 4.7,
  mv 2.5; near the memory's bandwidth for ~7 GB), the routed experts 12.7 (maya's ~12; the F32 lane's 19.5), the
  rest ~7 in ~1,600 small kernels. the idle 16.7: under 5 us between kernels 3.5 (1,690 launches); the misses'
  reads ~8-9 (before the lane's gather, 3.1 a token of ~2.7 ms and 2.5 of ~0.45; the counter 7.7-9.9 ms, ~3.9 waits
  a token); **the resident set's swaps ~1.9** - 4.6 a token, each a ~9 MB hipMemcpyAsync device to device that runs
  on the copy engine, outside the kernels, ~0.45 ms with the GPU idle behind it (the gaps after a layer's
  gather_sum, 4 a token, ~485 us); the bench's own logits to the host and argmax ~0.9 (the harness's).
  - **the swaps as a kernel** (ops_copy): hip_slots' memory is HSA's, not the runtime's, so hipMemcpyAsync took
    it for the host's and copied on the copy engine. ops_copy - 16 bytes a thread, the tail a byte a thread, any
    memory the GPU reaches (test_copy_gpu: tails of 1 .. 15 bytes and an expert, between hip_slots' allocations,
    every byte and none past; the tail dropped fails it): alone an expert 0.145 ms against the copy engine's 0.176,
    but in decode the gaps after gather_sum are gone (1.94 ms a token -> none) and the copies' kernels take 0.63
    (13 a token, three parts a swap): ~1.3 ms a token. interleaved fresh runs at 106 GiB, maya's prompts plain: 15.5
    / 15.4 / 15.7, 15.6 / 15.1 / 15.5, 16.0 / 15.6 / 15.8 tok/s against 16.1 / 14.9 / 15.1, 15.8 / 15.3 / 15.3,
    15.9 / 15.2 / 15.5 - +0.8%, within the runs' spread (the misses' reads vary more than it); kept for the trace's.
    the streams the same bits.
- **the prompt's scratch lent to the experts, measured before building** (2026-11-01; the GTT's cap 113 GiB, RAM
  125): the bound - a step's scratch as good as gone (max_tokens 512, ~1 GB of ~3.9) and its room given to the
  resident set: 109 GiB 15.4 / 14.9 / 15.1 tok/s, 110 15.5 / 15.0 / 15.3 (111 refused at a prompt), against 106 at
  2,048 tokens a step 15.58 a question (the swaps' A/B above). at 2,048: 107 GiB 11.5 / 10.9 / 10.3, 108 refused. a
  GiB more resident is a GiB less page cache - the same RAM, which held the misses' reads too - and past ~106 the
  page cache and the system run short. lending the scratch (the resident set's last arenas, each layer's
  least-picked experts, freed for a prompt's step and read back after) would buy no decode here: not built. 106
  GiB at 2,048 is this box's budget.
- **maya's decode profiled beside ours, and the bench's resident set** (2026-11-01; rocprofv3, maya 1.1.46 with
  the BF16 router, its first question, 256 tokens; scratch `drive.py`): 61.7 ms a token, its GPU busy 55.4 of
  which 5.3 is wait_seq_kernel (the GPU spinning on its CPU lane) - ~50 of work against our 50.4. stage for stage
  the same: dense ~29.9 / 30.3, attention 1.8 / 1.8, hyper-connections ~2.3 / 2.4, router ~0.8 / 0.8, ~1,740 / 1,690
  launches; its routed experts on the GPU ~10.5 against our 12.7 (ours the misses' too). the gap is the misses
  alone: its CPU lane's 5.3 against our reads' ~8.5-9.4. and the bench had built our resident set by id (no
  picks): forward_create_ranked's profile (`osmos-picks.bin`, the 16K-token real-text count, nothing of the
  bench's questions) at 106 GiB, interleaved fresh runs: 16.7 / 15.2 / 14.9, 16.9 / 15.9 / 15.9 tok/s against by
  id 15.2 / 14.9 / 15.3, 15.5 / 15.0 / 15.4 - +4.7% (15.9 against 15.2), the streams the same bits. ranked is the
  comparison with maya (which ranks its own); the profile stays the caller's.
  - **the two side by side, ranked** (interleaved fresh runs, the three questions, 106 GiB, no profiler): ours
    17.2 / 15.5 / 15.5 and 16.5 / 15.2 / 15.4 tok/s (15.9), maya 18.4 / 18.0 / 17.6 and 18.1 / 18.2 / 18.0 (18.0) -
    the first question within ~6%, the other two ~15%. the misses a token, ours (forward_guesses' streamed picks)
    ~2.1 / 4.9 / 4.0, maya's (its CPU lane's share of the experts) ~4.6 / 4.1 / 4.3: as many. a miss's cost is the
    gap - ours, the first question (a counter around fwd_pinned's wait, not kept): ~2.1 picks a token streamed, 42%
    of them guessed a layer ahead; the unguessed waited ~2.5 ms each (most 2-4 ms: the drive's latency), the guessed
    ~2.1 (a layer's ~1.5 ms is less than a cold read's ~2.7) - ~4.55 ms a token; maya ~5.3 for ~4.6 (~1.2 each, its
    CPU lane computing while the GPU goes on; its drive reads 0.86 GB/s in decode against our 0.35: more bytes, less
    waiting). and maya's routed experts on the GPU ~10.5 ms against our 12.7: its gate and up one kernel
    (gate_up_kernel), ours gate, up, round, swiglu, round, down apart.
  - **gate and up in one launch** (q8_gate_up, kernels/q8.h's gate_up_kernel): a lane's blocks decoded from both
    matrices and summed against x read once, each sum in group_tok_kernel's order and tree, then lane_swiglu on the
    two - the gate's and up's outputs never written, the activation's launch gone (gate and up are one type in every
    layer of ds1017). test_q8_fuse_gpu: q8_group_tok twice and ops_swiglu_lane against it on an IQ1_M, IQ2_XXS and
    IQ2_XS layer (experts 7 and 300 a launch, 1 to 8 tokens at two widths), at a clamp that takes ~40% of the gates
    and one that takes none - the same bits, the rows past a count untouched; gate and up swapped in the activation
    fails it. the trace (ranked, 106 GiB): the routed experts 12.7 -> 11.5 ms a token (gate_up 7.3, down 4.2; maya's
    ~6.7 and ~4.1), launches 1,690 -> 1,616. interleaved fresh runs, the three questions: 16.6 / 15.5 / 15.7, 17.1 /
    16.3 / 16.2, 17.4 / 15.5 / 15.8 tok/s against 16.2 / 15.5 / 15.7, 17.2 / 15.9 / 15.9, 17.1 / 16.0 / 15.9 - 16.23
    against 16.16 (+0.4%), the ~2% the trace shows within the runs' spread. the streams the same bits.
  - **maya's miss path, read** (its src/ds4, v1.1.46): a missed expert never goes to the GPU. route_publish_kernel
    marks the picks not in GPU memory and copies their tokens' FFN input to host-mapped memory; publish_kernel
    writes a sequence number there; a pinned host thread polling it computes the experts on 15 CPU threads straight
    from the GGUF's mmap (madvise WILLNEED on all three matrices first, the guessed ones a layer ahead first) and
    writes each token's weighted sum to host-mapped memory; the GPU meanwhile runs the shared expert and the
    resident experts' gate and up, then spins (wait_seq_kernel) on the answer, then down and the combine, which
    reads the CPU's sums in place. no host-to-GPU sync a layer; a layer with no CPU pick answers itself. route_ahead
    is ours (the next layer's router on this layer's input, top 6, WILLNEED behind it). at start it reads the
    non-resident experts into the page cache by its profile's rank (pread, 12 threads), and its resident set swaps
    in a CPU-hit expert past 1.25 x the coldest's heat + 1, a spare slot a layer.
  - **the page cache warmed as maya's is** (an experiment, not kept): after the load, the streamed experts pread by
    rank until ~4 GB of RAM is left - 2,964 experts, 25.5 GB in 6-8 s. by decode the page cache is ~9 GB again (the
    scratch, the caches and the prompt's step take the rest), and the drive still reads 0.47 GB/s in decode (0.42
    without): the RAM holds about a third of the experts the GPU does not, warmed or not. 17.1 / 16.3 / 15.9 tok/s
    against 16.8 / 15.9 / 16.0 (+1%; a first run after a while 14.4 / 9.5 / 8.4 - fadvise DONTNEED on the streamed
    experts did not make a cold baseline). so maya's misses are the drive's too; what makes them cheaper is the
    overlap - the CPU computing them from the pages as they come while the GPU does the rest of the layer - not
    where they are.
- **the lane's CPU half, built** (2026-11-01; forward_lane's cpu_threads; src/cpu/lane.c, src/run/hip_cpu_lane.c):
  maya's miss path in our order. at a small step's routing the FFN inputs and weights come to the host with the
  picks (no wait of their own); an expert neither resident nor in the pool goes to a pool of CPU threads
  (cpu_lane_submit, before the GPU's launches): x to int8, gate and up and lane_swiglu, the activations to int8, down
  - cpu_q8_round / cpu_q8_rows, the GPU's bits - straight from the model's files mapped read-only (madvise WILLNEED on
  its three parts first), rows split over the workers (a spinning barrier between phases, a condition between jobs);
  its rows into host memory the GPU maps, then a value posted. the stream runs the GPU's experts, waits on the value
  (ops_wait: one thread, a system-scope acquire), copies the CPU's rows into ye (ops_copy_rows) and sums as before.
  the expert's read into the pool begins too (held, not waited for), and the swaps now take the hottest of a layer's
  streamed experts anywhere in the pool's GPU memory, held or not. tests: test_cpu_lane (CPU only: 1, 3 and 8 workers
  the one thread's bits, two jobs, refusals; one weight an expert, a share's last row, the barrier before down - each
  fails it), test_cpu_lane_gpu (the pool's rows the GPU lane's bits on three layers' real experts, both clamps; a
  stream queued to wait before the job is given, then its copy the same bits), test_lane_gpu (20 layers at 12 GiB with
  6 CPU threads: 640 of 1,656 streamed picks on the CPU, fingerprint 0ec6bd52a8c8bb8a - the GPU's; the CPU's rows'
  weight wrong fails it).
  measured (ranked, 106 GiB, interleaved fresh runs, the three questions, every stream the same bits):
  - first wiring (the CPU's experts only wanted in the pool): 12 threads 17.5 / 16.6 / 14.9, 16.7 / 15.5 / 14.7, 8
    threads 17.2 / 15.1 / 14.2, 16.5 / 15.0 / 14.0, against none 16.7 / 15.8 / 15.4, 16.5 / 15.6 / 15.5 - the misses
    ~3x (the second question's streamed picks 2,934 against 1,262): a want seldom found room, so the hot ones never
    reached the pool and the resident set stopped following the text.
  - then (held, and the swaps' wider choice): the swaps alone, the GPU only, 16.7 / 15.6 / 15.3, 16.5 / 15.7 / 16.1
    (15.98) against before 16.3 / 15.0 / 13.6, 16.5 / 15.6 / 15.7 (15.45; ~+1.5% without its 13.6) - kept; and 12 CPU
    threads 17.0 / 15.4 / 15.0, 16.8 / 15.1 / 14.8 (15.68) - the first question a little faster, the others slower,
    the misses still ~1.6x: an expert the CPU takes is still reading into the pool when its layer's swap is chosen,
    so the set follows a step late. the CPU half stays opt-in (cpu_threads 0, the default, unchanged); what is
    next for it is that lag (a swap's choice at the expert's next visit, or a read straight into the slot), not the
    kernel.
  - **the lag, closed** (forward_lane's swaps per question, counted): with the CPU half the swaps had halved (1,405
    against 2,642 over the three questions) - a CPU expert's read ends ~2 ms after its layer's swap, its hold went at
    the next layer, and the pool's room (two layers') had taken it long before its layer came round. now such an
    expert stays held while it reads (cpu_pend) and is offered its swap at the first swap point after (fwd_cpu_settle,
    swap_in: the same choice - the layer's coldest resident not picked in the step, the same margin - its copy on the
    stream), then held with the layer's others till the stream is past it. the misses then 3,502 against the GPU's
    3,622, the swaps 2,732 against 2,638 (test_lane_gpu's CPU run 1,656 -> 1,482 streamed picks; its bits the same).
  - **and still even**: the GPU's wait for the CPU (wait_kernel, a trace of the second question) 1.4 ms a token
    (p50 0.72 ms a wait) against ~4.5 of reads without it - but the idle elsewhere grows as much: the host late to
    launch after a layer's last kernel (~3 ms a token: 12 workers, 4 readers and the host on 15 cores). the second
    question alone: 4 / 6 / 8 / 10 threads 16.5 / 16.9 / 16.4 / 16.7 tok/s against none 16.7; the workers at nice
    10 or 19 (12 or 8 threads) 16.8 / 16.5 / 16.4 and 13.9-14.5 on the third against none 16.6 / 15.2. on this box
    the CPU's half buys back in the GPU's waits what it costs the host: kept opt-in, off by default.
  - a bug of this change, caught by test_drafter_gpu before commit: fwd_swap_layer told "no swap" from "a swap
    failed" by an empty err - a stale message in err made an ordinary step fail. now each outcome its own return;
    the test prints the step's error.
- **drafted, against maya's DSpark, after today's changes** (2026-11-01; ranked, the int8 lane with gate and up
  fused, the swaps' wider choice; the drafter's ~8 GB beside a 97 GiB budget - 99 refused, 98 fits a short run;
  maya's maya-ds4-hip.json; interleaved fresh runs, the three questions, 256 tokens): ours 29.1 / 20.6 / 30.9 and
  28.9 / 20.4 / 30.5 tok/s (26.7), maya 30.3 / 23.6 / 32.1 and 30.3 / 26.7 / 34.1 (29.5) - 91% of it (plain: 16.0
  against 18.0, 89%). DSpark 1.67x our plain. the drafted streams the plain ones bit for bit (the same hashes).
  acceptance follows the text: the first question 236 of 273 drafts (86%, 3.7 tokens a round), the second 146 of
  182 (80%, 2.3 a round: prose), the third 206 of 223 (92%, 5.1 a round: code). (the bench's drafted loop:
  test_drafter_gpu's - draft, verify [y, drafts] in one batch, the longest agreeing prefix kept, seq_rollback, the
  accepted rows seeded.)
- **the idle that is left, and two levers tried** (2026-11-01; ranked, 106 GiB, plain; the fused build's trace:
  61.1 ms a token, the GPU busy 49.5): the misses' reads ~6.0 ms (1.7 a token of ~3.2 ms, the gap before the lane's
  gather), the gaps between kernels 3.4 (1,512 a token, p50 2.16 us), the token's end ~1.4 (the logits to the host
  and the bench's argmax, the next step's tables), the rest ~0.8.
  - **a graph** (scratch `graph.hip`: 1,500 tiny dependent kernels, launched, then captured and replayed): 2.02 us
    a kernel launched, 1.87 in a graph - ~0.2 ms a token, below the runs' spread, and a step's launches change with
    its routing (a graph rebuilt each step). not built. the ~2 us is the GPU's between two dependent kernels: only
    fewer kernels move it (a fusion's launches saved are worth ~2 us each a token, beside its bytes).
  - **the guesses** (an experiment, not kept; counters around fwd_pinned's wait, guessed or not; one round each of
    the three questions): two layers ahead too (layer l + 2's router on l's input, its top 6 wanted two visits on):
    the streamed picks guessed 58% -> 64%, the reads' waits 5.55 -> 4.99 s - 16.28 tok/s against 16.18 (two
    rounds), within the spread; a guessed read still waits ~2 ms, the drive's queue full of guesses (a layer's ~2-3
    of them ~27 MB in ~1.5 ms, past what it reads). and fewer: a token's best 0 / 2 / 4 / 6 guesses (the router's
    picks come best first) 16.13 / 15.97 / 16.37 / 15.90 tok/s, the waits 6.5 / 6.1 / 5.5 / 5.4 s - the waits move,
    the speed does not. what the misses cost is the drive's latency for picks no guess sees.
- **the hyper-connections' kernels fused** (ops_hc_post_dot, ops_hc_mix_pre_rms; test_fuse_gpu): post and the
  next mixes' dot one launch below a tile of tokens - a block a (part, token) computing the part's streams as
  hc_post_kernel's thread does, keeping them in LDS and dotting them as hc_dot_few_kernel's thread does (its loads in
  flight, its FMAs in order); the mixes and the pre-norm one launch - independent (the norm's pre is a mix before:
  the attention's the last layer's FFN mix, the FFN's this layer's attention mix), the block's first wave the mixes,
  a barrier (for a caller whose pre is the mix's own), then the norm. 3 launches a layer fewer (1,616 -> 1,497 a
  token). test_fuse_gpu: 1, 3, 9 and 70 tokens (70: the two launches), x16 or not, the pre another's or the mix's
  own - the bits; the dot reversed, the barrier gone - each fails it. alone: 8.9 -> 5.4 us, 9.5 -> 6.9 us. first
  without the dot's loads in flight it was slower than the two (20.5 against 14.4 us a call in decode). in decode
  the hc kernels 2.37 -> 2.29 ms a token, the gaps under 5 us 3.40 -> 3.15: ~0.3 ms (~0.5%); interleaved fresh runs
  16.24 against 16.41 tok/s (three rounds), within their spread. the streams and fingerprints the same.
- **the greedy pick on the GPU** (forward_greedy, ops_argmax): the token's end had the bench copying a row of
  logits (129,280 floats) to the host and scanning it. ops_argmax: a block a row, the greatest value and the lowest
  index among equals - an order-free choice, the first maximum a scan finds (-0 and +0 equal); -1 for a row with a NaN,
  which forward_greedy refuses; rows past what the last step's head computed refused (vocab_rows: the last sub-batch's,
  0 when it ran none). test_argmax_gpu (a vocabulary's length, a tie, -0 / +0, all -inf, a NaN; ties to the higher
  index fails it); test_model_gpu: at every position forward_greedy the first maximum of forward_logits' row, a second
  row refused. the bench's picks then forward_greedy's (and a verify's rows one call): plain 16.7 / 15.6 / 15.6 and
  16.7 / 15.4 / 15.4 tok/s against 16.6 / 15.4 / 15.2 (a first run, cold, 14.1 / 10.6 / 7.4 left out) - ~+1%;
  drafted 29.8 / 21.9 / 31.5 (27.7) against 29.5 / 21.2 / 31.3 (27.3) - ~+1.5%, 94% of maya's DSpark (29.5). the
  same streams.
- **prompts, and text a drafter knows** (2026-11-01; ours ranked with forward_greedy; maya's nodraft config for the
  prompts, its DSpark config for the drafts; each engine's first long prompt after a short warm-up, maya a fresh
  process for each; scratch `drive_ids.py`):
  - prompts (the GPL-3's text, the first 2,048 and 8,192 tokens of a summarize request): 2K - maya 236 / 242 tok/s,
    ours 202-255 the first in a process (283-300 the second); 8K - maya 377 / 394 (its chunks of 4,096), ours at
    2,048 a step and 106 GiB 406 / 394, and at 8,192 a step (the experts read once for all four sub-batches; its
    buffers take the budget to 92 GiB) 554 / 523 - ~40% past maya. (the bench's own error first: max_ctx 4,096
    refused the 8K prompt, and its last row past a step of 2,048 is the last sub-batch's - forward_greedy refused
    the other row, loud.)
  - drafted, text a drafter knows: a CSV of 15 rows to a JSON array - ours 35.7 / 35.2 tok/s, maya 35.8 / 38.2; a
    class of 12 fields' getters and setters - ours 38.8 / 38.7, maya 29.9 / 29.2. (a first run warmed up on the CSV
    prompt itself gave 50.7 / 52.1: the resident set had followed that very answer - the correction above, again;
    left out.)
- **Engram's rows read from the step's start** (2026-11-01; rowread_start / rowread_wait, fwd_engram_issue): a
  fresh trace (the second question, after the greedy picks) showed gaps of ~1.55 ms after a layer's hc_post, 1.5 a
  token (~2.7 ms): the Engram layers' (1 and 14) rows read synchronously when the host reached the layer, the GPU
  idle under the reads. maya issues them at the token's start (its STAT: engram_ms 0.42-0.56 a token). the rows are
  the tokens' alone - known once the step's tokens are placed - so every sub-batch's rows for every Engram layer now
  go to the readers in one batch then (into the stage one after another), and the first layer that needs them
  waits; a step that failed, or stopped before them, has its batch finished at the next start. (DeepSeek-V4.1's
  Engram layers are before dec, so no compacted token is read: refused otherwise.) test_rowread: start, the caller's
  work, wait - every row; one batch at a time; a short row through the wait; destroyed at once after a start, 2,000
  times, every row read (the destroy's wait left out fails it - a first version of the test, destroying after the
  rows were long read, did not). the gaps 2.77 -> 0.68 ms a token. interleaved fresh runs, the three questions after
  an unrelated warm-up (tools/bench/prompts.py's): 16.1 / 16.5 / 16.0, 16.2 / 16.2 / 16.1, 15.9 / 15.9 / 15.7 tok/s
  (16.07) against 15.8 / 15.8 / 15.6, 15.5 / 15.7 / 15.6, 16.0 / 15.8 / 15.5 (15.70) - +2.3%, the streams the same
  bits. (a correction: the plain numbers above warmed up on the first question itself, which put it ~1 tok/s high;
  after an unrelated warm-up, as maya's bench, the plain baseline was ~15.7.)
  - **and the drafted numbers, corrected the same way**: the drafted runs above (26.7, then 27.7 tok/s, "91-94% of
    maya") also warmed up on the first question. after the unrelated warm-up (four runs; the first cold after the
    sanitizers' rebuild, 17.3 / 16.2 / 23.2, left out): 23.5 / 20.3 / 28.4, 25.0 / 20.8 / 29.0, 25.4 / 20.8 / 29.3 -
    ~25.1 tok/s at steady state, ~85% of maya's DSpark (29.5, its own warm-up an unrelated "Say hello"). the same
    build warmed on the first question: 29.0 / 20.8 / 29.8 - the first question ~4 tok/s high, the others near. the
    README's table takes the corrected numbers; the prompts and the repetitive questions (warmed up on the first
    question, unlike them) stand.
- **a verify's experts in GPU memory first** (2026-11-01; hip_lane.c's entries): drafted, against maya on the same
  question (the second; the bench's draft / verify timers, maya's STAT draft_ms / verify_ms): the same rounds (111 /
  104) and drafting (1,385 / 1,035 ms) - its verifying 10,469 ms against 7,416. over the whole answer (rocprofv3, both
  engines): the GPU's work near (ours 9.10 s with the drafter's, maya's 8.56 beside 0.92 of its CPU lane's spin), the
  idle ours 3.15 s against 0.68 - a round 21.4 ms waiting on reads (11 a round, ~1.9 ms each): a verify of ~2.6 tokens
  picks more experts a layer, more of them streamed, and the host waited on the first one before it launched any of
  the layer's. the CPU half instead (12 threads) slower still: 21.5 / 20.5 / 26.3, 23.6 / 19.8 / 26.4 against 25.3 /
  21.5 / 31.8, 25.8 / 21.3 / 31.2 (a verify's expert of several tokens is the CPU's slow case). now the entries in GPU
  memory (resident, or read into the pool) launch first - gate and up, the activations to int8, down - then the host
  waits for the reading ones, sends their pointers and launches them (x to int8 again between: the ready ones'
  activations took its buffer - a first version without it failed test_lane_gpu, the budgets' bits apart). every row
  its own sums, ye's sum unchanged: the same bits (test_lane_gpu, test_drafter_gpu, the fingerprints). drafted 26.6
  / 22.0 / 32.6 and 27.8 / 22.0 / 32.5 (27.25) against 25.2 / 20.8 / 29.3 (a first run, cold, 18.1 / 16.2 / 24.8,
  left out) - +8.6%, ~92% of maya's DSpark (29.5); the verifying 9.2 -> 8.3-8.7 s (the first question), 8.0 -> 7.1
  (the third). plain 15.93 against 15.83 (a decode step's ready experts are little to hide a read behind: as before).
- **the experts' kernels, tuned to their limit** (2026-11-01; scratch `kbench.hip`: q8_gate_up and q8_group_tok on a
  layer's real experts, 6 a launch, cycled through 64 so they come from memory): alone, gate and up 193 / 210 / 213 /
  215 GB/s (IQ1_M, IQ2_XXS, IQ2_XS layers), down 164 / 173 / 183 - the memory's ~230 near for gate and up. in
  decode the same launches run ~8% slower (186.6 against 173.6 us, IQ2_XXS: the token's other kernels in the
  caches). maya's per launch in its decode: down even (97.6 against our 103.5 us, IQ1_M), gate and up ~13% faster
  (163 / 152 / 181 against 186.6 / 173.4 / 212.2). tried, the same bits: R rows a lane group in the down (their
  loads in flight together) - R 2 164 -> 158-180, R 4 slower still; the IQ2s' grids and signs in LDS (a lane each
  its own index) for gate and up - 1-2% alone (173.6 -> 171.4, 192.2 -> 188.8 us), not kept (its plumbing for less
  than a decode's spread). what is left of the gap is the cache's, not the kernel's.

## Engram's reads (measured, 2026-10-29)

a step reads each token's 24 rows of 144 B for each of the 2 Engram layers: **98,304 random reads** a 2,048-token
step, in a 110 GB file the page cache mostly does not hold. scratch `egread.c`, random rows of the Engram file:

| threads | reads/s | a step's rows |
| ---: | ---: | ---: |
| 1 | 16,700 | ~5.9 s |
| 4 | 87,000 | ~1.1 s |
| 16 | 297,000 | ~0.33 s |
| **64** | **832,000** | **~0.12 s** |
| 128 | 870,000 | ~0.11 s |

so `rowread.h`: 64 workers that live for the forward's life, a layer's rows handed to them at once (16 a take).
(fixed 2026-10-30, found by `make tsan`: a worker woken for a batch already done was still reading its fields when
the next call wrote them - harmless while it took no row, a wrong read had it taken one. a call now waits for every
worker to be out of the last batch first; test_rowread's 20,000 calls of a row or two catch it 5 runs of 5 without.)

## the experts' matmul (measured, 2026-10-29)

a real-text 2048-token step: a layer's ~306 experts, 54% of them one tile of 16 tokens, 15% two, 15% up to four, 17%
more (the hottest ~1100 tokens). each block decodes a 16-row slab once for its column tiles; decode and WMMA issue on
the same SIMD, so the floor is their sum (a real layer's gate: ~8.5 ms of decode, ~6.5 of WMMA). what moved it:

- **a launch a run of the same column tiles** (`mm_col_tiles`): one launch took its tiles from the hottest expert,
  so every one-tile expert ran the 8-tile kernel (164 VGPRs, fewer waves): 37% slower for it. 290 -> 294 tok/s.
- **IQ2's f16 decode**: each |value| rounded, the sign one xor a pair of halves (rounding to nearest is symmetric) -
  the decode's VALU 268 -> 172 a slab, the same bits (`test_quant_gpu`: every value gguf-py's, rounded).
- **the shape**: 16 waves of 64-value slabs ran 5 slabs a wave over 5120 columns, then a 16-way sum through LDS a
  tile (the counters: waves 52-76% waiting, 14% at the barrier). 4 waves of 128: a real layer's experts 17-31%
  faster, a decode token 74.0 -> 71.6 ms, a 4096-token prompt ~294 -> ~314 tok/s (interleaved runs, medians). a
  shape is a type's for every token count: its K split is the sums' order.
- not it: prefetching the next slab's raw words (no gain at one tile, slower at more), global instead of flat loads
  (none), the WMMAs' order (only 8 tiles, -12%). the activations' reloads (a 16-row block reads its tokens' x again)
  are ~23% at 8 tiles.
- **the dense Q5_K matrices** (q_b, wo_a, the shared expert's gate and up; ~18% of a pass): at 2048 tokens 19-24
  TFLOPS, the waves ~80% waiting - each tile's x loaded, then waited for, then its WMMAs, eight round trips a slab.
  with every tile real, the next tile's x now loads under this one's WMMAs (the same bits): wo_a 7.24 -> 5.69 ms, a
  4096-token prompt ~322 -> ~328 tok/s. only for slabs of few fragments (`kPipe`: G <= 2) - the IQ experts' 8 a tile
  doubled is 64 more VGPRs, a real layer 15.2 -> 22.0 ms. measured and not taken: Q5_K at 4 waves of 64 (q_b 8.75 ->
  7.55 ms, but a decode token +0.5 ms, and wo_a's gain the pipelining's already); 2 or 4 row tiles a block, each x
  for 2-4 WMMAs (the same bits, no faster); 16 column tiles (registers: slower).
- q_b (1280 -> 32768) is its own case: 5 slabs a wave, 268 MB of y a step, its weights read again by each of the 16
  token blocks; not the epilogue (without the cross-wave sum it was slower). open.
- found on the way: the compiler fused decode's last product and the f16 rounding into `v_fma_mixlo_f16` (a * b + 0,
  rounded once): +0 for -0, and Q6_K's 25-bit products not round(round(x)) - the matmul's Q6_K weights were never
  quite gguf-py's rounded. an empty asm keeps the two roundings (not volatile: a volatile one cost a decode token 11
  ms).

## determinism (operator, 2026-10-28)

maya's prompt path is not run-to-run deterministic on this box: two runs of the bar's maya-ds4 on the same 32 real
tokens (`~/osmos-oracle`, `dumpcmp.py layers`) part from layer 0 (worst 1.1e-8 at position 8) and grow - layer
10 median 1.3% (worst 18%), layers 30-38 median 4-7% - to logits KL 2.5e-3 mean, 1.4e-2 max, the argmax the same at
32 of 33 positions (the 4 greedy tokens the same). osmos diverges from maya here:

1. **run-to-run determinism - a rule.** the same request on the same build gives the same bits, every run: no
   floating-point atomics, a fixed reduction order in every kernel, hipBLASLt solutions taken only where
   deterministic (no atomic split-K), no CPU lane (the pool pins what the GPU reads). tested: two runs under different
   thread timing and load, every dump record and every logit bit for bit.
2. **schedule invariance - the target.** a request's answer does not depend on how the scheduler served it: its
   prompt read whole or in pieces of any size, alone or beside other conversations, DSpark drafting or not. so every
   kernel computes a row the same way whatever the batch or chunk around it (a row's tiling and reduction fixed by
   its own shape, never by T). then libevt's policies are correct by construction: no policy can change an answer.
   tested as a property (random piece sizes, random interleavings, against the request alone). its cost is
   **unmeasured**: measured on the bar, and if it costs more than scheduling gains, it is a mode, said so.

## the oracle (updated 2026-10-28)

- **numerics: maya's torch reference**, deterministic F32, sharing no engine code - `oracle/ref.py` and
  `oracle/ref_gguf.py`, copied from maya (MIT) with their attribution, run in `.oracle-venv` (CPU torch: a tool,
  never the runtime). fixed here: maya's `ref_gguf.py` reads Engram rows as Q6_K or MXFP8 only, and this model's are
  **IQ4_NL** (144 B a row) - read as MXFP8 they would be silently wrong. the weights' types it covers through gguf-py
  (IQ1_M, IQ2_XXS, IQ2_XS, Q5_K, Q6_K, Q8_0, IQ4_NL, BF16: checked).
- **measured, the first 8 positions of 32 real tokens** (`~/osmos-oracle`: `ref_gguf.py --dev cpu`, 334 s for 8
  tokens): KL(reference || maya) 0.0002-0.018 a position, maya against itself 0-0.0057; the argmax the reference's at
  7 of 8 (position 6: 67 against 75). osmos's numerics target: bit for bit run to run, and at least as close to
  the reference as maya is (int8 activations and FP16 prompt GEMMs put maya a few times its own spread away).
- fixed to read this model, besides the Engram rows (each `# osmos:` in `oracle/ref_gguf.py`): the embedding is
  Q5_K, not Q8_0; the Engram gate is `q_weight` and `k_weight`, not folded. maya's `ref_gguf.py` would have read the
  Engram rows wrong silently and refused the other two; all three are fixes to offer maya.
- **the bar and a sanity check: maya-ds4's dumps** - `DS4_DUMP_L` (every layer's residual streams), `DS4_DUMP_R`
  (routing), `DS4_DUMP_Y` (expert sets), `--dump-logits`. osmos writes the same records (`dumpcmp.py` unchanged);
  against maya it must stay within maya's own run-to-run spread, measured per layer.

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
- **C for the host code**, libevt's style (Allman braces, 2-space indent, `Type` / `module_verb`, notes on each
  function in a block after the header's `#endif`), and libtrdr's `scope.md` memory model: every type has
  `type_create` / `type_destroy`; ownership is a tree - a component frees what its scope allocated and never what it
  borrowed; a constructor that fails releases what it took and returns NULL; `destroy` takes NULL. HIP C++ only for
  the kernels, behind a C interface.
- **no leaks, no faults, checked**: every host test runs plain and under ASan + LeakSanitizer + UBSan (`make asan`) and
  ThreadSanitizer (`make tsan`); the device code stays behind the C interface so the host's tests need no GPU.

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
  copied through a staging ring at 1.4 GB/s - **measured 2026-10-29: in place stalls a long prompt for minutes**;
  copies straight from the files at the SSD's speed instead, "the memory, revised"), prompt kernels for this GPU, and
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

## UD-IQ1_M: the repack (inventory, 2026-11-02)

the operator's call: run unsloth's `DeepSeek-V4.1-Flash-1017-GGUF` UD-IQ1_M, a better quant at about the same size.
maya's scorer puts it at **KLD 0.198, 90.3% the same top token** as the original (`maya.py`, its `gguf-iq1_m` text;
the scale of its Q8 0.047, UD-IQ3_S 0.061, Maya-S 0.119). GSQ-RCO has no number on that scale: **the gap is
unmeasured**. the first step is both through the oracle on the same tokens (`compare.py`: KL, top-1), before any kernel.

read from the 5 shards' tensor directories and metadata only (`~/Models/ds1017/UD-IQ1_M`, shards 2-5 mid-download;
a GGUF's directory is at its start, no tensor data read). nothing was run.

### what changes

| part | GSQ-RCO (built for) | UD-IQ1_M | here |
| --- | --- | --- | --- |
| experts gate / up | IQ1_M 16, IQ2_XXS 17, IQ2_XS 7 | **IQ1_S 20**, IQ2_XXS 18, IQ2_XS 2 | IQ1_S new |
| experts down | IQ1_M 28, IQ2_XXS 8, IQ2_XS 4 | IQ2_XXS 15, **IQ2_S 10**, IQ2_XS 10, **IQ1_S 5** | IQ2_S, IQ1_S new |
| shared experts | | Q8_0 (27 layers), Q5_K (13) | have |
| attention q / kv / output, head | | Q6_K | have |
| compressors, indexer | BF16 | BF16 | have |
| embedding | Q5_K | Q8_0 | have |
| Engram tables | `engram_embd` a layer, IQ4_NL, own GGUF | one `per_layer_token_embd` [256, 768,022,850], IQ4_NL, in the backbone | the slicing new |
| Engram kv | `engram_wkv` Q5_K | `engram_kv` [6144, 25600] Q6_K | have (the name new) |
| Engram gate | `engram_q`, `engram_k` BF16 | `engram_gate` [5120, 4] F32, q·k folded | the folded form new |
| Engram hash | I32 / I64 tensors | `ple.*` metadata | the reader new |
| DSpark | maya's pack | not in the files (40 blocks; `dspark.*` metadata only) | maya's pack, as now |

despite the name there is **no IQ1_M** in it: its 1-bit experts are IQ1_S.

**the sizes**: experts 132.2 GB (IQ2_XXS 59.6, IQ1_S 39.8, IQ2_XS 18.3, IQ2_S 14.5), against GSQ-RCO's ~132 GB; the
Engram tables 110.6 GB (768,022,850 rows of 144 B), as before. the budget picture is unchanged in total; per layer it
is not, so the pick profile and the budget numbers do not carry over.

### the work, by part

1. **IQ1_S** (`block_iq1_s`: f16 `d`, `qs[32]`, `qh[8]` u16; 50 B a block of 256, 1.5625 bpw). `qh` carries a group
   of 32's high index bits (3 for each 8), its 3-bit scale (`d * (2s + 1)`) and the delta's sign. the grid is
   `kIq1sGrid`, already generated (`tools/gen_grids.py` asserts IQ1_M and IQ1_S share it). the lane's contract
   (`kernels/lane.h`) extends as IQ1_M's: integer dot products on the grid's values, the delta as a separate term,
   one scale a group of 32 instead of IQ1_M's packed ones. three places, as every expert type: the GPU decoder
   (`kernels/quant.h`), the int8 expert kernels (`mm`, `q8`), the CPU lane (`src/cpu/expert.c`).
2. **IQ2_S** (`block_iq2_s`: f16 `d`, `qs[64]`, `qh[8]`, `scales[8]`; 82 B, 2.5625 bpw). `qs`'s first 32 bytes are the
   index's low 8 bits, the last 32 the **signs, stored** (not `kKsigns`); `qh` 2 high bits for each 8; a 4-bit scale
   for each 16. its grid is its own: `kIq2sGrid` (1024 entries) to add to `gen_grids.py`; its values are the other
   IQ2s' (0x08 / 0x19 / 0x2b). the same three places.
3. **the metadata's spelling.** 17 keys `weights_geo.c` reads are not in this file:
   `attention.candidate_{block_size,source_layer_id,topk_blocks}`, `dspark_target_layer_ids`,
   `{index,kv}_source_layer_ids`, `expert_scoring_func`, `engram.{present,file,source,layer_ids,num_embeddings}`,
   `engram_{compressed_vocab_size,head_dim,max_ngram_size,n_heads,pad_token_id}`. most are llama.cpp's newer names
   for the same values (maya's `pack_gguf.py` `NEWER_KEYS` maps them: `attention.indexer.candidate_source_layer`,
   `attention.kv_source_layers`, `dspark.target_layers`, ...). one geometry, two spellings read into it at the load
   seam - not a second geometry.
   - `expert_scoring_func` "sqrtsoftplus" is absent; the file has `expert_gating_func = 4`. **checked** (2026-11-03):
     llama.cpp's `llama_expert_gating_func_type` at `272aad8b` (maya's vendored ref) has `SQRT_SOFTPLUS = 4`
     (`src/llama-hparams.h`). the gating func is the field read; the scoring text, where present, must agree; any other
     value refuses.
   - the Engram hash from `ple.*` (maya's `ple_to_engram`): `ple.layers` [1, 14], `ple.ngram_size` 4,
     `ple.heads_per_ngram` 8, `ple.layer_multipliers` (8: 2 layers x 4), `ple.head_vocab_sizes` (the primes, 48: 2
     layers x 24 columns), `ple.head_offsets` counted **across** both layers' tables (a layer's own: minus its first).
     a layer's rows: its last offset + last size - its first offset; head_dim from `embedding_length_per_layer_input`
     (256). `engram.token_map` (129,280 ids), `engram.pad_token_id` 2, `engram.compressed_vocab_size` 99,092 are
     metadata, not tensors. **checked against `engram_hash.py`'s derivation** (`10007 * layer` seeds, primes above
     `engram.vocab_size - 1`): the 8 multipliers equal, the primes' first 6 equal, the derived rows 384,006,168 +
     384,016,682 = 768,022,850, the tensor's. the reader checks the same at load: a mismatch refuses.
   - `engram.file` / `engram.source`: there is no second file; the tables are the backbone's.
4. **the Engram tables in one tensor.** each layer's table is a row range of `per_layer_token_embd`, the layers' in
   `ple.layers` order (maya's `split_ple`: the same bytes, read in place). `rowread.h` reads in place already; only a
   layer's base offset changes. the layers' rows must sum to the tensor's 768,022,850, or the load refuses.
5. **the folded gate.** `engram_gate` is q·k as F32. a product of two BF16 values is exact in F32 (8-bit
   significands, 16-bit product), so if the converter multiplied in F32 the folded weight is GSQ-RCO's q·k bit for
   bit - **unverified** (no GSQ-RCO file on this unit). whether the old bits carry over also depends on the kernel's
   order (`h·q·k` as `(h·q)·k` rounds differently from `h·(q·k)`). either way the guarantee is a model's own and holds.
6. **DSpark** is not in the GGUF: `dspark.bin` / `dspark.json` from maya's pack, as now. correctness does not depend
   on it (every draft is verified); its acceptance on this backbone is to measure.
7. **the pick profile** recounted (`forward_picks` on the calibration text) and the budget re-measured.
8. **the oracle.** gguf-py decodes IQ1_S and IQ2_S (`quants.py`), so `ref_gguf.py`'s weights are covered; its
   `# osmos:` fixes need the UD spellings: embedding Q8_0, the gate folded, the tables sliced from
   `per_layer_token_embd`. a new reference dump for `test_model_gpu`.
9. **the paths.** the tests look in `~/models/ds1017`; the files are in `~/Models/ds1017/UD-IQ1_M`.

### the order

1. the quality, measured (the oracle, both quants, the same tokens) - it decides whether the rest is worth doing.
2. schema: the geometry read from either spelling; the two types' ids in the type tables.
3. tests that fail first: IQ1_S and IQ2_S decoded against gguf-py on real blocks, with mutations (a sign bit, the
   delta's sign, a scale nibble) that must fail them; the CPU lane's bits equal the GPU's for both; the geometry read
   from the UD file equal to a hand-written one; the Engram row ranges' sum checked; the fingerprints across batching,
   budget, drafting and CPU / GPU on the new model.
4. the kernels, then the numbers against maya on this file (maya runs it: `--ds4-quant gguf-iq1_m`).

### the box

this is the second unit: **all 16 cores online** (the protocol's "core 1 offline" was the first unit's). the bar is
re-measured here before any comparison.

## another pack, another model: what to generalize (plan, 2026-11-03)

the operator's question: does a second quant pack (UD-IQ1_M, the Coder) or a second model (GLM-5.3, `ox-alpha`) need
the code generalized first. the answer: the quant types yes, the model no.

### what a second pack touches today

- **a type is spelled four times**: weights.c's enum, `quant::k*` (`kernels/quant.h`), `GGUF_*` (`src/cpu/expert.c`),
  `LANE_Q8_*` (`kernels/lane.h`). 15 `switch` on a type (`mm.hip` 4, `mv.hip` 3, `quant.hip` 3, `q8.hip` 2, `ops.hip`
  1, `src/cpu/expert.c` 1, `src/cpu/q8.c` 1); four lists of what a path takes (`mm_known`, `quant_known`,
  `cpu_mv_known`, `lane_q8_type`); block bytes written inline (`src/cpu/q8.c`: 66 / 74 / 56). a new type is edits in
  about a dozen files, and nothing says one was missed.
- **roles pin GSQ-RCO's types**, not what the kernels run: `token_embd` Q5_K, `attn_output_b` Q6_K, experts
  `IQ1_M|IQ2_XXS|IQ2_XS`. UD's embedding is Q8_0 and the Coder's `attn_output_b` Q5_K: each refused, on a type every
  path already runs.
- **the metadata has one spelling** (`deepseek41.*` as maya's pack writes it).
- the expert count is metadata everywhere (`OPS_MAX_EXPERTS` 1024): 192 needs nothing.
- `mem/`, the kernels and the lane name DeepSeek only in comments.

### the plan

1. **one type table.** each ggml type this build runs, once: its id, a block's elements and bytes, the paths that take
   it (GPU decode, matvec, int8 matmul, the CPU lane). the four enums, the known-lists and the inline sizes become
   projections of it; a test fails when a path the table names has no case for a type. a pure refactor: the
   fingerprints before and after equal, the proof nothing moved.
2. **roles by capability.** a role names the path it is read by (dense matvec, an expert's int8 matmul, an Engram row
   read) and the shape; the loader takes any type the table gives that path. a type no path takes still refuses, the
   role and the type named. shapes stay pinned.
3. **one geometry, two spellings.** an alias table at the load seam: maya's names and llama.cpp's newer ones for each
   field. read once; both present and unequal refuses; derived values (the Engram vocab sizes, a layer's rows)
   computed and checked against the file's, as the UD section's item 3.
4. **the model stays one.** no model interface until a second model is built. GLM-5.3 (`glm5-next`: 34 KDA
   linear-attention layers, 12 MLA + indexer, 288 experts top-8 sigmoid, nextn MTP, no Engram) would share the type
   table, the memory and the expert kernels, not the forward pass; an interface drawn now, from one model, would be
   drawn wrong. what is free now: the DeepSeek-specific code under one name (`model/deepseek41/` when it moves), the
   gating function a geometry field rather than a constant in the router.

### the Coder first (read 2026-11-03, download complete)

`Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF`, `~/Models/ds1017-coder`: 192 experts (pruned), 6
used; backbone shards 1-5 72.1 GB, Engram shard 6 110.6 GB, an mmproj (not read).

| part | the Coder | needs |
| --- | --- | --- |
| experts | IQ1_M / IQ2_XXS / IQ2_XS, GSQ-RCO's mix | nothing new |
| metadata | llama.cpp's newer spelling (`attention.kv_source_layers`, `attention.indexer.source_layers`, `attention.indexer.candidate_{source_layer,top_k_blocks,block_size}`); gating func 4, no scoring text; the hash in `ple.*` | plan 3 |
| Engram tables | one `per_layer_token_embd` IQ4_NL [256, 768,022,850] in shard 6, as UD's | UD item 4 |
| Engram gate / kv | `engram_gate` F32 [5120, 4] folded; `engram_kv` Q5_K (UD's Q6_K) | UD item 5 |
| `attn_output_b` | Q5_K (pinned Q6_K) | plan 2 |
| `exp_probs_b_vl.bias` | F32 [192], all 40 layers (UD has it too) | a role, checked, not read |

`exp_probs_b_vl` is the router's bias for image tokens: llama.cpp's `deepseek41.cpp` uses it only for a batch of
embeddings. this build runs text, so it is a role whose shape is checked and whose bytes are never read, not an
exception to "a tensor this build does not know".

the Coder is UD's items 3-5 without items 1-2: the half of the repack that needs no kernel, on a model whose
backbone fits in GPU memory whole. UD then adds IQ1_S and IQ2_S to a load path already proven. its quality is as
unmeasured as UD's: pruned to 192 experts, it goes through the oracle the same way.

### the order, revised (operator, 2026-11-03)

the operator's call: **UD-IQ1_M first** (the quant most people want), **the GSQ-RCO path kept**. the Coder is off this
box, and the GSQ-RCO set never was on it (`~/models/ds1017` absent: every model test skips here, exit 77). so:

- **the GSQ-RCO path is kept, not proven end to end here.** its types stay in the table; the CPU tests (each type's
  bits on generated blocks: `test_cpu_q8`, `test_cpu_lane`) and the GPU kernel tests that need no model hold it. its
  fingerprints wait for its files (ISTA-DASLab `cb02422c`, 249 GB, a download the operator has not asked for). said,
  not assumed: a refactor here is proven by the kernel tests, not by the model's fingerprints.
- **UD-IQ1_M is proven against the oracle and maya**, as the UD section's order: gguf-py decodes IQ1_S and IQ2_S
  (the oracle's weights), maya runs this file (`maya-ds4-hip.json`, its dumps the bar and the sanity check).

1. schema: the type table (`include/osmos/model/qtype.h`): each type once, its id, a block's values and bytes, the
   paths that read it. tests: each path the table names takes the type (a GPU call on a block of it is not refused),
   and every table today agrees with it - a mutation (a path named, its case removed) must fail them.
2. the projection: `gguf.c`'s sizes, `weights.c`'s enum and masks, `quant::Type`, the lane's and the CPU's ids, the
   known-lists from the table. the kernel tests unchanged, the CPU lane's bits unchanged.
   **done (2026-11-03)**: the kernel tests on real blocks skip here (they read ds1017), so the proof is the build's:
   every kernel's GPU code (`.hip_fatbin`: mv, mm, q8, quant, ops, attn, sparse_attn, round) byte for byte the
   commit before's, and every host object's `.text` the same but three, each changed by design and tested -
   `gguf.o` (sizes from the table: test_qtype's sweep, test_gguf), `quant.o` and `mm.o` (`quant_known`, `mm_known` the
   table's: test_qtype_gpu's sweep calls the kernels). `weights.o` the same bytes: its rename proven where test_weights
   skips. left as they are: the kernels' own `switch`es (a case each type; test_qtype_gpu finds one missing), Q5_KR
   (the forward's repack, no file's type) and `DECODE`'s fall-through to IQ1_M in `src/cpu/q8.c` (a new type there
   needs its own arm).
3. IQ1_S and IQ2_S: rows in the table, then their decoders against gguf-py on real blocks of the UD file, the lane's
   contract for both, the CPU lane's bits the GPU's.
   **done (2026-11-03)**: `kIq2sGrid` from gguf-py (`gen_grids.py`; the file regenerated byte for byte first), the
   rows (IQ1_S 19, 50 B; IQ2_S 22, 82 B), each path: the decode, the matvecs, the matmul, the lane's int8 experts
   (IQ1_S IQ1_M's 8 grid +- 1 with one s and one delta sign a 32; IQ2_S IQ2_XS's with its grid and the signs stored -
   on the CPU the pair's 8 sign bytes are its negatives' mask as they are), the CPU lane's both. the tests on the UD
   file where ds1017 is not (`tests/gpu/test_model.h`): test_quant_gpu bit for bit gguf-py's on every type the table
   decodes (IQ1_S and IQ2_S 3.67 M values each; IQ1_M not checked: no UD tensor), test_mv_gpu and test_mm_gpu within
   their bounds and schedule-invariant on 17 and 11 UD matrices, test_q8_gpu the contract's bits against the host's
   replay (IQ1_S's and IQ2_S's written from gguf-py's layout, not the kernel's), test_cpu_expert_gpu,
   test_cpu_q8_gpu and test_cpu_lane_gpu the CPU's bits the GPU's on four UD layers' type mixes. 12 GPU tests run
   here, 6 before. `DECODE`'s fall-through to IQ1_M is gone: an arm a type, else abort.
   - **found**: test_cpu_q8's loop ran its first 3 types whatever the list (`i < 3`): the new two passed untested
     until it took the list's length.
   - **found**: UD's router (`ffn_gate_inp`) is **F32**, GSQ-RCO's BF16 - a role's type for step 4.
   - **to measure**: the new types' launch shapes are their sibling's (IQ1_M's, IQ2_XS's), not measured. IQ1_S is
     the slow one: an expert's matvec ~75-80 GB/s at 1 token (the IQ2s 120-137), the matmul 43-53 (the IQ2s 75-108).
4. the UD file's load: roles by path, the alias table, the Engram slices, the folded gate, `exp_probs_b_vl`.
   read 2026-11-03 from the file (all 62 `deepseek41.*` keys) and the code that reads it; four parts, in order:
   - **4a, the geometry.** one `Geometry`, each field read once from a key or its alias (maya's spelling, then
     llama.cpp's: `kv_source_layer_ids` / `attention.kv_source_layers`, `index_source_layer_ids` /
     `attention.indexer.source_layers`, `dspark_target_layer_ids` / `dspark.target_layers`,
     `attention.candidate_{topk_blocks,block_size,source_layer_id}` / `attention.indexer.candidate_{top_k_blocks,
     block_size,source_layer}`, the Engram's below); both present and unequal refuses; an integer key read at its own
     width (UD's `candidate_source_layer` is INT32, the reader asked UINT32). `expert_scoring_func`, where present,
     must be "sqrtsoftplus"; `expert_gating_func` 4 is the field. the Engram from `ple.*`: layers `ple.layers`, heads
     `ple.heads_per_ngram`, n-gram `ple.ngram_size`, head dim `embedding_length_per_layer_input`, pad
     `engram.pad_token_id`, vocabulary `engram.compressed_vocab_size`, a layer's rows its columns' span (last offset
     + last size - first offset); the Engram in the model's own files (no second Gguf, no `engram.source` to match).
     test: the UD file's geometry equal to one written by hand from the dump above; each alias pair made unequal
     refuses; a missing key of each refuses; GSQ-RCO's spelling (the same file's keys renamed in memory) gives the
     same geometry.
     **done (2026-11-03)**: `weights_geo.c`'s `lookup` (the 6 pairs; an error names both spellings), integers at any
     width, `ple_engram` (the columns must tile from 0 in order, the layers ascending; neither `ple.layers` nor
     `engram.present` refuses naming both). `test_geo` on the UD file: the geometry by hand (rows 384,006,168 +
     384,016,682), ds1017's spelling the same, 2 unequal pairs and 11 missing keys refused, gating 2 refused, an
     overlapping column refused. the compress ratios are 0 / 1 / 2 in this file and maya's pack of it alike
     (`ds4.txt`), and the reference reads them literally (`ref.py`: 1 no pooling, 2 pairs): not a code to translate.
     not proven here: ds1017's own Engram path (`engram.present`, the second file), unchanged but for the reader's
     width and the error's wording.
   - **4b, roles by path.** a role names the path that reads it (dense matvec, the experts' int8, an Engram row, a
     vector read as F32) and its shape; any type the table gives that path. UD's own roles: `engram_kv` (GSQ
     `engram_wkv`), `engram_gate` F32 [dim, hc] (GSQ `engram_q` x `engram_k`), `per_layer_token_embd` (the layers'
     tables, sliced: its rows the layers' sum), `exp_probs_b_vl` (checked, never read: image routing). the router
     F32 (GSQ BF16).
     **done (2026-11-03)**: a role names its path - MAT (`fwd_mat`: the matvec, the matmul where it has one),
     GROUPED (`mm_group` alone: wo_a), EXPERTS (by tokens and grouped), EMBED, ROWS (the Engram's decode) - or reads
     raw and names its one type (norms, biases, the hyper-connections, the gate's halves or its fold, the hash
     tensors). each form's Engram roles on its form's layers only (`ENGRAM_FILE` / `ENGRAM_OWN`, `geo.engram_own`);
     wo_a's other spelling (UD's [4096, 8192] for [4096, 1024, 8]: the same bytes); the image bias optional and not
     stored; the own form's tables `eg_slice`, each a layer's rows of per_layer_token_embd in order. found too:
     `include/osmos/gpu/quant.h`'s `QUANT_Q5_K*`, a fifth copy of a type's numbers, now the table's.
     test_weights on the UD file: every role (Q8_0 embedding, F32 router, 2-dim wo_a, IQ1_S / IQ2_S experts, the
     gate folded, the slices tiling the tensor), 7 refusals (wo_a in BF16, the folded gate in BF16, the tables a row
     long, an I32 router, IQ1_S experts called IQ2_XXS, the image bias short, engram_kv moved off its layer).
     until 4c and 4d, the own form refuses loudly where it is run: `engram_create` (the hash) and `forward_create`
     (the rows' file, the gate) say which step, rather than reading NULL.
   - **4c, the hash.** `engram_create` from the tensors (GSQ) or the metadata arrays (UD: `engram.token_map`,
     `ple.layer_multipliers`, `ple.head_vocab_sizes`, `ple.head_offsets` less each layer's first) into the one
     `Engram`, its checks unchanged. test: both readers' rows the same for the same tokens (`engram_hash.py`'s rows
     on the UD file).
     **done (2026-11-03)**: `engram.c`'s `from_meta` (the offsets less each layer's first; the map an int32 an
     entry, its length the vocabulary's); the integer reader one function, `gguf_int` (gguf.c), for it and the
     geometry. `oracle/engram_rows.py` writes the reference: the layout, primes, offsets and multipliers from
     DeepSeek's recipe (`engram_hash.py`), only the token map the file's (deriving it needs the tokenizers package);
     the recipe's primes give the file's rows (384,006,168, 384,016,682). test_engram: 64 positions (seeded ids,
     the vocabulary's ends and the pad among them), 48 rows each, every one the oracle's; a map entry past the
     compressed vocabulary and an even multiplier refused. the token map itself is the file's, checked only for
     range: an independent check of it needs the tokenizer.
   - **4d, the forward.** one gate: an F32 [hc][dim] product, GSQ's folded at load (BF16 x BF16 is exact in F32: the
     kernel's own product, the same bits), UD's read; the Engram rows from a layer's slice of the model's file.
     **done (2026-11-03)**: `ops_engram_gate` takes the gate (F32); `gates()` builds each Engram layer's at load
     (UD's read, ds1017's q k multiplied on the host) into its own device buffer, the raw q / k / gate tensors no
     longer resident; `engram_fd` the rows' file (ds1017's second, or a dup of the model's shard holding
     per_layer_token_embd). the gate's bits: the kernel with q, k before and with the host's fold after, on the same
     seeded inputs (7 tokens x 4 streams x 5,120, a subnormal product among them) - the 143,360 outputs byte for
     byte the same (a one-time check in the scratchpad: no repo test calls the gate alone; the forward's tests do,
     on ds1017). `tools/bench/dbench.hip` takes `MODEL` (a first shard) and `ENGRAM` (where there is one) and
     `IDS` (the answers' token ids). **UD-IQ1_M runs**: loaded in 16 s; greedy answers to maya's bench questions
     read right (five greetings, a typed CSV parser with its docstring, photosynthesis step by step) at 12.1-13.3
     tok/s - no picks profile, no CPU lane, no DSpark: not a number to compare yet (maya on this file: 18.0
     undrafted, 29.3-29.9 drafted).
5. the oracle and maya: the logits within maya's spread, then the numbers.
   **the logits (2026-11-03)**: 32 real ids (the GPL-3 prompt's first, `~/osmos-oracle/ids32.txt`); the oracle
   `ref_gguf.py --dev cpu` on maya's pack of the file (735 s; ppl 20.3), maya `maya-ds4 --ids --dump-logits` on the
   same pack, test_model_gpu on the UD file (tests/gpu/test_model.h). before today's kernel changes: ours mean KL
   0.046, maya's 0.272; argmax the oracle's 30 of 32 (both misses ties), maya's 29 (its position 18 KL 5.8); the lane
   30 of 32 at 0.201; the logits bit for bit run to run and under 2,931 swaps. **position 18 is fragile**: after the
   Q6_K shape below (any of the shapes tried) ours goes where maya's does there (KL 5.9; the stream from layer 21 on
   ~10^3 the reference's largest at that position alone), while each layer forced onto the oracle's stream keeps its
   error (<= 0.17, as before) - a rounding's turn amplified, not a layer's fault; the other 31 positions' mean KL
   0.037-0.047 across the builds. one 32-token sample cannot rank builds whose sums differ: a longer evaluation is
   what would (the oracle's ~12 min a 32 tokens on the CPU is the cost).
   **the decode (2026-11-03; maya's three bench questions, 256 greedy tokens, an unrelated warm-up; drafting off on
   both; maya's config with `--dspark off`: 16.6 / 17.0 tok/s, 16.8)**:
   - by id 12.1-13.3; a profile (16,000 tokens: Strata's docs, the Apache and MPL licenses, Strata's Python,
     maya's C++ - `~/Models/ds1017/UD-IQ1_M/osmos-picks.bin`, dbench `PICKS_OUT`) ranked at 106 GiB ~15.0.
   - the budget: 110 GiB (this unit's GTT 113 GiB; 112 refused) 15.27 against 15.0 at 106, misses 3,479 / 4,368.
   - the CPU lane on all 16 cores (the first unit's note was 15): 0 / 4 / 8 / 12 / 16 threads 15.2 / 15.2 / 15.1 /
     15.2 / 14.6 tok/s (two rounds) - it takes ~85% of the misses and buys nothing; 16 threads slower. off.
   - traced against maya (rocprofv3, 127 decode steps): the GPU's work the same (ours 57.0 ms a token, maya 57.6),
     the idle not (12.2 against 4.7). Q6_K - UD's attention, the head - ours 26.9 ms, maya's matvec 23.5; the F32
     router 4.2 (its guess a second pass, maya's too).
   - Q6_K's short rows (mm.hip `Short`: q_b's 1,280 columns) 8 waves of 64-value slabs: 171 -> 197 GB/s, its error
     1.6 -> 1.4 u|wx|; ~15.5 -> 15.7 tok/s.
   - an F32 router's guess on a BF16 copy (hip_guess.c, `guess_routers`: the guess picks reads ahead, no bit): the
     three answers' hashes the same, ~15.5 -> 15.8 (with the 4-wave shape).
   - now (64 x 8): 15.6-15.9 tok/s (rounds 2-4; the first 14.6), **~94% of maya's 16.8**. the trace: busy 53.9 ms a
     token (maya 57.6), idle 10.4 (maya 4.7) - 3.7 in ~1,500 launch gaps, 6.6 in ~6 waits a token over 100 us
     (median 0.55 ms) behind the routed experts: the misses, which the CPU lane does not hide here. left: the misses'
     waits, the token's end (~1.5 ms: the logits to the host, the next step's tables), q_a / kv's matmul (~134 GB/s
     against the matvec's 171: 80 and 32 row tiles, too few blocks).
   - **the three, tried** (a host-side probe, not kept: clock_gettime around each wait):
     - the misses: the pool never full (0 waits for room); the host waits on a missed expert's read 5.9 ms a token
       (2.8 waits of ~2.1 ms: an expert ~7-9 MB, ~4 GB/s, in 1 MiB jobs on 4 workers - the urgent queue's chunks).
       the readers FWD_FETCH_THREADS 4 / 8 / 12 / 16 / 24: the waits 5.9 / 5.5 / 4.8 / 4.9 / 4.9 ms, 15.8 / 15.8 /
       16.2 / 16.1 / 16.1 tok/s; 16 readers with 512 KiB jobs worse (5.3 ms). **12 kept**: a load 17-18 s, an 8K
       prompt ~490 tok/s, a 2K ~310-340, both counts alike (106 GiB, two rounds each).
     - the token's end: Engram's rows waited 0.6-0.75 ms a token (maya 0.35, its row cache ~20% hits). their own
       open with POSIX_FADV_RANDOM (the dup shares the shard's readahead): no change - not kept. the rest (~0.4
       ms) is the host's between the pick and the next step's launches: the next token's ids are its input.
     - q_a / kv: Q6_K's long shape 64 x 16 140 GB/s on q_a (136), each other shape the head's or out_b's loss;
       128 x 16 and 256 x 8 past the LDS (64 KiB). a 2-5 MB matrix is its launch's latency, not its shape's: left.
   - **now: 16.0-16.2 tok/s (16.0), ~95% of maya's 16.8** - the three questions, two rounds, 110 GiB.
   **position 18, checked** (a temporary print in test_model_gpu: the position's drift and its 6 routed experts at
   each layer, the Q6_K shape before (128 x 8 everywhere) and after (64 x 8 on q_b)): the two builds' routes part by
   one expert at layers 7, 9, 11, 12 and 18 (near-ties, the streams staying near) and by two at **layer 20** - [3 62
   96 259 282 338] before, [96 180 259 282 284 338] after - where the after build's drift goes 0.03 -> 0.32, then 39
   at layer 21 and ~10^3 on; the before build's stays ~0.03 there. layer 20 at position 18 is where the oracle's own
   stream bumps (its largest value 48 against ~7 at the other positions' and the layers' around), and both builds,
   forced onto the oracle's stream, compute layer 20 to 3.6e-4: a router near a tie, tipped by an input ~3% off (the
   ordinary rounding of 20 layers), then a different expert pair's large output. not a kernel's fault; the before
   build was right there by chance, and maya's stream tips there too. what ranks two builds' accuracy is a longer
   evaluation (many such positions), not one 32-token sample's mean KL.

### the order (before 2026-11-03, kept for the record)

1. the type table and roles by capability: the pure refactor, fingerprints equal on the current GSQ-RCO pack.
2. schema: the alias table; the Engram slices of one tensor; the folded gate; the `exp_probs_b_vl` role.
3. tests that fail first: the Coder's geometry read equal to a hand-written one; each alias conflict refuses; a role
   given a type no path takes refuses; a table type with a path missing its case fails; the slices' rows summed
   against the tensor's.
4. the Coder loaded and generating; the fingerprints across batching, budget and CPU / GPU on it; the oracle.
5. UD-IQ1_M: IQ1_S, IQ2_S and the rest, as the UD section's order. its quality measurement (that order's step 1)
   waits on none of this.

## the KV in the model's formats (2026-11-03)

the operator's point: DeepSeek's design keeps ~0.92 KB a token (0.96 GB at 1M; "the model" above), and this build kept
~6.4 KB - each value of the attention's state in 4 bytes, though every one is already rounded to the model's own
grid (ref.py): the window's keys `fp8_round` (E4M3, a power-of-two scale a 32), the compressed rows `fp4_round(16,
e4m3)` (E2M1, an E4M3 scale a 16), the indexer's keys `fp4_round(32, pow2)` (E2M1, a power of two a 32). int8 would be
another rounding on another grid: not done. kept now as those codes (`include/osmos/gpu/kvfmt.h`): a row its
codes then a scale byte a block - 528, 288 and 68 bytes for 512, 512 and 128 values (2,048 / 2,048 / 512 before):
~0.89 KB a token of compressed rows and keys, the window's ring 13.5 MB a sequence (52 before). a value read back is
its code's value times its scale, one product - the rounding's own: the same F32.

- **written where rounded**: `ops_fp8_round` / `ops_fp4_round` take a packed output beside the F32 (their scale in
  hand: recovering it from rounded values fails at FP4-E4M3's subnormal scales); `ops_kv_finish` writes the window's
  row packed; `ops_put_rows` copies rows by (format, values) - a byte count it took first let the pending rows' F32
  callers pass a float count and copy a quarter of each row, caught only by the model's fingerprint (no unit test
  sees host wiring): the interface fixed so a caller names its row's format. values rounded elsewhere (the oracle's,
  a probe's stand-ins) through `ops_kv_pack`: a block's scale searched (E4M3's 126, a power of two's 254) for the
  largest under which each value is exactly its code's value; none: NaN's scale code, read back NaN (loud). an
  FP4-E4M3 scale past 448 (a block's amax past 2,688: none of the model's normed latents) packs NaN's too.
- **read where used**: `kernels/kvload.h` - fetch a quad's bytes, decode when staged (`ops_sparse_attn`'s scores and
  values, `ops_index_score`); the attention's second key source in its own format (the compressed rows FP4, the
  drafter's block's own F32).
- **its cost, and back**: first 15.0 tok/s against 16.0 - one token's attention 59 -> 147 us (rocprofv3 and a scratch
  microbenchmark: the scores 13 -> 44, the values 36 -> 83). decoding at staging rather than at the fetch 147 ->
  142; E4M3 values without branches 136; a small step's tile all 64 heads (16 before: four blocks each decoding the
  same keys) **70 us** (32 heads 86). decode 15.8-16.1 tok/s (16.0), drafted 31.3 / 23.4 / 34.1 (29.6, maya 29.6).
- **proven the same bits**: test_kvfmt_gpu (each format rounded, packed and read back the F32's bits; rounded values
  packed as they are; zero blocks, -0, subnormal scales, a swapped nibble order fails it); test_attn_gpu (packed keys
  within the bound of a double attention, and the F32 path's bits); test_fuse_gpu (kv_finish's packed row the
  separate ops'); test_model_gpu's fingerprint cb2ad57792ac72d2 before and after, the probed run the unprobed's;
  the bench's answers (plain and drafted) the same three hashes.
- **what it buys**: a 64K context at the 110 GiB budget now starts (before: the pool refused it) - 16.1 / 16.1 /
  16.0 tok/s, maya's own context. at 1M the state is ~1 GB a sequence, not ~6.7.

## less work a decode token (2026-11-03)

a fresh trace of UD-IQ1_M's plain decode (100 steps): 63.0 ms a token, the GPU busy 54.0, 1,510 launches. the idle
8.7 ms: ~4 ms the experts' reads from the SSD (the reading entries' waits, a few long ones), ~3 ms the gaps between
launches (~2 us each; a launch saved ~6 us of wall), the rest the host's turns (the picks' copy back, the tables'
upload) and Engram's rows.

- **the lane's inputs rounded once a token** (q8_group_tok / q8_gate_up take xin, win: an output row's input row and
  its weight's by table; NULL, the row's own). the lane gathered each token's FFN input once a pair (K copies), rounded
  every copy to int8, gathered the weights, and rounded x again before the reading entries (the activations had taken
  its buffer). now each token's row is rounded once into its own buffer (lane_xq) and gate and up read it by token,
  the fused activation its weight by pair: 2 gathers and a round a layer gone (-82 launches a token), 63.0 -> 62.5
  ms. a launch's rows past its count read the count's last input (in_rows): no table entry past it is read -
  test_q8_fuse_gpu's by-token check poisons them, and faults with the clamp taken out. the same bits:
  test_q8_fuse_gpu (by token: the gathered rows' bits, 3 layers of UD's types), the q8 and CPU-lane tests,
  fingerprint cb2ad57792ac72d2, the bench's answers' hashes.
- **Engram's rows spread over every reader** (rowread_take). the readers took 16 rows at a time: a decode step's 48
  rows (2 layers x 24) went to 3 of the 64 workers, 16 random reads one after another each (~1.3 ms), and layer 1
  waited ~0.6 ms a token (median) for them, the GPU idle. a batch's take is now ceil(n / readers), at most 16 (a long
  step's as before): one row a worker, every read in flight at once. layer 1's wait median 614 -> 8 us, the mean of
  both 358 -> 56 us. plain decode 16.0 / 16.0 / 15.9 -> 16.4 / 16.3 / 16.2 tok/s (two rounds, alternated with the
  last commit's build); drafted unchanged (30.8-31.3 / 23.3-24.0 / 33.7-34.4 both; its steps' batches were already
  spread). the answers' hashes the same.
- **the guess** (fwd_guess: the next layer's router on a BF16 copy, ~1.5 ms of GPU a token) measured skipped
  whole: plain 16.2 / 16.2 / 16.2 vs 16.1 / 16.1 / 16.2, drafted within the runs' noise, ~1% more expert reads from the
  SSD without it. kept: worth nothing measurable here, and its reads saved grow where less of the model fits.
- **Engram on the CPU** (asked): not taken. its projection (engram_kv, Q6_K 6144 x 25600, 129 MB a layer) is a
  matvec bound by memory's bandwidth (0.59 ms a layer on the GPU, ~220 GB/s); the CPU shares that memory and reads
  it slower, so moving it moves the same bytes onto a slower reader beside a GPU that wants them (the CPU lane's
  experts, measured on this box: nothing gained, 15.3 vs 15.2). layer 14's could run while layers 1-13 do (it
  depends on the tokens alone), but it needs the GPU's Q6_K sum order on the CPU for the same bits, for ~0.5 ms at
  best less bandwidth's contention. layer 1's wait was the reads, fixed above.
- **the lane's tables read where they are written** (lane_up mapped, coherent; lane_dev its device address). they
  were uploaded a layer: an 18 KB copy queued behind the shared expert, the lane's first kernel ~9.5 us after the
  shared expert's last (median). now no copy: 2.4 us, 62.08 -> 61.82 ms a token in the trace; plain 16.3 / 16.3 /
  16.2 -> 16.4 / 16.4 / 16.3 and 16.4 / 16.3 / 16.2 -> 16.4 / 16.3 / 16.3 (two rounds alternated), drafted 30.8-31.5 /
  24.1 / 34.0-34.1 -> 31.5-31.8 / 23.8-24.2 / 34.1-34.4; the hashes the same. safe by the stream's order: the host
  writes a layer's tables after its picks' wait, which the stream passes only after the last layer's experts; the
  reading entries' weights after the ready ones' launches, entries those do not read.
- **tried, not kept: the activations to int8 inside q8_gate_up** (the round after it, 40 launches a token). an
  int8 block is 32 neighbouring rows, and gate_up's workgroup holds 8 (16 at 16 lanes a row), so the block's max
  needs more than one: (a) each wave 4 rows in turn, a workgroup a block: gate_up 6.59 -> 6.90 ms a token, the wall
  61.8 -> 61.6; (b) a row a wave, 1,024-thread workgroups: 7.03, 62.0; (c) the shape unchanged, the last of a
  block's workgroups to end (an atomic count, a fence) rounding it: 7.09, 62.0. every one the same bits
  (test_q8_fuse_gpu's int8 check, which caught a first-not-last mutation of (c)), none worth it: each slows the
  lane's largest kernel by more than the round and its launch cost (0.09 ms and ~0.1 ms of gaps), (a)'s 0.2 ms one
  trace's noise, and a drafted step's wider tokens would feel the kernel more. the separate round stays.

## the experts' reads (measured, 2026-11-03)

UD-IQ1_M at the 110 GiB budget (the most this box gives it: ttm's 113.1 GiB less the pool's room, 1.26 GiB, the
scratch, ~1 GiB at 512-token steps, and the runtime's; the operator keeps ~8 GB of RAM for the rest of the machine).
experts 123.1 GiB of 15,360 (8.2 MiB each), the rest 7.1 GiB - every tensor but the routed experts always resident,
the experts by their picks after (choose: the budget refused if the layers' own weights pass it). the bench's
answers (3 x 256 tokens): ~4.6 streamed picks a token, 98% of picks held.

forward_waits (new): the host's waits on expert reads, split by whether the guess began them. the third answer:
441 waits on reads not guessed, 853 ms; 274 on guessed reads not yet done, 397 ms - ~4.9 ms a token, the trace's
idle on reads. the budget: 100 / 105 / 110 GiB streamed 5,927 / 4,660 / 3,512 picks, waited 6.2 / 4.9 / 4.0 s, 15.3-15.5
/ 15.6-16.0 / 16.0-16.3 tok/s.

tried, none kept (each A/B'd alternating, the answers' hashes the same):
- **the guess two layers ahead** (layer l + 2's router on layer l's input): guessed 1,861 of the streamed picks
  (2,136 a layer ahead), fewer of them late (515 against ~900), the waits the same (4.1 against 4.0 s), the speed the
  same.
- **the guess wider** (the router's top 10 / 16 a token, not 6): guessed 74% / 81% of the streamed picks (61%), the
  waits on reads not guessed 2.8 -> 1.9 -> 1.4 s but those on guessed ones 1.4 -> 2.4 -> 3.0 s: the total the same,
  16.3 / 16.2 tok/s against 16.4.
- **smaller chunks, more workers** (fetchq's 1 MiB / 12 against 128-512 KiB / 12-48): a miss 2.22-2.53 ms, the
  same.
- **O_DIRECT** (scratch rd/: one cold 8 MiB unit alone, the GPU idle - buffered 2.31 ms at 8 threads; O_DIRECT into
  malloc'd memory 1.39; into the GPU pool's pages straight 2.17, their pinning the cost; through a worker's aligned
  bounce, then copied, 1.55 at 256 KiB x 16-32). in the engine, each file's O_DIRECT twin through the bounces, a
  unit always in chunks (test_fetchq's twin check, a zeros file under it, caught a twin ignored): a miss's read
  2.29 ms, waits 1.82 ms each against 1.97 - nothing. its chunk reads 0.55 ms a 256 KiB under the GPU's load
  (~0.3 idle), twelve at once ~5.7 GB/s: the SSD's ceiling.

- **the guess narrower** (only its most confident: the router's top 4 / 2 / 0 a token prefetched, the guess still
  run; forward_guesses' new count of the reads it began): top 6 16.2-16.5 tok/s, 2,265 reads begun, waits ~4.1 s;
  top 4 16.1-16.4, 1,520, ~4.4 s; top 2 16.0-16.3, 602, ~4.8 s; none 15.8-16.1, 0, ~5.2 s (two rounds each). the
  guess began ~2,260 reads and ~2,135 streamed picks were guessed: nearly every read it begins is used. it is worth
  ~0.3 tok/s here, not wasted bandwidth (the skip's earlier "the same" was within its noise). top 6 stays.

what it says: a layer's reads come in a burst (its misses, the next layer's guesses) that fills the SSD, and they are
bytes the model needs: how a read is issued, how early, or how many of the guesses are read, moves the wait from one
read to another or adds to it. the lever is the bytes a token must read - the budget, which is the machine's.

## a prompt's attention on the model's formats (2026-11-03)

an 8K prompt (UD-IQ1_M, 2,048 a step, 106 GiB; rocprofv3) against the build before the KV's packing: the attention
values 925 -> 1,540 ms, scores 919 -> 1,186, softmax 433 -> 503 - ~0.95 s more of the GPU's time; prompts 450-457 tok/s
at 8K where ~490 was (DESIGN, the readers). two causes:

- **each token decodes its own keys**: a prompt's tokens select among the same compressed rows (and the window's
  ring), each decoding them again for its scores and again for its values. now a step's run unpacks the rows once
  when its tokens would decode more than there are (run x topk > the rows below its furthest position; run x window
  > the ring's slots): ops_kv_unpack into b.ckv_f (score_stride rows, at most FWD_CKV_F32_ROWS: 64 MB, a 128K context
  at ratio 4 - past it the rows are read packed) and b.ring_f (the slots). ops_sparse_attn takes the ring's format
  (KV_FP8 or KV_F32), and with every key F32 launches kernels of plain loads. the same bits: an unpacked row is the
  F32 the rounding gave (test_attn_gpu: the ring unpacked, alone and with the rows F32, a batch or a token at a time,
  each the packed path's bits - a load shifted by a float4 caught; test_model_gpu's fingerprint cb2ad57792ac72d2,
  the bench's hashes plain, drafted and the prompts').
- **the row's size chosen per key**: Keys.row picked ring_bytes or ckv_bytes for each key, and the compiler read the
  chosen one from the kernel's arguments at every load - 31 8-byte loads a values kernel (the assembly), 28% slower
  than the F32 kernel before the packing at every launch. the all-F32 kernels compute a row's size (hd floats): the
  loads the old kernel's, its spills too (29 / 29 against 34 / 37).

the 8K prompt's attention now: values 1,001 ms, scores 914, softmax 436, the unpacks 9; the GPU busy 16.2 -> 15.4 s.
prompts 2K / 8K / 2K: 297 / 415-422 / 305-308 -> 308-321 / 446-459 / 322-327 tok/s (two rounds alternated with the last
commit's build); decode 16.1-16.2 and drafted 31.0 / 23.8 / 33.1 the same, their answers the same.

tried first, not kept: **the IQ experts' mm shapes** (scratch shp/: 64 x 4 waves on gate and up, 64 x 2 on down -
alone, one expert a launch, 10-25% faster at 16-64 tokens, the error the same to the digit). in a prompt's grouped
launches (many experts each) slower: IQ1_S 3.58 -> 3.78 s, IQ2_XXS 1.86 -> 2.03 of an 8K prompt; the prompt the same.
a shape's measure is the grouped launch, as the shapes were first tuned.

## a prompt's step size, and the experts' kernel's ceiling (2026-11-03)

a larger step reads each expert once for more tokens (its slab decoded once a tile of up to 8 x 16 tokens), and
its buffers take budget from the resident set. UD-IQ1_M, prompts 2K / 8K / 2K (tok/s, one run each, the box faster
this hour than the A/Bs above):

| step | budget | 2K | 8K | 2K |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 106 GiB | 337 | 485 | 358 |
| 4,096 | 104 | 309 | 541 | 339 |
| 4,096 | 102 | 318 | 520 | 336 |
| 8,192 | 100 | 296 | 578 | 334 |
| 8,192 | 96 | 312 | 580 | 332 |

a long prompt +19% at 8,192 a step; a 2K one (one step either way) none. the budget is the decode's (~16.2 tok/s at
110 GiB, ~15.3 at 100): a server's choice, 2,048 the default. most of the larger step's buffers are the experts'
(scratch(): S = max_tokens, P = S x K - gate, up, the activations, down's rows, ~3.3 GB at 8,192 against ~0.8 at
2,048), held for every pair at once so each expert is read once for all sub-batches. the experts in chunks, a
token's sums in their order still, would make the large step nearly free of budget - not done.

tried, not kept: **16 column tiles** for the IQ experts (a hot expert's slab decoded once for 256 tokens): at 8 the
kernel holds 256 VGPRs (occupancy 5), at 16 its loops no longer unroll (-Werror). a restructured kernel, not a
constant.

## against maya 1.1.52 (measured, 2026-11-03)

UD-IQ1_M, 64K context, maya's bench questions (256 greedy tokens) and its 2K / 8K prompts, each engine a fresh
process, two rounds. maya v1.1.52 (pulled and built: its 17 commits since 1.1.51 include the indexer's scores
through WMMA on RDNA 3 and the Engram readers following the host's threads), its pack of the same files, configs
scratch maya-ud-nodraft.json / maya-ud-dspark.json (DSpark off / on); runs in ~/bench/maya-1.1.52 and
~/bench/osmos-final.

| | ours | maya 1.1.52 |
| --- | --- | --- |
| plain | 16.3 / 16.3 / 16.2, 16.4 / 16.1 / 16.3 (16.3) | 17.0 / 16.7 / 16.5, 16.9 / 17.2 / 16.9 (16.9) |
| DSpark | 31.1 / 23.9 / 33.5, 31.0 / 23.8 / 33.5 (29.5) | 31.4 / 27.4 / 29.1, 32.2 / 25.9 / 31.4 (29.6) |
| 2K prompt | 333, 332 | 232, 246 (no draft); 269, 269 (DSpark config) |
| 8K prompt | 484, 472 | 409, 407; 412, 411 |

plain 96% of maya's, drafted even (maya steadier: its second question 26-27 where ours is 24; ours the third, 33.5
against 29-31), prompts +24-43% at 2K and +15-18% at 8K. our budgets: plain 110 GiB, drafted 100, prompts 106 at
2,048 a step.

## serving: conversations kept, and several at once (design, 2026-11-03)

maya's DeepSeek engine has both, the operator's: **#356** (a conversation's state set aside and taken back across
switches and restarts) and **#587** (`--parallel` on an event scheduler: prompts read a piece at a time between the
answers' rounds; squashed into maya a73bdbc, 2026-10-29). libosmos has neither: one process, one `Seq` a
caller's, no server. this ports both, as one binary that speaks maya's line protocol, so maya's own harnesses
(`tools/ds4/slot_keep_test.py`, `parallel_test.py`, `agent_bench.py`) test it unchanged and time both engines alike.

**1. a conversation's state out and back (the library)**
- `seq_save(s, &bytes, &n, err)` / `seq_load(s, bytes, n, err)`: the whole state as one byte string - the position,
  the prompt's end, the tokens (all of them: the slot store matches prefixes on them; the Engram n-gram's last ones
  are among them), every layer's window ring (FP8 rows), each compressor's pending rows, each KV source's compressed
  rows (FP4) and indexer keys (FP4, rows 0 .. pos / ratio), the decoder's `dec_from`; and the drafter's ring where a
  drafter is attached (else the next draft rebuilds it - to settle when the drafter's state is read).
- `forward_identity(f, out)`: what a saved state must match - the build's version, the geometry, the model's files
  (names and sizes), max_ctx's bound on the rows. a load against another identity, a size that disagrees, or tokens
  that do not add up to the position: refused, the Seq reset (never half of two conversations).
- **the bits**: a state saved and loaded continues as the state never set aside - the same logits, the same answer's
  hash. maya could only check greedy tokens (its restore and a re-read differ in arithmetic); ours are the same
  bits by the engine's contract (a token's bits do not depend on the batch or the step), so the tests compare hashes.
- size: ~0.9 KB a token (the KV's formats) - a 70K-token conversation ~65 MB, as maya's.

**2. the slot store (#356's)**
- a request that does not continue the conversation in its Seq sets that one aside first, then takes back the kept
  conversation whose tokens are the longest prefix of the prompt (longer than what the Seq holds already); only the
  rest of the prompt is read.
- RAM first (`OSMOS_SLOT_CAP`, 4), past it the least recently used to a folder (`OSMOS_SLOT_DIR`, beside the
  model), the folder capped (`OSMOS_SLOT_GB`, 16: the least recently used removed, never the one just written),
  conversations under `OSMOS_SLOT_MIN` tokens (1024) not kept, a folder's older than `OSMOS_SLOT_DAYS` (7)
  removed. a slot is a `.bin` and a `.meta` (the identity, the tokens, the size), each written under a temporary name
  and renamed; at start the folder is indexed and a slot whose `.meta` is missing, foreign or disagrees is removed.
  QUIT sets every conversation aside and writes RAM's to the folder. `OSMOS_SLOT_KEEP=1` turns it on.

**3. several conversations at once (#587's)**
- `--parallel N`: N Seqs. the loop on libevt's engine (`~/Projects/libevt`): stdin's lines and the turns are
  events, a line outranking a turn, each turn queueing the next as it ends, the loop asleep with nothing to do.
- a turn: every answering conversation's next token in one `forward_batch` (each its Seq) - a round - or the next
  piece of a prompt being read. prompts take turns a piece each, a round of tokens between two pieces; a prompt whole
  in one piece goes ahead of longer reads (a short question does not wait behind a long read).
- a piece: a whole number of the step's tokens (max_tokens), so a prompt reads through the same step boundaries
  however it is interleaved - and its bits are the same anyway.
- drafting in a round (DSpark): later; spec=0 first (#587 measured greedy, the harness's default).

**4. the protocol (maya's `ds4_main.cpp`, its serve loop)**
- ready: `READY <max_ctx> stop[ multi=<N>]`.
- in: `GEN <max_new> [rid=<r>] [spec=<0|1>] [seed=<s>] <id,id,...>`, `STOP [<rid>]`, `QUIT`. maya's other keys
  (sampling, penalties, thinking budgets, logprobs, GENI's pictures) are refused with `ERR` naming the key: greedy is
  the engine's one mode.
- out: `T [<rid>] <token>`; `PP[ <rid>] <done> <total> <ms> <tok/s> <reused>`; `DONE [<rid>] <produced> <prompt>
  <prompt_ms> <decode_ms> <finish> <accepted> <drafted> <reused> <hits> <lookups>`; `ERR [<rid>] <reason>`; `NOTE`.
  each field's meaning maya's, so its harnesses read ours.

**done (2026-11-03): steps 1 and 2.**
- seq_state_bytes / seq_save / seq_load / forward_identity / seq_position (src/run/hip_seqstate.c): rows saved by
  position (the window's from pos - window - SEQ_MAX_BACK, the compressors' open groups from SEQ_MAX_BACK back, so a
  rollback after a load finds its rows), the drafter's rings whole, an FNV-1a trailer. test_seqstate_gpu: A's 301
  tokens set aside (3.5 MB), B run in its own sequence, A loaded into B's - every logits row of 24 greedy steps and of
  drafted rounds the never-set-aside run's; a damaged byte, a short size, another identity refused, the sequence
  reset. mutations caught: the compressors' open rows not restored, the compressed rows not restored.
- slots.h / slots.c (src/serve): RAM, the folder, its cap, the .meta (SUBSLOT1, identity, bytes, last used, tokens,
  FNV-1a), the index at start. test_slots (CPU; clean under ASan + UBSan): the minimum, replacement, the longest
  prefix past `beyond`, the cap's LRU to the folder, a flush and a new store over it, another identity's / a damaged
  .meta / a short .bin / a .bin alone removed at start, the folder's cap (never the one just written), `days`.
- bin/osmos-serve (tools/serve/serve.c, `make serve`): one conversation, maya's GEN / STOP / QUIT and T / PP /
  DONE / ERR / NOTE; a request continuing the conversation reads only its rest (one within SEQ_MAX_BACK of it rolled
  back to their common part); a drafted answer cut by max_new rolled back to its printed tokens (the state ends at
  the answer's last printed token less one, as maya's greedy one: the next turn continues it). its answers the
  bench's: q1's 256 tokens 7e8dbc0974070b10 plain and drafted.
- tools/serve/slot_test.py (maya's slot_keep_test.py, ported; tools/serve/drive.py the protocol): A (12,000 tokens of
  real text), B (8,000), A's next turn - every check passed: the turn reuses A's 12,031 in 3.7 s against 28.1 read from
  token 0, its answer the bits of A never set aside, in-run, after a restart from the folder (and drafted), the
  folder under its cap, eviction of the least recently used.
- **a prompt step and a decode step are two computations**: a turn read from token 0 answers other bits than the
  same turn continued (b560d8dc against f9bdef8f): the earlier answer's tokens are then in a prompt's step, whose
  routed experts are mm's (WMMA, f16 inputs), where decoded they were the lane's (int8, forward_lane). each is
  deterministic and batch-invariant within its kind (a decode alone or verified; a prompt however it is split - the
  model test's fingerprints); not across them. maya's restore and re-read differ too (its PR: chunk boundaries). so the
  slot test's reference is the conversation never set aside, not a re-read.

**done (2026-11-03): step 3.** `osmos-serve --parallel N` (tools/serve/multi.c on libevt; the server split
into serve.h / conv.c - a conversation, both modes' - / single.c / multi.c / main.c, each under 300 lines): a reader
thread posts stdin's lines as events (priority 2), the turns are events (1), one queued at a time, each queueing the
next while a lane is busy. a turn is a round (every answering lane's token printed, the ones going on fed in one
forward_batch - a decode step) or one piece of one prompt (max_tokens, alone - a prompt's step): never one step of
both, so neither changes the other's bits. a round and a piece alternate (#587's one each); the piece a prompt whole in
one piece first, else the reading lanes in turn. a request takes the idle lane whose conversation it continues
furthest (conv_reach), else the least recently used (its conversation set aside, the longest kept prefix taken back),
else waits for one. STOP <rid>: a reading lane's state dropped (its prompt's end past it), an answering one's kept to
its last fed token; a waiting request answered DONE cancel.
tools/serve/parallel_test.py (maya's parallel_test.py, ported; real text): four prompts at once each the bits they get
alone (26.2 s against 33.3 one at a time), a STOP ends one alone (the other's answer its own), A's next turn reuses
its lane (1,547 of 1,547), a 900-token question sent during a 24K-token read got 6 of its 16 tokens before the read's
last piece. the single path's slot_test.py again: every check passed.

**done (2026-11-03): drafting in --parallel's rounds** (tools/serve/round.c; multi.h the lanes' shapes). a round's
drafting lanes draft (drafter_draft reads only the sequence's rings: no other lane's step between changes a draft)
and are verified with every other lane's token in the same forward_batch; each then prints its token and accepted
drafts, rolls back to all but the last printed and is seeded from the verify's rows before the next step (taps are the
last step's). a round stays a decode step - at most Server.round_tokens = (experts - 1) / experts a token = 63 tokens
(the lane's P < E; more, and it is a prompt's step, other bits): --parallel at most that many lanes, a lane's drafts
cut to leave a token for each answering lane after it (cut drafts change no bit: a decode's tokens the same bits
however they are grouped into verifies). a prompt's last piece seeds its drafter when the request drafts.
parallel_test.py with --dspark: four at once all drafting, and two drafting with two greedy - every answer the greedy
one alone (120 of 151 / 60 of 77 drafts accepted). the bench's four questions x 256 tokens, one process, --parallel 4
(96 GiB): one at a time greedy 14.2 tok/s in all, four at once greedy 31.5, one at a time drafted 32.6, four at once
drafted 44.2 - every answer the bench's hashes (7e8dbc09, 4ea20c49, 1390d91b, q4's cdf5f24e).

**drafting the default, and a budget refused at create (2026-11-03).** a GEN without spec= drafts when the server
has a drafter (maya's: -1, as the engine was started); spec=0 a request's own greedy. the drafted server at its
best (512-token steps, 100 GiB - 102 does not fit with the drafter): the bench's four questions one at a time 28.8
tok/s in all (the bench's drafted decode 29.5, here with the prompts read), four at once 45.7 (~11.4 a conversation).
and 102 had started - READY - then refused every request: the pool takes its room (stream_budget, 1.3-1.7 GiB) as
experts stream, after create, so a budget that left the GPU less than that was found at the first miss. now
forward_create and drafter_create check the GPU's free memory against the pool's room (fwd_room_left) and refuse:
"the GPU's free memory (1.49 GiB) under the expert pool's room (1.71 GiB): a smaller budget", exit 2 before READY.
plain decode at 110 GiB (no drafter) still created.

**5. the order** (each its tests first): seq_save / seq_load with identity (test: a conversation set aside, another
run, the first taken back - the same hash as never set aside; a damaged or foreign state refused, the Seq reset);
the serve binary with one conversation and the slot store (maya's slot_keep_test.py against it; restart, eviction,
the folder's cap); `--parallel` on libevt (maya's parallel_test.py: each conversation's answer token for token its
answer alone - here its hash; a STOP ends one alone; a short request answered during a long read); then
agent_bench.py against maya 1.1.52 on this box.

**6. measured (2026-11-04): agent_bench 4 x 8, seed 1, against maya 1.1.52** (`tools/serve/agent_bench.py`, maya's
workload and timing ported; ours at budget 96 GiB, 2,048-token steps, the IQ2_XS drafter; maya its config, DSpark on;
both `--parallel 4`, 131,072 context, slots off; ~/bench/agent):

| | ours | maya 1.1.52 |
| --- | --- | --- |
| turns answered | 32 of 32 | 32 of 32 |
| wall | 508.7 s | 538 s |
| first token p50 / p95 | 6.83 / 58.1 s | 6.41 / 93.5 s |
| turn done p50 / p95 | 50.9 / 113.1 s | 57.6 / 126.6 s |

the first run answered 31: a drafting turn's drafter was seeded once, after its prompt's last piece, from that step
- whose taps hold only that piece's positions, so a last piece shorter than the window left the ring's earlier
positions unseeded. seeded in each piece instead, the step's own part of the prompt's last window (`conv_read`):
seeding a piece's whole range failed every first turn ("a position the last step did not tap": past the decoder's
compaction a step taps only the tokens within reach of the prompt's end) - the run that showed which positions a
step holds.

## past 16K: the indexer's candidate blocks (2026-11-03)

a context past candidate_topk_blocks x candidate_block_size compressed rows was refused ("the candidate masking is
not built"): the source layer (20) and the index layers after it (24 .. 36) compress at ratio 1, so 2,048 x 8 x 1 =
16,384 positions - every conversation past 16K (first written here as 64K at ratio 4: wrong, the geometry's ratios
read since). the agent bench's conversations reach it (4 of 32 turns failed), and maya serves them. the model (oracle/ref.py, `candidates` and `indexer`):
- at the candidate source layer (20), after its indexer's scores (rows past a token's length -inf): the rows in
  blocks of 8, a block's key its rows' max; the token's last (partial) block forced in (+inf); the top
  candidate_topk_blocks blocks kept (of those not wholly -inf).
- every later index source: its scores outside the kept blocks -inf before its top k. earlier layers, and the source
  itself, unmasked. with no more blocks than candidate_topk_blocks every one is kept: the mask nothing (why every bit
  so far is the same: a step under 64K never builds it).

here: one selection kernel for both (attn.hip's radix select - 8 bits a pass, integer counts, ties to the lower index,
a compaction in index order), its key a row's score or a block's max (the last forced), its output the rows' list
(ops_topk) or the kept blocks' bits (ops_cand_blocks, a word per 32 blocks a token). ops_topk takes the bits (NULL: no
mask): a masked row's key -inf (kept blocks hold >= 16,384 rows: k = 512 never reaches one). the bits a sub-batch's
own (Sub.cand, beside its sel: a prompt's sub-batches run each layer in turn), allocated only when max_ctx can pass
the blocks (at 1M: 1,024 words a token); built at the source layer when any token of the sub-batch passes them, read
by the later index sources. the source is the decoder's layer (dec = 20, the last KV source) and the index layers
24 .. 36 come after its compaction (a prompt's tokens past the decoder's reach skip them): the bits are gathered with
the kept tokens there, beside the selected rows (hip_skip.c) - found when the first long prompt reached a guard that
refused an index layer past the compaction rather than read misaligned bits.

tests (each seen to fail first, or by a mutation):
- test_select_gpu (new; ops_topk had no test of its own): ops_cand_blocks against a host replay of ref.py's
  `candidates` (40 tokens of scores from 23 values - ties everywhere - and negatives, rows a partial block, fewer
  blocks than kept and many more, at 16 kept blocks so the mask bites at test sizes), ops_topk plain and under the
  mask against the host's top k - every bit; a kernel without the forced last block caught.
- test_cand_gpu (new): a 22,000-token prompt (steps of 2,048) and 4 decode steps, the source layer's bits probed
  ("cand") and every later index layer's selection ("sel", the first token - past the compaction, the first kept):
  every selected row in a block its own token's bits keep (114 selections, 58,368 rows). the mask built and not
  applied: caught; the compaction's gather of the bits skipped: caught (at 17,500 tokens it was not - the one
  sub-batch past the blocks wholly within the decoder's reach, the gather the identity: hence 22,000).
- test_model_gpu's fingerprint cb2ad57792ac72d2 unchanged (under 16K the bits are never built).
- end to end: a 72,576-token prompt (the bench's text, three times) and 24 greedy tokens - ours the same 24 as maya
  1.1.52's (prompt 518 tok/s against maya's 496). the same 24 with the mask not applied, too: greedy tokens on
  repeated text are not the mask's test; the probes above are.

## the MXFP8 Engram pack (design, 2026-11-04)

the operator's: UD-IQ1_M's weights with unsloth's **MXFP8** Engram table in place of its IQ4_NL one - the table as the
checkpoint holds it (lossless), our pack. unsloth publishes the table apart (`engram/`, 202.8 GB against IQ4_NL's
110.6), stamped shard 1 of 1; `set_split.py <file> 5 5` restamps it as UD-IQ1_M's shard 5 of 5.
- **the type**: ggml type 43 in unsloth's GGUFs (`per_layer_token_embd.weight`, [256, 768,022,850]): a block of 32 -
  an E8M0 scale byte e (2^(e - 127)), then 32 E4M3 codes: 33 B. upstream llama.cpp's 43 is F8_E4M3 (a byte a value,
  no scales): read as MXFP8 its tensor needs a 32nd more bytes than the file gives it - into the next tensor or out
  of the file - so gguf.c refuses a tensor whose bytes overlap another's as well as one that leaves its file (maya's
  pack tells them apart by the same span). a tensor of fewer than 1,024 values could hide the difference in its
  alignment padding; the only MXFP8 tensor is the Engram table.
- **the decode**: `quant_dequant` (QTYPE_DECODE), each value e4m3 x 2^(e - 127) - exact in F32 (4 significant bits,
  the smallest 2^-136), so no order of operations changes a bit; the forward's Engram path is the type table's
  already (`gguf_bytes(type, head_dim)` a row: 264 B against IQ4_NL's 144).
- **the reads**: a row is still one random read (264 B, inside a 4 KB page) - the reads a step makes are the same
  count, the SSD's IOPS what bounds them (Engram's reads, above): the speed should hold; measured below.
- **the oracle**: gguf-py reads 43 as F8_E4M3, so `dequant_ref.py --mxfp8 <file>` walks the header itself and decodes
  with torch's float8_e4m3fn (the OCP MX spec's); test_quant_gpu checks every value bit for bit against it.
- **the pack**: a folder of shards 1-4 linked from UD-IQ1_M and the restamped table as shard 5 (nothing of the IQ4_NL
  pack moved or deleted). quality: the reference forward (`ref_gguf.py`, maya's pack of the new folder) over the
  MXFP8 table, our engine against it on both tables - what IQ4_NL costs, and where the pack lands.

measured (2026-11-04; ~/bench/mxfp8, ~/osmos-oracle/ref32-mxfp8):
- the decode: test_quant_gpu, 458,752 values of the real table (its start, a middle, its end) and 8,192 random
  blocks, bit for bit torch's; a bias of 128 for 127 caught (456,654 differ).
- speed, plain decode at 110 GiB (dbench, 512-token steps, 64K context, three questions): IQ4_NL 16.2 / 16.2 / 16.1,
  MXFP8 16.3 / 16.3 / 16.1 tok/s - the same (the reads are IOPS, not bytes). IQ4_NL's answers hash as before.
- KL against the reference forward (32 positions; test_model_gpu with OSMOS_TEST_UD / OSMOS_ORACLE_REF /
  OSMOS_ORACLE_MAYA; maya 1.1.52's logits on the same pack):

  | engine's table vs the reference's | ours mean / median | maya mean / median |
  | --- | --- | --- |
  | IQ4_NL vs IQ4_NL | 0.209 / 2.8e-3 | 0.216 / 2.6e-3 |
  | MXFP8 vs MXFP8 | 0.081 / 3.7e-4 | 0.016 / 5.4e-4 |
  | IQ4_NL vs MXFP8 | 0.208 / 4.8e-4 | 0.214 / 1.5e-3 |

  the means are one or two positions each (near ties: ours 1.25 and 0.84 at positions 15 and 21 on MXFP8; maya's
  5.9 and 0.49 on IQ4_NL), the medians the rest. the lane (decode's path) against the MXFP8 reference: 0.038 on the
  IQ4_NL table, 0.024 against IQ4_NL's own - what the 4-bit table costs at the logits, about 0.013. test_model_gpu's
  bar (our mean at or under maya's) fails on the MXFP8 pack, on those two positions.
- perplexity over 8,191 tokens of the bench's text (`bin/ppl`, steps of 512, budget 100): IQ4_NL **2.4579**, MXFP8
  **2.4591**; at 4,096 tokens 5.9324 and 5.9430. no difference the text can see: at 1-bit experts the Engram table's
  4 bits are not what limits the model.
- so: the MXFP8 pack is 92 GB more disk for the same speed and the same perplexity. a pack where the weights are
  larger (ISTA's IQ2_XXS, a Q2 / Q3 UD) may show what this one cannot.

## UD-IQ1_S, and the two packs (2026-11-04)

the operator's: UD-IQ1_S's weights on the same MXFP8 table, and both quants abliterated - two packs, one table on
disk. nothing new in the engine (IQ1_S was in the type table, the int8 path and the CPU lane already); a pack is data.
- **the layout**: `~/Models/ds1017/engram/` holds the table once (stamped 5 of 5: both quants' slot); each pack's
  shard 5 links to it. UD-IQ1_S's shards 1-4: 122.4 GB against UD-IQ1_M's 139.8 (12.5% smaller experts).
- **the profile**: UD-IQ1_S's own (`PICKS_OUT`, the same 16,000-token calibration text, MAX_CTX 16384: 16K tokens
  past dbench's default 4,096 context fail seq_prompt); the abliterated packs take their quant's (the patch is
  attention only).
- **abliterated**: huihui-ai's `patch-unsloth-DeepSeek-V4.1-Flash-1017.py` with its per-quant patch GGUF, shards 1-4
  rewritten (UD-IQ1_S: 18 `attn_output_b` tensors replaced; UD-IQ1_M's as before); gguf-py from maya's llama.cpp on
  PYTHONPATH (the script's own folder kept off the path: `python -s -P`).

measured (~/bench/mxfp8; 64K context, three questions, 256 tokens, fresh processes, interleaved):

| | UD-IQ1_M + MXFP8 | UD-IQ1_S + MXFP8 |
| --- | --- | --- |
| plain, 110 GiB (two rounds) | 16.2 / 16.2 / 16.2, 16.5 / 16.2 / 16.1 | 18.4 / 18.4 / 18.2, 18.5 / 18.4 / 18.4 |
| drafted (IQ2_XS DSpark), 103 GiB | 28.5 / 23.8 / 36.1, 27.9 / 23.8 / 36.6 | 35.6 / 27.0 / 47.4, 36.8 / 27.4 / 47.9 |
| drafted, 100 GiB | - | 33.8 / 25.5 / 46.2, 37.0 / 26.8 / 45.8 |
| abliterated (one round) | drafted 103: 28.0 / 23.5 / 35.6 | plain 18.5 / 18.4 / 18.5 |
| perplexity, 8,191 tokens (`bin/ppl`) | 2.4591 | 2.5006 (abliterated 2.5048) |

- the speed is the reads: on the third question plain, 615 experts streamed from the SSD against 3,452; drafted,
  2,036 against 6,174 (a round's verify picks more experts than one token does), verify time 4.7 s against 6.5 s.
  the drafter's acceptance is the model's (205 of 220 against 209 of 239), and a drafted answer hashes as its plain
  one.
- the cost: perplexity +1.7% (2.4591 to 2.5006). a mean over every token: where a lower-bit quant loses (a number,
  a line of code) it may not show; not yet measured against the reference forward (a maya pack of UD-IQ1_S and the
  oracle's ~12 min) nor on the agent bench.
- the drafters on UD-IQ1_S (100 GiB, drafted): maya's pack of DSpark (MXFP4 experts, 8.0 GB) 36.8 / 26.9 / 45.7
  tok/s, accepted 186/217, 144/173, 206/224; our IQ2_XS requant (4.7 GB) 37.0 / 26.8 / 45.8, 183/205, 140/171,
  205/220 - the accepted tokens within 3%, the speed the same.
- so: two packs, the operator's choice of speed or quality - UD-IQ1_S ~26% faster drafted, UD-IQ1_M the lower
  perplexity - over one 203 GB table.

## plain decode's work a token (design, 2026-11-04)

the operator's: plain decode as fast as it can be, a token's work at its least, on UD-IQ1_S and UD-IQ1_M - before the
drafter. on UD-IQ1_S the SSD is nearly out of it (615 experts streamed over three questions), so the GPU's own work is
the token.

**the trace** (rocprofv3 --kernel-trace, UD-IQ1_S + MXFP8, 110 GiB, the first question, 250 decode tokens; scratch
`ksum.py`): 55.5 ms a token, the GPU busy 51.2, 1,450 launches.

| what | ms a token | |
| --- | --- | --- |
| dense Q6_K (mm, type 14) | 22.3 | 4.39 GB at 197 GB/s |
| Q5_K (type 269) / Q8_0 | 5.7 / 2.4 | 1.13 / 0.49 GB, 198 / 204 GB/s |
| routed experts (q8 gate_up, group_tok) | 10.0 | ~1.9 GB (240 experts) |
| the F32 router (mv) | 2.1 | 0.31 GB at 148 GB/s |
| the guess (BF16 router copy, its top-k, its copy) | ~1.1 | |
| attention: scores / softmax / values | 1.16 / 0.15 / 0.62 | |
| hyper-connections (hc_dot, mix_pre_rms x2, post_dot, post) | ~2.4 | |
| the rest (norms, rope, kv_finish, x16, round, router top-k, gather_sum, swiglu) | ~2.5 | |
| launch gaps under 5 us | 3.0 | ~2.1 us before nearly every launch |
| gaps over 5 us | 1.3 | the bench's host argmax (0.37), missed experts (~0.5), the layer tables' copies |

the floor: ~8.5 GB a token at ~215 GB/s, ~39.5 ms (~25 tok/s). the dense matrices are at ~92% of it already; the
~16 ms past it is small kernels, their gaps, and a few kernels that leave the GPU idle. `engram_kv` (2 layers, 129 MB
Q6_K each, 583 us) is dense work like the rest.

**what each change is, and why it keeps the bits**:
1. **the guess an option** (measured: GREEDY=1, guess off, the answers' hashes the same): UD-IQ1_S 18.2-18.5 ->
   18.4-18.9 tok/s (+2.5%); UD-IQ1_M unsettled (its first round +2%, its second worse with the ISTA download on the
   SSD - its misses ~6x UD-IQ1_S's, the guess earns there). a `ForwardOptions` field, on by default; the bench and
   the server take it. which experts are read ahead changes no bit.
2. **decode's attention tiles**: a window-only layer's scores are 2 blocks of 64 heads x 64 keys (~30 us, 38 of 40
   CUs idle). every output is a sum over d ascending (scores) or keys ascending (values) whatever the tile, so a tile
   is free to change (test_attn_gpu: the bits). the cost: a block decodes its keys once for its heads - fewer heads a
   block is more decoding, cheap at 128 FP8 keys. aim ~1.9 -> ~0.6 ms.
3. **the router's top-k in one wave**: 6 passes of a 256-thread tree (~9 barriers each), ~9 us twice a layer. a wave
   of 32 lanes, 12 experts a lane, shuffles: the same picks (the larger key, ties to the lower id), the same weights'
   sum in pick order. aim ~0.7 -> ~0.2 ms.
4. **the mixes' Sinkhorn** (hc_mix_pre_rms, ~12 us however few its bytes: 20 serial iterations of a 4 x 4 matrix):
   one wave, the matrix in registers, the same operations in the same order. to measure before building.
5. **launches folded into their neighbours** (~2.1 us of gap and ~1-2 of tail each): the FFN's mix writes the shared
   expert's x16 and the routed experts' int8 input (the attention side's mix writes its x16 already); q's rope with
   kv_finish; the layer's end (gather_sum, hc_post) into the next layer's hc_dot (ops_hc_post_dot, which the
   attention side uses). ~5 a layer, ~0.8 ms. each a test that the fused output is the separate kernels' bits.
6. **the F32 router's matvec** at 148 GB/s where the dense matrices reach ~200: a shape for its 384 rows. ~0.5 ms.

**measured since** (2026-11-04):
- 1 built: `ForwardOptions.no_guess` (`--no-guess`, dbench `NO_GUESS=1`; no guess, no router copy made - ~160 MB of
  GPU memory). test_guess_gpu: at 40 GiB (experts streamed), on: 843 streamed, 389 guessed; off: 0 guessed, 0 reads
  begun; 32 steps' logits and tokens the same bytes. UD-IQ1_M's choice waits for a quiet SSD.
- 2 smaller than it looked (scratch `mbattn`: ops_sparse_attn at decode's shape, 1 token x 64 heads x 512, 128 FP8
  window keys + 512 FP4 rows, every shape the same output bytes): 64 x 64 keys 67.7 us a call (three kernels back to
  back), 64 x 32 keys and dims 62.6, 64 x 16 64.1; fewer heads a block slower (32: 85, 16: 135 - each block decodes
  its keys again), a 32-value stage slower (72), 8 x 4 or 4 x 8 a thread slower (108, 103). the blocks are not too
  few: each is a chain of 32 dependent stages (a load, a barrier, the sums). ~0.2 ms a token from a tile; the chain
  itself is a rewrite (to ~1 ms), later.
- the gaps: the host is ahead (the queue deep), so the ~2.1 us before a launch is the GPU's own, between dependent
  kernels - a graph of the step would not remove it. a launch removed is ~3.5 us (its gap, its ramp and tail); so 5
  first, as: q's rope with kv_finish; the softmax inside the values kernel; the FFN's mix writing the shared expert's
  x16 and the routed experts' int8 input; the router's picks written to mapped host memory (no copy kernel); the
  layer's end with the next one's hc_dot - ~7-8 launches a layer, ~1.2 ms. then 3, 4, 6, 2's rewrite.

- 5 built (four of its folds; the softmax inside values left - each values block would redo it): q's rope in
  ops_kv_finish (q, q_heads: test_fuse_gpu's queries the separate ops_rope's bits, seen failing first); the FFN's mix
  writes the shared expert's x16 when the step is one sub-batch (f->xf_x16); the routers write their picks into mapped,
  coherent host memory (r_ids, r_guess: no copy kernel; a probe's still through ridx); a layer's end writes the next
  layer's attention mixes' sums (ops_hc_post_dot) when one sub-batch, no probe, no Engram next and not the decoder's
  compaction (f->hcp_next). the bench's answers hash as before, the guess on and off (420 guessed either way); the x16
  and the next layer's sums each mutated (none written; the wrong layer's fn): the answers differ. traced as before:
  1,450 -> 1,255 launches a token, gaps under 5 us 3.01 -> 2.62 ms, the wall 55.48 -> 55.10 ms - ~2 us a launch
  removed, not 3.5 (the kernels that took the work run a little longer).

- 6 built: the F32 router's matvec and the BF16 matvecs (the guess's copy, the compressed layers') one token at 4
  chunks in flight a lane (mv.hip's tune; R 1-4, U 1-8 swept on 40 routers cold, the same bits each): F32 53.5 ->
  38.8 us, BF16 22.8 -> 19.5; traced, 2.09 -> 1.63 and 1.54 -> 1.38 ms a token.
- 3 tried, not kept: the router's top-k as one wave (32 lanes, 12 experts a lane: 15 us a call in decode, against
  9), then as 256 threads with each one's keys in registers and one barrier a round (11 us) - both the same picks and
  weights (test_router_gpu, kept: the old kernel as its reference, a flipped tie rule caught in the wave's step and in
  the waves' step). the barriers were not its time: the scores (expf, log1pf, sqrt) and the launch are.
- traced now (the guess on, as the first): the wall 55.48 -> 54.51 ms; plain decode, the guess off, 18.2-18.5 ->
  19.1-19.3 tok/s (the ISTA download on the SSD meanwhile), the answers' hashes as ever.

- 2, a tile (traced in decode this time - the scratch benchmark's fixed 640 keys were not decode's: a short
  context's layers attend a few hundred): scores' keys a block 64 / 32 / 16 / 8 / 4 at 64 heads 1.18 / 1.16 / 0.97 /
  0.88 / 1.03 ms a token, 8 heads a thread 1.78; values 0.32 whatever its dims; kept at 8 (kSmallKeys) - the bench's
  hash and test_attn_gpu as before; ~0.27 ms a token, more where a context selects more rows.
- hc_post_dot's fn rows staged in LDS by the whole block before its 25 threads' sums (the same sums, the same order:
  test_fuse_gpu, and a staged row one off caught): slower in decode, 1.14 -> 1.58 ms a token (an index divided a
  load), 1.28 row by row unrolled - the barrier and the LDS round trip cost more than the loads in flight saved. not
  kept.

together, revised: ~3.5 ms past the guess (~6%), UD-IQ1_S plain ~18.9 -> ~20 tok/s. the first estimate, ~4-5 ms of
~54 (~8-9%) past the guess: UD-IQ1_S plain ~18.9 -> ~20.5 tok/s, UD-IQ1_M in proportion. the
order: 1, 2, 3 (each small, each its own measure), then 5, 4, 6. a HIP graph of a step (most of the 3 ms of gaps) is
the larger lever and the larger change - the host decides a layer's experts between launches - and waits until these
are in. every change: the bench's answers hash as before, interleaved fresh runs on both packs.

## the quiet-box bench, and GSQ-RCO's new layout (2026-11-04)

every pack on the MXFP8 table, nothing else on the SSD (the morning's numbers ran beside a 250 GB download), two
interleaved rounds of the three questions, GREEDY=1 (the server's GPU pick; the bench's host argmax was ~0.4 ms a
token of its own); ~/bench/clean.

| | UD-IQ1_M | UD-IQ1_S | GSQ-RCO IQ2_XXS |
| --- | --- | --- | --- |
| plain, guess on | 16.6 / 16.6 / 16.4, 15.9 / 16.4 / 16.2 | - | 18.8 / 18.9 / 18.8, 18.1 / 18.5 / 18.8 |
| plain, guess off | 16.5 / 16.6 / 16.1, 16.5 / 16.8 / 16.6 | 18.8 / 19.2 / 19.2, 19.0 / 19.2 / 19.1 | 19.1 / 19.0 / 19.2, 18.9 / 18.9 / 19.2 |
| drafted, guess on | 28.3 / 24.2 / 35.9, 29.4 / 24.1 / 36.0 | - | 34.3 / 24.3 / 39.7, 35.0 / 24.6 / 40.7 |
| drafted, guess off | 28.9 / 23.7 / 36.4, 28.8 / 23.6 / 36.3 | 38.3 / 27.6 / 48.4, 38.1 / 27.5 / 48.2 | 34.9 / 24.0 / 40.3, 34.7 / 24.1 / 40.4 |
| perplexity, 8,191 tokens | 2.4591 | 2.5006 | 2.5461 (its IQ4_NL table 2.5489) |
| package power decoding, plain / drafted | - | 122 / 127 W (6.4 / 3.3 J a token) | 125 / 125 W (6.6 / 3.8 J) |

- the guess: a wash on UD-IQ1_M (plain 16.35 on against 16.5 off, drafted 29.65 / 29.62) and on GSQ-RCO, a loss on
  UD-IQ1_S - off for every pack (the pack's setting). the morning's power ran 133-138 W: fewer launches and no guess
  pass, ~125 W now.
- GSQ-RCO: its IQ2 experts at UD-IQ1_S's plain speed (2,873 streamed against 615 - the int8 IQ2 kernels' cheaper
  decode a byte); drafted below UD-IQ1_S (a verify streams 4,991). its perplexity the worst of the three, as unsloth's
  card measures it (higher KL, lower top-1; six benchmarks 83.68 against UD-IQ1_S 83.53, UD-IQ1_M 85.55): not a pack.
- ISTA's files since 2026-10-27 (llama.cpp's layout: five shards, the Engram file linked as shard 6) name neither
  DSpark's target layers nor the Engram's compressed vocabulary (nor embedding_length_out, unread) - both refused at
  load until now:
  - the target layers: the model's where its GGUF names them, else the drafter's dspark.json ("target_layers",
    DeepSeek's config's 37, 38, 39 - ours carries it, requant_dspark copies it); both named and unequal refused,
    neither refused; the Forward taps the drafter's (fwd_taps_on's at). test_dspark (parsed, four malformed lists
    refused), test_weights (UD without the key: loaded, none set), test_seqstate_gpu (a drafter naming 36, 38, 39
    against the model's: refused) - each failing before.
  - the compressed vocabulary: absent, the token map's greatest entry + 1 (unsloth's file: 99,092 either way - the
    map's 99,092 distinct entries; the two checks it feeds, a map entry's range and the multipliers' overflow bound,
    are the map's own). test_engram: UD without the key, every hashed row of 64 positions the oracle's, failing before.

## an overlay: abliteration without a second copy (design, 2026-11-04)

the operator's: the packs' abliterated forms opt-in, off by default, from huihui-ai's patches (unsloth/patch-UD-IQ1_M,
patch-UD-IQ1_S: 18 attn_output_b each, Q6_K and Q5_K, 0.6 GB) applied as the model loads - not 120-140 GB patched
copies beside the originals.
- **the shape**: `weights_create_with(&(WeightsOptions){model, engram, overlay}, err)`; weights_create its no-overlay
  form. an overlay is a GGUF of tensors; each must name one of the model's (refused: a name the model lacks, dims
  not the model's tensor's, a tensor no role reads) and stands in for it in every role - its type its own, taken if
  that role takes it (huihui's Q5_K for a Q6_K out_b: the matmul takes both). its metadata unread.
- **the files**: a tensor's bytes are found by its file index; the Weights now own the file list - the model's, then
  the overlay's (`n_files`, `paths`) - and the overlay's tensors are its copies with their index past the model's.
  the Forward opens `w->paths` (the CPU lane maps them), not the model's.
- **the identity**: forward_identity hashes `w->paths` (names and sizes): a sequence's state saved with an overlay
  refused without it, and the other way.
- **the proof**: the overlaid UD-IQ1_M decodes the bits of huihui's own patched files (UD-IQ1_M-abliterated: their
  script's output, on disk) - every logits row of a greedy decode; test_weights: the roles' tensors the overlay's,
  the refusals. then perplexity both packs, and a small fixed prompt set's refusals with and without.
- **the tools**: dbench / ppl `OVERLAY=file`, osmos-serve `--overlay file`.

## a second model: Qwen3.8-Flash-Next (qwen4exp) (design, 2026-11-04)

the operator's: unsloth's UD-IQ3_XXS of Qwen3.8-Flash-Next (83.6 GB with its n-gram table; >99% recovery by
unsloth's measure) in osmos and its installer. 125B (6B active) + 51B n-gram embedding + 4B MTP: the whole model in
GPU memory (~121 GB) with ~37 GB to spare - no expert streams; a token ~2-3 GB of weights.
- **references**: llama.cpp's own graph (src/models/qwen4exp.cpp, MIT; its build the reference forward for KL and a
  speed baseline on this box), Strata (github.com/Niko1221/Strata, MIT: src/split - its GDN, QSA, gated residual and
  MoE kernels and their parity tests). credited where code follows them.
- **the file** (`general.architecture` qwen4exp; 48 layers, D 2560, 4 streams, vocab 248,320):
  - layers l % 4 != 3 (36): Gated DeltaNet - attn_qkv [2560 -> 10240: q, k 16 groups x 128, v 48 heads x 128],
    ssm_conv1d [4 x 10240] (causal, its state the last 3 rows), ssm_alpha / ssm_beta [2560 -> 48], ssm_a, ssm_dt
    [48], attn_gate [2560 -> 6144], ssm_norm [128] (the output's gated norm a head), ssm_out [6144 -> 2560]; a
    sequence's state 48 x 128 x 128 F32 a layer (~113 MB for the 36).
  - layers l % 4 == 3 (12): Qwen Sparse Attention - attn_q [2560 -> 12288: 24 heads x 256, and their gates],
    attn_k / attn_v [2560 -> 512: 2 KV heads x 256], q / k norms, attn_output [6144 -> 2560]; an indexer (BF16
    q_proj [2560 -> 4 x 128], k_proj [-> 128], norms) choosing top-k 2048 tokens (compress ratio 4: blocks).
  - every layer: the gated residual (hc_{attn,ffn}_{norm,down,up,inject}: x normed a stream, gate =
    sigmoid(up(silu(down(x) / 4))), the block's input the streams' gated mean; after it, each stream += 2
    sigmoid(inject_s) x out); the MoE - router F32 [2560 -> 512] top 10, a shared expert gated by sigmoid
    (ffn_gate_inp_shexp), experts 640 wide (gate / up IQ3_XXS, IQ2_S, IQ4_XS; down IQ4_NL, Q2_0).
  - layer 1: the n-gram embedding (per_layer_token_embd IQ4_NL [160 x 320M], ple.* metadata: 3-grams, 8 heads,
    conv 4; ple_key / ple_value Q8_0) - Engram's cousin, its own hash.
  - the head: output_hc_{norm,down,up} (the streams mixed once more), output Q6_K.
- **the kernels**: three types new to the table, all in llama.cpp's ggml - Q2_0 (ggml 42: 64 values, an f16 scale,
  2-bit codes -1 .. 2), IQ3_XXS (a grid type, IQ2_XXS's sibling), IQ4_XS (IQ4_NL's values in K-superblocks); IQ4_NL
  onto the int8 expert path (45 of the 48 downs). new ops: the GDN's conv and delta rule (a token's recurrent update;
  a prompt's chunked form - the same bits either way, as every path here), QSA's block indexer, the gated residual.
- **the shape**: osmos's model layer is DeepSeek's (Geometry, LayerWeights, forward.h). the second family beside
  it, not inside it: its own geometry, roles and forward (src/q4x/, include/osmos/q4x/), the shared parts shared -
  gguf, the type table and every kernel, the resident set, the int8 experts, the server through an interface both
  forwards present (step, batch, greedy, logits, a sequence's state). one shape per concept: the server's interface
  the one place the two meet.
- **the order**: llama.cpp built (HIP) - its logits on fixed ids the reference, its tok/s the baseline; the three
  types (each against ggml's dequant, bit for bit); weights and geometry (every tensor in a role, refusals); the
  forward one block at a time against llama.cpp's intermediate tensors (llama-eval-callback); then decode speed,
  then prompts, then MTP.

### Qwen's forward, op by op (design, 2026-11-06)

from llama.cpp 68d20b8's src/models/qwen4exp.cpp (the reference: its names in quotes are its tensors' cb names, what
the dumps hold) and ggml's CPU ops. D = 2,560, hc = 4 streams, r = 320, eps = 1e-6. a token's residual is hc
streams of D (h[s][d]), the embedding copied into each ("hc_init").

- **the mixer** (twice a layer: before the token mixer, before the MoE; once at the end as the output norm):
  xn[s] = rms_norm(h[s]) * w_norm[s] (a norm a stream, eps; the weight per stream - the converter folded 1 + w);
  lo = silu(down . xn / hc) (down: [hc D] -> r); gate = up . lo (r -> [hc D]); mixed = mean over s of
  xn[s] * sigmoid(gate[s]) ("hc_mixed"); inject = (inject_w . xn) / hc ([hc D] -> hc; none at the end).
- **the combine**: h[s] += 2 sigmoid(inject[s]) * out (every stream the same block output, its own weight;
  "hc_combine").
- **a layer**: [PLE on its layer] -> mix(attn) -> DeltaNet or attention -> combine -> mix(ffn) -> MoE -> combine
  ("l_last"); after the last layer the end mixer's `mixed` ("result_norm") through output.weight ("result_output").
- **the Gated DeltaNet** (36 layers; 16 key heads, 48 value heads of 128): qkv = attn_qkv . x (10,240: q 16x128,
  k 16x128, v 48x128); z = attn_gate . x (6,144); beta = sigmoid(ssm_beta . x); g = softplus(ssm_alpha . x + dt) *
  ssm_a (ssm_a holds -exp(A_log): g <= 0). the conv: a depthwise causal conv of 4 taps over the last 3 tokens' qkv
  and this one (its history the state: 3 x 10,240), silu ("conv_output_silu"). q, k: an l2 norm a head (rms_norm
  with eps / 128, times 1 / sqrt(128)). the delta rule a value head h at a time, its key group **h % 16** (ggml's
  head broadcast, not h / 3): S *= exp(g); delta_j = (v_j - sum_i S[i][j] k_i) beta; S[i][j] += k_i delta_j;
  out_j = sum_i S[i][j] q_i / sqrt(128) - S 128 x 128 F32 a head, the recurrent state ("new_state"). then out =
  rms_norm(out) * ssm_norm * sigmoid(z) a head (sigmoid: Qwen3.8's difference from Qwen3.5's silu), ssm_out . out.
- **the sparse attention** (12 layers; 24 heads, 2 KV heads, 256 dims): attn_q gives a head's q then its gate,
  interleaved a head at a time (512 a head); q, k rms-normed a head (q_norm, k_norm); IMRoPE (rope_multi: sections
  11, 11, 10, 0 of the 64 rotated dims, base 1e7; text positions all four the same); attention at 1 / sqrt(256),
  GQA 12 heads a KV head; out * sigmoid(gate); attn_output. the selection (QSA): keys pooled in blocks of 4, the
  indexer (4 heads of 128: q_proj, k_proj, their norms) scores the pooled blocks, the top 2,048 tokens' blocks plus
  the incomplete tail attended, the rest masked - at 2,048 tokens or fewer every token is kept: dense. the first
  build is dense (and refuses a context past 2,048 until QSA is built and held to llama.cpp's selection).
- **the MoE**: logits = router . x (F32 512); softmax; the top 10 by probability; their weights / their sum
  (clamped at 6.1e-5); experts silu(gate) * up -> down (640 wide); plus the shared expert (the same shape) times
  sigmoid(ffn_gate_inp_shexp . x) ("ffn_out").
- **the n-gram table (PLE, layer 1, before its mixer)**: a token's 16 rows: n = 2, 3 (8 heads each) over the token
  and its 2 predecessors (none, or an EOS at or before one: the rest EOS; the token's own EOS cuts nothing);
  mixed = t0 m0 ^ t1 m1 [^ t2 m2] in uint64; row = mixed % rows[h] + offset[h]; the 16 rows of 160 (IQ4_NL)
  flattened to 2,560 ("ple_embd"). key = ple_key . e, value = ple_value . e; key and the residual each rms-normed a
  stream (ple_norm_key, ple_norm_query); a stream's gate = sigmoid(sgn(s) sqrt(max(|s|, 1e-6))), s = key[s] .
  query[s] / sqrt(D); gated[s] = value * gate[s]; a depthwise causal conv over gated (rms-normed a stream,
  ple_norm_conv) of 4 taps dilated by 3 (its history 9 tokens), silu; h += gated + conv.
- **a sequence's state**: a DeltaNet layer's S (48 x 128 x 128 F32, 3 MB) and conv history (3 x 10,240); the PLE
  layer's conv history (9 x 10,240); the attention layers' K and V; its last 2 tokens (the n-grams).

### the reference's tensors (2026-11-06)

oracle/qwen_dump.cpp: llama.cpp's own (68d20b8, HIP, all layers on the GPU) through its eval callback - the README's
first 256 tokens (prose, tables, code; the model's own tokenizer) one batch, each tensor of the spec's names written
as F32 with its shape (~/osmos-oracle/qwen-dump-256: 1,246 tensors, 5.7 GB, 19 s). a layer's two mixers share
their names: the second is <name>#1. llama.cpp's matmuls round the activations to 8 bits (q8_1) against a quantized
weight, so its tensors are near ours, not ours: a layer is held to them within a measured tolerance, the logits by
their KL divergence - never a bit-for-bit claim.

### the mixer, held to exact math (2026-11-06)

kernels/q4x.hip: the hyper-connections' norm, gated mean and combine (the down / up / inject through mv_strided).
test_q4x_mix_gpu on layer 0's real residual (hc_init) and weights: the norm 6.0e-8 of llama.cpp's (relative L2),
the inject 1.5e-6, the combine 2.8e-8 - where llama.cpp does not round. through the rank's two quantized matmuls it
does: so the gate and mixed input in double, from the exact weights, as well - osmos against that **1.1e-7 / 6.7e-8**,
llama.cpp **7.2e-3 / 1.1e-2**: its activations at 8 bits before each Q8_0 matmul, ~10^5 further from the math than
ours. the tests hold osmos to the exact sum (1e-5) wherever one is computed, and to llama.cpp within its own rounding
(1% a matmul, 2% two in a row) where not. (the operator's sense that osmos answers better: here a measure of it.)

### the Gated DeltaNet (2026-11-06)

kernels/q4x_ssm.hip: the conv (its history the sequence's), the gates, the delta rule - a block a value head, a
column of its 128 x 128 state a thread, in registers, a token at a time - and the gated norm. test_q4x_deltanet_gpu,
layer 0, each op from llama.cpp's own inputs (its weights F32, so no rounding of llama.cpp's between): conv 5.6e-8,
the decay 6.2e-7, beta 1.5e-7, the delta rule over 256 tokens 2.2e-7, the gated norm 6.5e-8 (relative L2); the
whole layer from its mixed input 1.0e-2 (llama.cpp's 8-bit activations into attn_qkv, attn_gate, ssm_out). the
state carried: 128 + 128 tokens the bits of 256. a mutant reading key group h / 3 (not h % 16) is 79% off.

### the attention, dense (2026-11-06)

kernels/q4x_attn.hip: q's and k's norm and rotation - IMRoPE on text (every section the token's position) is the
NeoX pairing over the first 64 dims, theta_i a running product as ggml's cache builds it - and the gated attention
over an F32 cache (a block a head and token, its keys' scores in LDS: 8,192 keys at most, half a block's 64 KB;
dense - the QSA selection keeps every token below its 2,048-token budget). GQA is h / 12 (llama.cpp's flash
attention), unlike the DeltaNet's h % 16. test_q4x_attn_gpu, layer 3: q and k 4.6e-7 / 5.3e-7 of llama.cpp's; the
attention 6.0e-7 of exact (double), llama.cpp 7.8e-4 of it (its F16 cache); the whole layer 1.0e-2; two halves
through the cache the bits of one run; a mutant grouping h % 2 96% off.

### the MoE (2026-11-06)

kernels/q4x_moe.hip: the routing (ggml's softmax, the top 10 - ties to the lower id - their weights over their sum,
clamped at f16's least normal), an entry a (token, expert) with its own copy of its token's input (mv_group_tok
writes the row it reads), silu(gate) up, and the weighted sum with the shared expert's (sigmoid-gated).
test_q4x_moe_gpu, layer 0: the router 7.3e-7, the routing weights 1.8e-6 of llama.cpp's - the same 10 experts in the
same order for all 256 tokens; the routed experts 1.3e-7 of exact (16 tokens, double from the exact weights),
llama.cpp 8.4e-3; the whole MoE 5.6e-3 of llama.cpp's. the decode's grouping by expert (an expert read once for
its tokens) is the speed work's: this is an entry a (token, expert), correct first.

### the n-gram table (2026-11-06)

src/q4x/ple.c: a token's rows hashed on the host (as llama.cpp does: ggml has no 64-bit integers or xor);
kernels/q4x_ple.hip: the rows decoded, the per-stream gate, the residual's add; q4x_conv takes a dilation (the
DeltaNet's 1, this layer's 3, its history 9 tokens). test_q4x_ple_gpu, layer 1, the dump's own tokens: the decoded
rows **bit for bit** llama.cpp's embedding (every token's 16 rows the same - the first two's missing predecessors
read as EOS); the gate 1.2e-4 and gated value 6.0e-4 (through Q8_0 key / value: llama.cpp's rounding); the dilated
conv 5.7e-8; the residual after it 6.8e-8; the conv's history carried, two halves the bits of one run; a mutant
taking the predecessors in the wrong order 132% off. the test's tensor upload reads 256 MB at a time (the table is
28.8 GB; a read takes ~2 GB at most).

### the whole forward (2026-11-06)

include/osmos/q4x/forward.h, src/q4x/hip_{forward,layers,step}.c: every weight resident (~78 GB, read through a
pinned 256 MB piece), a step's buffers, a sequence's state (36 DeltaNet states and conv histories, the n-gram conv
history, 12 F32 attention caches, its last tokens), the layers as the op tests hold them, the end mixer, the
logits. test_q4x_forward_gpu, the dump's 256 tokens: 128 + 128 and 255 + 1 the bits of one batch; free-running,
the residual drifts from llama.cpp's - 1.7e-2 after layer 0, 0.15-0.2 from layer 32 - the logits' KL mean 2.9e-2
(max 1.07), the greatest token the same for 93.4%. teacher-forced (each layer from llama.cpp's own residual): every
layer alone 0.5e-2 .. 2.7e-2 (worst layer 32), none an outlier - the DeltaNets, the attentions, the IQ4_XS and Q2_0
experts, the n-gram layer: no layer wrong, the drift two roundings compounding over 48 layers (and osmos's the one
near the exact sum, the op tests show). which is the better model is perplexity's to say: next.

### perplexity: osmos at llama.cpp's, and below (2026-11-06)

the text: the 16,000 ids of ~/bench/osmos-prompts/calib16k.txt (DeepSeek's perplexity text) decoded back by osmos's
own tokenizer (~/bench/calib16k-text.txt: Strata's docs, then license text) and tokenized by Qwen's own (qwen_dump).
one protocol both sides: one sequence, every next token scored by log-softmax in double (oracle/qwen_dump -ppl,
tools/bench/q4x_ppl). 2,048 tokens - llama.cpp's QSA keeps every token below its 2,048-token budget, so both
attentions dense:

| UD-IQ3_XXS, 2,047 tokens scored | perplexity | mean log-probability |
| --- | ---: | ---: |
| llama.cpp 68d20b8 (HIP) | 6.9478 | -1.938422 |
| **osmos** | **6.9471** | **-1.938324** |

at llama.cpp's, by a hair below (the weights' 3 bits the error that matters; osmos's arithmetic, ~10^5 nearer the
exact sum an op, shows only at the fourth digit here). the same number at steps of 256 and of 100: the batches'
split changes no bit. over 8,191 tokens llama.cpp is 2.7193 - with its sparse attention past 2,048: osmos's
comparison there waits for its QSA.

### serving it: tokenizer, template, dialect, family (2026-11-06)

what osmos-serve and osmos-api need beyond the forward, each held to its reference:

- **the tokenizer**: Qwen's `pre` is qwen35, one expression llama.cpp splits with its own code
  (unicode_regex_split_custom_qwen35); src/text/pretok_qwen.c is that code branch for branch, its classes the code
  point's own (not the regex path's collapsed ASCII). 242 cases against llama-tokenize over the model's vocabulary.
  two of its branches cannot show in ids on this vocabulary (no token crosses punctuation into a newline, or a
  newline into letters); the digit branch is redundant with the last (both take one code point).
- **the chat template**: the GGUF's own tokenizer.chat_template (Qwen's, unsloth's fixes) ported byte for byte as
  transformers renders it (jinja2 sandbox, trim and lstrip blocks, json.dumps for tojson, Python's str.strip for
  trim). its differences from DeepSeek's: tools a variable of their own; the effort a word (xhigh default, high read
  as xhigh, medium no line, low; anything else the template's refusal); every assistant turn's reasoning kept unless
  preserve_thinking is false; tool calls as `<tool_call><function=..><parameter=..>`, a string parameter raw and
  anything else its JSON. the parse is the template read backwards: a parameter's text is JSON where that parses to
  a non-string of a type its schema allows - a nullable string's null comes back null, a string "3" stays a string
  (the generator round-trips every assistant turn it renders, and found the nullable case).
- **the API's dialect**: tools/api/dialect.c, a family's reader, render, parse and stream markers, chosen by
  general.architecture. Qwen's template refuses tool arguments given as a JSON string, so the reader parses them
  into objects first (llama.cpp's server does the same); its parse trims, so its stream starts past whitespace.
- **the engine family**: every weight resident, so the server builds once (OsmosFacts.streams false: no budget
  search). a batch is run one sequence's consecutive tokens at a time, each run's logits at its own rows - every row
  kept, the bits of each sequence alone. a sequence saves and loads (its DeltaNet states, conv histories, caches
  below the position, a checksum). no rollback reach: a recurrent state cannot step back without snapshots, so a
  conversation that does not extend its last prompt is prompted again.

end to end on this box (performance, UD-IQ3_XXS, low effort): "What is 17 * 23?" - reasoning, then 391; a
get_weather call with {"city": "Paris", "days": 3}, the integer typed by its schema. 53 prompt tokens at 92 tok/s,
decode 14.8 tok/s - llama.cpp's 26.3 the mark for 0.5.0, whose work this is not yet.

### Qwen's speed (2026-11-06, performance)

the gate: qwen-forward merges at 40-60 tok/s decode. measured with tools/bench/q4x_bench (a 512-token prompt, then
128 greedy tokens; the decoded ids' FNV-1a printed, so a change that moves the answer shows):

| step | plain decode | drafted (MTP, 3 a round) | prompt 512 / 2,048 |
| --- | ---: | ---: | ---: |
| the first build | 15.2 | - | 105 |
| the router's picks a block's argmax (thread 0 scanned 512 ten times: 13 ms a token) | 24.5 | - | 106 |
| the expert types' matvec shapes (tools/bench/mvsweep) | 25.2 | - | 126 |
| the MTP head drafting (q4x_mtp_*), rollback rings | 25.2 | 38-40 | 126 |
| matmuls on mm (f16 WMMA), a prompt's experts a tile a read | 21.7 | 34 | 352 / 323 |
| mm's expert shapes (mmsweep), an input's f16 once, pairs one launch | 22.8 | 38.1 | 370 / 331 |
| the DeltaNet loop (norms out of it, two threads a column), 320-wide on mm | 22.3-22.9 | - | 424-472 / 404-413 |

- **the mm decision**: prompts are compute: the F32 matvec's decode-then-FMA ran 1.4-3 TFLOP/s. mm (DeepSeek's: f16
  activations and weights into WMMA, F32 sums) takes them to llama.cpp's (415 at 2,048). its cost: precision - the
  MoE 2.3e-4 from exact (F32: 1.3e-7; llama.cpp's q8_1: 8.4e-3) - and perplexity says it is none: 6.9365-6.9503
  across the shapes tried (F32 6.9471, llama.cpp 6.9478). the operator's rule kept: one kernel a matrix at every
  token count, so a token's bits are its own whatever its batch (the forward test's 128 + 128 and 255 + 1). a
  hybrid (mm for prompts, the matvec for decode) measured 24.6 / 37.6 / 355 and was declined for breaking that.
- **grouping a small step's experts** (each expert read once for the tokens that picked it) did not pay on the
  matvec: 4 consecutive tokens share a third of their experts, but an expert read for several tokens cost that
  back (61.8 ms a 4-token step ungrouped, 64.8-67.4 grouped). on mm it is a tile's work, and pays.
- **the drafts a round**: 1-5 measured over 192 tokens (34.4, 35.2, 35.0, 33.4, 32.6 tok/s on the matvec): 2-3, as
  llama.cpp's default 3. a step's extra token costs ~6-8 ms (its experts' reads), a draft ~3.8 ms (the head's
  output projection mostly).
- **next**: the attention (dense, a block a head and token: ~1 s of a 2,048-token prompt, quadratic - ~16 s at
  8,192; +7 ms a token decoding at 2,048) to a tiled kernel on the matrix cores; a prompt's experts read once for
  up to 128 tokens (DeepSeek's 8 tiles; now 16); the decode's f16 conversions folded into their producers.

### the cache in Q8_0, the native context (2026-11-06, performance)

the attention's cache in ggml's Q8_0 (a block of 32 values: an f16 scale, amax / 127, then each rounded to nearest):
~13 KB a token a sequence for its 12 layers and the MTP head's, 48 KB in F32 - the native 262,144 positions
(qwen4exp.context_length; OPS_Q4X_MAX_KEYS) 3.4 GB a conversation. k and v written by q4x_kv_put (a thread a block)
after their projection and rotation; fa_kernel loads a key as its scale times its int8, to f16 - the matrix cores'
own, so a token's bits stay its batch's. the state's format 2 (a format-1 state refused). YaRN past 262,144 (1M) is
0.5.1's: the GGUF names no rope scaling.
- **test_q4x_attn_gpu**: the cache's bytes ggml's quantize_row_q8_0's (a host copy), bit for bit; k and v 5.3e-3 /
  6.3e-3 of their F32 (a block's bound sqrt(32) / 254, 2.2e-2: all of it in one value); the attention 2.0e-4 of
  double on the cache's values, 5.6e-3 of double on F32's (llama.cpp's F16 cache 7.8e-4); the halves, one at a time
  and past 1,000 positions the bits of one run.
- **perplexity** (the same ids; llama.cpp on its Q8_0 cache: oracle/qwen_dump, QWEN_DUMP_KV=q8_0):

  | tokens | osmos F32 cache | osmos Q8_0 | llama.cpp F16 | llama.cpp Q8_0 |
  |---|---|---|---|---|
  | 2,048 | 6.94-6.95 | 6.9765 | 6.9478 | 6.9738 |
  | 8,192 | 2.7136 | **2.7075** | 2.7193 | 2.7110 |

  the Q8_0 cost at 2,048 llama.cpp's own (+0.04% of its Q8_0, +0.4% of its F16); at 8,192 below both. the operator
  took Q8_0 as the default for the context it buys.
- **speed**: plain decode 23.3 tok/s (the same), drafted 40.5 (41.1: ~1%, the puts' launches); prompt 2,048 663
  tok/s (649), 8,000 454 (417) and its decode 22.0 (21.9) - the cache's reads a quarter of F32's.

### QSA: the sparse attention past 2,048 keys (design, 2026-11-06)

why: the agent bench (4 x 8, ~99K prompt tokens) 638 s against DeepSeek's 440 - the steps shared (q4x_step) and
the drafts too (q4x_mtp_draft_all) moved the wall 733 -> 638 and then not at all: a prompt's pieces and the decode
rounds take turns (multi.c), so the prompts' speed is the wall's, and a dense attention's falls with the context -
2,048 tokens 651 tok/s, 8,000 454, 16,384 315. llama.cpp's QSA (src/models/qwen4exp.cpp, build_qsa_sel; its
cache's rules, llama-memory-hybrid-idx.cpp set_input_kpool, in order mode) caps a token's keys at the budget.

**the math** (one sequence; R = 4, compress_ratios; T = top_k / R = 512 blocks; 4 indexer heads x 128):
- **the keys' blocks**: block j positions R j .. R j + R - 1. its pooled key K_j = rope(rmsnorm(mean of its members'
  raw keys) x k_norm) at position R j (its first member) - the raw key index_k_proj(x), neither normed nor rotated;
  the mean the members summed in order, then x 1 / R (llama.cpp's slices).
- **a token's query**: q_h = rope(rmsnorm(index_q_proj(x)_h) x q_norm) at its position - the main attention's
  IMRoPE (the first rope_dim dims of 128), x the attention's input (the mixer's mixed, as q, k, v).
- **visible**: token p sees the blocks complete at or before it, nv = (p + 1) / R; its tail the n_tail = (p + 1) % R
  positions R nv .. p.
- **the score**: block j < nv: sum over heads in order of relu(q_h . K_j) / sqrt(128) (ggml_lightning_indexer).
- **the pick**: the top min(nv, T) blocks by score - ties to the lower block (ggml_top_k leaves them unordered:
  ours fixed); the attention over their positions and the tail. nv <= T: every block - the dense attention.

**osmos's shape**:
- **one attention, a key list**: a token's keys one flat list - its picked blocks' positions ascending, then the
  tail - in chunks of 128 merged in order (fa_kernel's chunks of the list, not of positions). every block picked
  (nv <= T, below 2,052 positions) the list is 0 .. p: today's chunks, today's bits. a token's chunks at most
  ceil((R T + R - 1) / 128) = 17: the scratch a step's tokens x 17, whatever the context.
- **the cache**: a QSA layer's pooled keys F32 [max_ctx / R][128] a sequence (~1.5 KB a token for the 12, the head's
  too); the raw keys a ring of the last R - 1 + Q4X_MAX_BACK positions (a rollback's partial block in it), the step's
  own from its buffer. a step pools the blocks it completes before any score (a token sees a block its step ends).
- **the kernels** (kernels/q4x_qsa.hip): the indexer's projections (q4x_mat: BF16, mv); its q norm and rope; the
  pooling; the scores - a token's against its visible blocks, tiled (tokens x blocks); the pick - a token's top T by
  a radix select on the scores' bits (non-negative: relu sums), ties to the lower block, the picked in block order;
  the attention reading the list.
- **the bits**: a score's dot products in one order, a pick by (score, block): a token's list its own, however
  batched - the halves, one at a time and shared steps the bits of one run, as now.
- **the state**: the pooled keys below the position's blocks and the ring - the format's 3.

**held to** (tests/gpu/test_q4x_qsa_gpu): llama.cpp's tensors at 3,000 tokens (qwen_dump: indexer_q, indexer_k, its
scores and picks): q and the pooled keys to F32's rounding of their F16 cache, the picks the same blocks but near
ties (named); a double reference of the pick from our own keys and queries - the same blocks exactly; the
attention against double over its picked keys; the halves, one at a time and shared steps the bits of one run past
2,052 positions; perplexity at 8,192 and 16,384 against llama.cpp's QSA (8,192: F16 2.7193, Q8_0 2.7110) - at or
below; mutants (a tie to the higher block, the tail off by one, a block's rope at its last member).

**the measure**: prompt tok/s at 2,048 / 8,000 / 16,384 / 65,536, decode at 8K and 64K, then the agent bench.

**measured** (2026-11-06, performance):
- **test_q4x_qsa_gpu** (layer 3, 3,000 tokens): the raw keys 1.9e-3 of llama.cpp's (its BF16 matmul), the query
  1.9e-3; the pooled keys from its raw keys 2.1e-4 (its F16 cache); the picks of all 949 tokens past the budget the
  double's blocks exactly and llama.cpp's exactly (no tie at the cut in this text; a zero query's every score tied:
  blocks 0 .. 511); the attention over them 1.9e-4 of double, 5.7e-3 of llama.cpp's (its F16 cache, our Q8_0); pieces
  of 1000 + 999 + 1 + 1000 and a token at a time the bits of one run; below the budget the dense bits. mutants: ties
  taken wholesale, the tail one short, a block rotated at its last member - each caught.
- **perplexity** (the same ids):

  | tokens | osmos dense | osmos QSA | llama.cpp QSA F16 | llama.cpp QSA Q8_0 |
  |---|---|---|---|---|
  | 2,048 | 6.9765 | 6.9765 (the same bits) | 6.9478 | 6.9738 |
  | 8,192 | 2.7075 | 2.7143 | 2.7193 | 2.7110 |
  | 16,384 | 2.2714 | 2.2732 | 2.2721 | 2.2675 |

  QSA's own cost +0.25% / +0.08% of our dense; llama.cpp's two caches 0.2% apart at 16,384 - the spread a
  forward's roundings make. at 16,384 above llama.cpp's (+0.05% of F16, +0.25% of Q8_0), dense too: not the picks'.
- **speed**: prompt 2,048 672 tok/s (651 dense), 8,000 588 (454), 16,384 567 (315), 65,536 500; plain decode at 8K
  21.8, 16K 21.6, 64K 19.6; at 512 23.2 (the same), drafted 39.5 (40.2: the indexer's launches in a draft step).

### 0.5.5: the prompt's GEMM - the operands shared, the order kept (2026-11-07, low-power)

why: Strata's fast set reads a 16K Qwen prompt ~40% faster at every length (attention is not it: QSA keeps ours 198
-> 295 us a token from 4K to 32K; the matmuls are ~830 of ~1,450 us a token). the matrix cores' own peak here
(wmma_rate, low-power): f16 45 TFLOP/s, iu8 48 TOPS - int8 no faster on RDNA 3.5, and with the q8 contract's per-32
scales after each block (on the same SIMD) at most ~29 before a single load (i8epi): not the way for the dense mm.
0.5.4's tries said the bound is each wave's own operands, none shared (tile's waves are the K split, the sums' order).
- **mm.h's gemm_kernel**: a block 64 rows x 256 tokens, 8 waves of 32 x 64; each slab (64 rows x 128 values) decoded
  into LDS once by the block, the next under this one's WMMAs (two slabs' LDS, a barrier a slab). the order kept:
  partition p (tile's wave p) its slabs p, p + 8, ... one WMMA chain a 16 x 16 tile from zero, the partitions in turn
  into a running sum (tile's part[0] + part[1] + ...) - the same bits at every count, a prompt's as decode's (mmbits:
  every bit of y at 1-2,048 tokens; test_mm_gpu; a mutation - the chain not reset a partition - caught).
- the matrices (Q6_K / Q8_0, Qwen's shapes, 2,048 tokens): 10240 x 2560 20.6 -> 27.0 TFLOP/s, 2560 x 6144 22.9 ->
  27.6, 6144 x 2560 20.7 -> 26.7, 12288 x 2560 20.9 -> 26.8, Q8_0 20.5 -> 28.7. a single slab's LDS 25.5; 64 x 128
  22.8; 32 x 256 22.4; 128 x 128 past the 64 KB LDS; 128 x 256 spills.
- Qwen (against a v0.5.4 build, two interleaved rounds, every hash the same): 16K 590 / 584 -> 637 / 623 tok/s
  (+7%), 4K 634 / 647 -> 703 / 745 (+13%), 2K 705 / 668 -> 776 / 735 (+10%), 512 +3%; decode unchanged (it is tile's).
- DeepSeek (UD-IQ1_M, dbench, greedy; two interleaved rounds): every stream 5ca48d476414fd50 (0.5.3's); 2K 259 / 251
  -> 263 / 258 tok/s (+2%), 8K 387 / 386 -> 391 / 387 (+0.6%) - its prompt the SSD's, its dense mm a small share.
- **a step past MM_MAX_TOKENS a layer's part at a time** (hip_step.c: the n-gram layer, the mixers, the DeltaNet or
  attention and the shared expert a sub-batch of 2,048 - b's token rows moved to it, its runs' parts at their
  positions and rings' cur - then the routed experts over the step whole): the same bits (test_q4x_forward_gpu: a
  step of 3,000, two sequences and one, against steps within 2,048; a ring's cur or a position one off caught). a
  16K prompt (profiled, one session) 2K -> 8K steps 25.89 -> 25.24 s: the experts' matmuls -0.97 s at ~160 tokens
  an expert, their activation +0.31 and moe_sum +0.19 (the experts' F32 rows out of the cache) - timed 2-3% (the
  box drifted 3% between rounds; the parked whole-step x16 not needed: each part's f16 its own).
- **the experts' activation straight into the down's f16** (q4x_swiglu_rows: q4x_swiglu then mm_x16_rows's bytes,
  test_q4x_moe_gpu): those kernels 1,946 -> 1,634 ms at 8K steps, 1,480 -> 1,388 at 2K; and the shared expert with
  the layer's part (b.x16 its mixer's): its conversions at 8K 143 -> 12 ms. what is left: gate and up's F32 (420 MB
  a layer at 8K) written then read.
- **gate + up in one kernel, swiglu's f16 its epilogue** (mm_group_swiglu: a block's gate tile kept in LDS, then its
  up tile, each tile_to's own sums - the down's output the same bits, test and a mutation): slower. 16K at 8K steps
  (profiled, back to back) gate + up + q4x_swiglu_rows 3.66 s -> 4.11 s with two tile copies inlined (IQ3_XXS at 8
  tiles 248 VGPRs, 36 B spilled, 5 waves), 5.70 s with one in a two-pass loop (222, none, 6): a block's latency
  doubled, half the blocks, and the up's epilogue (silu, the f16 put) wave 0's alone. not kept (the diff in the
  0.5.5 session's notes); the F32 round trip is cheaper than the block it saves.

### 0.5.5: DeepSeek's resident set settled to a prompt - measured, not kept (2026-11-07, performance)

why: after a prompt the resident set is the profile's (a prompt's steps pick nearly every expert, so the swaps - the
coldest resident not picked in the step - all but stop), and a decode's first picks stream behind their routers.
UD-IQ1_M, an 8K prompt then 64 decode tokens (dbench, 8 against 72 tokens, the difference): 4.7 experts streamed a
token, 8.2 ms of a ~64 ms token waited on their reads - ~13%, the rest the GPU's.
- **the settle** (after a step of 32 tokens or more, before the next smaller one: each layer's hottest streamed
  experts - the prompt's picks - in place of its coldest resident ones, FWD_SWAP_MARGIN hotter, up to 16 a layer and
  2 GiB of reads, pinned together then waited on and copied): the same bits (test_model_gpu: the decode's logits with
  it off; every moved expert's memory its file's bytes - a map left unmoved caught there, not by the logits, the
  coldest resident ones too rarely picked). 289 experts settled; plain 15.6 -> 14.9 tok/s, drafted 19.6 -> 18.4: the
  decode's streamed picks 779 -> 738, 1,680 -> 1,585 - its reads cost more than they saved. a prompt's hottest
  experts are not the ones its continuation reaches for next.
- **the pool's alone** (no read: only the experts the pool still holds): 16 settled, 15.7 / 19.7 against 15.6 / 19.6.
not kept (the diff in the session's notes). and the ~13% the reads are is the most any read-side lever can give
here: the decode's ~56 ms on the GPU is the bigger part - next, profiled.

### 0.5.5: Qwen's decode measured - what does not help (2026-11-07, performance)

a plain decode token (rocprofv3, 64 steps after a 512-token prompt; the logits launch each step's end): 1,743
kernels, 42.3 ms on the GPU - busy 38.4, the gaps 3.9 (2.2 us each). the dense mm 21.0 ms (400 launches), the
experts' group 8.7 (144), the small F32 matvecs 3.0 (288), the rest ~5.7. llama.cpp's plain decode 26.2 tok/s to
our 23.7: the gap is these kernels' efficiency at a token, not the launches.
- **the dense matrices at a token** (dec1: each read from DRAM, the copies rotated past the cache): ~3.85 GB a
  token at ~181 GB/s - the DeltaNet's qkv + gate 200 GB/s, ssm_out / attn_output 178, the logits 210, the hc
  mixers' Q8_0 320 x 10240 / 10240 x 320 156-160, the shared expert's 129-134. at 230 GB/s 16.7 ms against 21.2:
  ~4.5 ms the most a dense-mm change could give.
- **the experts at a token each** (exp1: 10 experts a launch, the 512 rotated): IQ2_S gate / up 91 GB/s, IQ3_XXS
  114, IQ4_XS 129, IQ4_NL down 160, Q2_0 98 - 8.2 ms a token. a launch's time its experts' (1: 9 us, 2: 21, 5: 41,
  10: 58, 20: 97): each wave's K partition a chain of slabs, each slab's loads after the one before's work.
- **the unroll at a token** (U, a lane's chunks in flight - the sums unmoved): 8 (the shapes') 8.2 ms, 4 8.6, 2
  10.5, 1 13.6 - the loads in flight are what the chain needs, not the occupancy fewer registers buy.
- **the next slab touched** (a lane's row of it loaded at the slab's start, its wait at the slab's end): experts
  8.2 -> 8.4 ms, dense 21.2 -> 22.2. not kept.
- **HIP graphs** (a token's step captured - 1,737 nodes - and replayed; built on the per-thread default stream):
  decode 23.42 / 23.54 tok/s against 23.63 / 23.56 launched - ROCm's replay no faster than the launches. not kept.
what is left at a token needs the K partitions' chains shorter, which is the sums' order: decode as it is stands
until a design keeps prompt and decode the same bits another way.

### 0.5.4: the dense mm explored - what does not help (2026-11-06, performance)

why: Strata's fast set (its docs: a 128 x 256 WMMA GEMM for the prompt's FP16 projections, from 4,096 tokens) reads a
16K Qwen prompt at 946 tok/s against our 676; a 16K prompt here 24% dense mm, 22% experts, 21% small kernels, 18%
attention. the dense mm (both models' Q6_K / Q8_0) timed alone (Q6_K, Qwen's shapes, 2,048 tokens: 27-31 TFLOP/s,
~half the matrix cores' ~59 by our estimate) and counted (SQ_*: 70% of wave cycles waiting - 26% on memory, 24% on
instruction dependencies; VALU 4%; no LDS conflicts; 234 VGPRs, 6 waves a SIMD). every variant below the same
checksum (the same operands in each tile's order - a token's bits unmoved), each slower:
- **the tiles' WMMAs interleaved** (8 chains, not 8 WMMAs back to back): 9 TFLOP/s - 256 VGPRs and spills; 2 or 4
  chains a group, no spills, 9 too: a tile's 8 x loads in flight together were what hid the memory, not the chain.
- **A from LDS a step at a time, x double-buffered** (kPipe's for long slabs): 21-25 at 8 tiles (spills, 176 B); at 4
  or 2 tiles a block (no spills) 13-18 - a weight slab decoded for fewer tokens costs more than the 4% VALU said.
- **fewer waves (less K split)** at a token (mmsweep): 8 waves 195 / 178 GB/s (10240 x 2560 / 2560 x 6144), 4: 191 /
  160, 2: 169 / 176, 1: 160 / 132 - a big tile without the K split costs decode 18-26% (decode and prompt share the
  order), the head the exception (4 waves 228 against 208).
so the current tile is the best of its family here; a gain needs a different design (a block's x shared across more
rows through LDS at the same K split, or per-shape splits - each its own experiment at 1 and 2,048 tokens).

then the experts (a 16K prompt's expert path 33%: the matmuls 5.3 s at ~15 TFLOP/s, the gather 1.17, moe_sum 0.64,
the tables 0.55, swiglu 0.27; their counters 83% waiting, 54% on memory, VALU 2.3%):
- **kept**: the tables' placement an expert a block (tiles_kernel's one block scanned every pick an expert): 551 ->
  73 ms, the same tables (test_q4x_moe_gpu against a host stable sort).
- **the gate / up read in place** (mm_group by a slot's token, not the gathered copy): the same bits, slower - 16K 693
  -> 640 tok/s, 2K 794 -> 718: the gathered tiles' contiguous reads are what the kernel is fast on.
- **8K steps** (each expert ~160 tokens, its weights read once for them, not ~40): the experts 5.39 -> 4.16 s, but
  the side copies' f16 lost past MM_MAX_TOKENS (+1.19 s of conversions) and the DeltaNet's per-token kernels +0.76
  (their 4x buffers out of the cache) - net slower (24.6 against 23.7 s). worth it only with a step-wide x16 and the
  DeltaNet's passes chunked: ~1.5% left after the conversions, not yet.
- **the DeltaNet's columns over 4 or 8 blocks a head** (a column's sums its own: the same bits): 1,084 -> 1,029 /
  1,229 ms - its cost is a token's chain (q / k in, a sync, two 64-value dots, shuffles), not its 48 blocks.
- **the step-wide x16** (b.x16 a step's tokens, each chunk of MM_MAX_TOKENS at its own tiles: parked on
  park/x16-whole-step): the same hashes at 2K, 4K and 8K steps, the conversions gone (x16_kernel 941 -> 0 ms) - and a
  16K prompt at 8K steps level with 2K (23.17 against 23.27 s): the experts -1.05 s, the router -0.28, but gated_norm
  +0.30, swiglu +0.20, the DeltaNet +0.20, the dense mm +0.20, conv +0.17 - a 2K step's activations (~32 MB) stay in
  the 32 MB Infinity Cache, an 8K step's do not. the gain needs a layer's per-token part in 2K sub-batches and only
  the experts over the step (DeepSeek's FORWARD_SUBS shape): ~1 s of a 16K prompt (~5%, prompts past 8K only) for the
  forward's most delicate code (the DeltaNet's history and the attention's positions split mid-sequence) - not now.

the dense mm again (a harness hashing every bit of y at 1, 7, 16, 33, 100, 512 and 2,048 tokens against the current
kernel, Qwen's shapes; the current kernel itself 25.6-28.4 TFLOP/s run to run on 10240 x 2560). every variant below
the same bits at every count, each slower at 2,048 tokens:
- **R row tiles a block** (each wave all R over its own slabs, a slab decoded SC values at a time: an x fragment one
  load for R WMMAs): R 2 at 8 tiles 13 TFLOP/s (256 VGPRs, spills); R 2 at 4 tiles 23.6 (212 VGPRs, no spills - x's
  traffic halved, slower); R 4 at 2 or 4 tiles 10-15; R 1 decoded 64 at a time 27.1 (the parts cost nothing). so x's
  loads are not the bound: a slab decoded for fewer tokens is.
- **P token groups sharing a K-wave's decoded slab through LDS** (a decode for 256 tokens, not 128; every wave the
  same trip count, block barriers): P 1 (the current tile with block barriers, nothing else) 22.5 against 27.2 - the
  barriers alone 18%; P 2 at 8 tiles 22.7, at 4 tiles 18.5; P 4 15.6. the waves' independence (each its own slabs,
  no barrier until the sum) is what hides the latency.
what is left is the GEMM rewritten whole (the next slab's weights loaded under this one's WMMAs, at the same K split):
a large change, and its smaller form (x double-buffered, above) was slower. the dense mm is 24% of a 16K prompt, so a
20% faster kernel is ~5% of it; the attention (18%, and growing with the prompt's square) is the better next look.
- **the weights decoded to f16 once, then the same tile on them** (no decode, no LDS but the sum's: the same bits):
  slower - 10240 x 2560 at 2,048 tokens 3.76 -> 5.28 ms with the decode, 4.93 without (the f16 2.4x the Q6_K's
  bytes); the row blocks' token blocks side by side 7.3-7.7. the decode in the tile is cheaper than the bytes it
  saves: the bound is the waves' own operands, none shared - a GEMM sharing them (a block a 64 x 128 tile, the
  partitions in turn, a running sum: the same order) is what is left, its barriers to be hidden.
- **the tables' placement for few picks** (kPlaceInline, 256): a decode token's 10 picks placed by the tiles kernel
  itself, not a second launch - the second cost decode 1.2% (23.8 -> 23.5 tok/s, 3 of 3 rounds); now 23.65 against
  0.5.3's 23.67, the same tables (test_q4x_moe_gpu: 30 picks and 2,560).
- **DeepSeek's prompt step, the packs' settings** (UD-IQ1_M, 1M context, drafted, budget auto; apibench through
  osmos-api, two rounds each): 2K steps 262 / 355 / 455 tok/s (2K / 4K / 16K prompts), drafted answers 32.7 / 20.0 /
  27.8; 4K 262 / 390 / 536, -1 to -2%; 8K 253 / 381 / 578, -6 to -10% (the budget 99.9 -> 98.6 -> 95.8 GiB: fewer
  experts resident). the packs keep 2K; osmos-api's `max_tokens` the choice.

### 0.5.3: DeepSeek against Maya 1.1.56, and bigger prompt steps (2026-11-06, performance)

- **a prompt step's reads**: DeepSeek's prompt is the SSD's (a 2,048-token step touches nearly every expert, the
  streamed ones read again each step). the step was already a setting (FORWARD_SUBS x 2,048; a layer's experts run once
  over every sub-batch): UD-IQ1_M's 8,192-token prompt 470 -> 589 tok/s at 8,192-token steps (2K: 304 -> 288), the
  budget 105.9 -> 102.9 GiB (the experts' step buffers), decode -2 to -3%, every answer's hash the same.
- **the head-to-head** (~/bench/h2h-056/perf.sh: Maya 1.1.56 rebuilt and calibrated, warmed; two interleaved rounds a
  pack, the order swapped in the second; memory sampled a second apart, memlog.sh): UD-IQ1_M prompts 300 / 445 against
  241 / 408, drafted 33.5 / 27.6, plain 17.1 / 17.0; UD-IQ1_S 327 / 488 against 258 / 436, 40.6 / 34.5, 19.4 / 19.3;
  osmos's 8K prompts at 8K steps 557 / 595. agents (UD-IQ1_M, under tools/power.py): osmos 440.5 s, first token p95
  55.5 s, 6.96 J an answer token, peak GTT 109.7 / RAM 116.9 GiB; at 8K steps 421.7 s, 32.8 s, 6.68 J; Maya 547.7 s,
  100.3 s, 8.81 J, 112.5 / 120.9 GiB. Maya 1.1.56 on these GGUFs is 1.1.55 (its 1.1.56 Halo gains are EXL3's).
- **at Maya's memory**: osmos at a 108 GiB budget (auto: ~106): plain 17.0-17.1, no faster; drafting and 8K steps
  refused at 108 ("GPU memory refused": fails closed) - auto already at the useful edge.
- **perplexity** (the first 8,192 ids of calib16k.txt, BOS + 8,191 scored; osmos bin/ppl budget 100, steps of 512;
  Maya maya-ds4 --ppl, one forward): UD-IQ1_M **2.4579** against 2.4649, UD-IQ1_S **2.5006** against 2.5023 - osmos's
  own published numbers to the digit. Maya's --ppl keeps every row's logits on the host (~4 GB) beside its full
  memory: the OOM killer took it (exit 137) until --reserve-mib 8192 (it moves experts, not arithmetic).

### 0.5.2: YaRN to a million positions (2026-11-06)

the model card (Qwen/Qwen3.8-Flash-Next, "Processing Ultra-Long Texts"): native 262,144; past it static YaRN,
rope_type yarn, factor 4.0 over original_max_position_embeddings 262,144 for 1M - and, its note, static YaRN "may impact
performance on shorter texts": turn it on only when the context needs it, its factor the context's ratio (2.0 for
512K). so:
- **q4x_rope(dim, base, native, ctx)**: plain at or below the native context (today's rotation, its bits); past it
  factor ctx / native, llama.cpp's YaRN - ext_factor 1, beta_fast 32 / beta_slow 1 its correction dims
  (ggml_rope_yarn_corr_dims' floor / ceil), the angle freq_scale theta mixed with theta by the ramp, the pair's cos
  and sin times 1 + 0.1 ln(factor). one helper (kernels/rope.h, rope_cs) for every rotation - the attention's q and
  k, the indexer's query and its pooled keys, the head's.
- **the angle at a million positions**: an F32 angle's last bit is ~0.06 rad at 700,000; the plain rotation keeps
  its running product (ggml's CPU rope cache: the bits of 0.5.1), YaRN's takes llama.cpp's GPU form, pos
  theta_scale^i - against llama.cpp's own at 700,000: 1.7e-6 (the running product: 2.1e-3, the two roundings apart).
- **the forward's rotation** its context's (q4x_forward_create), in its identity: a 262K state is not a 1M one's.
  OPS_Q4X_MAX_KEYS 1,048,576: the Q8_0 cache ~13.6 GB at 1M, the pooled keys ~1.7 GB, the weights 78 - ~93 GB.
- **held to** (test_q4x_attn_gpu, qwen_dump QWEN_DUMP_YARN=4 and QWEN_DUMP_POS0): q and k normed and rotated
  against llama.cpp's at positions 0 .. 255 (6.9e-8 / 6.4e-8) and 700,000 .. 700,255 (1.7e-6 / 1.9e-6); mutants (no
  magnitude, the betas swapped) caught; the plain rotation the same ids as 0.5.1 (4K and 16K prompts).
- **perplexity with YaRN** (factor 4: a 1M forward on 8K / 16K of text):

  | tokens | osmos plain | osmos YaRN | llama.cpp YaRN F16 | llama.cpp YaRN Q8_0 |
  |---|---|---|---|---|
  | 8,192 | 2.7143 | 2.7256 | 2.7067 | 2.7196 |
  | 16,384 | 2.2732 | 2.2839 | 2.2859 | 2.2911 |

  YaRN's cost on short text +0.4 / +0.5% (the card's note); against llama.cpp on its Q8_0 cache +0.2% / -0.3%.
- **past the native context** (osmos-api, a 1M config): a fact at 10% of a 394,045-token document (the calibration
  text over and over, numbered) asked for at its end - 355K tokens back, past the native 262,144 itself: "The woman's
  name was Marisol Quintero, and she painted the lighthouse's door green." word for word; the prompt 952 s (414
  tok/s). the first needle, a "secret passphrase for the vault", the model found and refused - "a prompt injection
  attempt": the needle's wording, not the context.
- **the 1M forward on short text** (performance; the 262K's beside it): prompt 2K 745 tok/s (810), plain decode 22.97
  (23.18), drafted 34.9 (40.4 - the head's acceptance 59 -> 50%: trained without YaRN, it and the model agree less
  under it). the prompt's -8% not the rotation's cost (a few VALU a pair): open, for 0.5.3.
- **the pack**: qwen38-ud-iq3_xxs-1m, the same files (installer: packs.json), rig's row 1M with the card's 256K of
  reasoning a turn; the 262K pack the default - the card's advice, and the numbers above.

### 0.5.1: decode and prompts, profiled (2026-11-06, low-power)

rocprofv3 on both (a decode token: a 65-token run less a 1-token one), each change A/B'd at low-power against a v0.5.0
build from its tag (interleaved, twice; tools in the scratchpad: the old q4x_bench and the new, the same ids):
- **Qwen decode, 40 ms of GPU a token** (1,886 launches): Q6_K's dense matmuls 16 ms (~195 GB/s), the experts 9.1
  (~110 GB/s), our small kernels 6.2, the Q8_0 mixers 5.7, the F32 matvecs 2.7, ~3.7 ms of gaps. **DeepSeek's**: 51
  ms GPU of ~63 (the rest the SSD's experts), Q6_K ~22 ms.
- **the bound the guarantee sets**: mm at one token is ~180 GB/s on Q6_K at best (every shape swept), the matvec 230
  - mm's sums at every token count are what keep a token's bits its batch's. the experts the same: decode-bound (IQ3_XXS
  ~100-130 GB/s, 10, 20 or 40 matrices a launch within 12%: not the launch).
- **kept** (each the same ids as v0.5.0):
  - the attention's cache read 8 values a load, a block's scale once; then K and V a block ahead in registers.
  - **the attention's V^T stores**: a wave's lanes along a key's dims wrote 16 lanes to one bank pair - 16-way
    conflicts, 5.1 G cycles against the LDS's 1.6 G (SQC_LDS_BANK_CONFLICT). V's lanes along the keys: 0.16 G, the
    kernel's wave cycles -45%.
  - **the DeltaNet's q / k rows**: a lane's half and its pair's 64 floats apart, one bank (4.08 G conflict cycles to
    1.38 of LDS); the high half a float on: 0.03.
  - a block's output combined in the next mixer's norm launch (q4x_hc_combine_norm: the MoE's, the attention's, the
    head's) - ~100 launches a token fewer.
- **measured and not kept**: the attention's S once a round of 64 keys (bits the same, 37% slower: 52 KB of LDS, one
  block a CU); S's dims split over the waves (7.5% of the kernel, its bits moved: not worth them).
- **the Q8_0 mixers' K split across blocks** (320 x 10240: 20 blocks for 40 CUs, the guess its idle CUs): 2 parts
  127 GB/s, 4 parts 119, against 154 unsplit (test_mm_gpu at a token) - the second launch and the parts' round trip
  cost more than the CUs gave; the unsplit already at the matvec's 159 there. not kept (a split by the rows also made a
  row's bits its matrix's height's - test_mm_gpu's pair caught it; by the columns alone it held). test_mm_gpu now times
  and holds Qwen3.8's two mixer matrices.
- **the scan**: every kernel's SQC_LDS_BANK_CONFLICT against SQ_INST_CYCLES_LDS, both models - the matmuls clean;
  DeepSeek's worst its scores kernel (30% of its LDS cycles, 1.4% of the time). DeepSeek's prompt is the SSD's (the
  GPU busy 11 s of a 132 s run): no kernel moves it.
- **Qwen at low-power** (v0.5.0 -> now, two rounds each): prompt 4K 569 -> **671 tok/s (+18%)**, plain 23.1 -> 23.5
  (+1.8%), drafted 39.0 -> 39.8 (+2%).
- **at performance** (the release's once, v0.5.0's build beside it; every ids hash the same):

  | | v0.5.0 | 0.5.1 | |
  |---|---|---|---|
  | prompt 2,048 | 700 | **790** | +13% |
  | prompt 4,096 | 635 | **756** | +19% |
  | prompt 8,000 | 591 | **714** | +21% |
  | prompt 16,384 | 561 | **696** | +24% |
  | prompt 65,536 | 501 | **640** | +28% |
  | decode at 16K, plain | 21.2 | 22.4 | +6% |
  | decode, plain / drafted | 23.1 / 39.5 | 23.7 / 40.5 | +2.6% |

  the gain grows with the context: the attention's share of a prompt does (llama.cpp at 2K / 8K / 16K: 520 / 489 /
  463 - osmos now +52% / +46% / +50%).

### 0.5.0: DeepSeek unchanged (2026-11-06, performance)

the kernels Qwen shares (mm, mv, quant.h, block.h) and the engine's and server's draft path (oseq_draft_all;
DeepSeek's draft_all NULL: one lane at a time, as before) moved for Qwen. held: osmos-serve's token streams the
v0.4.0 binary's (built from its tag, the same libevt) on 16 requests - each the same hash as 0.4.0's own check
(~/bench/v050-streams.log); the DeepSeek GPU tests pass (test_forward_gpu, test_lane_gpu skipped: GSQ-RCO's dumps
not on this box, as before); dbench UD-IQ1_M plain 17.1 / 17.1 / 17.0 tok/s, drafted 35.1 / 27.7 / 38.0 (33.6 the
mean: 0.3.2's), each question's hash the same plain and drafted.

### correctness first: Qwen held as DeepSeek is (2026-11-06)

the operator's: osmos's DeepSeek answers read better than maya's on the same files - the arithmetic held to a
reference, every sum in a fixed order. Qwen is held the same way before any speed work:
- **the types**: Q2_0, IQ3_XXS, IQ4_XS each against ggml's dequant, bit for bit (as every type here).
- **the forward**: a block at a time against llama.cpp's intermediate tensors on fixed ids (llama-eval-callback), then
  the logits' KL divergence over a real text - ~0, its max named.
- **perplexity**: bin/ppl on the 8,191 tokens of mixed text and code DeepSeek's 2.459 is measured on, against
  llama-perplexity on the same file and text - osmos's at or below llama.cpp's, never above; a README row.
- **the limit**: the BF16 model (~250 GB) does not fit here, so the reference is llama.cpp on the same quant, not a
  PyTorch oracle: it holds our forward, not the quantization's cost (unsloth's own KL figures speak to that).

### the baseline: llama.cpp on this box (2026-11-05, performance)

llama.cpp 68d20b8 (2026-10-29), its HIP build for gfx1151 (~/Projects/llama.cpp/build-hip), UD-IQ3_XXS (77.87 GiB;
llama.cpp names it A3B: ~3B active a token), all layers on the GPU, flash attention:
- coherent: "What is 17 times 23?" - its thinking short and right, the answer 391 (greedy).
- llama-bench, 3 runs: prompt 512 415 tok/s, 2,048 419; decode (128) **26.3 tok/s**; the package 67 W mean (87
  peak) - at performance, under half the DeepSeek decode's ~130 W: not the memory's rate that bounds it.
- its decode traced (rocprofv3, 64 tokens; the trace's own overhead inflates the wall, not the kernels): 3,040
  kernels a token, the GPU busy ~34 ms of a token's ~43; mul_mat_vec_q 73% of that (545 launches a token: Q6_K's
  dense projections and head the most, ~13 ms; the experts' IQ4_NL / IQ3_XXS / IQ3_S ~6 ms; Q8_0 ~5 ms), the F32
  matvecs 9%, every other kernel (the GDN's delta rule 0.55 ms, its conv, the gated residual, top-k) a few percent
  each and most of the launches.
- the bound, from the file's tensors (not llama.cpp's "A3B" label - unsloth's 6B activated is the right one): a token
  reads ~3.5 GB of dense weights (the GDN / attention projections Q6_K ~2.6 GB, the gated residual's down / up Q8_0
  0.67 GB, the F32 router 0.25 GB, the head 0.52 GB), ~1.0 GB of experts (10 of 512), and the GDN state in and out
  (~0.23 GB F32) - ~4.7 GB; the n-gram table (28.8 GB) and the token embedding are lookups. at the ~170-200 GB/s
  osmos's kernels read here: ~26 ms, ~38 tok/s - llama.cpp's 26.3 (~124 GB/s) is ~70% of it: ~1.4x for a token at
  a time, not more. the gain is the verify's: a drafted block reads the dense 3.5 GB once for all its tokens (as
  DeepSeek's 17 -> 33) - MTP (the separate mtp-*.gguf; unsloth: 1.3-1.7x). targets: plain ~33-36, drafted ~45-60.

## 0.4.0: one engine interface, two families (design, 2026-11-05)

why: osmos-serve, dbench and the tests call DeepSeek's forward by name (Forward, Seq, Drafter: ~30 functions). a
second family behind the same server needs one shape both forwards present - the server's, nothing more. 0.4.0 is
that shape with DeepSeek alone behind it: no Qwen code, no feature, every bit the same.

- **what the server uses** (tools/serve: main, build, conv, single, multi, round): create from the files and options
  (budget, max_ctx, max_tokens, picks, no_guess, the table, the overlay, the CPU lane, the drafter's dir) and destroy;
  a sequence - create, destroy, prompt(end), rollback(n), position, state bytes, save, load (osmos-serve's slots,
  forward_identity their key); forward_batch over sequences and their tokens, forward_greedy on a step's rows,
  forward_logits; the drafter - seed(seq, from, to), draft(seq, y, floors) -> its drafts and confidences; the
  facts the server sizes by (a decode step's most tokens - (n_experts - 1) / n_used today -, the drafter's window,
  its block); counters (budget, room, swaps, waits, guesses, CPU-lane picks). forward_step only as batch of 1.
- **the shape**: include/osmos/run/engine.h - `Engine` (opaque), `EngineSeq` (opaque), `EngineOptions` (the union of
  the options above; a family refuses one it has no use for by name - fail closed), and the calls above as
  engine_* / eseq_*. inside, one table of the family's functions (a struct of pointers, filled by the family's
  engine_<family>.c); engine_create reads `general.architecture` and picks the family - deepseek41 (the forward as it
  is), later qwen4exp - or refuses the file by its name. a family's own headers (forward.h, drafter.h, weights.h)
  stay for its tests and tools; the server includes engine.h alone.
- **what moves**: src/run/hip_engine.c (the dispatch), src/run/hip_engine_ds4.c (DeepSeek's table: each a call into
  the forward as today); tools/serve onto engine.h; the server's sizing facts asked of the engine, not read from
  Geometry. nothing in the forward changes but seq_reach (rollback's reach, said). dbench stays DeepSeek's: its
  probes (ROUTES_OUT), picks dump and CPU lane's threads are the family's instruments - the interface would carry
  them for one family; a family's bench is its own. the names: libevt has Engine / engine_create, so the interface
  is Osmos / OsmosSeq / osmos_* / oseq_*.
- **a step's rows**: DeepSeek keeps a long step's last sub-batch's logits (MM_MAX_TOKENS), the server computed the
  row ((k - 1) % MM_MAX_TOKENS); the interface takes the batch's own row numbers and refuses a row not kept - the
  family maps them.
- **done when**: every CPU and GPU test passes (test_engine_gpu: the interface's tokens, drafts, logits and saved
  state the family's own calls' bits); osmos-serve's token streams the v0.3.2 binary's on the same requests (single
  and --parallel, drafted and not); its parallel and slot tests pass; the agents bench level with 0.3.2 in one
  session (the call is one indirect jump a step: nothing to measure, measured anyway).

- **measured** (2026-11-06): every CPU and GPU test passes (test_engine_gpu: 22 tokens, 20 drafts, the last step's
  129,280 logits and a saved state the family's own calls' bits; a refused load leaves the sequence empty, as
  seq_load's contract says; a 2,100-token step's rows 2,048 - 2,099 served by their own numbers, 2,047 refused; a
  mutant without forward_lane caught); osmos-serve's token streams the v0.3.2 binary's on 16 requests (the five
  questions greedy and drafted, a 2K prompt, a continued turn reusing 75 tokens, four at once - ~/bench/v040-streams.log);
  parallel_test and slot_test pass; the agents bench at performance 447.6 s, 7.07 J an answer token (v0.3.2 the
  evening before: 442.5 / 448.7 s, 7.07 J) - level.

what the second family brings to the shape (designed now, built in 0.4.x, so 0.4.0 does not have to move again):
- **a recurrent state**: Qwen's 36 Gated DeltaNet layers hold ~113 MB a sequence (48 x 128 x 128 F32 a layer) and a
  conv tail, not a cache a position indexes. rollback(n) - a verify's rejected drafts, a conversation cut back - is a
  restore, not a move: the verify writes the state after each of its rows (a block of k: k snapshots, ~0.7 GB a
  sequence at k = 6; ~37 GB of the box is free with the whole model resident), and rollback takes the row's.
  SEQ_MAX_BACK's reach beyond a verify (conv_reach's common part) holds for attention only: the family says how far
  back it can go (engine_reach), and the server reads the prompt again from the last state it holds when farther.
  save / load carry the state as they carry the cache (its bytes the family's).
- **its drafter**: MTP (the file's 4B multi-token head) behind the same seed / draft pair - its block its own
  (engine_block), its floors the server's.
- **its text**: Qwen's pre-tokenizer (the file's tokenizer.ggml.pre names it `qwen35`: llama.cpp's name for the split
  Qwen has kept since 3.5, not the model's version - the file is Qwen3.8-Flash-Next) - a split of its own, beside
  deepseek-v3's in src/text, its chat template (the file's jinja, ported to C and held byte for byte to its output on
  fixtures, as DeepSeek's encoding is), eos 248046, a native context of 262,144 (its own max). osmos-api picks the
  family's text by the architecture as the engine does.
- **resident whole**: ~78 GB, nothing streamed - the budget, picks, swaps and guess counters report zeros, a CPU lane
  is refused (its experts IQ3 / IQ4 / Q2 have no lane kernels).

## the installer (design, 2026-11-04)

the operator's: one command from a stock Halo to a pack ready to run - and, a nice touch, from the files a user has
already. `tools/install/` (python, the standard library alone; `hf` for downloads - installing is not running).
- **the manifest** (`packs.json`, the source of truth): every file a pack reads, by id - its repo, the commit it is
  pinned to (never a branch), its path there, its bytes and SHA-256; and each pack - its model shards, its Engram
  table and the shard slot the table takes (no, count), its picks profile, its drafter, its overlay (abliteration,
  opt-in) and its tested settings (budgets, the guess off, drafted). a file of ours not yet published names the repo
  and no commit: found among the user's files or refused, never fetched unpinned.
- **the table's stamp**: unsloth's MXFP8 table is published stamped 1 of 1; a pack reads it as its shard slot. its
  hash checked canonically - the two header values (split.no, split.count) read as published - so a table already
  restamped (this box's) verifies too. the installer stamps only files it downloaded; a user's file another stamp
  than the slot is copied (a reflink where the filesystem has them) and the copy stamped. it never writes a file it
  did not create.
- **the user's files** (`--have DIR ...`): each manifest file looked for by name under the directories given, and
  taken only if its bytes and SHA-256 are the manifest's (`--trust-size`: bytes alone, the user's call, said so).
- **the host**: a gfx1151; ROCm; the GPU memory limit (ttm's pages, reported - raising it is a system setting, the
  user's); free disk for what is missing. a failed check stops with why.
- **the result**: `<models>/<pack>/`, the shards linked by their names, the table as its slot, `osmos.json` (the
  model's first shard, the picks, the drafter, the overlay if asked, the budgets and flags) and the command to run.
  the engine built from this checkout (`make`, the HIP library, the tools).
- **the launch** (the operator's): the installer ends at a model one can talk to - `--serve` starts osmos-api on a
  port, or `--rig` registers the pack in rig (~/Projects/rig: the agent runtime that launches and routes local
  models): a row merged by id into `~/.rig/models.json` - the provider half (window, maxTokens, reserve,
  keepRecent) and the child half rig launches (`command` with `${PORT}`, `port`, `aliases`, `stopGrace`, `places`)
  - no other row touched, the file replaced whole (written aside, renamed); and, asked first (or by flag), the
  default model in `~/.rig/settings.json`. rig probes `/health`: 503 while loading, 200 when ready.
- **the tests** (`tools/install/test_install.py`, unittest, no network): the manifest's checks (an unpinned file,
  an unknown id, a slot out of range, refused); the stamp read, written and hashed canonically on a small GGUF; the
  user's files found, a wrong hash refused, a wrong stamp copied not written; a pack laid out from local files.

## osmos-api: the engine as an OpenAI-compatible server (design, 2026-11-04)

the operator's: osmos served as rig serves its models - `POST /v1/chat/completions` (streamed as server-sent events,
or whole), `GET /v1/models`, `GET /health` (503 loading, 200 ready) - in C, no Python at runtime: a new binary,
`osmos-api --config <pack>/osmos.json --port N`, over the engine osmos-serve drives (the conversations kept, several
at once, drafting).
- **the tokenizer**: the GGUF's own (tokenizer.ggml.tokens / merges / token_type, pre "deepseek-v3"): byte-level
  BPE - the pre-tokenizer's split (its regex's classes over Unicode categories), bytes to the GPT-2 alphabet, merges
  by rank. checked token for token against Maya's (its pack's tokenizer, the corpus the bench used) and llama.cpp's
  (llama-tokenize), decoded back byte for byte; special tokens matched whole.
- **the template**: DeepSeek-V4.1's chat encoding (its encoding.py, as maya's serve/ds41_encoding.py ports it, MIT):
  system, user, assistant, tool turns, thinking on or off, reasoning effort - checked against maya's golden chats
  (serve/chat_golden.json) byte for byte.
- **the wire**: HTTP/1.1 on libevt (which osmos-serve links), one connection a request's stream; a request's tokens
  to a lane (osmos-serve's multi.c), each token detokenized as it lands (a partial UTF-8 sequence held); `stop`,
  `max_tokens`, `stream`; sampling greedy (the engine's) - temperature and top-p refused by name until sampling is
  built, not ignored.
- **the order**: the tokenizer and its tests; the template and its tests; the server - a request against a fake
  engine (the protocol's), then the real one; then the installer's `--serve` / `--rig`.
- **the process** (decided with the tokenizer and template done): osmos-api runs osmos-serve as its child (the
  protocol over pipes: `--parallel N --serve`, the pack's flags from osmos.json) and is the rest - HTTP, the
  tokenizer, the template - as maya's front drives maya-ds4. one engine reader thread routes the child's lines by
  rid (T, PP, DONE, ERR) to the request waiting on them; a thread a connection, their number capped; a request's
  disconnect STOPs its rid.
- **the mapping** (maya's frontend.py for DeepSeek, and what rig sends): `tools` onto the first system message (an
  empty one put first if none); role `developer` as system; an assistant's `reasoning` as its reasoning_content.
  thinking on unless `chat_template_kwargs.enable_thinking` is false, `reasoning_effort` is "none" or
  `thinking.type` is "disabled"; the effort `reasoning_effort` (or chat_template_kwargs') - "minimal" / "low" 50,
  "medium" / "high" 75, "xhigh" / "max" 100, an integer 1-100 itself (rig's row names low / high / max).
- **the answer**: streamed - reasoning_content deltas until </think>, content deltas after, nothing past the start
  of a DSML calls block; a text that could still become one of those markers (or ends inside a UTF-8 character)
  held until it cannot. at DONE the whole completion parsed (chat_parse): its tool calls one last delta (ids
  call_<rid>_<n>), finish_reason tool_calls / stop / length, usage and timings, then [DONE]; a completion out of the
  format answered as its text (content), stop. whole (stream false): the same, one message.
- **built** (2026-11-04): tools/api (`make api`) - request.c (the mapping; sampling, stop strings, n > 1 and a
  forced tool_choice refused by name), answer.c (the stream's holding back), reply.c, engine.c (the child dies with
  the server: PR_SET_PDEATHSIG), http.c, config.c (osmos.json read whole; an unknown key refused). tested against a
  scripted engine (tests/api, `make api-test`: health while loading, thinking and tool-call streams, a whole answer,
  length, refusals, a client gone -> STOP, the engine gone -> the server exits; each check seen failing on a mutant),
  then on the real one: the installed ud-iq1_s pack (drafter and overlay) answered a thinking question, a tool call
  and the tool's result round trip, 29-37 tok/s drafted on short answers.
- **one context: 1M** (the operator's, 2026-11-04): both packs, osmos-serve's default, 1,048,576 tokens; rig's row a
  reply of 256K tokens (maxTokens, its thinking and answer: rig's thinkingBudget is Anthropic's alone), as much kept
  free (reserve). the state is ~0.9 KB a token as measured before (~0.85 GiB at 1M); what did not fit was the
  indexer's scores - a step's 2,048 tokens by every row they could see, F32: this model's ratio-1 layers make that
  every position, 8 GiB at 1M (measured: 1M took 8.35 GiB more than 64K). the step is now scored in chunks of
  tokens, at most FWD_SCORE_BYTES (512 MiB, the 64K size: one chunk there) - each token's scores, bits and picks its
  own, so no bit moves. tests/gpu/test_longctx_gpu: 1M 0.71 GiB over 64K (8.21 before), the last logits the same bits
  at both; seen failing before the change and on a chunk-offset mutant.
- **the budgets at 1M** (osmos-serve to READY with the packs' drafter, dspark-iq2xs; the edges to half a GiB): IQ1_S
  plain 107.5, drafted 102.5; IQ1_M plain 107.5, drafted 101.5 - the packs 1 GiB under: IQ1_S 106.5 / 101.5, IQ1_M
  106.5 / 100.5. (IQ1_M's drafted edge was first measured at 98 beside the original 7.97 GB drafter by mistake - the
  packs ship the 4.68 GB requant; that drafter is deleted now, its tests on dspark-iq2xs.)
- **decode at those budgets, 1M** (~/bench/1m-b, the quiet-box bench's method: three prompts, two interleaved rounds,
  every answer's hash the quiet-box run's): IQ1_S plain 18.8, drafted 37.5; IQ1_M plain 16.0, drafted 28.6 tok/s -
  against the 64K run at 110 / 103 GiB: 19.1 / 38.0 and 16.5 / 29.6 (-1.4 to -3.4%: the experts' smaller room, each
  GiB fewer misses read from the SSD).
- **published**: huggingface.co/mrsirg97-rgb/osmos-evo-1017 (public, MIT: DeepSeek's DSpark weights, maya's packing,
  ggml's quantizers) - the drafter and both picks profiles; packs.json pins its commit.

## the budget sized to the GPU, and Maya 1.1.53 (2026-11-04)

- **why**: a pack's fixed budgets were this box's limits less 1 GiB - another Halo (a BIOS split, a desktop holding
  GPU memory) starts short of them and is refused, and a budget picked for one setup (the bench's 100 GiB drafted)
  leaves memory idle in another. first shipped as a shrink-only correction (the free memory the pack was measured
  beside, less what is missing): it never fills a bigger room. replaced the same day, before the release.
- **the sizing** (osmos/run/fit.h, `--budget auto`, osmos-serve's and dbench's default way): fit_sized builds the
  engine whole - the forward, every lane's sequence, the drafter - at budget 0 (the forward's dense weights alone:
  forward_budget says how much), reads forward_room (the GPU's free memory less the expert pool's room, which the pool
  takes only as experts first stream), and builds again with that room but FIT_HEADROOM (1 GiB) given to experts,
  until the room is within FIT_SLACK (256 MiB) past the headroom, FIT_BUILDS (5) at most. the pool's room is twice a
  layer's streamed experts, so it shrinks as the budget grows: each step's room is an underestimate the next corrects,
  never past the fill. a first version took one step from the dense build and undersized by ~6 GiB (that build's pool
  is every expert's): caught by test_fit_gpu and by its plain run choosing 103.3 where 110 had run.
- **measured**: test_fit_gpu (UD-IQ1_M, 64K, 2,048-token steps): 8.00 -> 100.40 -> 106.24 GiB in three builds, 1.21
  GiB left. the bench's runs, the same budget both rounds: UD-IQ1_M plain 109.97, drafted 104.75, prompts 106.75;
  UD-IQ1_S 110.79 / 105.55 / 107.55 (dbench's 512-token decode steps take less than osmos-serve's 2,048).
  against the hand-picked budgets (110 / 100 / 106): plain the same, drafted +3.7% (UD-IQ1_M 28.6 -> 29.7) and +4.8%
  (UD-IQ1_S 36.5 -> 38.2), prompts +0-4% (~/bench/auto).
- **Maya 1.1.53** (`./maya.sh --bench`, its configs `maya-ud-{nodraft,dspark}.json` / `maya-s-*.json`: 64K, its own
  8 GB DSpark, `--threads on`; UD-IQ1_S packed by its tools/ds4/pack_gguf.py, its lane calibrated (`--calibrate
  --no-start`) and its expert profile warmed by a discarded run first - the warm-up 17.7, the next 19.2 tok/s), each
  engine in a fresh process, two interleaved rounds (~/bench/maya-1.1.53, ~/bench/maya-1.1.53-s): plain a tie (UD-IQ1_M
  16.5 / 16.5, UD-IQ1_S 19.2 / 19.3); drafted osmos 29.7 / 28.1 and 38.2 / 35.4; prompts 2K 298 / 237 and 330 / 261,
  8K 450 / 404 and 482 / 429. the router's guess on lifted osmos's 2K prompt 290-297 -> 321 (one run): not taken - off
  is the packs' setting and wins the agents below.
- **the headroom kept** (0.1.3): the first rig session left 0.61 GiB free, under the 1 GiB the sizing promises.
  osmos-serve's builds (now each a NOTE line): 0 -> 94.97 -> 101.16 GiB left 87.97 / 7.19 / 0.61 - the last step's
  6.19 GiB of budget took 6.58 of memory: a resident expert's units take ~6% past its bytes, and fit_sized assumed a
  step could only undershoot. the memory also moves in ~0.93 GiB steps here (100.85 GiB left 1.54, 101.08 left
  0.61): no budget lands in a 256 MiB window. fit_sized now aims each step by the cost measured between builds,
  keeps a bracket (the largest budget that left the headroom, the smallest that did not or was refused), stops in
  the window or where the bracket closes within FIT_SLACK, and never keeps a build short of the headroom when one
  that left it is known. on this box: 100.85 GiB, 1.54 GiB left, in six builds (~90 s to READY, ~40 before);
  test_fit models the pool's shrink, a 6% and a 12% overhead, refused steps and 0.9-1.0 GiB memory steps.
- **agents** (tools/serve/agent_bench.py under tools/power.py, the energy over the bench's own window: power.py's
  samples between agent_bench's printed epochs; ~/bench/agents, ~/bench/agents-auto), UD-IQ1_M, 4 x 8 seed 1, 131K:
  osmos (auto) 440.1 s, first token p50 / p95 5.75 / 58.3 s, turn done 44.1 / 100.1 s, 134.2 W, 7.38 J an answer
  token; osmos at the searched budget (100.5 GiB, the limit in that setup less 1) 453.7 s, 7.47 J; Maya 546.6 s,
  6.63 / 90.9 s, 55.9 / 124.9 s, 123.6 W, 8.44 J. the UD-IQ1_S agents runs were not run (UD-IQ1_M decided it).
- **a 131,072-token prompt** (tools/serve/drive.py, one GEN of ~/bench/osmos-prompts/p131072.txt: BOS, 130,879 tokens
  of this repo's DESIGN and README text, 192 of calib16k; ~/bench/long), UD-IQ1_M, one run each: osmos (1M context,
  auto, guess off) its first token at 258.9 s (506 tok/s), Maya (`--max-context 139264`, its drafter off) 273.5 s (479
  tok/s); the 8 tokens after the same (hash 0de05d1e3180f2b6). the lead narrows with length (2K +26%, 8K +11%, 128K
  +6%): at 128K the sparse attention over the whole context is most of a step's work, and the two are close there.

## 0.2.0: the forward's reads on libevt (design, 2026-11-05)

- **why**: osmos-serve's loop is libevt's, but the forward's own events are not: a finished read reaches the forward
  through a hand-built inbox (`Inbox`: a mutex, a condition, a growing array, a `broken` flag; hip_held.c), and the
  reads' order is fetchq's two FIFO rings (urgent, ahead) plus `hurry`, which moves a read ahead to the urgent ring
  once it is waited on. that is an event queue with priorities written a second time - one shape per concept says
  collapse it into the one the project already has. pool.h's contract already names the model ("the scheduler's
  consumer is its only caller (libevt: one consumer, so nothing locks)"); the code never followed.
- **the facts it rests on** (this file, "the fetcher" and decode's traces): 4 readers reach the SSD's 6.85 GB/s; a
  decode's reads are paced by the GPU, not the SSD (8 -> 32 readers, a 4x ring, 8x read-ahead: no change); a decode
  token's misses wait ~6 ms, behind the dependency that a layer's picks are known only at its routing. so the order
  of the reads cannot make one conversation's decode faster: what 0.2.0 can buy is the structure, and the order where
  reads do queue - a prompt's read-ahead (many layers' experts wanted at once) and several conversations.
- **what was wrong in the story told for it** (said in chat, checked here): "a decode's miss waits behind another
  conversation's prompt reads" - it does not: a miss is a pin, its read urgent, the ahead ring waits for it. what
  the FIFO does get wrong: within the ahead ring, a read wanted at layer L+6 that was asked first goes before one
  wanted at L+1 (a prompt's step asks for the next layer's experts; guesses and swaps ask too).
- **the decision**:
  1. **the forward's inbox is a libevt Engine.** a reader's `done` is `engine_add(context(read: unit, ok),
     PRIO_READ)`; the forward thread is the consumer: `fwd_drain` pumps without waiting until none is left,
     `fwd_wait_read` pumps one, waiting. the event runs pool_ready on the consumer - pool.h's contract as written.
     an add the engine refuses (no memory, stopped) fails the forward closed, as `broken` does now. `Inbox` goes.
  2. **the reads ordered by when they are needed.** every read carries `need` - the pool's clock (pool_want's
     `next`; a pin's is now), the order pool_release already evicts by - and readers take the read needed soonest:
     libevt's `cap` on the consumer, its slots the reads given to readers at once (`FETCH_INFLIGHT`: a measured
     depth, not a guess - see "to measure"), the rest waiting in cap's order by `before_fn(need, then asked)`,
     asked at each release so a read whose need moved up (a want then a pin: today's `hurry`) is simply first.
     `hurry` and the second ring go; a pin is a read with need = now. the readers stay fetchq's threads (several
     preads in flight are the SSD's speed) and keep its chunking (a lone read in FETCHQ_CHUNK pieces): they pread
     what the consumer gave them and post done; they no longer choose.
  3. **CPU-lane jobs as events under a cap** (the balanced-power work, next): the lane's jobs a second `cap`, its
     slots the threads that may take memory bandwidth beside the GPU - the policy in one place when it is built.
- **the rule engine (libtrdr's forward chaining), asked**: not in 0.2.0. the chain it would encode - a layer
  routed -> its non-resident picks read -> a read done -> the GPU released at its wait - is three calls on the
  consumer today, run ~240 times a token; facts, activations and a working set would add a heap event and an
  allocation to each, on the hot path, to say what three lines say. where rules would earn their place is a policy
  with many conditions that changes as it is tuned - the balanced-power CPU lane's (when to take a miss on the CPU:
  power profile, bandwidth, the GPU's queue, the read's need) is the candidate. revisit there, measured.
- **the bits**: no bit moves. which experts are resident or read when is already outside the arithmetic's contract
  (test_model_gpu: 1,983 swaps in a forward, the same fingerprint); the order of reads only changes when.
- **to measure, before it is kept** (a quiet box, `performance`, interleaved, two rounds, against 0.1.3):
  - plain and drafted decode, both quants (dbench auto): unchanged within the spread is the bar - it cannot gain.
  - prompts at 2K / 8K / 128K: the read-ahead's order is where it can gain.
  - agents (agent_bench.py, 4 x 8, UD-IQ1_M): first token and turn done p50 / p95, wall time, J an answer token.
  - FETCH_INFLIGHT: the reads given out at once, swept (readers x 1, x 2, x 4): the cost of the consumer choosing
    is that a read waits for the consumer's next pump to be given out - decode pumps at each layer's routing
    (~1.5 ms apart), a read takes ~1.8 ms; too few in flight leaves the SSD idle between pumps, too many is FIFO
    again.
- **the order** (each its tests first): the inbox as an Engine (test: reads finished on many threads all reach
  pool_ready on the consumer, none lost, an add refused fails closed - under tsan); then the reads by need (a
  fake fetcher's log: given a want at L+6 then one at L+1, L+1 is read first; a want then a pin of the same unit: it
  goes first, once; the depth never exceeds FETCH_INFLIGHT; destroy with reads waiting drops them, no done); then
  the GPU tests unchanged (fingerprints); then the benches above. 0.2.0 if they hold.
- **built** (the order above; libevt 2f456c3 first: `cap_submit` returns whether cap took the work - a refusal used
  to destroy it unseen, here a read lost and a forward waiting forever; merged, the submodule at c70c58c):
  - libevt's sources compiled into libosmos.a with this library's flags (every test and sanitizer run covers them;
    osmos-serve no longer links libevt.a beside it).
  - `mem/reads` (reads.h): a read record a unit, an Engine whose events are the reads' ends, a cap (`sooner`: need,
    then asked) - `fetch` submits, `need` moves a waiting read, `reads_pump` runs the ends (pool_ready, the slot
    freed, the next given out). the forward's `Inbox` and fwd_on_read deleted; fwd_drain / fwd_wait_read a pump each.
    (the cap: replaced below - the agents bench.)
  - `fetchq`: one queue in the order given (the urgent / ahead rings and `hurry` deleted); `fetchq_read(.., split)`.
  - pool.h's Fetcher: `need` (a pin's 0, a want's `next`) for `urgent`, `need(unit, need)` for `hurry` - also told
    when a reading unit's want is corrected.
  - tests: test_reads (the order with one out: by need, ties as asked, a need corrected while waiting; 2,000 reads on
    8 readers, at most `inflight` out (the cap's, since gone), each pinned once with its bytes; a read whose file is gone FAILED; destroy
    busy; the split below), test_fetchq / test_pool ported, test_pool_gpu on test_model.h's UD-IQ1_M (it had
    skipped since ds1017 left this box): a cold layer's 384 experts through reads.h at 6.70 GB/s. host tests, asan,
    tsan, and the GPU suite (27 ok; forward, lane - no ds1017 or oracle here - skip as before): the fingerprints the
    same.
- **the batch, found by the bench**: the first build read 2K prompts 4% slower (279-280 against 292). fetchq chose
  to split a read across its workers when its own queue was short - in 0.1.3 a prompt's read-ahead arrived at once
  and kept it long (each read whole); behind the cap it is always short, so every read was split, and chunks of many
  reads side by side defeat the file's readahead (this file, "the fetcher": a load 16 -> 22 s). the giver knows the
  batch: reads.h splits a read only when it is given out alone (none other out or waiting - a decode's miss) and
  gives the rest whole (`fetchq_read`'s `split`; reads_split counts them). test_reads' count; a never-split and an
  over-split mutant each fail it. 2K back to 292.
- **measured** (2026-11-05, `performance`, UD-IQ1_M, budget auto, 64K, dbench as the 1.1.53 comparison; one round
  of each in-flight depth - the operator's call: three depths agreeing - against 0.1.3 the same morning, ~/bench/v020):

  | | 0.1.3 | 0.2.0, 12 out | 24 out | 48 out |
  | --- | ---: | ---: | ---: | ---: |
  | plain, tok/s | 16.4 | 16.5 | 16.4 | 16.5 |
  | drafted | 29.8 | 30.0 | 29.9 | 29.9 |
  | prompt 2K | 292 | 292 | 290 | 286 |
  | prompt 8K | 444 | 447 | 454 | 461 |

  decode level (every 0.2.0 run at or above 0.1.3's, within the spread); the depth trades prompts - more out, a long
  prompt faster and a short one slower. FWD_READS_INFLIGHT 24 (2 x the readers): at or above 0.1.3 everywhere, 8K +2%.
  (superseded: the cap is gone - the agents bench, below.)
  not measured for it: the agents bench (several conversations - where the order by need should matter most).
- **the agents bench, and the order moved to where the readers take** (2026-11-05, ~/bench/agents-v020,
  ~/bench/agents-stats; agent_bench 4 x 8 seed 1, 131K, DSpark, budget auto 99.73 GiB both, guess off): the build
  above was level alone and 4% slower with four agents - 461.6 s against 0.1.3's 443.8 the same afternoon (first token
  p95 64.0 / 59.3 s). tried, not the cause: misses past the cap to the readers' front, split unless another miss out
  (461.5 s); the cap at 1,024 (460.2). counted instead (osmos-serve OSMOS_STATS=1, forward_waits at exit): 0.1.3 waited
  on 39,012 reads for 96.5 s (451.8 s that run). the cause, by elimination and by 0.1.3's code: a read ahead already
  given to the readers and then waited on kept its place behind every whole read queued before it - 0.1.3's hurry had
  moved it to the urgent queue. the fix is the design's own point taken one step further: the order lives where the
  readers take. fetchq's queue is libevt's heap (each job an event, its priority its read's need: 0 the top, then by
  need, ties the order given - a unit's jobs side by side), `fetchq_need` moves a unit's jobs not yet begun with libevt's
  new `queue_update` (32988ad: each event knows its heap index; O(log n); a randomized test against a model, two
  mutants fail it), and reads.h gives every read out at once - no cap. measured: 450.6 s, 37,319 reads waited for 85.4
  s (-11.5%), first token p50 / p95 6.00 / 57.6 s, turn done 44.6 / 102.5 s, 7.25 J an answer token (0.1.3: 7.30);
  a cold layer through reads.h 7.05 GB/s (test_pool_gpu); one conversation (dbench, as above): plain 16.4, drafted 29.8,
  prompts 300 / 466 at 2K / 8K (0.1.3: 292 / 444), the answers' hashes 0.1.3's. the waits fell 11 s and the wall only matched: with four
  conversations the GPU has other lanes' work while one waits - the reads are not the agents bench's only limit.
- **the rule engine, measured offline instead** (a routing trace: dbench's ROUTES_OUT, 1,604 steps of UD-IQ1_M; the
  analysis ~/bench/routes/analysis): a transition rule (expert e at L -> f at L+1, counted online) catches 7.7% of
  the next layer's misses at 6 reads a token-layer, 0.4% of its reads needed (~40 extra reads a token); L+2 / L+3
  the same; the router guess catches 56%, ~79% of its reads needed. 82% of decode's misses are an expert's first pick
  at its layer in the run - no count learned online predicts them; the repeats are what the resident set's heat
  already swaps in. not built. the lever it points at: the guess's reads earlier (1,316 guessed reads still waited
  1.8 s), and a better starting profile.

## 0.3.0: energy a token (design, 2026-11-05)

- **why**: the operator's frontier: the least energy a token, measured at the box's low-power profile (power-profiles-
  daemon's power-saver = platform_profile low-power, a stock profile) and kept only where performance does not
  regress. the box is for a dev or a small team - a few conversations at once - so the agents bench's J an answer
  token leads, single-stream plain / drafted / prompts beside it. less work is less work at any power: launches, bytes,
  copies and spins cut, not speed traded.
- **the tools**: dbench prints `window prompt|decode <start> <end> <tokens>` (epoch seconds) for each prompt read and
  answer; tools/bench/energy.py matches tools/power.py's samples (amdgpu PPT, the package: CPU + GPU + NPU, not the
  wall, not the board's DRAM) to them - J a token by window, and agent_bench's J an answer token (it reproduces the
  agents run above: 7.25 J). osmos-serve's OSMOS_STATS gives the read waits.
- **the baseline** (0.2.0, UD-IQ1_M, budget auto 99.73 GiB, ~/bench/v030/base-low):

  | | performance (earlier today) | low-power |
  | --- | ---: | ---: |
  | plain decode | 16.4 tok/s | 14.8 tok/s, 69.5 W, **4.80 J** a token |
  | drafted decode | 29.8 | 25.8, 69.3 W, **2.83 J** |
  | prompts 2K / 8K | 300 / 466 | 268 / 396, 70.2 W, **0.19 J** a prompt token |
  | agents (4 x 8) | 450.6 s, 128.8 W, 7.25 J | 527.2 s, 62.6 W, **4.12 J** an answer token |

  low-power costs 10-17% of the speed for half the power: decode waits on memory and the SSD, which the profile does
  not slow (the agents' read waits 85.4 s / 86.0 s at both), and the last tenth of a clock is the dearest in watts.
  for scale (the operator's measures, not this file's): a 2x5090 desktop drafts this model at 90 tok/s at 500-600 W
  (cards, board, CPU package): 5.6-6.7 J a token.
- **where the work is** (the trace of 2026-11-04 above: 55.5 ms a token, 8.5 GB read; the dense matrices 30 ms at ~92%
  of the bandwidth) **and what the CPU does** (osmos-serve in the agents bench, top -H over 2 s): the forward's thread
  85%, the 12 readers ~48% each - 6-7 cores on an APU whose CPU and GPU share one package limit.
- **the targets, ranked by joules**:
  1. **the readers copy every expert twice**: a buffered pread lands it in the page cache (the SSD's DMA), then the CPU
     copies it into the slot - 48% of a core a reader, and the bytes through DRAM three times. O_DIRECT reads straight
     into the slot. the facts: btrfs (sector 4 KiB, no compression on this mount), the NVMe's logical block 512 B; no
     expert slice of UD-IQ1_M starts on 4 KiB (0 of 46,080), every slice 2 KiB past a multiple of it - a direct read
     takes the aligned range around a slice (at most 4 KiB before and after: ~24 KiB an expert, ~0.3% of the pool) and
     the kernels read the slice inside it. the page cache is no longer a second tier for streamed experts (1-4 GB of it
     under load: above). open: whether O_DIRECT can DMA into hip_slots' memory (a driver mapping get_user_pages may
     refuse) - else page-aligned anonymous memory registered with HIP (the GPU read registered host memory at 237.5
     GB/s: above). a prototype first (scratch odirect.hip).
  2. **the forward's thread spins on the GPU**: the routing event (40 a step) is hipEventDisableTiming only, and the
     step's end, the greedy pick and the drafter's rounds are hipDeviceSynchronize / hipStreamSynchronize - all
     busy-waits. one shape: fwd_wait - a blocking-sync event recorded and slept on - for every one, the routing event
     created hipEventBlockingSync. the price to measure: a wake's latency (~10-50 us?) x ~45 a step.
  - **target 1, measured, then designed** (scratch odirect.hip: 256 MiB of a UD-IQ1_M shard, cold, 8 MiB 4 KiB-aligned
    reads, one thread): buffered pread into hip_slots memory 3.63 GB/s at 76% of the reading thread's wall in CPU;
    O_DIRECT into hip_slots memory 4.74 GB/s, 61% (the kernel still copies: a driver mapping, not plain pinnable
    pages); **O_DIRECT into page-aligned anonymous memory registered with HIP 6.14 GB/s, 13%**; buffered into it 4.75,
    61%. every read's bytes the file's. the GPU reads registered host memory at its own memory's speed (237.5 against
    239.7 GB/s: "in place", above). so the win is both together:
    - **the pool's memory a registered arena** (mem/host_slots: one mmap'd, page-aligned anonymous arena, hipHostRegister
      once at the forward's creation, a Slots over it - first fit on 4 KiB, freed back in place). only the pool: its
      room is two layers' streamed experts (stream_budget, ~2 GB); the resident set stays where it is (read once).
    - **reads O_DIRECT** (the forward opens the model's files a second time O_DIRECT for the pool's units; a file that
      refuses it refused loudly, no silent buffered fallback). the pool lays a part out at its slice's 4 KiB-aligned
      range (pool_create's read alignment: the range begins at the slice's offset rounded down, ends rounded up; the
      slice's place inside it is the part's pointer) - ~24 KiB an expert of slack.
    - **the sizing's footprint the same**: the arena is counted against the budget as the pool's room was (fit sized
      to the same RAM: the operator's ~8 GB kept free), though registered memory is outside ttm's pages_limit.
    - streaming no longer goes through the page cache, so the page-cache cliff (decode collapsing past ~106 GiB of
      experts: above) is not the streamed experts' any more - a larger resident budget is a later question, measured.
      Engram's rows stay buffered (small random reads the page cache helps).
    - the bits: the same bytes in a different place - every kernel reads the slice through the part's pointer; the
      fingerprint tests and the answers' hashes the check.
    - the tests first: test_pool (a read alignment: every read range aligned, its slice inside it at the part's
      pointer, the slack bounded); test_host_slots (alloc / free / reuse in the arena, its bounds, a registration
      refused failing closed); test_pool_gpu (the cold layer through O_DIRECT into the arena: the GPU's sums the
      file's, GB/s); then the benches (the readers' CPU, J a token).
  3. drafting for joules: a drafted token is 41% less energy than a plain one (2.83 / 4.80 J) - its length and
     thresholds tuned on J, not tok/s.
  4. launches: the hyper-connections (~2.4 ms a token), the small rest (~2.5), the gaps (~3.0).
  5. the F32 router (0.31 GB a token; BF16 halves it) - it moves bits once: the operator's call, not a default.
- **the rule for keeping a change**: J a token down at low-power, nothing slower at performance, the answers' hashes
  unchanged (1-4 move no bit); each measured on single-stream and the agents bench.
- **target 2, measured** (~/bench/v030/sleep-low, split-low; test_sleep_gpu, test_wait_gpu): an event's own
  hipEventBlockingSync is not honoured on ROCm 7.15 - a 300 ms wait spun 300 ms of a core; the device's
  hipDeviceScheduleBlockingSync is (7 ms), set before the process's first kernel (after one it does not take:
  gpu_sleep_on_waits, the top of osmos-serve's and dbench's main). sleeping on every wait lost: plain 14.7 -> 13.2
  tok/s, 4.78 -> 5.36 J; drafted 25.6 -> 23.9, 2.83 -> 3.04 J (low-power) - a wake ~0.2 ms, 40 a step, and the package
  at its cap gave the GPU the CPU's watts anyway. kept as a split: a decode-sized step (its routing pairs fewer than a
  layer's experts: the lane's small step) spins, a prompt's step sleeps (fwd_wait, hip_wait.c; test_wait_gpu: a 2K
  step at 17% of the thread's core, 32 decode steps at 86%; sleeping always, 23% and the decode 18% slower - the
  test's mutant). its agents run was stopped for target 1; it is in target 1's numbers below.
- **target 1, built and measured** (mem/arena, mem/host_slots, pool_create's read_align, the forward's O_DIRECT
  twins; test_arena, test_pool's read alignment and arena room, test_pool_gpu's direct path: a cold layer's 384 experts
  O_DIRECT into registered pages 7.06 GB/s, every part's GPU sum the file's). the pool, refused by an arena in pieces
  with the budget's room made, now frees more (spares, then the furthest) until it fits or nothing can go - the arena
  test's invariant (a pin refused only with no room it could have made) fails the single try. the arena: the pool's
  room and two of its largest unit. at low-power, against 0.2.0 the same day (~/bench/v030/direct-low, base-low),
  with target 2's split:

  | low-power | 0.2.0 | 0.3.0 (targets 1 + 2) |
  | --- | ---: | ---: |
  | plain | 14.8 tok/s, 4.80 J | **15.9, 4.45 J** (-7%) |
  | drafted | 25.8, 2.83 J | **29.3, 2.50 J** (-12%) |
  | agents (4 x 8) | 527.2 s, 62.6 W, 4.12 J | **479.0 s, 63.1 W, 3.78 J** (-8%); first token p95 76.2 -> 62.6 s |

  the answers' hashes 0.2.0's. at the package's cap the readers' watts (12 copying at ~48% of a core) go to the GPU:
  the same power, more tokens. drafted at low-power now 0.2.0's drafted at performance (29.8) at ~71 W against ~125.
  the agents' forward waited on more reads (44,058 against 37,598) for about the same time (89.3 against 86.0 s):
  direct reads take no page-cache hits - a RAM tier for streamed experts is the lever that points at.
- **and at performance** (the release's check: nothing slower; ~/bench/v030/direct-perf): plain 16.4 -> **17.1**
  tok/s (7.71 J), drafted 29.8 -> **32.0** (4.10 J), agents 450.6 -> **433.2 s**, 7.25 -> 6.98 J an answer token
  (first token p95 57.6 -> 51.8 s); the answers' hashes 0.2.0's. prompts within the bench's spread: 0.2.0 measured
  300 / 466 in the morning and 281 / 458 beside 0.3.0's 296 / 443 (2K / 8K), 0.31 -> 0.29 J a prompt token. the
  agents' read waits 85.4 -> 106.5 s with the wall shorter: the page cache's hits again. low-power against performance
  on 0.3.0: plain -7% speed for -42% J, drafted -8% for -39%, agents +11% time for -46%. from here every change is
  developed at low-power and checked at performance once a release (the operator's).
- **against Maya 1.1.55** (2026-11-05 afternoon, `performance`, ~/bench/h2h-055: Maya pulled, rebuilt, its lane
  calibrated on both packs - the engine's own threads, as 1.1.53's - its expert profile warmed by a discarded run, two
  interleaved rounds a pack; osmos 0.3.0's UD-IQ1_S interleaved beside it, its UD-IQ1_M the direct-perf runs above):
  UD-IQ1_M prompts 296 / 443 against 243 / 406 (+22% / +9%), drafted 32.0 against 27.5 (Maya's rounds 25.8 and 29.1:
  +16%, +10% against its better), plain 17.1 / 17.0; UD-IQ1_S prompts 326 / 473 against 259 / 431, drafted 37.8 /
  34.9 (+8%), plain 19.3 / 19.2. agents (UD-IQ1_M): osmos 433.2 s, first token p50 / p95 5.87 / 51.8 s, turn done
  42.96 / 97.8 s, 128.9 W, 6.98 J an answer token; Maya 542.4 s, 6.32 / 81.3, 55.3 / 121.9, 126.9 W, 8.74 J - 20% less
  time and energy. Maya's 1.1.54 / 1.1.55 prompt changes moved it a few percent on these GGUFs (1.1.53: 237 / 404);
  its plain decode rose 16.5 -> 17.0, as osmos's 16.4 -> 17.1.

## 0.3.x: drafting for joules (2026-11-05)

- **why**: a drafted token is ~44% less energy than a plain one (0.3.0, low-power: 2.50 against 4.45 J), and the
  drafter's thresholds were maya's (0.5 each confidence, 0.5 their product), tuned on no one's energy. every draft is
  verified, so a threshold moves cost, never bits - checked: the answers' hashes the same at every value below.
- **the sweep** (dbench DRAFT_CONF = DRAFT_CUM, UD-IQ1_M, drafted, low-power, one run each, ~/bench/v030/draft-sweep;
  a first pass discarded - two sweeps ran at once, then a stale binary ignored the settings; the script now checks the
  log shows the settings it asked for):

  | floor | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | tok/s | 29.5 | **30.7** | 30.4 | 30.1 | 29.9 | 29.4 | 29.0 | 27.6 | 26.3 |
  | J a token | 2.49 | **2.35** | 2.38 | 2.41 | 2.45 | 2.49 | 2.55 | 2.66 | 2.79 |

  lower floors let more drafts through to a verify pass that runs anyway (the third question: 262 drafted at 0.1, 256 at
  0.2, the same 207 accepted - the extra rows ride along); at 0.0 the drafter loses its early stop (a row's head and
  markov step run only once its confidence passes) and does every row's work for 3 more drafts and none more kept.
- **confirmed** (~/bench/v030/draft-confirm, interleaved 0.5 / 0.1 / 0.1 / 0.5): 0.5 29.4 / 29.4 tok/s, 2.51 / 2.51 J;
  0.1 30.6 / 30.7, 2.35 / 2.34 J - -6.6% energy, +4.3% speed. the agents bench with 0.1: 483.0 s, 3.84 J an answer
  token, first token p95 62.9 s, turn done p95 103.2 s (0.3.0: 479.0, 3.78, 62.6, 112.7) - level: 99K of its tokens
  are read, 8K written. kept: DRAFTER_CONF_MIN / DRAFTER_CUM_MIN 0.1 (drafter.h), one setting for osmos-serve's single
  and parallel paths and dbench (three copies of 0.5 before).
- not swept to the end: a cap on a round's drafts (DRAFT_MAX 2-4) - the floors' trend says more drafts through is
  cheaper, the operator's call to stop there.

## 0.3.2: the low-power work (2026-11-05)

each change measured at the low-power profile (UD-IQ1_M; dbench drafted and plain, then the agents bench), kept in its
own commit when it is no worse and no bit moves.

- **the experts' registers** (kernels/q8.h kFullWaves): the drafted decode's kernel trace (rocprofv3, 33.2 ms a token,
  the GPU busy 85%) put the experts at 11.7 ms; q8bench (tools/bench, 19 of a layer's experts, the forward's launch)
  showed the int8 kernels near the memory's rate at 1-4 tokens and a cliff at 5-6 - 98-107 registers a lane, 12 waves a
  SIMD not 16: gate+up 1,143 us at 5 tokens against 704 at 4. asking for 16 waves (amdgpu_waves_per_eu) gives 5 tokens
  798 us, 6 tokens 1,004 (from 1,543), down at 6 563 (from 909); 1-4 the same. in the forward, level - a drafted step's
  experts are mostly 1-2 tokens each: drafted 30.9 / 30.9 tok/s, 2.35 J against 30.9 / 30.8, 2.35 / 2.36 (interleaved,
  ~/bench/v032/waves); plain 15.9 against 15.7; agents 481.3 s, 3.79 J an answer token, first token p95 62.0 s, turn
  done p95 103.6 s (0.3.1: 483.0, 3.84, 62.9, 103.2). kept: no loss, the answers' hashes the same, and a verify block
  of 5-6 tokens on an expert no longer falls off the cliff.
- **the drafter's greedy pick on the GPU** (hip_drafter.c): a draft row's 129,280 logits were copied to pinned memory
  (517 KB) and scanned on the CPU; now ops_argmax (its ties the scan's: the lowest index) and 4 bytes back - the same
  drafts (279 / 282 / 262 drafted, 199 / 173 / 207 accepted, both builds), drafting 854 / 1,180 / 757 ms against 831 /
  1,179 / 750. end to end below the bench's noise: drafted 30.6 / 30.5 tok/s against 30.7 / 30.7, 2.36 J against 2.35
  (~/bench/v032/argmax; plain - no drafter - 15.6 against 15.8 in the same run: the noise's size). kept: less work, a
  row with a NaN now ends the drafts (the scan picked from it).
- **measured, not kept**:
  - the resident set's swaps off (FWD_SWAPS_A_LAYER 0: their copies 0.7 ms a drafted token, ~2.5% of the GPU's busy
    time): drafted 19.0 / 18.9 tok/s, 3.32 / 3.33 J against 30.9 / 30.7, 2.32 / 2.34; plain 10.4 against 15.8, 6.47 J
    against 4.47 (~/bench/v032/noswap). the copies are what keeps the text's experts in GPU memory: kept as they are.
  - hc_post_dot reading fn once for a block of tokens (staged in LDS) instead of a block a token: the same bits, and
    slower - drafted 30.6 / 30.6 against 30.9 / 30.8, plain 15.5 against 15.8 (~/bench/v032/hcdot). alone, at the
    decode's shapes (40 layers' fn in turn, so none stays cached): 5 tokens 30.0 us a block a token, 30.7 - 47.3 the
    staged ones (1 - 8 tokens a block). a token's blocks after the first read fn from the cache, not memory: the
    re-reads cost little, and the staging's barriers and serial sums more.
  - the drafter's head through mm (rms_x16, then mm_run; the trace had the target's head by mm at 195 GB/s, the
    drafter's by mv at 151): the same drafts, drafting 2-4% slower - 846-859 / 1,185-1,188 / 748-764 ms against
    833-850 / 1,142-1,178 / 711-720; drafted 30.6 / 30.6 tok/s against 30.8 / 30.7 (~/bench/v032/head-mm). at 5 rows
    the x16's conversion and the flag's round trip cost what the faster read saves.

### 0.3.2 at performance (2026-11-05, evening)

the release measured as 0.3.0 was (~/bench/v032/perf, the script beside it: one run each, power.py around each):

| | 0.3.0 (afternoon) | 0.3.2 |
| --- | ---: | ---: |
| UD-IQ1_M drafted (q1 / q2 / q3) | 33.1 / 24.8 / 38.0, 32.0 tok/s, 4.10 J | 35.1 / 27.7 / 38.1, **33.6**, **3.87 J** |
| UD-IQ1_M plain | 17.1, 7.71 J | 17.1, 7.72 J |
| UD-IQ1_M prompts 2K / 8K | 296 / 443 | 293 / 441, 0.30 J a prompt token |
| UD-IQ1_S drafted | 37.9 / 27.5 / 47.9, 37.8 | 41.0 / 31.4 / 49.9, **40.8**, 3.35 J |
| UD-IQ1_S plain | 19.3 | 19.3, 7.15 J |
| UD-IQ1_S prompts 2K / 8K | 323-328 / 471-474 | 331 / 473 |

the drafted gain mostly 0.3.1's floors (the hardest question most: 24.8 -> 27.7, 27.5 -> 31.4 - more of its drafts
reach the verify), carried from low-power to performance. the answers' hashes 0.3.0's, both packs, drafted and plain.
the agents bench: 0.3.2 442.5 / 448.7 s, 7.07 / 7.07 J an answer token (first token p95 58.1 / 51.3 s, turn done p95
92.9 / 95.1); 0.3.0 rebuilt and run beside them, 444.6 s, 7.01 J (52.9, 99.4) - level. 0.3.0's afternoon run, 433.2 s,
was the box then (~2.5% faster; the profile switched to low-power and back between): a comparison across a profile
switch wants its baseline rerun.

## open questions

1. the NPU's memlock limit (8 MB): only when an NPU experiment is on the plan.

## the protocol (carried from #356 / #557 / #587)

- every claim is a measured number on this box, with the command that reproduces it, and the box's state with it:
  today the second unit, **all 16 cores online**. the first unit ran with core 1 (cpus 1, 17) offline for idle MCE
  errors; its numbers are that box's, and are re-measured here on the same commands before any comparison.
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
