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
  since the day's start as the commit's own. a baseline is built by its library files (`make lib/libsubstrate.a
  lib/libsubstrate_hip.a`), never `make lib`.
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
tokens (`~/substrate-oracle`, `dumpcmp.py layers`) part from layer 0 (worst 1.1e-8 at position 8) and grow - layer
10 median 1.3% (worst 18%), layers 30-38 median 4-7% - to logits KL 2.5e-3 mean, 1.4e-2 max, the argmax the same at
32 of 33 positions (the 4 greedy tokens the same). substrate diverges from maya here:

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
- **measured, the first 8 positions of 32 real tokens** (`~/substrate-oracle`: `ref_gguf.py --dev cpu`, 334 s for 8
  tokens): KL(reference || maya) 0.0002-0.018 a position, maya against itself 0-0.0057; the argmax the reference's at
  7 of 8 (position 6: 67 against 75). substrate's numerics target: bit for bit run to run, and at least as close to
  the reference as maya is (int8 activations and FP16 prompt GEMMs put maya a few times its own spread away).
- fixed to read this model, besides the Engram rows (each `# substrate:` in `oracle/ref_gguf.py`): the embedding is
  Q5_K, not Q8_0; the Engram gate is `q_weight` and `k_weight`, not folded. maya's `ref_gguf.py` would have read the
  Engram rows wrong silently and refused the other two; all three are fixes to offer maya.
- **the bar and a sanity check: maya-ds4's dumps** - `DS4_DUMP_L` (every layer's residual streams), `DS4_DUMP_R`
  (routing), `DS4_DUMP_Y` (expert sets), `--dump-logits`. substrate writes the same records (`dumpcmp.py` unchanged);
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
