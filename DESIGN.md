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
  / 28.3 tok/s (mean 26.4), no drafting 16.7 / 16.6 / 16.8 (16.7); substrate DSpark 16.2-16.3 / 14.4-14.9 / 18.6-19.1
  (16.6), plain 9.4-9.6 / 10.1-10.3 / 10.0-10.2 (9.9) - the drafted answers the plain ones token for token. ~60% of
  maya both ways; DSpark 1.7x here (maya 1.6x).
- **a correction**: the first numbers here (256 tokens after a 512-token prompt: 21.8 / 26.0 tok/s, "1.9-2.2x the best
  plain") ran the drafted decode after the plain one on the same prompt in one process - the resident set had
  already followed that very answer (hip_adapt.c), its misses gone. the same on maya's prompts: 26.3 / 20.4 / 28.4
  tok/s, within ~5% of maya's DSpark. so what parts substrate from maya on a fresh prompt is the experts it reads from
  the files (maya holds ~104 GB of experts in GPU memory and computes the rest on the CPU from RAM; substrate's GPU
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
    copy of them twice); substrate has no CPU lane (determinism's rule 1).

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

- **the lane's experts over int8 inputs** (2026-11-01; kernels/q8.h, include/substrate/gpu/q8.h): a kernel-by-kernel
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

- **the CPU's half of the int8 contract** (2026-11-01; src/cpu/q8.c, include/substrate/cpu/q8.h): cpu_q8_round
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
  picks): forward_create_ranked's profile (`substrate-picks.bin`, the 16K-token real-text count, nothing of the
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
   `# substrate:` fixes need the UD spellings: embedding Q8_0, the gate folded, the tables sliced from
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

1. schema: the type table (`include/substrate/model/qtype.h`): each type once, its id, a block's values and bytes, the
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
5. the oracle and maya: the logits within maya's spread, then the numbers.

### the order (before 2026-11-03, kept for the record)

1. the type table and roles by capability: the pure refactor, fingerprints equal on the current GSQ-RCO pack.
2. schema: the alias table; the Engram slices of one tensor; the folded gate; the `exp_probs_b_vl` role.
3. tests that fail first: the Coder's geometry read equal to a hand-written one; each alias conflict refuses; a role
   given a type no path takes refuses; a table type with a path missing its case fails; the slices' rows summed
   against the tensor's.
4. the Coder loaded and generating; the fingerprints across batching, budget and CPU / GPU on it; the oracle.
5. UD-IQ1_M: IQ1_S, IQ2_S and the rest, as the UD section's order. its quality measurement (that order's step 1)
   waits on none of this.

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
