# maya upstream: patches and findings

what building libosmos found in [Project Maya](https://github.com/mw00/project-maya), measured on the Ryzen AI
Halo (Ryzen AI MAX+ 395, Radeon 8060S gfx1151, 128 GB, ROCm 7.15 / hipBLASLt 1.4.1). each: what, the evidence, the
fix, its status. none is filed unless it says so.

## landed (mrsirg97-rgb's PRs, committed by maya's maintainer and credited in its CHANGELOG)

| # | what | measured | in maya |
| --- | --- | --- | --- |
| [#628](https://github.com/mw00/project-maya/pull/628) | the DFlash2 drafter on one Ryzen AI Max APU | GLM decode 27.4 -> 29.8 tok/s (+9%) | 23586f7 |
| [#631](https://github.com/mw00/project-maya/pull/631) | gfx1151's GLM hipBLASLt table for hipBLASLt 1.4.1 | GLM prefill 159 -> 306 tok/s at 8K (x1.9) | 955c4dc, then 23d24b3 (every table kept, per shape the first the build accepts) |
| [#634](https://github.com/mw00/project-maya/pull/634) | the fused GLM prompt experts at 3 blocks a multiprocessor on gfx1151, bit for bit | the kernel -13 to -16%, GLM prefill +3.5% | 29024d0 |
| [#643](https://github.com/mw00/project-maya/pull/643) | DeepSeek's prompt experts through the fused int8 experts | DeepSeek prefill 234 -> 257 (2K), 430 -> 468 (8K) | acb5828 |
| [#745](https://github.com/mw00/project-maya/pull/745) | DeepSeek's router weights kept BF16 on AMD | decode +3.5% (17.0 -> 17.6 tok/s), the same logits | 3ab0908 |

(checked 2026-11-04 against v1.1.53: every one's change in the tree; GitHub shows them closed, not merged.)

## to file

### 1. `tools/ds4/ref_gguf.py` reads an IQ4_NL Engram table as MXFP8 - silently wrong

- **what**: `RefGguf.engram` sizes a row as Q6_K (210 B a 256-value block) or else MXFP8 (33 B a 32-value block),
  and decodes every non-Q6_K table as MXFP8. the GSQ-RCO IQ2_XXS GGUF's Engram tables
  (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`) are IQ4_NL (18 B a 32-value block, 144 B a 256-value row).
- **effect**: rows read at the wrong stride and decoded as FP8 - no error, wrong numbers in the oracle that checks
  maya-ds4's GGUF path.
- **fix**: the row's bytes from its type, decoded by gguf-py for any type but MXFP8; refuse a type of unknown size.
  done in `oracle/ref_gguf.py` (`# osmos:`).

### 2. `tools/ds4/ref_gguf.py` reads every non-BF16 embedding as Q8_0

- **what**: `forward` reshapes `embed.weight` into 34-byte Q8_0 blocks; this GGUF's embedding is Q5_K (3,520 B a
  row of 5,120).
- **effect**: `ValueError: cannot reshape array of size 455065600 into shape (5440)` - it refuses (loudly, at
  least).
- **fix**: the row size by type, decoded by gguf-py. done in `oracle/ref_gguf.py`.

### 3. `tools/ds4/ref_gguf.py` needs a folded Engram gate

- **what**: it reads `layers.N.engram.gate.weight` (q_weight * k_weight folded, as maya's own packs store it); this
  GGUF keeps `q_weight` and `k_weight` apart (`KeyError: 'layers.1.engram.gate.weight'`). `ref.py` multiplies them
  itself, and maya-ds4 folds them at load (`ds4_model.cu`, "a GGUF: q_weight * k_weight, folded already").
- **fix**: fold them when `gate.weight` is absent, as `ref.py` does. done in `oracle/ref_gguf.py`.

### 4. `ref_gguf.py`'s gguf-py error hides the cause

- **what**: any `ImportError` while importing gguf-py prints "no gguf-py with a dequantiser for it" - here the cause
  was gguf-py's own `import yaml` (`metadata.py`), with pyyaml not installed.
- **fix**: print the ImportError's message too.

### 5. maya-ds4's prompt path is not run-to-run deterministic on this box (a finding; the cause not found)

- **what**: two runs of maya-ds4 (v1.1.36 + #643 + #587) on the same 32 real token ids, one process each, the same
  config: the `DS4_DUMP_L` records part from layer 0 (worst 1.1e-8 at position 8), layer 10 median 1.3% (worst 18%),
  layers 30-38 median 4-7%; the logits KL 2.5e-3 mean, 1.4e-2 max, the argmax the same at 32 of 33 positions (the 4
  greedy tokens the same). `~/osmos-oracle`, `dumpcmp.py layers L_a.bin L_b.bin`.
- **against the F32 reference** (`ref_gguf.py`, the first 8 positions): KL 0.0002-0.018 a position, maya's own spread
  0-0.0057; the argmax the reference's at 7 of 8.
- **candidates** (not checked): the CPU lane's multi-threaded sums (it takes a few experts a run), a GPU reduction
  whose order varies.
- **why it matters**: maya's own exactness checks compare greedy tokens; at a near-tie the run decides.

### 6. `./maya.sh --bench` ignores the config's `exe`

- **what**: `bench_exe(cfg)` returns the build's own binary (`exe_for_target`) whatever the config's `exe` says, so
  `--bench --config <a config naming another build>` measures `build-hip/` silently.
- **effect**: an A/B of two builds through `--bench` measures one build twice (it happened here: four "old / new"
  runs, all the old engine).
- **fix**: use the config's `exe` when it names an existing file, or say which binary it runs.

### 7. `MAYA_DS4_EXPERT_MMQ=1` stops the engine where VRAM is full

- **what**: with ds1017 filling the pool (~1 GB free beside the reserve), the opt-in MMQ path's workspace
  (`cudaMalloc 32 MiB on CUDA0 failed`) ends the engine at start.
- **fix**: fall back to the existing path (the docs say "allocation failure keep[s] the FP32 tiled kernel"), or
  reserve the workspace before the pool is sized.

### 8. `--parallel` (#557 / #587) measured on AMD

- **what**: both PRs list AMD as untested. on this box, the bar (v1.1.36 + #643 + #587), `agent_bench.py` 4x8 seed 1:
  32 of 32 turns answered, 28 of 28 later turns continued in place, two runs - first token 13.02 / 12.84 s mean,
  p95 40.25 / 37.43 s; turn done 60.37 / 60.11 s mean; wall 514.8 / 512.7 s.
- **use**: data for #557 / #587's descriptions.

### 9. `tools/ds4/parallel_test.py`'s `Util` reads `nvidia-smi` only

- **what**: on AMD the GPU utilisation series stays empty.
- **fix**: `rocm-smi --showuse` (or amdgpu's `gpu_busy_percent` in sysfs) on HIP.

### 10. `tools/test_setup_qwen.py` fails on Node 20

- **what**: `test_the_web_apps_names` imports `serve/web/lib.js`, whose `navigator.platform` is not defined in Node
  20 (`ReferenceError: navigator is not defined`); it fails the same way on `main`.
- **fix**: guard the `navigator` read, or set the test's Node version.

## to look at: libosmos's findings of 2026-11-01 (each checked against maya's code before a PR)

### 11. device-to-device copies of memory the HIP runtime did not allocate go to the copy engine

- **found**: libosmos's resident set swaps an expert by copying it between two HSA allocations
  (`hsa_amd_memory_pool_allocate`); `hipMemcpyAsync(..., hipMemcpyDeviceToDevice)` took them for host memory and
  copied on the copy engine, the GPU idle ~0.45 ms behind each (4.6 a token: ~1.9 ms). a copy kernel (16 bytes a
  thread) took 0.63 ms of kernels a token for the 1.9 of gaps.
- **check in maya**: any `cudaMemcpyAsync` on the decode stream between buffers maya allocates itself (the VRAM
  expert slots, the swap path's `cache_loop`), and whether its swaps' copies stall the stream.

### 12. the hyper-connections' small kernels, fused

- **found**: post then the next mix's dot (one block a part: the part's values computed, kept in LDS, dotted - the
  same sums), and the mixes with the pre-norm (independent: the norm's pre is the mix before) - 3 launches a layer
  fewer; alone 8.9 -> 5.4 us and 9.5 -> 6.9 us a token's pair; in decode ~0.3 ms a token (~0.5%), the same bits.
- **check in maya**: its trace shows `hc_mix_kernel`, `hc_pre_norm_kernel`, `hc_post_kernel`, `hc_split_kernel` as
  separate launches (80 a token each) - the same fusions may apply.

### 13. the resident set's swaps: choose from every streamed expert in GPU memory

- **found**: libosmos chose a swap only from the experts held in the step; choosing from every streamed expert
  the pool still holds (held or not) +1.5-3.4% decode.
- **check in maya**: maya swaps CPU-hit experts in from a background thread with a spare slot a layer
  (`cache_tick` / `swap_tick`); whether experts streamed to the GPU (the PCIe lane, off on this box) are candidates.

### 14. negative results worth knowing on Strix Halo (no PR: data for the next person)

- the CPU and the GPU share the memory's bandwidth: an AVX-512 int8 expert kernel running 12 threads beside decode
  costs the GPU 16% and runs 36% slower itself; a CPU lane for misses only measured even with the GPU's waits here
  (15 cores).
- a GiB more GPU memory is a GiB less page cache (one RAM): past ~106 GiB of experts (113 GiB GTT) decode
  collapses; warming the page cache at start does not survive to decode (~9 GB left).
- a captured graph saves ~0.15 us of the ~2 us between two dependent kernels (gfx1151).

## to look at: libosmos's findings of 2026-11-04

### 15. DSpark's experts at IQ2_XS: the drafter 7.97 -> 4.68 GB, the same acceptance

- **found**: maya's DSpark pack keeps the drafter's 3 x 128 experts MXFP4; requantized to IQ2_XS with ggml's own
  quantizers (libosmos's `tools/requant/requant_dspark`, the dense matrices Q8_0 and the heads BF16 as they were) the
  drafter is 4.68 GB, its acceptance on the bench's three questions the same (DESIGN.md). on one 128 GB Halo the
  3.3 GB goes to the model's experts: UD-IQ1_M drafted streams fewer from the SSD.
- **offer**: the requantized drafter is public (huggingface.co/mrsirg97-rgb/osmos-evo-1017, MIT); or the tool, for a
  `tools/ds4/drafter_pack.py --experts IQ2_XS`.
- **check in maya**: its drafter's experts' kernel for IQ2_XS (the trunk's IQ2_XS path) and the acceptance on its own
  calibration text before a default changes.

### 16. a 1M context: the indexer's per-token scores, not the keys, are what grows

- **found**: libosmos's 1M context cost 8.35 GiB over 64K, all but ~0.85 of it one scratch: the indexer's scores for a
  step's 2,048 tokens by every row they could see (F32) - this model's ratio-1 layers make that every position. scored
  in chunks of tokens (each token's scores, top-k and bits its own: no bit moves) it is 0.71 GiB.
- **check in maya**: v1.1.53's `b4f78bb` sizes the indexer's slab rows by the run's keys rather than max_context
  (opt-in, `DS4_AI_RUN`); whether its default path holds a [chunk x max_context] score buffer at a long context.

### 17. `--bench` on a new DeepSeek pack measures a cold expert profile

- **found**: a pack without its own `expert-profile.bin` starts from `data/expert-profile-ds4.bin` and learns this
  PC's routing as it runs, so a first `--bench` on it measures a profile still learning: UD-IQ1_S on this box, a
  fresh pack's first `--bench` decoded 17.7 tok/s, the next 19.2 (+8.5%), the one after 19.4.
- **fix**: say so in the bench's output (or the docs), or warm the profile in `--bench` before it measures.

