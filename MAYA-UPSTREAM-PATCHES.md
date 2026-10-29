# maya upstream: patches and findings

what building libsubstrate found in [Project Maya](https://github.com/mw00/project-maya), measured on the Ryzen AI
Halo (Ryzen AI MAX+ 395, Radeon 8060S gfx1151, 128 GB, ROCm 7.15 / hipBLASLt 1.4.1). each: what, the evidence, the
fix, its status. none is filed unless it says so.

## open as PRs (mrsirg97-rgb)

| # | what | measured | status |
| --- | --- | --- | --- |
| [#628](https://github.com/mw00/project-maya/pull/628) | the DFlash2 drafter on one Ryzen AI Max APU | GLM decode 27.4 -> 29.8 tok/s (+9%), `--bench` three runs each; same tokens with `MAYA_GLM_NO_PREFILL=1` | open, CI pass |
| [#631](https://github.com/mw00/project-maya/pull/631) | `gfx1151-glm-hipblaslt-100401.txt` retuned: the shipped table's one solution (2390) is rejected for every GLM shape on hipBLASLt 1.4.1 (0 launches, 2,979 fallbacks a prompt) | GLM prefill 159 -> 306 tok/s at 8K (x1.9) | open, CI pass |
| [#634](https://github.com/mw00/project-maya/pull/634) | the fused GLM prompt experts at 3 blocks a multiprocessor on gfx1151, bit for bit the same | the kernel -13 to -16%, GLM prefill +3.5% | open, CI pass |
| [#643](https://github.com/mw00/project-maya/pull/643) | DeepSeek's prompt experts through the fused int8 experts (IQ1_M, IQ2_XXS / IQ2_XS downs, DeepSeek's shape) | DeepSeek prefill 234 -> 257 (2K), 430 -> 468 (8K); perplexity 8.4907 -> 8.4937 | open, CI pass; conflicts with #634 at the grid line (the description says how) |

all four are rebased locally onto v1.1.36 (clean) and not yet force-pushed.

## to file

### 1. `tools/ds4/ref_gguf.py` reads an IQ4_NL Engram table as MXFP8 - silently wrong

- **what**: `RefGguf.engram` sizes a row as Q6_K (210 B a 256-value block) or else MXFP8 (33 B a 32-value block),
  and decodes every non-Q6_K table as MXFP8. the GSQ-RCO IQ2_XXS GGUF's Engram tables
  (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`) are IQ4_NL (18 B a 32-value block, 144 B a 256-value row).
- **effect**: rows read at the wrong stride and decoded as FP8 - no error, wrong numbers in the oracle that checks
  maya-ds4's GGUF path.
- **fix**: the row's bytes from its type, decoded by gguf-py for any type but MXFP8; refuse a type of unknown size.
  done in `oracle/ref_gguf.py` (`# substrate:`).

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
  greedy tokens the same). `~/substrate-oracle`, `dumpcmp.py layers L_a.bin L_b.bin`.
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
