# 0.7.0: the NPU - what is known before a line of it is written (2026-11-12)

## this box
- NPU Strix Halo (1022:17F0, 0000:c3:00.1): 8 columns; amdxdna in kernel 7.2.7; XRT 2.21.75 (libxrt2, libxrt-npu2,
  python3-xrt: pyxrt opens it); firmware 1.1.2.65; memlock unlimited (/etc/security/limits.d/99-npu.conf).
- xrt-smi reports an estimated power (0.001 W idle): a counter for A/B.
- missing: libxrt-dev (apt, sudo), the AIE compiler (mlir-aie/IRON + Peano: pip wheels from GitHub releases), and
  xrt-smi validate's archive (Xilinx/VTD 2.21.75 archive/strx/xrt_smi_strx.a -> ~/.local/share/xrt/2.21.75/amdxdna/bins).

## measured here (2026-11-12, xrt-smi validate, Power Mode Default - setting another needs root)
- gemm (int8): 51.0 TOPS, 222 cycles average - Krackan's number to the digit (destevez.net: 51.0), so Halo's NPU is
  the same array at the same clock: no Halo-only headroom.
- latency: 56.0 us a launch end to end, average (the literature's 72).
- throughput: 71,765 ops/s - ~14 us a launch when they queue: dispatch pipelines; a dependent chain pays ~56.
- no bandwidth test in this xrt-smi (2.21.75); mlir-aie 1.4.5's getting_started/00_memcpy (every shim DMA in-out
  pair, 8 columns x 2 channels, a passthrough kernel; ~/npu-venv, llvm-aie 22.0.0.2026111001): 64 MiB in and 64 out
  in 1,628 us on the NPU (1,734 dispatched, 2,022 end to end, 5 runs, MAD 11 us) - **82.4 GB/s read + write together**,
  ~41 each way: above the literature's ~50 (Krackan, DDR5-5600). a read-only rate, and the GPU's loss beside it, next.
- zero copy (spikes/npu/share.hip; libxrt-dev, a uuid.h shim for its headers):
  - host pages registered with HIP (host_slots' kind: the expert cache) as a userptr BO, flags host_only: accepted;
    its map the same pages, the GPU's bytes seen, its device address the process's own (SVA: the NPU's PASID in the
    IOMMU). normal: refused ("unsupported buffer type"); cacheable: refused ("must be AMDXDNA_BO_SHARE").
  - hipMalloc's memory exported as a dma-buf (hsa_amd_portable_export_dmabuf) and imported (xrt::bo(device, fd)):
    accepted, 64 MiB, the CPU's map of it the GPU's bytes.
  so either kind of osmos's memory reaches the NPU with no copy - its kernel reading them still to be shown (the AIE
  compiler).

## Q2_K decoded on one AIE2P core (spikes/npu/q2k, 2026-11-12)
real evo-A bytes (blk.20.ffn_down_exps, Q2_K), 4,096 blocks (1M values) to bf16, checked against gguf-py's dequantize:
| version | one core | max err / max |
| --- | --- | --- |
| scalar (ggml's loop) | 0.010 Gvalues/s | 4.1e-3 (bf16's rounding) |
| vector, f32 multiply-add, scales in scalar float | 0.078 | 4.1e-3 |
| vector, 4 values a sub-block chosen by selects | 0.041 | 4.1e-3 |
| vector, bf16 multiply-add, scales in scalar float | 0.085 | 1.07e-2 |
| **vector, bf16, the 16 scales and mins as two vector products, no scalar float** | **0.99** | 1.07e-2 |
| (timing alone: the scales' math removed) | 1.45 | - |
the AIE's scalar unit has no FPU - each scalar float op is software: Q2_K's per-row scale math was 92% of the time.
the 1.07e-2: the scales rounded to bf16 before the product (keep them f32). linear in REPS: the core's, not the DMA's.
so 32 cores ~32 Gvalues/s from ~10 GB/s of Q2_K; beside a bf16 GEMM on the same cores at ~128 rows an expert the
decode takes ~2/3 of a core - the NPU ~5 TFLOP/s of experts, against the GPU's ~15 (Qwen, 0.6.3). next: the scales
f32, the GEMM fed (mm_bfp's block float), the codebook formats (IQ2_XXS, IQ1).

## IQ2_XXS decoded on one AIE2P core (spikes/npu/iq2xxs, 2026-11-12)
UD-IQ1_M's own bytes (blk.5.ffn_down_exps, IQ2_XXS: its expert downs; gate and up are IQ1_S), 1M values to bf16:
| version | one core | max err / max |
| --- | --- | --- |
| grid bytes (2 KB) unpacked to int16, converted, f32 product; lanes built by 8 inserts a group | 0.33 Gvalues/s | 4.06e-3 |
| **the grid stored as bf16 (4 KB: 8, 25, 43 exact), a group 4 row loads concatenated, signs by integer XOR, one native bf16 product** | **1.11** | 8.73e-3 (the scale rounded to bf16) |
codebook lookups are no obstacle: the grid lives in the core's 64 KB, a row one 16-byte load. as fast as Q2_K's
arithmetic decode. (Peano lowers no bf16 vector negate: sign flips go through int16. an fp16 scale guard that zeroed
d below 2^-11 hid the whole output once: these tensors' d run ~3e-4.) next: IQ1_S (gate, up): a 2,048-row grid -
16 KB as bytes, 32 KB as bf16, beside the core's buffers.

## IQ1_S decoded on one AIE2P core (spikes/npu/iq1s, 2026-11-12)
UD-IQ1_M's own bytes (blk.0.ffn_gate_exps, IQ1_S: its gate and up experts - most of its expert bytes): the 2,048-row
grid of {-1, 0, 1} as bf16 (32 KB of the core's 64; 8 blocks a buffer beside it), a group four row loads, one bf16
multiply-add (dl * grid + dl * delta; the 8 scales and shifts a block two vector products): **0.94 Gvalues/s**, error
7.2e-3 (the scales rounded to bf16), every sign right - the first build.

so every format of UD-IQ1_M's and evo-A's experts decodes on one core at ~1 Gvalues/s (Q2_K 0.99, IQ2_XXS 1.11,
IQ1_S 0.94): 32 cores ~30 Gvalues/s. what is left: the decode feeding a GEMM on the core (the effective rate at a
prompt's ~128 rows an expert), the scales kept f32 (error to bf16's 4e-3), and the numerics against the KLD protocol.

## one core's matmul (spikes/npu/mm, 2026-11-12)
64x64x64 bf16 tiles in the core's memory, back to back (mlir-aie's matmul_vectorized; the run at REPS 257 against 1):
native bf16 4x8x8 5.61 us a tile, **0.09 TFLOP/s** a core (~3 for the array); bfp16-emulated 8x8x8
(-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16: both operands converted to bfp16ebs8 a call) 1.01 us, **0.52** (~16.6:
the literature's ~15). so the float path is block float: an 8-bit mantissa, an exponent shared by 8 values - the
activations too (finer than the int8-a-32 0.6.x simulated on Qwen's experts, +1.3% perplexity: still the KLD's to
judge). at ~128 rows an expert a weight's GEMM is 256 flops, ~0.49 ns of a core, against ~1 ns of decode: ~1.5 ns a
weight, the array ~5.5 TFLOP/s of experts. native bfp16 (mm_bfp.cc's matmul_vectorized_bfp16, the operands
already bfp16ebs8): 64x64x64 0.80 TFLOP/s a core, 64x128x64 1.06, **64x256x64 1.39 (~44 for the array: the GPU's
f16 ceiling is 47-49, its Qwen experts 14.7)**. K = 256 is a quant block's 256 values. the balance turns: a weight's
GEMM 0.18 ns at ~128 rows against ~1 ns of decode - the decode 85%, the array ~7 TFLOP/s. the lever now: the decode
faster (1.45 Gvalues/s with Q2_K's scales removed - the rest overhead), written straight as bfp16 (8 int8 mantissas,
a shared exponent: an IQ grid row is 8 small integers under one scale).

## the decoders written as bfp16 (2026-11-12)
each decoder's rows two to a 64-lane accumulator, converted by the core (acc.to_vector<bfp16ebs8>(), floor) and
pushed as 72-byte block vectors - the native bfp16 matmul's operand. checked with mlir-aie's host codec against the
reference as the core would round it (aie.utils.bfp.quantize):
| decoder -> bfp16 | one core | exact | worst |
| --- | --- | --- | --- |
| Q2_K | 0.95 Gvalues/s | 71.4% | 2 LSB (its scale and min each bf16) |
| IQ2_XXS | 1.13 | 86.9% | 1 LSB |
| IQ1_S | 0.91 (0.59 until its group loop carried no accumulator across a branch) | 88.7% | 1 LSB |
the conversion costs ~nothing (bf16 out: 0.99 / 1.11 / 0.93). the 1-2 LSB: the scales rounded to bf16 before their
product - f32 scales next. so the pipeline's operand is ready: decode to bfp16 beside the 1.39 TFLOP/s matmul.

## an expert matmul fused on one core (spikes/npu/fused, 2026-11-12)
fused_iq2xxs.cc: a 64-row x 256 IQ2_XXS weight tile (UD-IQ1_M's bytes, a block a row) decoded straight into the native
bfp16 matmul's operand order - 8 rows' grid rows at one 8-value K octet make a 64-lane vector, each 8-lane slice its
row's group scale, converted and pushed: the sub-tiles in raster order, no shuffle pass - then C = A x W^T
(matmul_vectorized_2x2_bfp16, A 64 x 256 the host's encode + shuffle). C against Aq x Wq^T on the host: corr 0.999956,
max err 1.5e-2 (C's own bfp16 rounding). a tile: decode 19.8 us (0.83 Gvalues/s; 117 us when its 64-lane vectors were
built by inserts, not concats), matmul 1.0 us (C zeroed by vector stores - a scalar byte loop had cost ~20 us),
together 19.9 us: 0.105 TFLOP/s a core at M = 64. the decode a weight's fixed cost, the matmul the rows': at
DeepSeek's ~128 rows an expert in 8K steps ~6.6 TFLOP/s for the array (M = 512: ~22). half of a layer's experts on
the NPU beside the GPU: a prompt step ~20-40% faster on a rough model (to profile: the experts' share of a step).

## the fused decode, faster (2026-11-12)
| fused IQ2_XXS, one 64 x 256 tile | decode | + matmul | at M = 64 |
| --- | --- | --- | --- |
| v1 nested loops, row-major words | 19.8 us | 19.9 | 0.105 TFLOP/s |
| v2 group-major words, one counted loop | 17.9 | 19.9 | 0.106 |
| **v3 no setup pass: words read in place, d/8 a row, (2s + 1) from a 16-entry bf16 table, the scale a vector product a band-group** | **12.0** | **13.4** | **0.157** |
the setup pass (rows' words unpacked, 8 lane inserts and an emulated f32 product a row) had been 7.5 of v2's 18 us.
v3: 1.36 Gvalues/s, corr 0.99994 (d/8 and its product rounded to bf16: worst 1.8e-2). at ~128 rows an expert ~9.1
TFLOP/s for the array: ~25% of DeepSeek's routed experts, a step ~9% faster. what is left a 64 values: 8 grid rows'
index extracts and loads, 8 sign lookups into a mask, the words as 16-bit halves (66-byte blocks: 2-byte aligned).

IQ1_S fused (fused_iq1s.cc): its 2,048-row grid packed as 2-bit codes (4 KB; as bf16 its 32 KB would not fit beside
A and the decoded tile), expanded in registers (a row's code over its 8 lanes, a lane's field masked, two compares,
two selects), + delta in bf16, x the band-group's scale vector: corr 0.999949 (worst 2.0e-2), decode 16.6 us a tile
(0.99 Gvalues/s), + matmul 18.2. at ~128 rows an expert ~6.9 TFLOP/s for the array (IQ2_XXS ~9.1); DeepSeek's mix
(IQ1_S 57% of its routed time, IQ2_XXS 32%) ~7.7: ~22% of the routed experts the NPU's, a step ~8% faster.

IQ2_XXS's signs (fused_iq2xxs.cc): ggml's ksigns_iq2xs[k] is k with even parity in bit 7 - computed, not looked up.
v4: a band-group's 8 rows x 4 octets in 32 int32 lanes (shifts 0/7/14/21 by row selects, & 127, parity by xor folds),
an octet's 8 bytes over 64 lanes by an 8 x 8 transpose, each lane's bit tested: 15.0 us - slower than v3's scalar
lookups (12.0): the transpose an octet cost more than its 40 scalar ops. **v5: the same bytes narrowed to uint8 [l][r] -
an octet's 8 sign bytes end to end are its 64-lane mask, two 32-bit lanes: 10.97 us** (corr 0.999945), + matmul
12.9: ~9.7 TFLOP/s for the array at 128 rows. DeepSeek's mix ~9.9: ~27% of its routed experts, a step ~9.5% faster.

IQ1_S's decode taken apart (fused_iq1s.cc, a tile): full 17.4 us; its codes constant (the vector work alone) 5.7;
the index computed but no table load 15.2 - so ~2/3 is the scalar index path: 5 ops a row-octet to join qs's byte
and qh's 3 bits into the 11-bit index, then a broadcast. the grid as int8 rows would drop the broadcasts, but AIE
vectors are 128 bits at least (no 8-byte vector). the indices pre-assembled (an NPU layout's stand-in): 8.7 us.
**v5: the same indices by vector ops a row inside the decode** - its 32 qs bytes one load, qh's 8 words over 32 lanes
(Peano lowers no 128-bit AND: everything 512-bit), the shifts 0/3/6/9 merged by row selects, aie::transpose to the
octet order - **9.95 us** (corr 0.999949; the scalar reference overwritten by it each decode), + matmul 11.9:
~10 TFLOP/s for the array at 128 rows. DeepSeek's mix ~9.6: ~26% of the routed experts, a step ~9% faster.

## the whole array: one IQ2_XXS expert matmul on 32 cores (spikes/npu/fused/fused32.py, 2026-11-12)
C (64 x 2048) = A (64 x 4096) x W^T, W UD-IQ1_M's IQ2_XXS bytes: core (row r, column c) owns 64 output rows, all 16
K-steps of 256 through it (fused_iq2xxs.cc's fused_step, v5 decode). a K-step's A tile (18 KB bfp16) from each
column's shim through its memtile, broadcast (forward, 4 consumers); a column's W stream split at its memtile (4
cores, double-buffered); C in the core over the K-steps, joined at the memtile. corr 0.9995 against Aq x Wq^T (max
7e-2: C re-rounded to bfp16 at each of the 16 K-steps - accumulate f32 next).
| columns x 4 | us | TFLOP/s | a core | of one core alone (0.163) |
| --- | --- | --- | --- | --- |
| 1 | 323 | 0.42 | 0.104 | 64% |
| 2 | 256 | 1.05 | 0.131 | 80% |
| 4 | 279 | 1.92 | 0.120 | 74% |
| 8 | 271 | **3.96** | 0.124 | 76% |
linear in columns: DRAM and the memtiles keep 32 cores fed (~20 GB/s of W and A). a core's ~24% loss: the A tile
single-buffered (memory: 54 of 64 KB) - a K-step waits for its 18 KB. next: M = 128 with one decode (two A halves in
turn: the decode's cost a FLOP halved, the halves' fetch overlapped), f32 accumulation.
**M = 128, one decode for both 64-row halves of A (fused_step's half): 306 us, 7.03 TFLOP/s for the array (0.220 a
core), corr 0.999487** - DeepSeek's ~128 rows an expert in 8K steps, measured: 72% of the 32 x one-core projection
(~9.7). (M = 64 again beside it: 351 us, 3.06 - the box busy with the BF16 download; +-15% between runs until quiet.)
against the GPU's ~27 TFLOP/s on DeepSeek's routed experts: ~21% of them the NPU's, an 8K step ~7% faster. memory caps
M at 128 (192: C and the single-buffered A past 64 KB). next: A's fetch overlapped (~28% lost), C in f32 (its memory
against that), the single-core decode gains (each the array's).
**A's fetch overlapped**: a K-step's A as four 9 KB quarters (two 64-row halves x two 128 K halves), double-buffered (18
KB, as the one full buffer was), Bm decoded as two contiguous K halves (64 x 128 x 64 matmuls): back to back at M = 128,
**342 -> 288 us, 6.28 -> 7.46 TFLOP/s (+19%)** - but corr 0.99949 -> 0.99839, worst 7e-2 -> 1.3e-1: C re-rounded to
bfp16 at every partial sum (32 an output now, 16 before; 8-bit mantissas, an exponent shared by 8). the accumulator's
precision, apart from A's: C in f32 - 32 KB at 128 x 64, past the budget; at 32 rows a core 16 KB (~51 KB in all), a
core's efficiency the same (the decode a weight, the rows an expert unchanged). next: a matmul keeping C f32 (mm_bfp's
8x8 block MACs, accumulated into f32), N 32 a core.
**C in f32** (fused_iq2xxs.cc's mm_f32acc: a C sub-tile's partial sums loaded from f32 memory into the accumulator, the
same mac_8x8_8x8T loop, stored back f32; 32 rows a core, ~52 KB): **corr 0.999971, worst 9.0e-3** (from 1.3e-1: 14x;
what is left bfp16's operand rounding), **5.63-5.65 TFLOP/s** (0.176 a core; -24%: C's f32 traffic a quarter, the
fixed costs a call against 32 rows). a half's two quarters in one pass (C loaded and stored once) does not fit: two
quarters held at once and IRON sizes A's fifo for a third (27 KB), ~1 KB past 64. so f32 is the baseline (the NPU's
results feed the model: 13% worst-case accumulation error is what the KLD protocol would refuse) and speed is won back
inside it: the kernel's 4.3 KB stack, the single pass, the per-core decode.
a core's memory (llvm-size on its elf): .data 4.3 KB (the bf16 grid, kodd, krow32), .bss 9.3 (Bm), fifos ~39 (A 2 x 9,
W 2 x 2.1, C 16), stack 4.3 - ~57 KB. the single pass's third A buffer fits with W single-buffered (a 2 KB tile once a
K-step): **5.89 TFLOP/s (+4.6%)**, the same 9.0e-3 - less than the ~15% hoped: C's traffic was not the cost. a core's
K-step ~11.4 us: decode ~6 (32 rows x 256), two matmuls ~1.2, ~4 unaccounted - waits on A and W (a column's 4 cores
in lockstep on one A broadcast). next: trace a core's stalls (mlir-aie's event trace) before changing more.
**traced** (fused32.py, TRACE=bytes: @iron.jit(trace_config=TraceConfig(...)) - the runtime then supplies and writes
the buffer; parse.py + get_trace_summary; the full 8 columns leave no channel for trace packets, so 1 column, egress
through column 2): every core alike - a K-step's half 0 (decode + matmul) ~12,200-12,580 cycles, half 1 (matmul)
1,747-1,803; **86% of the span in kernels, 14% lock stalls (a constant ~1,175 cycles before each call - handoff, not a
starved buffer), memory stalls 1% of kernel time, the vector unit busy only 34% of kernel time**. so not waits: the
cores are scalar-bound inside the decode (index extraction, row addresses, sign words - no vector gather on AIE2P);
the matmul ~12% of a K-step. next: the decode's scalar path; a K-step's two halves one call (half the handoffs).
**v6: the grid indices read as bytes from the block** (ib[66 r + l]: no 32-bit word assembled from two halves, no shift
and mask an octet - a byte load, a shift, an add, the row's load): one core, back to back, decode 11.65 -> 9.42 us
(-19%), corr unchanged. the array (3 runs each, interleaved, the download running: +-10% a run): v6 167 us mean
(~6.4 TFLOP/s), v5 180 (~5.9) - ~7% (the handoffs and the noise dilute it). the scalar ops a value ~1 cycle: the
op count is the decode's time.
v7 (measured away): a band's 8 scale vectors and sign masks first into small arrays, then one flat 32-octet loop with a
short even body for the software pipeliner: decode 10.17 / 9.52 us against v6's 8.87 / 9.79 back to back - no gain
(the arrays' round trip as dear as the overlap it bought). v6 stays: ~9.2 us a 64 x 256 tile, ~1.8 Gvalues/s a core.
single-core runs scatter ~5%, the array's ~10%, the download writing beside them.

## where a DeepSeek prompt step's time goes (2026-11-12, low-power, rocprofv3, UD-IQ1_M 0.6.9, an 8K step)
14.91 s, the GPU busy 98.2% (538 tok/s under the profiler, 560 without):
| | s | share |
| --- | --- | --- |
| routed experts (grouped matmuls: IQ1_S 2.93, IQ2_XXS 1.63, IQ2_XS 0.32, IQ2_S 0.28) | 5.16 | 35.2% |
| dense matmuls (attention, shared expert, head: Q6_K / Q8_0) | 3.98 | 27.2% |
| attention (sparse, compressed, indexer) | 3.97 | 27.1% |
| hyper-connections | 1.05 | 7.2% |
| other | 0.49 | 3.3% |
the routed experts' work ~139 TFLOP a step (8,192 x 6 x 40 x ~35M x 2): the GPU at ~27 TFLOP/s on them - far past
Qwen's 14.7 (0.6.3). so with the NPU at ~6.6: ~20% of them its, the step ~7% faster; at ~20 (decode 3x faster) ~15%;
the routed experts' ceiling 35%; the dense matmuls (27%: Q6_K, Q8_0 - simpler decodes) would raise it. the energy
is the steadier case: what moves runs at ~2 W, not ~60. (rocprofv3's --memory-copy-trace failed silently here.)

## the GPU beside the NPU (2026-11-12, low-power, DeepSeek UD-IQ1_M, 0.6.9, an 8K prompt + 128 tokens)
alone 507 / 509 tok/s prompt; beside the NPU's memcpy loop (60-64 GB/s read + write) 489 / 498 / 511 - the low two
with the cache squeezed (100.9-104.9 GiB: the BF16 download's Xet RAM growth), the same budget (106.8 / 107.3 GiB) 509
against 511: no prompt loss. decode 15.0 -> 13.2 (-12%, bandwidth-bound - the literature's -14%). the NPU's work is
the prompt's, never the decode's.

## the literature (Strix Point / Krackan unless said; Halo's own numbers unmeasured)
- array: 8 x 4 compute tiles (64 KiB data + 16 KiB program each), 8 memory tiles (512 KiB), shim row; 16 hardware
  contexts. ~1.8 GHz. int8 512 MAC/tile/cycle: 59 TOPS peak, 56.3 by hand, 51.0 xrt-smi gemm (destevez.net/?p=23380);
  IRON GEMM 38 TOPS int8, 14.7 bf16 (arXiv 2512.13282). MX block-FP: 1024 MAC/cycle theoretical. no native int4.
- DRAM: ~50 GB/s measured (2512.13282), < 40 read (Gemma3 on NPU, arXiv 2602.06063) - the NoC's, not the bus's.
- buffers: amdxdna imports/exports dma-buf and maps user VAs (ubuf); XRT bo from a 4K-aligned userptr. HIP <-> XDNA
  zero copy unproven.
- launch: ~72 us a no-op (xrt-smi); ~4.9 ms a reconfiguration.
- power: 1.1-1.8 W under LLM load (2602.06063) against the iGPU's 53-58.
- concurrency on Halo (sleepingrobots.com, FastFlowLM beside llama.cpp): 1.42x together; the GPU's decode -14%, the
  NPU's -6% - the memory's bandwidth shared.
- FastFlowLM (Linux since 2026-03, NPU kernels closed): Halo Llama-3.2-1B 39.5 tok/s decode, Qwen3-8B 8.2 (GPU 19.7).

## osmos's questions
1. measured here: the NPU's read bandwidth, a launch, int8 GEMM TOPS, and the GPU's loss beside it (a DeepSeek
   prompt step, a decode).
2. a HIP buffer (hipMalloc, and host_slots' registered pages) imported by the NPU: zero copy or not.
3. the format: osmos's experts are IQ1_M / IQ2 / IQ3_XXS codebooks (and Q2_K, Q5_K). the NPU wants int8 / bf16 /
   MX. an int8 copy is 4x their bytes (no room); dequant on the AIE's vector units (codebook gathers?) unknown.
   DeepSeek's originals are MXFP4: does AIE2P take MX4?
4. where it pays: work the GPU does far under its ceiling - prompt experts at 14.7 TFLOP/s (Qwen, 0.6.3) - beside the
   GPU, not instead of it; never bandwidth-bound decode.
