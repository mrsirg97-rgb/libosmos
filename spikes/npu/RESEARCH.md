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
