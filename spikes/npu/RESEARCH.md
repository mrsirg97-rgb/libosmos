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
- no bandwidth test in this xrt-smi (2.21.75): the DRAM read rate still to be measured with a kernel of our own.
- zero copy (spikes/npu/share.hip; libxrt-dev, a uuid.h shim for its headers):
  - host pages registered with HIP (host_slots' kind: the expert cache) as a userptr BO, flags host_only: accepted;
    its map the same pages, the GPU's bytes seen, its device address the process's own (SVA: the NPU's PASID in the
    IOMMU). normal: refused ("unsupported buffer type"); cacheable: refused ("must be AMDXDNA_BO_SHARE").
  - hipMalloc's memory exported as a dma-buf (hsa_amd_portable_export_dmabuf) and imported (xrt::bo(device, fd)):
    accepted, 64 MiB, the CPU's map of it the GPU's bytes.
  so either kind of osmos's memory reaches the NPU with no copy - its kernel reading them still to be shown (the AIE
  compiler).

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
