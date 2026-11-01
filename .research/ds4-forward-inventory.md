# DeepSeek-V4.1-Flash-1017: the forward pass as maya v1.1.36 implements it

source: a read of `~/project-maya` (v1.1.36). MC = `src/ds4/ds4_model.cu`, KC = `src/ds4/ds4_kernels.cu`. line numbers
are at v1.1.36. the cleanest spec is maya's oracle, `tools/ds4/ref.py:339-369` (the whole forward in ~250 lines of
torch). claims marked **unverified** were not checked.

## per layer, in order (a token = a window of T=1; decode windows up to 16)

`forward` MC:16810 -> `forward_serial` MC:16875 -> `layer()` MC:17419; attention `attn_core` MC:17086. the prompt
chunk path `layer_chunk` MC:14535: the same math, other GEMMs.

0. **embedding**: `embed.weight` Q5_K [129280 x 5120], read as BF16 rows from a pinned host copy (`emb_row`
   MC:8396); copied into all 4 hyper-connection streams (h = [4][5120] F32); the first pre-mix [1,0,0,0].
1. **Engram** (layers 1 and 14, before attention), MC:17435:
   - the host computes 24 row ids a token (`eg_issue` MC:8425): n-grams 2..4, 8 hash heads, a rolling xor of
     `ctok * mult`, then `% prime + offset`.
   - rows IQ4_NL [384M x 256], decoded by `engram_dequant` KC:2483.
   - `engram.wkv` (Q5_K [25600 x 6144]): 6144 -> (4+1) x 5120, a key per stream and one value.
   - `engram_gate` KC:2502: h_k += sigmoid(signed-sqrt(rms(h_k) . w_k . rms(key_k) / sqrt(d))) . value, with
     w = q_weight . k_weight [4 x 5120] BF16, folded at load.
2. **hyper-connections, the attention's pre**:
   - `hc_mix` KC:313: 24 dots of the flat h (20480) with `hc_attn_fn` (F32 [24 x 20480]), times rsqrt(mean h^2).
   - `hc_split` KC:333: pre = sigmoid(. scale0 + base) + eps; post = 2 sigmoid; comb = a 4 x 4 softmax then 20
     Sinkhorn iterations (`hc_attn_scale` [3], `hc_attn_base` [24]).
   - `hc_pre_norm` KC:408: x = rms(sum pre_s h_s) . attn_norm.
   - **the pre-mix is lagged**: attention uses the previous layer's FFN pre, the FFN this layer's attention pre
     (ref.py:356/361/363).
3. **q**: `wq_a` (Q8_0 1280 x 5120) -> rms(q_norm) -> `wq_b` (Q5_K 32768 x 1280): 64 heads x 512. RoPE on the last
   64 dims (`finish` KC:3392, MC:17478-17507): theta 1e4 without YaRN on window-only layers; theta 1.6e5 with YaRN
   (factor 16, original 65536) where ratio > 0 (`rope_freqs` MC:531/2911).
4. **window KV** (every layer): `wkv` (Q8_0 512 x 5120) -> kv_norm -> RoPE -> FP8-E4M3 rounding; K = V (MQA,
   d = 512); into the 128-row FP8 ring (`ring_to_ext` / `ext_to_ring` KC:4036/4039).
5. **compressor** (KV-source layers only; `attn_in_jobs` MC:13687):
   - ratio 2 (layers 2, 8, 14): `compressor.wkv` + `wgate` (BF16 512 x 5120); `compress_win` KC:849/3511 keeps slot
     p % 2; a completed group writes softmax(gate over its slots) . kv -> rms(compressor.norm). groups do not overlap.
   - ratio 1 (layer 20): wkv + norm, no gate.
   - the latent: RoPE at the group's first position, FP4 E2M1 with E4M3 scales per 16 (MC:17289-17302).
6. **indexer** (index-source layers 2, 8, 14, 20, 24, 28, 32, 36), MC:17197-17288:
   - keys (KV sources): `indexer.wk` (BF16 128 x 512) on the latent -> k_norm -> RoPE -> FP4, UE8M0 scales per 32.
   - query: `indexer.wq_b` (BF16 4096 x 1280) on qr -> 32 heads x 128 -> RoPE -> FP4.
   - head weights: `weights_proj` (BF16 -> F16 32 x 5120) on x, x 1/sqrt(128)/sqrt(32) (`gemv_f16_rows` KC:3459).
   - score `index_score` KC:3516: sum_h relu(q_h . k) . w_h; `topk` KC:3577 keeps min(512, n), n = (p+1)/R completed
     groups.
   - candidates: layer 20 also runs `cand_blocks` KC:3617 (the top 2048 blocks of 8 by max score, the newest block
     pinned); layers 24-36 score only those positions once n > 16384.
   - the other layers reuse the previous index layer's top-k (`plan_window` MC:16686).
7. **sparse attention** `sparse_attn` KC:3962: a query attends its own window (up to 128 rows, BF16 on the FP8 grid)
   plus its <= 512 selected FP4 compressed rows; a learned sink logit a head (`attn_sink` [64]), scale 1/sqrt(512);
   oracle ref.py:195.
8. **output**: inverse RoPE on o, FP8 rounding; `wo_a.slice.0..7` (Q5_K 1024 x 4096) a group of 8 heads each ->
   1024; concatenated (8192) -> `wo_b` (Q6_K 5120 x 8192) (MC:17520-17526). `hc_post` KC:441:
   h'_k = post_k . y + sum_j comb[j][k] . h_j.
9. **hyper-connections, the FFN's pre**: as 2, `hc_ffn_*` and `ffn_norm`.
10. **router**: `ffn.gate.weight` (BF16 384 x 5120, held as **F32** on the device), `gemv_f32` KC:3477 (one summation
    order on every path); `route` KC:752/3484: score = sqrt(softplus(logit)); top-6 of score + `gate.bias`; the
    weights the unbiased scores, normalized (+1e-20), x 1.5.
11. **shared expert**: w1 / w3 Q5_K, w2 Q8_0 (2304); the same clamped SwiGLU (`swiglu` KC:804, `shared_down`
    MC:13699).
12. **routed experts**: w1 / w2 / w3 types per layer, mixed IQ1_M / IQ2_XXS / IQ2_XS (layers 0-6 all IQ1_M; later
    w1 / w3 mostly IQ2_XXS / IQ2_XS, w2 mostly IQ1_M). SwiGLU silu(min(g, 10)) . clamp(u, +-10). kernels: `gg::moe_gate_up`
    / `moe_down` (`ds4_gguf.cu`:1548/1575) at T = 1, `win_experts` (`ds4_gguf.cu`:1614) for windows, the IQ kernels
    (`ds4_iq.cuh`:72-147); activations rounded to int8 per 32, dp4a. then `combine`, `hc_post`.
13. **final**: `hc_pre_norm` with the last f_pre and `norm.weight` -> `head.weight` Q5_K -> 129,280 logits
    (`head_rows_dev` MC:16785).

the prompt path (`layer_chunk`) differs only in its products: weights dequantized to FP16, hipBLASLt / cuBLAS FP16
GEMMs (`gemm` MC:4351) on FP16-rounded activations (`rows_f16`); the WMMA / BF16 attention (kAttnWmma); the experts
sorted per expert (`moe_sort`); the router still `gemv_f32` (routing bits as the token path's).

## the layers that differ (MC:3189-3235)

- **0-1**: ratio 0, window only: no compressor, no indexer.
- **2-19**: ratio 2. sources 2, 8, 14 (a compressor, index keys, compressed KV each); 3-7, 9-13 and 15-19 read the
  preceding source's compressed KV and top-k (cross-layer sharing, "YOCO" in the code).
- **20-39**: ratio 1. 20 is the source (gateless compressor) and the candidate source; 24, 28, 32, 36 recompute
  queries and top-k over layer 20's keys within the candidates; the rest reuse the previous index layer's top-k.
- **Engram**: layers 1 and 14.
- **entries 40-42: `mtp.0..2`, the DSpark blocks** (ratio 0, MC:3196-3209): no compressor / indexer / Engram;
  128 MXFP4 experts top-3, `gate.bias_vl`; Q8_0 attention; `mtp.0.main_proj` (Q8_0 5120 x 15360) and `main_norm`;
  `mtp.2` also `norm`, `markov_head.{embed,head}` (BF16 129280 x 256), `confidence_head` (1 x 5376).
- layers 21-39 see the past only through layer 20, so `MAYA_DS4_PROMPT_SKIP` (docs/SETTINGS.md:566) runs layer 20
  "kv only" and skips 21-39 on the early part of long prompts.

## a conversation's state (`alloc_seqs` MC:7539; `ds4_model.hpp`:5-10; row sizes `kv_row_bytes`, `ds4_kernels.cuh`:31)

fixed:
- window rings: 43 x 128 x 528 B (FP8 + UE8M0 scales), ~2.9 MB.
- compressor partials (2, 8, 14): 2 slots x 512 x F32 x (kv + score), 8 KB each.
- the last index layer's top-k and candidates; the Engram history (the last 3 compressed ids).
- 2 checkpoints (rings + partials) and the speculative backups (`ring_bak`, `c_snap`).

per position (append only):
- compressed KV 288 B a row: 2 / 8 / 14 add 144 B a token each, 20 adds 288: **720 B a token**.
- index keys 80 B a row: 3 x 40 + 80 = **200 B a token**.
- **~0.92 KB a token** in all: ~30 MB at 32K, ~0.96 GB at 1M. `slot_save` (MC:7791) saves and loads all of it.

## Engram

- `engram.bin` header (MC:429): vocab 129280, 2 layers, n-gram 4, 8 heads, hd 256, pad 2; hash parameters from the
  GGUF's metadata (`tools/ds4/pack_gguf.py`:362).
- tables `layers.1/14.engram.embed`: IQ4_NL, 384,006,168 and 384,016,682 rows x 256 values (144 B a row), 110.6 GB,
  `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`.
- a token: 48 random reads of 144 B (~6.9 KB of payload); from an SSD each is a ~4 KiB page: ~200 KB of I/O and 48
  IOPS a token.
- issued when a window's tokens are known, collected before layer 1 / 14 (`ds4_engram_io.hpp`); a clock-evicted row
  cache (16-256 MB, `DS4_ENGRAM_CACHE_MB`, MC:3727-3741); mmap / pread on 4-16 threads; `DS4_ENGRAM_RAM` locks them
  in RAM; a prompt prefetcher.
- prompts are the cost: 28K tokens = 1.36M rows, 13-17 s on Windows (MC:3731). `engram.wkv`'s GEMVs read ~220 MB a
  token (**unverified** against a measurement).

## DSpark (MC:13743-14050; `verify` MC:14341; `rollback` MC:14421)

1. **seed**: after every trunk window, `tap` (MC:16797) takes the mean of the HC streams entering layers 37, 38, 39
   (15360 values); `seed_drafter` (MC:13748): main_proj -> main_norm, then each block's wkv, kv_norm, RoPE, FP8 into
   its 128-row ring. ~80 MB more weights a trunk token.
2. **draft**: the block [y, noise 128799 x 4] (5 rows) through mtp.0..2, whole layers (HC, attention, MoE); each query
   sees the ring and the whole block (no causal mask within it); the ring unchanged.
3. **heads**: hc_pre_norm without a weight -> mtp.2.norm -> the trunk's head; then row by row `markov_add`
   (E[previous token] . H[v], rank 256), argmax (or a sample, `DS4_DSPARK_SAMPLE`); confidence = sigmoid(w . [x | E[tok]]).
4. **gate**: drafts verified while conf >= 0.5 and their running product >= 0.5 (`ds4_main.cpp`:226).
5. **verify**: the window [y, drafts] (<= 16 rows) through the trunk, logits for every row; greedy compares tokens;
   sampled keeps a draft with probability min(1, p/q).
6. **rollback**: ring rows and compressor partials restored in batches (`ring_restore_batch` /
   `comp_restore_batch`); the accepted rows re-seeded into the drafter; compressed KV past the kept position is
   overwritten later. oracle `tools/ds4/dspark_ref.py`, fed by `DS4_DSPARK_DUMP`.

## the numerics maya holds to

- "no fast math" (`ds4_kernels.cu` header): RoPE angles, softmaxes, roundings the reference's F32 arithmetic.
- the FP8 / FP4 roundings exact on their grids, emulated bit for bit on HIP (`ds4_numerics.cuh`): FP8 E4M3 (ring,
  attention output, query, kv); FP4 with E4M3 / 16 scales (compressed KV); FP4 with UE8M0 / 32 scales (index keys and
  queries).
- BF16 working rows for window attention; the router in F32, one summation order everywhere; rms eps 1e-20, hc eps
  1e-6, routing +1e-20; top-k ties to the lowest index, a deterministic order.
- GGUF GEMVs: activations int8 per 32, exact dp4a dots. optional fused paths (`MAYA_DS4_HC_FUSE`, `ROUTE_FUSE`,
  `DECODE_TRIM`, `Q8_BATCH`) byte-checked against the plain ones at load (`hc_fuse_check` MC:10367 on).
- AMD long verify windows are not always bit-identical to short ones (docs/SETTINGS.md:579). `MAYA_DS4_DENSE_REQ`
  and `DISK_SKIP` change answers.

## parity tooling

- `ds4_main.cpp`: `--ids`, `--dump-logits`, `--ppl FILE` (:2240), `--multi-check`.
- dumps: `DS4_DUMP_H` (a layer's residual), `DS4_DUMP_L`, `DS4_DUMP_R` (routing), `DS4_DUMP_Y`, `DS4_DSPARK_DUMP`;
  checks `DS4_INDEX_CHECK`, `DS4_TOPK_CHECK`, `DS4_TP_CHECK`.
- `tools/ds4`: `ref.py` (the EXL3 oracle), `ref_gguf.py` (a GGUF oracle sharing no engine code), `compare.py` (cosine,
  KL, top-1, `--layers`), `dumpcmp.py`, `dspark_ref.py`.
- tests: `tests/ds4_topk_order.cu`, `ds4_index_score_tc.cu`, `ds4_attn_mma_test.cu`, `ds4_gguf_test.cu`,
  `ds4_iq1_rows_test.cpp`.
- **gap**: `ref_gguf.py`'s dequantizers - BF16 / Q8_0 / MXFP4 / MXFP8 in its docstring, IQ2_XXS the only i-quant a grep
  found. whether it covers this pack's IQ1_M / IQ2_XS / Q5_K / Q6_K / IQ4_NL is **unverified**.

## `ds4_model.cu` by subsystem (18,463 lines; rough, from function spans)

| subsystem | ~lines | spans |
| --- | ---: | --- |
| model math (any runtime needs it) | ~3,300 | pack / geometry / RoPE 300-553; loaders 2144-2891 (part), 5390-5547; Engram 8386-8593; chunk GEMMs 4130-4479; DSpark / verify / rollback 13687-14493; `layer_chunk`; forward / head 16686-17060; `attn_core` / `layer()` 17060-17844 |
| multi-GPU (TP, pipeline, helper GPU, all-reduce, checks) | ~4,000 | 9981-11378, 11378-12312, 14929-15527 |
| expert tiering and streaming (VRAM / RAM / SSD, cache, swaps, PCIe lane, flow) | ~5,500 | 4479-5390, 5547-7539, 9117-9981, 12312-13687 |
| CPU-lane experts | ~520 | 8593-9117 |
| serving (`--parallel`, checkpoints, slots, step_multi, sampling) | ~1,300 | 7539-8170, 17844-18320 |
| scheduling / calibration (path pick, borrow / warm, yoco, forward_chunks) | ~1,150 | 15527-16686 |
| vision | ~220 | 8170-8386 |
| load(), env flags, profiling, OS / NUMA | ~2,000 | |

the other files: `ds4_kernels.cu` 4,458 (~500 TP / transfer / all-reduce), `ds4_gguf.cu` 1,819, `ds4_iq.cu` 1,942,
`ds4_main.cpp` 2,499. CUDA-only code is thin (~800 `cuda*` calls, hipified; 8 `#if`s).

estimate (the reader's, not measured): a single-APU runtime needs ~3-4K lines of host code and ~4-5K of kernels.

weight traffic a token (the reader's estimate, **unmeasured**): ~5.4 GB dense, ~2 GB routed, ~0.3 GB Engram / DSpark.
