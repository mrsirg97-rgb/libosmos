# The attention core (0.17.0, design for review)

One attention for every family, built for gfx1151, so that a prompt's attention runs near the matrix cores' rate
and one shape replaces today's copies. Status: **design**. Nothing here is built. Each phase below lands only after
its proofs pass.

## Why

- **Speed.** Mini's prompt attention runs at about **17%** of the matrix cores' practical rate (4.8 TFLOP/s against
  ~29, at performance, on a 2,968-token prompt). It is **16%** of the prompt step, and its share grows with context.
  The rest of the step's matmuls run at 70-86% of their rates (DESIGN.md "0.17.0: every pack's ceilings" and "Mini's
  prompt attention - what limits it"). Attention is the one large gap between Mini's prefill (~1,215 tok/s at
  performance) and its target (~2,670).
- **One shape per concept** (AGENTS.md). There are about eight attention implementations today, each with its own
  loop, masks, softmax and cache reads. Each family's speed work lands in one of them and nowhere else.

## What the core replaces

| kernel | family | role | head / KV | keys | extras |
| --- | --- | --- | --- | --- | --- |
| `ink2_fa`, `_fa16`, `_fa8`, `_fa85` (`kernels/ink2.hip`) | Inkling (Mini, Small) | prompt and split decode | 128; GQA 3 (Mini), 6 (Small); F32 / f16 / Q8 / Q8-Q5 records | causal; window 1,024 (sliding layers) | relative bias (16 states, extent 1,024), sink logit, sigmoid gate, the 256-key units |
| `q4x_attn`, `q4x_attn_lists` (`q4x_attn.hip`) | Qwen | prompt and decode; LSA's lists | 256; GQA <= 16 | causal; lists (any order) | gate (span: a value or a head); `AttnShape` window and ring |
| `q4x_qsa` (`q4x_qsa.hip`) | Qwen | its sparse attention's scores | indexer heads 128 | top-k selection | |
| `q35_attn_decode`, `q35_attn_read` (`q35_attn.hip`) | 27B, osmos1.7b | decode; LSA's read | Q8_0 cache | causal; a list | gate; `AttnShape` |
| `hy4_mla_attn` (`hy4.hip`) | Hy4 | prompt and decode | MLA: latent 512 + rope 64, Q8 option | window 512; the indexer's selection | sinks; latent output (`hy4_mla_out` after) |
| `ops_sparse_attn` (`sparse_attn.hip`), `attn.hip` | DeepSeek | prompt and decode | MLA-like; FP8 / FP4 rings | window; compressed and selected rows | sinks; ring slots |
| `lsa.hip` | osmos1.7b, 27B | the long context's read lists | | lists | |
| drafters: `ink2_mtp`, `q4x_dflash`, Hy4's DFly | each family's drafter | small | | | |

## The contract the core keeps (non-negotiable)

These hold for every family on the core. Each is tested before a family moves.

1. **Deterministic and batch-invariant.** A token's output bits depend on its own queries, keys and positions,
   never on how a step is split or batched.
2. **Prompt == decode, where a family continues in place** (Inkling and Qwen: their `reads_apart` is false). A
   token's attention gives the same bits in a prompt step and in a decode or verify step. Inkling's way today, kept
   exactly:
   - a query's keys are cut into **units of 256 keys at absolute positions**;
   - each unit runs its own online softmax from nothing;
   - the units are **folded in order**, starting from the sink (`fa_fold_coef`);
   - within a unit, keys go in **16-key tiles in ascending order**;
   - scores come from one WMMA chain over the head dim, in order, and P·V is one WMMA per 16-key tile, accumulated;
   - the row max and sum are DPP butterflies.

   Everything outside that, such as how many queries a block takes, how many keys a stage loads, or where operands
   live, is free.
3. **Drafted == plain**, token for token (it follows from 1 and 2).
4. **Fail closed.** A shape or format the core doesn't take is refused by name, never computed approximately.

## What limits today's kernel (measured)

- **91%** of its wave cycles are waiting.
- About **1,160 VALU instructions per 16-key tile per wave**, against 16 WMMAs.
- **240 VGPRs** (the cap at 768 threads), plus 196-344 bytes per lane spilled to scratch.
- One block per WGP, so latency-bound.

The VALU work per tile:
- the relative bias: 16 gathered loads and FMAs per score, about 640 instructions;
- decoding Q8 / Q5 records and storing them into LDS element by element;
- the softmax, plus rescaling all of O.

The registers: Q's fragments (64) and O's accumulators (64), plus the softmax state, the prefetched raw records and
addressing.

Already measured, and kept or not:
- **kept:** 64 keys per stage (-18% on a sliding layer, the same bits);
- **not kept:**
  - the bias table in LDS (more spills, slower);
  - smaller row tiles (fewer spills, more K/V decoding, slower);
  - a later prefetch point (no change).
- **upper bound only:** half of Q in registers (-20%, before paying for reloading the other half).

## Design

### A. Mini's prompt kernel to the contract's limit (Inkling only, bit-identical)

Each step is proven byte for byte against today's kernel over the 40-case harness (F32 / f16 / Q8 / Q8-Q5 caches,
windows, prompts of 37-2,968 tokens, a step at depth, decodes and verifies), plus a mutation check. Then Mini's
hashes, then an A/B against v0.16.0.

1. **Decode the step's K / V once, not once per block.** Today every block of 128 query rows decodes every Q8 / Q5
   key it reads: about 23x redundant at 3K. A pre-pass would write the keys this step reads as f16 into scratch
   (`kv8_vals` / `kv5_vals`, the same f16 values, so the same bits). The kernel then reads f16 with no decode VALU
   and no raw-record registers. Sliding layers are f16 already; this is for full layers.
2. **A register audit from the ISA.** Find which values spill (`-save-temps`, the kernel's VGPR liveness), then
   restructure those. The likely ones:
   - the causal cut's per-row WMMA copies (`bc`, `d`): diagonal tiles only, but allocated always;
   - the bias loop's address arithmetic;
   - `alpha[8]` held across P·V.
3. **The bias as its own pass over the scores.** Today the 16-term bias is summed inside the score loop, while Q
   and O are live. Instead: compute the tile's scores (WMMA), store them to LDS, then a second phase in which each
   lane adds the bias to the scores, with Q's registers no longer needed for that phase. The terms and their order
   stay the same, so the same bits. This trades LDS traffic for registers; measure it.
4. **Then occupancy.** With spills gone, try a 512-thread block (RT 8 with fewer heads per block, or a head split)
   so that two blocks fit a WGP.

Target: Mini's prompt attention from ~17% to >= 50% of the matrix rate. That is about -230 ms per prompt step at
~3K (~1.3x prefill).

### B. The core's shape (shared)

One descriptor and one kernel skeleton:
- **`AttnCore`**, growing `AttnShape`:
  - heads, kv heads, head dim (128 / 256);
  - scaling;
  - window and ring slots;
  - the key source: positions, a list, or a selection;
  - the KV format: F32 / f16 / Q8 / Q8-Q5 / FP8 / FP4;
  - sinks;
  - gate span;
  - the bias hook (Inkling's relative states; none elsewhere);
  - unit size (256, or 0 for "one softmax", for families without a prompt == decode contract).
- **One skeleton** in `kernels/attn_core.h`: the stage loop, the unit fold, the tile's WMMA chain, the online
  softmax, and the mask, as template parameters for head dim, group, format and hooks. Families supply only their
  prologue (rope, clips, which happen in `attn_prep` today) and their epilogue (gate, MLA's latent output).
- **MLA** (Hy4, DeepSeek): K = [latent; rope] and V = latent, a 576-wide K and a 512-wide V. That is the same
  skeleton with a different head shape: two head dims, the scores' chain over 576 and P·V over 512.

### C. Moving the families (one per commit, each with its proofs)

| family | bits | proof |
| --- | --- | --- |
| Inkling (Mini, Small) | **the same** (the contract is Inkling's today) | byte comparison against `ink2_fa*`, hashes |
| Qwen | **change once**: `q4x_attn`'s order isn't the unit contract | KL against today's (DESIGN.md's method), prompt == decode on the core, drafted == plain, continue-in-place serve tests |
| DeepSeek, Hy4, osmos1.7b, 27B (reads apart) | **change once**: their own orders today | KL, drafted == plain, their tests |
| drafters | follow their family | their tests |

## Proofs and checks

- **Per kernel change:** a byte comparison against the old kernel on seeded inputs (`fa_bits`-style, every format,
  window, length and decode / verify case), and a mutation of the comparison.
- **Per family move:** that family's GPU tests, its hashes (or KL, where its bits change by decision), drafted ==
  plain, and the serve slot / parallel tests.
- **Per phase end:** the full GPU suite, every pack's hashes and sizes, and an A/B against the last release's tag.
- **Speed:** stamps for each piece (the profiler's timestamps are broken on this box since 2026-11-23), plus
  end-to-end tok/s.

## Decisions (the operator's, 2026-11-24)

1. **One-time bit changes are acceptable** when KL matches today's, nothing regresses (speed, quality, drafted ==
   plain, the serve tests), and the gain outweighs what's there now.
2. **0.17.0 = Phase A + building B.** C's size is measured once B exists; C goes in 0.17.0 or in the release after,
   decided then.
3. Unit size: decided per family when it moves (C).

## Open questions to measure

- Whether gfx1151 has an asynchronous global-to-LDS load usable from HIP. If so, the prefetch stops costing registers.
- The causal cut's real cost on diagonal tiles, against a separate diagonal pass.
- Whether a 512-thread block (two per WGP) beats one 768-thread block once spills are gone.
