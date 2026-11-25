# The attention core (0.17.0, design for review)

One attention for every family, built for gfx1151, so that a prompt's attention runs near the matrix cores' rate
and one shape replaces today's copies. Each phase below lands only after its proofs pass.

**Status after 0.17.0:** Phase A's measurements are done. One change shipped: 64 keys per stage (sliding layers
-18%, the same bits). The rest of A (the decode pre-pass, the register audit, the bias as its own pass) was measured
in pieces and found limited by LDS fragment traffic and tile latency under the unit contract, not by spills (DESIGN.md
"Mini's prompt attention - what limits it"). B and C are not built; they move to a later release, where B is mainly a
consolidation (one shape replacing ~8) rather than a speed win.

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

## The spec (0.18.0, for the operator's review)

0.17.0 measured Phase A and shipped its one win; 0.18.0 builds the core (AGENTS.md's order: this spec, then the
schema, then failing tests, then the implementation, Inkling first).

### The inventory, as the headers state it

| family | kernels | keys | cache rows | head | the sums' order | precision | contract |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inkling (Mini, Small) | `ink2_fa`, `_fa16`, `_fa8`, `_fa85`, the split decode | positions: causal, window | F32, f16, Q8 records, Q8 + Q5 records | 128, GQA <= 8 | 256-key units from 0, folded in order; 16-key tiles | WMMA f16, F32 sums | prompt == decode |
| Qwen | `q4x_attn` (prompt and decode) | positions, or QSA's picked blocks + tail | Q8_0 (ggml blocks) | 256, GQA <= 16 | 128-key chunks merged in order | WMMA f16 | prompt == decode |
| q35, osmos1.7b | `q35_attn_decode`, `q35_attn_read`, `q4x_attn_lists` | positions; LSA's list (any order) | Q8_0 | 128 / 256 | runs of >= 16 keys, <= 512 parts; a list in order | F32 decode, f16 lists | reads apart |
| Hy4 | `hy4_mla_attn` (and its read path, `mla_mm`) | window; or `pool`-position groups (sel) | F32 or Q8 latent rows, a ring | MLA: K 576, V 512, MQA | splits of 64 merged with the sink | F32 decode, f16 read | reads apart |
| DeepSeek | `ops_sparse_attn` | window (ring) + selected compressed rows | FP8 ring, FP4 rows, F32 | 256, key = value | keys in order | F32 | reads apart |
| drafters | `ink2_mtp_attn`, `q4x_df_attn`, Hy4's DFly | ring context + the block's own (not causal) | f16 / F32 | 128 / its own | spans merged in order | F32 | their family's |

Five row formats, four sums' orders, two precisions - in 3,346 lines (ink2.hip's attention, q4x_attn.hip,
q35_attn.hip, sparse_attn.hip, attn.hip), with Hy4's MLA in hy4.hip and the drafters' beside them.

### The core's contract (proposed)

1. **One sums' order for every family: Inkling's units.** A query's keys in 256-key units at absolute positions,
   each unit's online softmax from nothing, the units folded in order from the sink (none: zero), 16-key tiles
   ascending in a unit, one WMMA chain over the head dim. A list or a selection (LSA, QSA, Hy4's groups, DeepSeek's
   compressed rows) is the same: its keys in the list's order, cut into units of 256 of the list. Then prompt ==
   decode for every family, and drafted == plain with it, from one place.
2. **One precision: f16 operands on the matrix cores, F32 sums and softmax.** The F32 decode paths (q35's, Hy4's,
   DeepSeek's) and the F32 drafters move to it: their bits change once, held by KL against today's.
3. **A row decoder per cache format, to f16 in LDS** - F32, f16, Q8 records, Q8 + Q5 records, Q8_0, FP8, FP4,
   Hy4's Q8 latent: each a decode of a row's chunk, the core's loads around it.
4. **The hooks a family supplies, and only these:** a logit term (Inkling's relative bias; none elsewhere), a sink
   (a logit, or none), an epilogue (the gate: a value's or a head's; MLA's latent output stays hy4_mla_out's).
   Rope, clips and norms stay before it (attn_prep's and each family's).
5. **Head shapes:** Dk and Dv separately (128 / 128, 256 / 256, MLA 576 / 512), GQA to 16, MQA.
6. **Fail closed:** a shape, format or source the core does not take is refused by name at build (the family's
   forward_create), never at a step.

### Ownership (SCOPE.md)

`attn_core_create(const AttnCoreDesc *, ledger, err)` / `attn_core_destroy` - the descriptor checked and the scratch
(the split decode's partials, the units' fold) sized and allocated whole at create, from the forward's ledger; a
step's call takes the core and its tensors, never allocates. The forward owns its cores (one per layer kind), and
destroys them.

### Tests, written before the core (they fail until it exists)

- byte comparisons against each kernel it replaces, on seeded inputs: every row format, window, list, length (37 -
  4,096 tokens), a step at depth, decodes and verifies - Inkling's must be byte-identical (fa_bits' 40 cases, made a
  test); the others' KL against today's on real weights (a family's own tolerance, from today's run-to-run: no number
  picked by hand)
- properties: a token's bits whatever its batch, its step's split, its decode; drafted == plain
- refusals: a head dim, a GQA, a format, a list longer than its stride - each named
- a mutation of each comparison (it must fail when the core is wrong)

### Phases

- **B1: schema + tests** (`include/osmos/gpu/attn_core.h`, the tests above, failing).
- **B2: the skeleton + Inkling**, byte-identical, `ink2_fa*` callers moved, the old kernels deleted.
- **C: a family a commit** - Qwen, q35 / osmos1.7b (with LSA's lists), Hy4, DeepSeek, then the drafters: each its KL,
  its prompt == decode test, its hashes re-based, its old kernel deleted. Sized once B2 exists.

### Decisions for the operator

1. **One order (Inkling's units) for every family** - recommended: it is what makes prompt == decode a property of
   the core rather than of five kernels; the cost is one KL-checked bit change for each family but Inkling.
2. **One precision (f16 operands, F32 sums), the F32 paths included** - recommended, by KL; DeepSeek's FP4 rows
   and Hy4's MLA are where it is least certain, measured before they move.
3. **The drafters on the core too** - recommended, last.

**Decided (the operator, 2026-11-24): yes to all three** - Inkling's units for every family, f16 operands with F32
sums everywhere (the F32 paths included, by KL), the drafters on the core last. B1 next.

### C1: Qwen (design)

Qwen3.8-Flash-Next: 24 query heads over 2 KV heads (12 a KV head), heads of 256, 12 full-attention layers of 48, QSA's
top 2,048 keys (picked blocks, then the tail); today every step on `q4x_attn` (prompts and decodes alike), its gate a
value's, no sink, no bias; its output also as the out matrix's f16 (Q4xX16, the side).

- **Rows: a token's query heads.** The matrix cores are not blind to a key a weight zeroes (`q4x_attn.hip`: a masked
  key's v moves the last bit; Inkling's kernel zeroes its diagonal tile's values row by row for the same reason). A
  tile's rows can share its keys only when they read the same keys - and QSA's picks are a token's own. So Qwen's block
  is one token and one KV head, its rows the token's 12 query heads (16 with padding) - `q4x_attn`'s layout - where
  Inkling's are 16 tokens of a head. The skeleton takes the row map as a parameter; the units, tiles and fold are the
  same, so the contract is.
- **Its keys: the tiles its list touches, the rest of a tile zeros.** QSA's blocks are 4 keys (compress_ratios 4:
  its top 2,048 keys 512 blocks), so a 16-key tile can be part picked. A block's rows are one token's, so every row
  reads the same keys: a tile's keys the token did not pick are loaded as zeros and masked (`q4x_attn`'s rule for
  the keys past a token) - tiles at absolute multiples of 16, units at absolute multiples of 256, tiles the list
  never touches skipped. Every block picked reads exactly the dense tiles: all-picked == dense bit for bit (today's
  property, kept), and a token's bits are its picks' alone.
- **A head of 256.** Q's fragments and O's accumulators double (~256 registers before the softmax). A wave pair a row
  tile: each the tile's scores (one WMMA chain over all 256 dims, in order - the contract's), each half of O's columns
  (`q4x_attn`'s split). Measured against keeping Q in LDS before it is chosen.
- **Rows: Q8_0** (ggml's blocks: an f16 d, then 32 int8s), decoded to f16 as Inkling's Q8 records are (q d, rounded).
- **No sink** (the fold from nothing: gm -inf, gl 0), **no term**, **the gate a value's**, **the side x16 at any
  step** (today's split decode's alone: Qwen's every step).
- **Proof (its bits change once, by decision):** against exact math (an F64 host attention on the same inputs), the
  core's error no larger than `q4x_attn`'s on every case - no tolerance picked by hand; then KL against today's run on
  the real weights (q4x_kld: wiki and an agent trace), the hashes re-based, drafted == plain and the serve tests.
  Properties: prompt == decode, all-picked == dense, a token's bits its own whatever its batch.
- **Speed: no slower than `q4x_attn`** (Qwen's prompt 954 tok/s and decode 50.0 / 44.3 at performance, 0.17.0's
  README) - or it does not move.

`q4x_attn` is also osmos1.7b's and q35's prompt kernel: Qwen moves first, `q4x_attn` stays for them until C2.

**C1 decided (the operator, 2026-11-24):** Qwen's unit 128 (the desc's `unit`; Inkling's stays 256) and its units
merged max-first, `q4x_attn`'s merge - bit-identical to today (10 of 10 cases byte for byte), so Qwen moves with its
hashes as they are. Measured first: at 256 three accuracy cases were 0.2 - 1.0% from `q4x_attn`'s and decode +72%; at
128 with the running fold the error was `q4x_attn`'s to every digit but an ulp apart in 6 of 10 cases (the merge);
max-first, byte for byte. Every Qwen step a block a (token, KV head, unit), then the merge - in pieces of 8,192 pairs
(q4x_attn's scratch, borrowed). A family's tiles are its list's entries (picked blocks, then the tail): every block
picked tiles exactly as dense.

### C2 and C3: the reads-apart families, Hy4 and DeepSeek (moved whole)

C1 settled the method: a family moves with its own kernels' code, whole, into the core - the core's desc and step
naming what each takes, its refusals by name at create - and its bits stay as they were, proven by hashes of the old
kernels' outputs frozen before their code moved (each set 3 runs the same) and a mutation of each path (the hashes
must fail). The one sums' order and the one precision (decisions 1 and 2) are then a change inside the core, measured
per family by KL - not a condition of moving.

- **C2: osmos1.7b and q35** (`apart`: the step's `read` picks the prompt's kernel or the decode's): `q4x_attn`,
  `q4x_attn_lists`, `q35_attn_decode` and `q35_attn_read` as `attn_q4x` / `attn_q35` (19 frozen cases).
- **C3: Hy4's MLA** (`ATTN_ROWS_MLA_F32` / `_Q8`: one latent row for every head, keys the row - latent and rope -
  and values its latent; `kv_heads` 1, `dk` 576, `dv` 512; a head's sink, `ATTN_SINK_HEAD`; unit 32, its splits;
  apart: a read on the matrix cores, mla_mm's, else the F32 splits; the indexer's groups as `ratio` = its pool, each
  query's groups in `sel`, the last its own; each query's position in the step's `pos`) - `hy4_mla_attn` as
  `attn_hy4` (`kernels/attn_core_mla.h`; the row layout `mla_row.h`, its writer's and its reader's). 19 frozen cases
  (dense, sliding over a ring, selected groups, F32 and Q8, 64 / 32 / 16 / 8 heads), both kernels mutated (9 and 10
  cases fail, every case one of the two). A core a layer (its heads, a sliding one's window or a sparse one's groups),
  the DFly2 drafter on its target layers' cores. Hy4's answers through osmos-api the pre-C3 build's to the hash.
- **C3: DeepSeek's latent** (`ATTN_ROWS_FP8`: a token's window over the FP8 ring, then its selected rows of
  `AttnKeys.rows` in `s_rows` - FP4 compressed rows, or the drafter's block's own F32 - at most `max_sel`; a floor;
  `k_wide` / `rows_wide` when a step unpacked them to F32, ops_kv_unpack's, the same bits; unit 0, one softmax) -
  `ops_sparse_attn` as `attn_ds4` (`kernels/attn_core_latent.h`). 11 frozen cases (packed and unpacked, the window
  alone, a floor, small and large tiles, 128 heads, the drafter's block). The forward's core and the drafter's.

The drafters left: Inkling's MTP (`ink2_mtp_attn`) and Qwen's DFlash (`q4x_df_attn`) - Hy4's DFly2 and DeepSeek's
drafter moved with their families.

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
   decided then. (Revised at 0.17.0's release: A measured and its one winning change shipped; B and C deferred.)
3. Unit size: decided per family when it moves (C).

## Open questions to measure

- Whether gfx1151 has an asynchronous global-to-LDS load usable from HIP. If so, the prefetch stops costing registers.
- The causal cut's real cost on diagonal tiles, against a separate diagonal pass.
- Whether a 512-thread block (two per WGP) beats one 768-thread block once spills are gone.
