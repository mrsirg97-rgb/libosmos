# libosmos

A deterministic inference engine for **DeepSeek-V4.1-Flash** (~768B parameters: ~557B in routed experts, ~196B in
Engram tables) and **Qwen3.8-Flash-Next** (~177B, 512 experts, a preview of the Qwen4 architecture) on a **single AMD
Strix Halo** (Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB).

It runs these quantizations of DeepSeek:

- **unsloth's UD-IQ1_M and UD-IQ1_S** ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF),
  5 shards each): experts in 1-2-bit IQ types, attention Q6_K. The two packs below run both with unsloth's lossless
  MXFP8 Engram table (one 203 GB file the two share) and, optionally, huihui-ai's abliteration patch. Against Maya
  each engine reads the same files (UD-IQ1_M with its stock IQ4_NL table, UD-IQ1_S with the MXFP8 one).
- **ISTA-DASLab's GSQ-RCO IQ2_XXS** ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)):
  experts IQ2_XXS / IQ2_XS / IQ1_M, its Engram a shared file linked as shard 6 (llama.cpp's layout since 2026-10-27;
  their earlier four-shard layout too). What the engine was first built on; measured below, not one of the packs.

And of Qwen:

- **unsloth's UD-IQ3_XXS** ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  4 shards, 78 GB): every weight resident in GPU memory, drafted by the model's own MTP head (unsloth's shared Q8_0
  `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`), a 262,144-token context - or 1,048,576 with YaRN (its own pack).

**The guarantee:** the same input gives the same output, bit for bit, however it is run:

- alone or batched with other tokens or sequences,
- at any memory budget, whichever experts happen to be resident in GPU memory,
- decoded one token at a time or drafted (DSpark, or Qwen's MTP head) and verified,
- with an expert computed on the GPU or on the CPU (the CPU lane's AVX-512 kernels give the GPU's exact bits).

Each of these is a test that compares bits, not tolerances: fingerprints of every output stream must match across
the configurations (`tests/gpu/test_lane_gpu`, `test_drafter_gpu`, `test_model_gpu`, `test_forward_gpu`; Qwen's
`test_q4x_forward_gpu`, `test_q4x_qsa_gpu`, `test_mtp_q4x_gpu`, `test_engine_q4x_gpu`), and the kernels are checked
against reference replays with deliberate mutations that must fail them.

C, HIP and a little C++. No Python at runtime.

## Quick start

```sh
python3 tools/install/osmos_install.py ud-iq1_s --serve    # or ud-iq1_m; --rig to register it in rig instead
```

The installer builds the engine, fetches the pack's files at pinned commits (unsloth's shards and Engram table, our
drafter and pick profile from [mrsirg97-rgb/osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)),
checks every SHA-256, writes the pack's `osmos.json` and starts **osmos-api**: an OpenAI-compatible server
(`POST /v1/chat/completions`, streamed or whole, with tools and reasoning; `GET /v1/models`; `GET /health`) on port
8091. `--have <dir>` uses files you already have; `--abliterated` adds huihui-ai's patch as an overlay; `--rig`
registers the pack in [rig](https://github.com/mrsirg97-rgb/rig) and asks before making it the default model.
`"max_tokens": 4096` in a pack's `osmos.json` reads prompts in 4K steps (the default 2K): DeepSeek's long prompts
10-18% faster, its drafted answers 1-2% slower (fewer experts fit beside the bigger step).

Both packs run one context length: **1,048,576 tokens**, with up to 256K tokens a reply (its reasoning and answer).

Qwen3.8-Flash-Next: its pack, unsloth's UD-IQ3_XXS 4 shards and its MTP head (~81 GB) at a pinned commit, a
**262,144-token** context, up to 64K tokens a reply, prompts read in 4,096-token steps (`"max_tokens"` in its
`osmos.json`):

```sh
python3 tools/install/osmos_install.py qwen38-ud-iq3_xxs --serve      # or --rig (its row's alias: qwen)
python3 tools/install/osmos_install.py qwen38-ud-iq3_xxs-1m --serve   # 1M context (alias qwen-1m): the same files
```

The 1M pack is the model card's static YaRN (factor 4 over the native 262,144): it recalls a fact 355K tokens back
(a 394K-token document's needle, answered word for word), but costs some quality and speed on short text - perplexity
+0.4%, a 2K prompt 745 tok/s against 810, drafting 34.9 tok/s against 40.4 (the MTP head agrees less under it). So the
262K pack stays the default; take the 1M one when a conversation needs it. ~93 GB of GPU memory at 1M.

## Performance

The short version, for people who want one page: [`docs/STRIX-HALO.md`](docs/STRIX-HALO.md).

One Strix Halo (128 GB, ~113 GiB GPU memory), 2026-11-06, nothing else running, AMD's `performance` profile. The
comparison is [Project Maya](https://github.com/mw00/project-maya) **1.1.56** on the same model files, the same session:
Maya as its setup configures it (rebuilt, its pack, its own 8 GB DSpark, its CPU lane calibrated on this machine, its
expert profile warmed first), osmos **0.5.2** as it ships (the budget sized to the GPU, `--budget auto`; the router's
guess off; our 4.7 GB IQ2_XS DSpark). Greedy decoding throughout. Every run and its command: `DESIGN.md`
(`~/bench/h2h-056/perf.sh`).

### Agents

What an agent waits on is mostly reading: Maya's agent bench (4 agents at once x 8 turns, seed 1, 131K context, DSpark
on) sends ~99K new prompt tokens against ~8K answer tokens. UD-IQ1_M, one run each that evening; power is the APU
package's (amdgpu PPT: CPU + GPU + NPU, not the wall) and memory the session's samples (a second apart), over each
run's own window:

| | osmos 0.5.2 | Maya 1.1.56 | |
| --- | --- | --- | --- |
| Wall, all 32 turns | **440 s** (7.3 min) | 548 s (9.1 min) | 20% less |
| First token p50 / p95 | 6.6 / **55.5 s** | 6.3 / 100.3 s | p95 45% lower |
| Turn done p50 / p95 | **42.0 / 89.7 s** | 54.5 / 127.9 s | 23% / 30% lower |
| Package power, mean | 126 W | 127 W | |
| Energy an answer token | **6.96 J** | 8.81 J | 21% less |
| Peak GPU memory (GTT) / RAM in use | **109.7 / 116.9 GiB** | 112.5 / 120.9 GiB | ~3 / 4 GiB less |

osmos's prompts in 8,192-token steps (a step size the next release's packs take; the same answers, bit for bit): 422 s
(7.0 min), first token p95 32.8 s, 6.68 J an answer token. At Maya's memory (a 108 GiB expert budget against auto's ~106)
osmos's decode is the same 17.0-17.1 tok/s: the budget is not what holds it.

### Both packs against Maya 1.1.56

64K context (Maya's configuration), 256 tokens an answer to Maya's three bench questions and its 2K / 8K prompts,
each engine in a fresh process; tok/s the mean of two interleaved rounds, the engines' order swapped in the second
(Maya's prompts the four reads of its runs, its range in brackets):

| | UD-IQ1_M osmos | UD-IQ1_M Maya | UD-IQ1_S osmos | UD-IQ1_S Maya |
| --- | --- | --- | --- | --- |
| Prompt, 2,048 tokens | **300** | 241 (237-245) | **327** | 258 (248-262) |
| Prompt, 8,192 tokens | **445** | 408 (402-414) | **488** | 436 (431-440) |
| Prompt, 8,192 tokens, osmos 8K steps | **557** | | **595** | |
| Decode, DSpark drafted | **33.5** | 27.6 (27.5-27.7) | **40.6** | 34.5 (34.3-34.6) |
| Decode, plain | 17.1 | 17.0 (16.9-17.1) | 19.4 | 19.3 (19.3-19.3) |
| Perplexity, 8,191 tokens | **2.4579** | 2.4649 | **2.5006** | 2.5023 |
| Output reproducible bit for bit | yes | not measured (1.1.46: no) | yes | not measured |

Plain decode is a tie; reading prompts (+9 to +27% as shipped, +36 to +37% at 8K in 8K steps), drafting (+18 to
+21%) and several conversations at once (20% less time, 21% less energy, ~3 GiB less memory) are where osmos leads.
0.5.5 against a 0.5.4 build (2026-11-07, `performance`, UD-IQ1_M, the same three questions, every hash the same):
plain decode **+5%** (16.3 -> 17.1 tok/s), drafted and prompts level (DeepSeek's prompts wait on the SSD).
osmos's answers hash the same in every run and configuration, across 0.1.3, 0.2.0, 0.3.0, 0.3.2, 0.4.0 and 0.5.x;
Maya's greedy decode varied run to run when measured on 1.1.46 (DESIGN.md), not re-measured since.

### Energy at the low-power profile

osmos 0.3.2 at the box's `low-power` profile (power-profiles-daemon's power-saver, a stock AMD profile), UD-IQ1_M:

| | tok/s | package W | J a token | against `performance` |
| --- | --- | --- | --- | --- |
| Decode, plain | 15.8 | 71 | **4.49** | -8% speed, -42% energy |
| Decode, DSpark drafted | 30.7 | 71 | **2.35** | -9% speed, -39% energy |
| Agents (4 x 8) | 481 s | 63 | **3.79** an answer token | +8% time, -46% energy |

At low-power osmos drafts faster than 0.2.0 did at `performance` (29.8 tok/s), at ~71 W against ~125. At the packs'
own 1M context (0.1.x, fixed budgets), decode is UD-IQ1_S 18.8 plain / 37.5 drafted, UD-IQ1_M 16.0 / 28.6 tok/s: a 1M
context's attention state is ~0.85 GiB.

### Qwen3.8-Flash-Next against llama.cpp

UD-IQ3_XXS, 2026-11-07, `performance`, osmos `tools/bench/q4x_bench` (greedy; prompts in 4,096-token steps as the
pack reads them, two rounds); llama.cpp 68d20b8 (HIP, all layers on the GPU, flash attention), `llama-bench` 2 runs
the same session, each length at its better batch (2,048 or 4,096); its plain decode from 2026-11-06:

| | osmos 0.5.5 | llama.cpp | |
| --- | ---: | ---: | ---: |
| Prompt, 2,048 tokens | **903** | 483 | +87% |
| Prompt, 8,000 tokens | **831** | 493 (8,192) | +69% |
| Prompt, 16,384 tokens | **810** | 498 | +63% |
| Prompt, 65,536 tokens | **727** | 362 | 2.0x |
| Decode, MTP drafted | **45.9** | not measured | |
| Decode, plain | **27.1** | 26.2 | +3% |
| Decode at 16K context, plain | **25.7** | 23.1 | +11% |
| Perplexity, 8,192 tokens | **2.7143** | 2.7193 (its F16 cache) | |
| Perplexity, 16,384 tokens | 2.2732 | **2.2721** (F16), 2.2675 (Q8_0) | +0.05% |

osmos leads at every length and both ways of decoding (0.5.4: plain decode 23.5, -10%; 0.5.5's same-bits gains in
CHANGELOG.md). Both run Qwen's sparse attention (QSA: past
2,048 keys, a token attends to its top 512 blocks of 4); osmos picks llama.cpp's own blocks on its tensors (949 of 949
tokens past the budget, `test_q4x_qsa_gpu`). osmos keeps the cache in Q8_0 (llama.cpp in F16: on its Q8_0 cache its
perplexity is the right column's). The perplexity difference at 16K is the spread two forwards' roundings make
(llama.cpp's own two caches are 0.2% apart there), our dense attention level with it (2.2714).

**Agents**, the bench above (4 agents x 8 turns, ~99K new prompt tokens), 65K context, the MTP head drafting:

| | Qwen3.8 osmos 0.5.0 | DeepSeek UD-IQ1_M osmos 0.3.2 |
| --- | ---: | ---: |
| Wall, all 32 turns | **292 s** | 446 s |
| First token p50 / p95 | **3.0 / 50.4 s** | 6.2 / 54.7 s |
| Turn done p50 / p95 | **24.1 / 68.9 s** | 44.1 / 94.0 s |
| Answer tokens | 5,850 | ~8,000 |

The steps are shared (each expert read once for every conversation's tokens), the drafts too, and the prompts read
at the sparse attention's speed; the guarantee holds across all of it - a conversation's bits its own however the
others share its steps (`test_q4x_forward_gpu`, `test_mtp_q4x_gpu`).

### The two packs: quality and power

64K context; perplexity over 8,191 tokens of mixed text and code (`bin/ppl`); power and energy at `performance`:

| | UD-IQ1_M (quality) | UD-IQ1_S (speed) |
| --- | --- | --- |
| Perplexity (abliterated) | 2.459 (2.470) | 2.501 (2.505) |
| Package power decoding, plain / drafted | 131 / 127 W | 138 / 131 W |
| Energy a token, plain / drafted | 7.72 / 3.87 J | 7.15 / 3.35 J |
| Expert shards on disk (plus the shared table) | 140 GB | 122 GB |

UD-IQ1_S's experts are 12.5% smaller than UD-IQ1_M's, so more of them stay in GPU memory: it reads fewer from the SSD
and decodes faster; UD-IQ1_M scores better on unsloth's six benchmarks (85.55 against 83.53). The abliterated packs
decode at the same speed. Both run unsloth's MXFP8 Engram table, speed- and quality-neutral against the IQ4_NL one
(DESIGN.md, "the MXFP8 Engram pack"). ISTA-DASLab's GSQ-RCO IQ2_XXS, the quant osmos was first built on, decodes as
fast as UD-IQ1_S but scores worst of the three (perplexity 2.546): measured, not packed.

## How it runs a 250 GB model in 128 GB

- **Experts by residency.** The dense weights and as many experts as the budget holds live in GPU memory, chosen by
  how often a calibration text picks them (a profile beside the model). The rest are read from the GGUF files when
  the router picks them, guessed a layer ahead, and the resident set follows the text (hot experts swapped in). The
  readers take the read needed soonest from [libevt](https://github.com/mrsirg97-rgb/libevt)'s priority queue - a
  miss the forward waits on first, a read ahead moved up in place when it becomes one - and each read's end comes back
  to the forward as an event. A read goes straight from the SSD into registered host memory the GPU reads (`O_DIRECT`:
  no copy through the page cache, no CPU copy).
- **Int8 experts.** A routed expert's input is rounded to int8 a block of 32, and its 1-2-bit weights are summed with
  exact integer dot products (`v_dot4` on the GPU, `vpdpbusd` on the CPU), then scaled in one fixed order - the
  contract in `kernels/lane.h` that makes the CPU and the GPU agree bit for bit.
- **Engram** rows are read from their file at each step's start, every row a reader of its own (~7 KB a token).
- **The attention's state in the model's own formats**: the window's keys FP8, the compressed rows and the indexer's
  keys FP4, as DeepSeek's design keeps them - ~0.9 KB a token, so a 1M context (~0.85 GiB) fits beside a 106 GiB expert budget.
  A packed row read back is exactly the F32 it was rounded to; a prompt's step unpacks the rows once for all its
  tokens.
- **DSpark**, DeepSeek's own drafter, proposes up to 5 tokens a round; one batched forward verifies them.

## How Qwen runs

- **Every weight resident** (78 GB of GPU memory): no streaming. The matmuls run on the matrix cores with f16
  activations and F32 sums at every token count, so a token's bits are the same however it is batched; the experts of
  a step are read once for up to 128 of its tokens.
- **The cache in Q8_0** (~13 KB a token for its 12 attention layers): a 262,144-token conversation ~3.4 GB, a
  1,048,576-token one ~13.6 GB.
- **YaRN** past the native context (the 1M pack): llama.cpp's static YaRN, its rotation held to llama.cpp's at
  positions to 700,000; off at 262K, where the rotation keeps its exact bits.
- **QSA**, Qwen's sparse attention: keys pooled in blocks of 4, a token's top 512 blocks by its indexer's score; below
  2,052 positions every block (the dense attention's bits).
- **The MTP head** drafts up to 5 tokens a round (a 65,536-token shortlist of the vocabulary); one batched step
  verifies them. Its 36 Gated DeltaNet layers' states keep 8 steps back in rings, so a rejected draft rolls back.

## Requirements

- AMD Ryzen AI MAX+ 395 (gfx1151) with 128 GB. The CPU lane (optional) needs AVX-512 VNNI and VBMI.
- Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14).
- GPU memory raised to ~113 GiB: `options ttm pages_limit=29637703` in `/etc/modprobe.d/ttm.conf` (a system setting:
  yours to make).
- The model, on an NVMe SSD (the faster its random reads, the faster the decode), on a filesystem that takes
  `O_DIRECT` reads (ext4, btrfs and xfs do; tmpfs and some FUSE ones do not - osmos refuses them, named), one of:
  - UD-IQ1_M's or UD-IQ1_S's 5 shards (`DeepSeek-V4.1-Flash-1017-UD-IQ1_M-0000N-of-00005.gguf`; shard 5 is the
    Engram table: the stock IQ4_NL one, 110 GB, or the MXFP8 one - below),
  - GSQ-RCO's IQ2_XXS shards: llama.cpp's layout (5 shards, the shared Engram file linked as shard 6) or their earlier
    one (4 shards and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`). Its GGUFs name no DSpark target layers: the
    drafter's `dspark.json` does (`target_layers`, ours carries it),
- for drafting, DSpark's weights: our IQ2_XS requant of Maya's pack (4.7 GB,
  [mrsirg97-rgb/osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)) or Maya's own (8 GB),
- optionally (UD-IQ1_M's decode ~12-13 -> ~16 tok/s with one), an expert pick profile beside the model (`osmos-picks.bin`: every
  pick on calibration text, `bin/dbench` with `PICKS_OUT` writes one).
- or, for Qwen, unsloth's UD-IQ3_XXS 4 shards and its MTP head (`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`) - ~81
  GB of GPU memory with the cache; the GPU-memory setting above.

## The packs

A pack is unsloth's shards 1-4 of a quant, the MXFP8 Engram table as its shard 5, a pick profile and the drafter;
`tools/install/` (the manifest `packs.json`: every file's repo, commit and SHA-256) lays one out. By hand:

```sh
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-IQ1_S/*0000[1-4]-of-00005.gguf" --local-dir .
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "engram/*MXFP8*" "engram/set_split.py" --local-dir .
python engram/set_split.py engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file 5 5   # both quants' slot: 5 of 5
ln -s $PWD/engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file \
      UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00005-of-00005.gguf                    # UD-IQ1_M's: the same link
MODEL=UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf PICKS_OUT=UD-IQ1_S/osmos-picks.bin \
  MAX_CTX=16384 bin/dbench 104 2048 1 warmup.txt calib16k.txt                           # the profile
```

`calib16k.txt`: ~16,000 token ids of ordinary text and code, nothing of the bench's questions (ours: DESIGN.md,
"the decode (2026-11-03)").

Abliterated: huihui-ai's patch for that quant ([huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF),
`unsloth/patch-UD-IQ1_S/` or `patch-UD-IQ1_M/`, and its `patch-unsloth-DeepSeek-V4.1-Flash-1017.py`) applied to
shards 1-4 (it replaces the attention output projections, `blk.N.attn_output_b`), the same table linked as shard 5,
the unpatched quant's profile. The smaller drafter: `make requant` (llama.cpp's ggml beside this checkout), then
`bin/requant_dspark <maya's dspark dir> <out dir> IQ2_XS`.

## Build and test

```sh
make                        # lib/libosmos.a (the host library)
make lib/libosmos_hip.a     # the GPU library
make tests                  # CPU tests (no GPU, no model)
make asan && make tsan      # the same under AddressSanitizer / ThreadSanitizer
make gpu-tests              # GPU and whole-model tests (skip without a gfx1151 or a model)
make serve api              # bin/osmos-serve (the engine over stdin / stdout), bin/osmos-api (the HTTP server)
make api-test               # osmos-api against a scripted engine
python3 tools/install/test_install.py
```

The whole-model tests read GSQ-RCO from `~/models/ds1017` where it is, else UD-IQ1_M from `~/Models/ds1017/UD-IQ1_M`
(`tests/gpu/test_model.h`); `test_forward_gpu` still needs GSQ-RCO and its layer dump, and skips without them.
`test_model_gpu` also needs a dump of the reference forward (`oracle/`, see its README).

## Use

```c
#include "osmos/run/forward.h"

char err[FORWARD_ERR];
Gguf *g = gguf_create(".../DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf", err);
Weights *w = weights_create(g, NULL, err);   /* GSQ-RCO: its Engram GGUF as the second argument */

ForwardOptions o = {.n_layers = w->geo.n_layers, .max_ctx = 1u << 20, .budget = 106ull << 30,
                    .picks = picks /* or NULL */, .max_tokens = 2048};
Forward *f = forward_create_with(w, &o, err);
forward_lane(f, 0, err);               /* decode's experts in the CPU-repeatable order; 0: all on the GPU */

Seq *s = seq_create(f, err);
seq_prompt(s, n_prompt);
/* a prompt of at most max_tokens (longer: several forward_batch calls; forward_logits says which rows a step keeps) */
forward_batch(f, seqs /* s, n_prompt times */, prompt, n_prompt, err);
int32_t t;
forward_greedy(f, n_prompt - 1, 1, &t, err);   /* the next token, picked on the GPU */
for (;;) {
  forward_step(f, s, t, err);
  forward_greedy(f, 0, 1, &t, err);
}
```

Every call returns `false` (or `NULL`) with the reason in `err` rather than guess. Drafting: `osmos/run/drafter.h`
(`drafter_create`, `drafter_draft`, then verify with `forward_batch` and `seq_rollback`; `tools/bench/dbench.hip`
shows the loop).

## Reproduce the numbers

```sh
python tools/bench/prompts.py <maya checkout> <maya pack>/tokenizer bench   # token ids (Maya's tokenizer)
hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/dbench.hip -Llib -losmos_hip -losmos \
      -lhsa-runtime64 -o bin/dbench
export MODEL=~/Models/ds1017/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf
export PICKS=~/Models/ds1017/UD-IQ1_M/osmos-picks.bin MAX_CTX=65536 NO_GUESS=1
GREEDY=1 bin/dbench auto 512 256 bench/warmup.txt bench/q1.txt bench/q2.txt bench/q3.txt                 # plain
GREEDY=1 DRAFT=~/Models/ds1017/dspark-iq2xs bin/dbench auto 512 256 bench/warmup.txt bench/q1.txt ...  # drafted
bin/dbench auto 2048 8 bench/warmup.txt bench/p2048.txt bench/p8192.txt                                # prompts
python tools/power.py --csv agents.csv -- python tools/serve/agent_bench.py --only osmos --model $MODEL \
  --picks $PICKS --dspark ~/Models/ds1017/dspark-iq2xs --budget auto --osmos-flag=--no-guess        # agents
echo "8 0 bench/p131072.txt" | python tools/serve/drive.py -- --model $MODEL --picks $PICKS --no-guess  # 128K
```

Qwen (its ids from its own tokenizer: `oracle/qwen_dump.cpp` writes them; llama.cpp's numbers from `llama-bench`
and `qwen_dump -ppl`):

```sh
hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/q4x_bench.hip -Llib -losmos_hip -losmos -lhsa-runtime64 \
      -o bin/q4x_bench                                           # and tools/bench/q4x_ppl.hip the same way
export Q=~/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf Q4X_BENCH_STEP=2048
bin/q4x_bench $Q ids.txt 16384 64                                # a 16K prompt, then 64 plain tokens
bin/q4x_bench $Q ids.txt 512 128 ~/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf   # drafted
bin/q4x_ppl $Q ids.txt 8192 2048                                 # perplexity
```

`dbench budget_gib|auto max_tokens n_new warmup q...`: `auto` sizes the budget to the GPU as `osmos-serve` does;
`MODEL` a model's first shard (GSQ-RCO: `MODEL_DIR` and `ENGRAM`),
`PICKS` the profile, `DRAFT` DSpark's directory, `PICKS_OUT` writes a profile from the run. Run each configuration in
a fresh process and interleave them; a first run after the machine has been idle is slower (a cold page cache). Each
answer prints a hash of its tokens: the same hash in every configuration is the guarantee.

## Scope

Two models (DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next) behind one engine interface (`include/osmos/run/engine.h`), one
machine (Strix Halo, 128 GB), greedy decoding. The quant kernels (Q8_0, Q5_K, Q6_K, Q2_0, IQ1_S, IQ1_M, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ4_NL, IQ4_XS, BF16, MXFP4, MXFP8, from one type table, `include/osmos/model/qtype.h`; the
int8 expert path and the CPU lane: the IQ1 and IQ2 types) are model-independent; the expert streaming, attention,
hyper-connections, Engram and DSpark are DeepSeek's (`src/run/`), the Gated DeltaNet, QSA, n-gram embedding and MTP
head Qwen's (`src/q4x/`).

## Layout

| | |
| --- | --- |
| `include/osmos/` | the public headers, each with its contract in a comment at the end |
| `src/model/` | GGUF, weights, Engram, DSpark index |
| `src/mem/` | the expert pool, the resident set, the file readers |
| `src/run/` | the engine interface; DeepSeek's forward: steps, layers, the lane, drafting |
| `src/q4x/`, `include/osmos/q4x/` | Qwen's forward: weights, steps, layers, its state, the MTP head |
| `src/text/` | the tokenizers and chat encodings (DeepSeek's, Qwen's) |
| `src/cpu/` | the CPU lane: AVX-512 int8 experts, its thread pool |
| `kernels/` | the HIP kernels and the shared arithmetic contracts (`lane.h`, `quant.h`) |
| `tests/`, `tests/gpu/` | CPU tests; GPU and whole-model tests |
| `oracle/` | the reference forward in PyTorch the numerics are held to (from Maya, MIT) |
| `tools/api/` | osmos-api: the OpenAI-compatible server (osmos-serve its child) |
| `tools/serve/` | osmos-serve: the engine over stdin / stdout, several conversations at once |
| `tools/install/` | the installer and the packs' manifest |
| `tools/` | grid generation, the bench, the drafter's requant |
| `DESIGN.md` | every decision and measurement, including what was tried and did not help |

## Credits

- [libevt](https://github.com/mrsirg97-rgb/libevt) (MIT, the same author): the event scheduler under the engine -
  the forward's expert reads (their order by need, their ends as events) and `osmos-serve --parallel`'s loop, several
  conversations at once on one engine (`third_party/libevt`, a submodule, built into `libosmos.a`).
- [Project Maya](https://github.com/mw00/project-maya) (MIT): the reference this engine is measured against and
  learned from - the DeepSeek reference tools in `oracle/` (copied, with `oracle/LICENSE.maya`), DSpark's packing.
  Several of Maya's AMD paths are this project's author's own contributions there, carried here: the int8 experts on
  gfx1151 ([#643](https://github.com/mw00/project-maya/pull/643), on Maya's main as a co-authored commit), the
  conversation slots and prompt caching ([#356](https://github.com/mw00/project-maya/pull/356)) and libevt as its
  scheduler ([#587](https://github.com/mw00/project-maya/pull/587)). Findings from this work go back to Maya:
  `MAYA-UPSTREAM-PATCHES.md`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the IQ grid tables in `kernels/quant_grids.h` are
  generated from its `gguf-py` (`tools/gen_grids.py`), and the quant formats are ggml's.
- [DeepSeek](https://github.com/deepseek-ai) (MIT): the model, its inference reference, and the Engram hashing that
  `oracle/engram_hash.py` follows.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): the GSQ-RCO quantization this was developed on.
- [unsloth](https://huggingface.co/unsloth): the UD-IQ1_M and UD-IQ1_S quantizations and the MXFP8 Engram table.
- [huihui-ai](https://huggingface.co/huihui-ai): the abliteration patches.
- [Qwen](https://huggingface.co/Qwen): Qwen3.8-Flash-Next; [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  its UD-IQ3_XXS quantization and MTP head. llama.cpp's `qwen4exp` (its forward, QSA, the qwen35 pre-tokenizer) is
  the reference Qwen's forward is held to (`oracle/qwen_dump.cpp`); the chat template is ported from the model's.

## Contributing and changes

`CONTRIBUTING.md` (the process, the house rules, how numbers are measured); `CHANGELOG.md` (each release).

## License

MIT - see `LICENSE`. Third-party material keeps its own license (`oracle/LICENSE.maya`).
