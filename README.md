# libosmos

An inference engine for five mixture-of-experts models, and one small long-context one, on one **AMD Strix Halo**
(Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB):

- **DeepSeek-V4.1-Flash** (~768B parameters, ~250 GB quantized), more than fits in memory: the experts it needs are
  streamed from the SSD as it runs.
- **Qwen3.8-Flash-Next** (~177B, 512 experts, the Qwen4 architecture's preview), resident in GPU memory on this
  machine (on less, its experts streamed from the SSD).
- **Inkling-2-Small** (Thinking Machines; 384 experts a layer, a 158 GB pack), its experts streamed from the SSD beside
  the ~84 GB osmos keeps in memory, drafted by its own multi-token-prediction head.
- **Hy4-flash** (Tencent; 220B, ~11B active a token, AngelSlim's Q3_K), its backbone in GPU memory (on less, its
  experts streamed) and its 63 GB Engram table read from the SSD, drafted by the release's own DFly2, to a 1M context.
- **Inkling-2-Mini** (Thinking Machines; 26B, its middle layers looped three times a token), the release's own
  NVFP4 / MXFP8 / BF16 weights - 19.6 GB, all of it in GPU memory - drafted by its own head, to a 1M context:
  prompts at ~1,000 tokens a second.
- **osmos1.7b** (LFM3-1.7B, 1.36 GB in Q4_0), its full-attention layers reading a 512K context through latent sparse
  attention: prompts at thousands of tokens a second, each token's attention ~4,600 keys however long the context.

**The same input gives the same output, bit for bit**, however it runs: alone or batched with other conversations,
at any memory budget, decoded one token at a time or drafted and verified, with an expert computed on the GPU or the
CPU. Each of those is a test that compares bits, not tolerances (`tests/gpu/`), and the kernels are checked against
reference replays with deliberate mutations that must fail them.

C, HIP and a little C++. No Python at runtime.

## Quick start

```sh
python3 tools/install/osmos_install.py osmos-evo-1017 --serve      # DeepSeek; or ud-iq1_s, ud-iq1_m, ud-iq2_xxs
python3 tools/install/osmos_install.py qwen38-ud-iq3_xxs --serve   # Qwen; or qwen38-ud-q4_k_xl
python3 tools/install/osmos_install.py osmos-evo-inkling --serve   # Inkling-2-Small
```

The installer builds the engine, downloads the pack's files at pinned commits, checks every SHA-256, writes the
pack's `osmos.json` and starts **osmos-api**, an OpenAI-compatible server on the first free port from 8091, or
`--port` (`/v1/chat/completions` streamed or whole, with tools and reasoning; `/v1/models`; `/health`). It serves
local clients - SDKs, curl, agents - and refuses web pages: a request with an `Origin` header, or a `Host` that is not
`localhost` / `127.0.0.1`, is refused (0.15.2). Options:

- `--abliterated` adds huihui-ai's abliteration as an overlay (the DeepSeek and Qwen packs; the stock files stay as
  they are).
- `--npu` puts the NPU to work beside the GPU on prompts (decode the same): it checks for the NPU and
  XRT, builds with them, and fetches the pack's NPU kernels (the four DeepSeek packs and `qwen38-ud-iq3_xxs`).
  Opt-in: its output is its own, the same every run but not the GPU-only bits.
- `--have <dir> ...` uses files you already have (several folders, or the flag again).
- `--doctor` checks the machine as osmos will run on it, and says what to change: the GPU and the memory it may
  map, the driver and ROCm, the power profile (balanced is slower than the profiles osmos is measured at), the RAM,
  `/tmp` as RAM, direct reads on the models' folder, an SSD under it, and a wireless download.
- `--rig` registers the pack in [rig](https://github.com/mrsirg97-rgb/rig) instead of serving it, as
  `deepseek-1017-evo`, `deepseek-1017-ud-s` / `-m` / `-l` (UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS), `qwen3.8-flash-next`
  (`-l`: UD-Q4_K_XL), `inkling-2-small-evo`, `osmos1.7b` or `hy4-flash` - its row on this version's release, built under
  `~/.local/share/osmos/<version>` (so `--upgrade` moves it later; `--rig-dev`: this checkout's own build, to test a
  branch).

**Upgrading** (0.11.0): each osmos version is built from its tag beside the others, in `~/.local/share/osmos/<version>`,
and rig's rows move between them by their command alone - their ports and settings stay yours.

```sh
python3 tools/install/osmos_install.py --upgrade            # the newest release, for the rows on the newest version here
python3 tools/install/osmos_install.py --upgrade 0.11.0 --rows deepseek-1017-evo,inkling-2-small-evo   # or --rows all
python3 tools/install/osmos_install.py --rollback           # the last upgrade's rows back where they were
python3 tools/install/osmos_install.py --versions           # the versions built here, the rows on each
python3 tools/install/osmos_install.py --prune [--yes]      # the versions no row runs (removed with --yes)
```

Each version reads its own frozen `osmos.<version>.json`, so a newer installer's settings never break an older row.
A running row takes the new version at its next start.

**Sampling**: requests take `temperature`, `top_k`, `top_p` and `seed`, and each pack's `osmos.json` sets the default
that measured best here (0.15.2: replayed on an agent session that looped, greedy and sampled). DeepSeek's packs sample
at temperature 1.0 / top_p 1.0, inkling-2-mini and osmos-evo-inkling at 1.0, osmos1.7b at 0.1 / top_k 50; Qwen3.8 and
hy4-flash decode greedily (they did not loop greedy, and draft faster). Each token is drawn from (seed, position)
alone, so a seed reproduces an answer, drafted or not; without one a seed is drawn and returned in `x-osmos-seed`.

**Loop signals for agent harnesses** (#47): every answer carries an `osmos.loop` object and an `x-osmos-loop`
header - turns since the last tool call, the most repeated recent call, how many times in a row the last call was
made again after the same result, orphaned calls and results (a client's history compaction can drop one side), and
`stuck`. Nothing the model reads changes; `"osmos": {"fill_orphans":
true}` puts a placeholder after an orphaned call. Context length: the pack's own (DeepSeek 1M, Qwen 512K, Inkling-2-Small 256K, Inkling-2-Mini 1M, Hy4-flash 1M, osmos1.7b 512K), or
`--ctx` (installer or osmos-api: `512k`, `1m`, ...).

## The packs

| Pack | Files | Context | Pick it for |
| --- | --- | --- | --- |
| `osmos-evo-1017` | 147 GB (UD-IQ1_M's files + a 34 GB delta + a 23 GB patch, made where you install it) + 203 GB Engram table | 1M | **the default**: about halfway from UD-IQ1_M's quality to UD-IQ2_XXS's, faster than either (`osmos-evo` and `osmos-evo-a` still install it) |
| `ud-iq1_s` | 122 GB + 203 GB Engram table | 1M | speed |
| `ud-iq1_m` | 140 GB + 203 GB Engram table | 1M | quality (unsloth's benchmarks 85.55 against 83.53) |
| `ud-iq2_xxs` | 165 GB + 203 GB Engram table | 1M | the closest to the model, the slowest decode |
| `qwen38-ud-iq3_xxs` | 84 GB + 3 GB MTP head | 512K | Qwen, fully in GPU memory |
| `qwen38-ud-q4_k_xl` | 111 GB (82 GB in GPU memory, its 29 GB n-gram table read from the SSD) + 3 GB MTP head | 512K | the larger quant: slower than UD-IQ3_XXS, no better perplexity on our text |
| `osmos-evo-inkling` | 158 GB (unsloth's UD-IQ1_S files + a 55.5 GB delta in two parts, made where you install it) + 1.3 GB drafter | 256K | Inkling-2-Small: UD-IQ1_S with its experts' down projections IQ2_S / IQ3_XXS and its attention and head Q5_K |
| `osmos1.7b` | 1.36 GB + 0.8 MB of LSA maps | 512K | a small, fast agent model: LFM3-1.7B, its long contexts read sparse (`vibes1.7b`: its old name) |
| `hy4-flash` | 92 GB (~79 GiB in GPU memory) + 63 GB Engram table + 1.7 GB DFly2 drafter | 1M | Tencent's Hy4-flash: 220B, ~11B active a token, its own drafter - AngelSlim's Q3_K |
| `inkling-2-mini` | 19.6 GB (all in GPU memory) + 0.4 GB drafter | 1M | Inkling-2-Mini as released: fast prompts, long contexts, the most room to spare |

## Speed

osmos 0.15.0 on one Strix Halo (osmos-evo-inkling: 0.15.2, its new draft floor), the `performance` profile, nothing
else running. Every number from the same harness:
`osmos-api` with the pack's own `osmos.json` (its context, its drafter, its budget sized to the machine), a fresh
process, a warm-up request, then each request alone; decode is the answer's own rate from the server's timings. Drafted
and plain answers are the same tokens, bit for bit (their hashes compared); drafting moves only the speed.

| Pack | Prompt, ~3K tokens | Decode, drafted: code / prose | Decode, plain |
| --- | ---: | ---: | ---: |
| `inkling-2-mini` | 1,018 tok/s | **45.7** / **42.2** tok/s | 28.8 tok/s |
| `qwen38-ud-iq3_xxs` | 921 | **50.3** / **44.4** | 28.7 / 29.2 |
| `osmos-evo-1017` | 385 | **28.4** / **24.1** | 18.8 / 19.3 |
| `hy4-flash` | 412 | **23.3** / **20.4** | 17.4 / 17.4 |
| `osmos-evo-inkling` (0.15.2) | 317 | 14.3 / 14.3 | 12.6 / 13.2 |
| `osmos1.7b` (no drafter) | **4,781** | | **126.3** / **126.2** |

Code is a duration parser with tests, prose a B-tree explanation, 256 tokens each at temperature 0.
`inkling-2-mini` samples at its release's temperature 1.0 by default: the same code question at its default draws
45.4 tok/s drafted (28.6 plain).

**Maya's three bench questions** (a CSV parser, photosynthesis for a student, a bouncing-ball web page; 256 tokens,
temperature 0, drafted / plain):

| Pack | Code | Explanation | Web page | Plain |
| --- | ---: | ---: | ---: | ---: |
| `inkling-2-mini` | **57.5** | 36.9 | 38.7 | 28.9 |
| `qwen38-ud-iq3_xxs` | **57.3** | **51.2** | **45.2** | 29.1 |
| `osmos-evo-1017` | 31.4 | 25.3 | 30.9 | 18.3-18.5 |
| `hy4-flash` | 21.4 | 21.4 | 20.2 | 17.5 |
| `osmos-evo-inkling` (0.15.2) | 15.6 | 12.7 | 13.5 | 12.0-12.6 |
| `osmos1.7b` | 126.1 | 126.4 | 122.4 | |

**Agents** (`tools/serve/agent_bench.py`, Maya's harness ported): 4 agents at once, 8 turns each - a 4K-12K system
prompt and task, then tool results of 150-2,000 tokens (one in ten a 5K-20K file read) and 0.5-5 s of tool time
between turns: 98,757 new prompt tokens against up to 8,003 answer tokens, each model's own tokens, at a 131K
context, every turn drafted where the pack has a drafter.

| | `osmos1.7b` | `qwen38-ud-iq3_xxs` | `inkling-2-mini` | `osmos-evo-1017` | `hy4-flash` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wall time, all 32 turns | **90.2 s** | 200.2 s | 207.7 s | 463.1 s | 520.1 s |
| First token, p50 / p95 | **0.87 / 6.2 s** | 1.93 / 25.5 s | 1.56 / 18.9 s | 7.52 / 53.9 s | 4.90 / 48.3 s |
| Turn done, p50 / p95 | **8.8 / 13.9 s** | 17.2 / 43.2 s | 19.6 / 43.4 s | 49.5 / 95.0 s | 53.3 / 111.6 s |

Every turn answered, and every later turn continued its conversation in place. osmos sizes itself to leave 8 GiB of
RAM free beside the model (`"fill_gib"` in `osmos.json`, osmos-serve `--fill`: the output the same at any value).

### The machine

| | |
| --- | --- |
| System | AMD Ryzen AI Halo (RAH-001), BIOS 04.01 |
| APU | Ryzen AI MAX+ 395: 16 Zen 5 cores, Radeon 8060S (gfx1151, 40 CUs) |
| Memory | 128 GB LPDDR5X, shared by CPU and GPU; GPU memory raised to ~113 GiB (`ttm pages_limit=29637704`) |
| SSD | Micron 4600 2 TB (MTFDLBA2T0THJ, firmware PBMA002): a PCIe 5.0 drive, linked at PCIe 4.0 x4 here; btrfs |
| Software | Linux 7.2.7, ROCm with HIP 7.15, gcc 14.2 |
| Placement | raised off its surface, open air beneath it |

DeepSeek streams its experts from the SSD, so the drive is part of its numbers: a prompt reads at ~7.1 GB/s (near
the PCIe 4.0 x4 ceiling); decode reads experts a few MB at a time (`O_DIRECT`), where random-read latency counts.

**NOTE - let your Strix breathe.** On a warm table beside another machine, a sustained
DeepSeek run took the GPU to 102 °C and its clock down to 925 MHz, and drafted decode after a 64K prompt read 17 tok/s
against 29 raised in open air (the same code, the same tokens). These numbers are from a warm room: under sustained
load the GPU still runs 95-100 °C with its clock near 2.6 GHz.

The `low-power` profile costs decode little - it is bound by memory, not clocks (inkling-2-mini's plain decode 28.1
tok/s there against 28.8) - and prompts more.

## The packs in detail

### osmos-evo-1017

**Recall to the window's edge.** An agent conversation built from our own rig sessions (1,038,695 tokens) with 11 facts
planted in user turns ("the access code for the red locker is RW-6315-KD"), each asked for at the end, plus a locker
never mentioned. Each answer scored teacher-forced - the probability of the whole code:

| Distance back | 985K | 881K | 787K | 687K | 587K | 487K | 388K | 267K | 188K | 78K | 12K | never planted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| osmos-evo-1017 | 0.986 | 0.992 | 0.995 | 0.993 | 0.998 | 0.996 | 0.998 | 0.994 | 0.955 | 0.995 | 0.996 | 0.0000 |
| UD-IQ2_XXS | 0.998 | 0.998 | 0.999 | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.983 | 0.999 | 0.998 | 0.0000 |

A session rarely reads 1M at once - a cached prompt reads only what a turn adds; a full read follows a client's
compaction.

**How close each DeepSeek pack is to the model itself** - the KL divergence from DeepSeek's own weights (MXFP4
experts, BF16 dense) over the whole vocabulary, against a reference that runs the original 296 GB of experts
(DESIGN.md, "0.6.6"); lower is closer:

| Pack | Wiki text (8,191 tokens) | Agent traces (27,459 assistant tokens) | Top-1 agreement (Wiki) |
| --- | --- | --- | --- |
| `ud-iq2_xxs` | 0.126 ± 0.005 | 0.036 ± 0.001 | 92.4% |
| `osmos-evo-1017` | 0.158 ± 0.006 | 0.042 ± 0.001 | 91.2% |
| `ud-iq1_m` | 0.190 ± 0.007 | 0.048 ± 0.001 | 90.3% |
| `ud-iq1_s` | 0.259 ± 0.008 | 0.066 ± 0.001 | 88.1% |

**What osmos-evo-1017 is.** UD-IQ1_M with the routed experts' down projections of layers 20-39 requantized to Q2_K from
DeepSeek's MXFP4 originals (unsloth's imatrix), the attention and the head requantized to Q5_K by GPTQ (Hessians
from our agent sessions), and the routed experts' gate and up of layers 10-19 at IQ2_XXS (UD-IQ2_XXS's tensors) -
found by probing the one fact the layers-20-39 pack wavered on and bisecting its layers by overlay (DESIGN.md, "0.7.6";
`tools/evo`). It ships as a delta and a patch on UD-IQ1_M
([osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)), assembled at install with every shared byte
cloned on btrfs or XFS, each shard SHA-256 checked.

The UD packs are [unsloth's quants](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) with the lossless
MXFP8 Engram table, our 4.7 GB DSpark drafter and expert pick profile. The abliteration (huihui-ai's) costs 0.2-0.5%
perplexity at the same speed.

### qwen38-ud-iq3_xxs

[Unsloth's UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (or UD-Q4_K_XL) of Qwen3.8-Flash-Next and
its MTP head, every weight in GPU memory on this machine.

**Recall at its 512K** (YaRN past its native 262K; osmos-evo-1017's rig sessions in its own template, 457,506
tokens), recalled the same way: the facts 10K-418K back (all 11, every token its first choice; never planted 0.0000):

| Distance back | 418K | 375K | 344K | 298K | 255K | 210K | 159K | 110K | 60K | 20K | 10K | never planted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 UD-IQ3_XXS | 0.988 | 0.991 | 0.998 | 0.995 | 0.995 | 0.994 | 0.964 | 0.994 | 0.982 | 0.997 | 0.993 | 0.0000 |

### osmos-evo-inkling

Inkling-2-Small, its experts streamed from the SSD; its long prompts and quality (osmos 0.9.0, the `performance`
profile, osmos-serve at the pack's 256K and 4,096-token steps, its Q5_K drafter):

| | |
| --- | ---: |
| Decode after a 64K / 248K prompt, drafted / plain | **18.1** / 9.2, **22.0** / 7.3 tok/s (the 248K text one its drafter predicts: 94% accepted) |
| Prompt, 8,192 / 65,536 / 253,952 tokens | **355** / 302 / 185 tok/s |
| KL against Inkling-2-Small in BF16 (Wiki, 10,239 tokens) | 0.197, top-1 85.8% |

### inkling-2-mini

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines; 26.4B, 3.4B active, its
middle layers 6-17 run three times a token) served from the release itself: its routed experts NVFP4 (layers 0-19)
and MXFP8 (20-23), its dense weights BF16 as released - on the GPU's matrix cores, each weight's f16 in the kernel -
and its own DSpark drafter (Q8_0). Sampled at temperature 1.0, the release's own setting (greedy decoding looped on one
tool call in long agent sessions); a request's `temperature`, `top_p` and `seed` override it, and a seed reproduces an
answer, drafted or not.

| | |
| --- | ---: |
| Decode after a 131K prompt, drafted / plain (0.12.0) | **24.2** / 17.0 tok/s |
| Prompt, 37K / 131K tokens (0.12.0) | **1,045** / 775 tok/s |
| KL against upstream llama.cpp's own run of the release (64 / 2,048 tokens) | 2.9e-3 / 1.1e-3 (its own GPU against CPU: 3.0e-3 / 1.2e-3) |

Each pass of a looped layer keeps its own cache (48 slots for 24 layers); a full-attention layer's keys and values
are cached Q8 (the pack's `kv_q8`), 17 KiB a token, so a 1M context's cache is 17 GiB beside the 19.6 GB model.
A fact hidden in a 1,000,652-token prompt, 900K tokens before the question, was found (read in 79 minutes, 212
tok/s); 128K and 512K needles at 10 / 50 / 90% likewise.

### osmos1.7b

[osmos1.7b](https://huggingface.co/mrsirg97-rgb/osmos1.7b) (vibes1.7b until 0.11.1): LFM3-1.7B (Liquid's hybrid: 13
short-convolution, 5 sliding and 6 full-attention layers) fine-tuned on 32K-token examples and trained on the Q4_0 grid
it ships in, its full layers reading through its latent sparse attention maps - a token's read its 1,024 best buckets
of the closed 512-key blocks, then the window and the sinks: ~4,600 keys, however long the context. Its long contexts
(osmos 0.11.1, the `performance` profile, one conversation):

| | |
| --- | ---: |
| Decode at 32K / 128K / 512K | 100 / 97 / 85 tok/s |
| Prompt, 2K / 32K / 128K / 512K tokens | **6,479** / 3,948 / 3,591 / 3,128 tok/s |
| A code placed at 10 / 50 / 90% of a 524K prompt (source code), asked at its end | 3 of 3 |
| The same at 10 / 30 / 50 / 70 / 90% of a 1.04M prompt (`--ctx 1m`) | 4 of 5 |
| KL against Liquid's reference (64 / 2,048 / 24,576 tokens) | 3.1e-5 / 2.0e-5 / 3.2e-5 (top-1 63, 64, 64 of 64) |

Its rope is native at every length: on the last 2,048 tokens' loss YaRN x4 read best past ~192K, but it lost the 524K
prompt's code 52K from its end, which native found (DESIGN.md, "osmos1.7b: rope"), so there are no stages and no
cached key is ever turned. The pack is 512K; `--ctx 1m` reaches 1M, where the model found 4 of 5 codes (the miss
answered with a path from the source beside it, as YaRN did). Prompts read sparse past 4,612 tokens (the first
position a read can leave a key out), as the model was trained; every token's bits are the same however its prompt
is split into steps.

### hy4-flash

Tencent's [Hy4-flash](https://huggingface.co/tencent/Hy4-flash): 220B (155.5B backbone + 63.5B of Engram n-gram
tables), ~11.3B active a token, 1M context - DeepSeek's MLA and sparse indexer, a latent MoE, 30 of its 42 layers a
512-token window. Served from AngelSlim's official [Q3_K](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) (Tencent's
compression team: on their agent benchmarks Q3_K matches Q4_K_M and BF16 within a point): the backbone in GPU memory,
the Engram table read from the SSD a step at a time, and the release's own drafter, **DFly2**, which reads the target's
KV cache and indexer picks rather than keeping its own. Its long contexts (osmos 0.11.0, the `performance` profile, one
conversation):

| | |
| --- | ---: |
| Decode with DFly2, served: a 128K / a 913K prompt | 18.8 / 17.0 tok/s |
| Prompt, 32K / 128K tokens | 448 / 414 tok/s |
| Prompt, 913K tokens (a book, its needle found) | 45 minutes |
| GPU memory at 1M context (weights, drafter, a conversation's cache) | ~98 GiB |
| KL against Tencent's reference (64 tokens; F32 cache / the Q8_0 cache osmos runs) | 1e-11 / 0.033 |

The context costs ~11 KB a token: MLA's latent rows Q8_0 (the rope part F32, as vLLM's `fp8_ds_mla`), the
indexer's keys a small ring, its pooled groups F32 - the Q8_0 cache's teacher-forced NLL over 8K tokens no worse than
F32's. DFly2's drafts are vLLM's (its speculator transcribed and checked token for token), and a drafted round's
tokens are always the plain decode's; it accepts ~2.4 drafts a round on code, ~3 on prose. A long prompt's prefill is the limit: the indexer's scores
and selection grow with the square of its length (on the matrix cores; at 913K ~9 of its 45 minutes) - a
conversation's cached prefix is not read again.

## Against other engines

Measured with osmos 0.9.0, the `performance` profile, every run a fresh process, greedy, one run each; DeepSeek at
64K context in the packs' 4,096-token steps.

### DeepSeek against Maya

[Project Maya](https://github.com/mw00/project-maya) v1.1.65, built and calibrated on this machine (`--calibrate`: its
own tensor-parallel and CPU-thread settings), run with its own pack and its own DSpark, reading the same model files.
Both engines at 64K context; Maya's own `--bench`, osmos's `dbench` on the same three questions and prompts.

| UD-IQ2_XXS | osmos | Maya |
| --- | ---: | ---: |
| Prompt, 2,048 tokens | **330** | 220 |
| Prompt, 8,192 tokens | **580** | 440 |
| Decode, drafted (three questions) | **29.0 / 22.1 / 27.4** | 17.6 / 10.9 / 11.0 |
| Decode, plain | **14.7 / 14.4 / 13.7** | 12.6 / 10.8 / 9.8 |

osmos reads prompts 1.3-1.5x as fast as Maya and decodes 30% faster plain and 2.0x drafted: osmos's one cache keeps
the experts decode comes back to.

### Qwen3.8 against llama.cpp

llama.cpp 7d7f028, with HIP, all layers on the GPU and flash attention, at its faster batch for each length (2,048
or 4,096). osmos reads prompts in 4,096-token steps, the way its pack does (`q4x_bench`).

| UD-IQ3_XXS | osmos | llama.cpp |
| --- | ---: | ---: |
| Prompt, 2,048 tokens | **1,012** | 574 |
| Prompt, 8,192 tokens | **952** | 549 |
| Prompt, 16,384 tokens | **925** | 515 |
| Prompt, 65,536 tokens | **865** | 374 |
| Decode, plain | **29.1** | 26.5 |
| Decode at 16K context | **28.5** | 24.0 |
| Decode, drafted by the MTP head | **50.7** | not measured |

osmos reads Qwen's prompts 1.7-2.3x as fast as llama.cpp, the more the longer the prompt. Its decode is 10% faster
plain and 19% at 16K of context.

### Qwen3.8 against Strata

[Strata](https://github.com/Niko1221/Strata) 0.2.13.1 with its signed engine 0.1.59 for this chip, launched with its
own defaults (`strata launch ... --context 128k --vision off`), against osmos-api - both on the same files, Unsloth's.
The same requests go to each server's OpenAI API: greedy, thinking off, prompts of ~2K / 4K / 16K tokens asked for a
summary, and three questions answered in 256 tokens, one round. Energy is the APU package's for the whole run.

| UD-IQ3_XXS | Prompt 2K | 4K | 16K | Code answer | Explanation | Web page | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| osmos, no drafter | **893** | **882** | **894** | 29.1 | 29.1 | 29.2 | **2.14 Wh** |
| Strata, no drafter | 518 | 667 | 744 | **29.4** | **29.5** | **30.7** | 2.45 Wh |
| osmos, DFlash2 | **898** | **884** | **891** | **83.5** | **63.2** | **75.4** | **1.44 Wh** |
| Strata, DFlash2 | 517 | 666 | 742 | 71.4 | 54.3 | 59.9 | 1.85 Wh |
| osmos, MTP head (its default) | 895 | 880 | 890 | 69.7 | 60.0 | 68.6 | 1.46 Wh |

osmos reads prompts 20-72% faster; without a drafter the answers are level (Strata 1-5% ahead) on 13% less energy,
with DFlash2 on both osmos answers 16-26% faster on 22% less energy. Setups differ: Strata packs DFlash2 as Q4_0
(osmos Q8_0); its opt-in fast set (documented as changing the output's bits) was not used, its `--drafting mtp` ran at
no-drafter speed here, and its DeepSeek engine is not shipped for Strix Halo on Linux.

## How it works

**DeepSeek in 128 GB.** The dense weights live in GPU memory, and the routed experts share one cache (the membrane):
- **At load** the cache fills with the experts a calibration text picks most.
- **As it runs**, the least recently used experts are evicted first, and a new expert takes the memory of an old one
  the same size.
- **On a miss**, the expert is read from the model's files into registered host memory the GPU reads directly
  (`O_DIRECT`, so no page cache and no CPU copy). [libevt](https://github.com/mrsirg97-rgb/libevt)'s priority queue
  orders the reads by when they're needed.
- **Prompts** go through a pool of their own and read ahead only the experts the next layer's router will pick, so a
  long prompt doesn't push out the experts decode wants back.
- **Bit-exact experts.** A routed expert's input is rounded to int8 in blocks of 32 and summed with exact integer dot
  products, then scaled in one fixed order (`kernels/lane.h`). That's why the CPU lane (AVX-512) and the GPU give the
  same bits.
- **Attention state** stays in DeepSeek's own formats (FP8 and FP4, about 0.9 KB a token), so a 1M context takes only
  ~0.85 GiB.
- **Engram** rows (about 7 KB a token) are read from their file at each step.
- **DSpark** drafts up to 5 tokens a round, and one batched forward checks them.

**Qwen** is resident where the machine has the room (on less, its experts streamed through the membrane):
- **Matmuls** run on the matrix cores, with f16 activations and F32 sums at every batch size, so a token's bits don't
  depend on its batch.
- **The cache** is Q8_0: 6.8 GB for 512K tokens.
- **QSA**, Qwen's sparse attention, attends to a token's top 512 blocks of 4 keys.
- **The MTP head** drafts up to 5 tokens. A pack can name DFlash2 instead (a block of 7 drafts from one pass of a
  small drafter fed the model's own layers): ahead on chat, behind on real agent work, so an option.
- **The n-gram table** (29 GB) can stay on the SSD, a step's rows read as it runs: the same bits, and how
  UD-Q4_K_XL fits.
- **Rollback.** The 36 Gated DeltaNet layers keep 8 steps of state, so a rejected draft rolls back.
- **The abliteration** is an overlay: 96 tensors loaded in place of the model's own.

`DESIGN.md` has every decision and measurement, including what was tried and didn't help.

## Requirements

- A Ryzen AI MAX+ 395 (gfx1151), and Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14). The optional
  CPU lane needs AVX-512 VNNI and VBMI.
- GPU memory: every number here is from a 128 GB machine with its GPU memory raised to ~113 GiB (`options ttm
  pages_limit=29637704` in `/etc/modprobe.d/ttm.conf`; a system setting, so it's yours to make). Less runs: every
  engine sizes itself to the machine (osmos-serve's `--budget auto`, 8 GB of RAM kept free), the experts it cannot hold
  streamed from the SSD - slower. A pack starts in its engine's smallest build (measured: osmos1.7b and osmos-evo-1017
  ~7 GiB, ud-iq1_m ~8, inkling-2-mini ~10, hy4-flash ~17, osmos-evo-inkling ~21, qwen38-ud-iq3_xxs ~40), a
  conversation's cache on top (inkling-2-mini: ~17 GiB at 1M tokens). The installer checks a pack's own need, and the
  measured machine's for a pack not measured.
- The model on an NVMe SSD, on a filesystem that takes `O_DIRECT` reads (ext4, btrfs and xfs do; osmos refuses one that
  doesn't, and names it). The faster the SSD's random reads, the faster DeepSeek decodes. The numbers here are from a
  Micron 4600 on a PCIe 4.0 x4 link ("The machine", above).
- For `--npu`: the XDNA driver (`amdxdna`, `/dev/accel/accel0`) and XRT with its headers (`libxrt-dev`). Building the
  kernels yourself (`tools/npu/build.sh`) needs mlir-aie's IRON; the installer fetches them prebuilt.

## Build and test

```sh
make && make lib/libosmos_hip.a   # the host and GPU libraries
make tests                        # CPU tests (no GPU, no model); make asan, make tsan
make gpu-tests                    # GPU and whole-model tests (skipped without a gfx1151 or the model)
make serve api                    # bin/osmos-serve (stdin / stdout) and bin/osmos-api (HTTP)
python3 tools/install/test_install.py
```

## Use as a library

The engine interface over both models is `include/osmos/run/engine.h`; DeepSeek's forward is `osmos/run/forward.h`,
and its drafter is `osmos/run/drafter.h`. Each header's contract is in a comment at its end, and every call returns
`false` or `NULL` with the reason in `err` rather than guess. `tools/bench/dbench.hip` is a complete example loop.

## Reproduce the numbers

The Speed tables: each pack through `osmos-api` on its own `osmos.json`, a fresh process each, one request at a time
(prefix reuse off, so every prompt is read whole), the timings osmos-api returns:

```sh
powerprofilesctl set performance
OSMOS_SLOT_KEEP=0 bin/osmos-api --config ~/Models/osmos/<pack>/osmos.json --port 8241 &
curl -s localhost:8241/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16}' > /dev/null   # the warm-up
curl -s localhost:8241/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages": [{"role": "user",
  "content": "Explain how photosynthesis works, step by step, for a high-school student."}], "max_tokens": 256,
  "temperature": 0}' | python3 -c "import json,sys; print(json.load(sys.stdin)['timings'])"   # (plain: the pack without "drafter")
python3 tools/serve/agent_bench.py --only osmos --model <first shard> [--picks <file>] [--dspark <drafter>] \
  --budget auto --max-tokens 4096 --prompts <the model's own tokens: calib16k.txt, p8192.txt>          # agents
```

`osmos-api --tokenize --model <first shard> < text` writes a text's token ids for a model (the agents' and the serve
tests' token files). The older measurements, per engine:

```sh
# DeepSeek (token ids from Maya's tokenizer: tools/bench/prompts.py)
cfg() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" ~/Models/osmos/osmos-evo-1017/osmos.json $1; }
export MODEL=$(cfg model) PICKS=$(cfg picks) DSPARK=$(cfg drafter) MAX_CTX=65536 NO_GUESS=1   # the installed pack's own paths
GREEDY=1 bin/dbench auto 4096 256 warmup.txt q1.txt q2.txt q3.txt                               # plain
GREEDY=1 DRAFT=$DSPARK bin/dbench auto 4096 256 warmup.txt q1.txt q2.txt q3.txt                 # drafted
bin/dbench auto 4096 8 warmup.txt p2048.txt p8192.txt                                         # prompts
MAX_CTX=139264 GREEDY=1 bin/dbench auto 4096 256 warmup.txt p65536.txt p131072.txt           # decode after a long prompt
NPU=<pack>/npu bin/dbench auto 4096 8 warmup.txt p2048.txt p8192.txt                          # prompts, the NPU beside
python tools/power.py --csv agents.csv -- python tools/serve/agent_bench.py --only osmos --model $MODEL \
  --picks $PICKS --dspark $DSPARK --budget auto --max-tokens 4096 --osmos-flag=--no-guess    # agents (+ --osmos-flag=--npu
                                                                                              #  --osmos-flag=<pack>/npu)
# Qwen (token ids from its tokenizer: oracle/qwen_dump.cpp; llama.cpp's perplexity: qwen_dump -ppl)
export Q=~/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf Q4X_BENCH_STEP=4096
Q4X_BENCH_CTX=17408 bin/q4x_bench $Q ids.txt 16384 64                                         # prompt, then decode
NPU=<pack>/npu Q4X_BENCH_CTX=17408 bin/q4x_bench $Q ids.txt 16384 64                          # the NPU beside
bin/q4x_bench $Q ids.txt 512 128 ~/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf   # drafted (or a DFlash2 GGUF)
python tools/power.py --csv qagents.csv -- python tools/serve/agent_bench.py --only osmos --model $Q \
  --dspark <the MTP GGUF> --budget auto --max-tokens 4096 --prompts <Qwen-tokenized prompts>      # agents
python3 tools/bench/rig_replay.py ~/.rig out/ --shard $Q --gguf-py <llama.cpp>/gguf-py \
  --tokenize <llama.cpp>/build-hip/bin/llama-tokenize                                          # prompts from your rig sessions
bin/q4x_ppl $Q ids.txt 8192 2048                                                              # perplexity
```

The bench tools are built with `hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/<tool>.hip -Llib -losmos_hip
-losmos -lhsa-runtime64` (with the NPU: `make NPU=1` first, and `-lxrt_coreutil`). Run each configuration in a fresh
process, and interleave them. Every answer prints a hash of its tokens: the same hash in every run of a configuration
is the guarantee (the NPU's is its own).

## Layout

| | |
| --- | --- |
| `include/osmos/` | public headers, each with its contract in a comment at the end |
| `src/model/`, `src/mem/` | GGUF, weights, Engram; the expert cache and the file readers |
| `src/run/`, `src/q4x/` | DeepSeek's forward and the engine interface (every family's); Qwen's forward |
| `src/ink2/`, `src/hy4/`, `src/lfm3/`, `src/lsa/` | Inkling-2's forward and drafter; Hy4-flash's; osmos1.7b's (LFM3) and its sparse attention |
| `src/serve/` | the kept conversations (osmos-serve's slots) |
| `src/text/`, `src/cpu/` | tokenizers and chat encodings; the CPU lane |
| `kernels/` | HIP kernels and the shared arithmetic contracts (`lane.h`, `quant.h`) |
| `tests/`, `tests/gpu/` | CPU tests; GPU and whole-model tests |
| `oracle/` | reference forwards the numerics are held to (DeepSeek's from Maya; Qwen's against llama.cpp) |
| `tools/` | osmos-api, osmos-serve, the installer and packs manifest, the benches |

## Credits

- [libevt](https://github.com/mrsirg97-rgb/libevt) (MIT, the same author): the event scheduler under the expert reads
  and `osmos-serve --parallel`.
- [Project Maya](https://github.com/mw00/project-maya) (MIT): the reference this is measured against and learned from.
  The DeepSeek reference tools in `oracle/` are copied from it (`oracle/LICENSE.maya`). Several of Maya's AMD paths
  are this author's contributions there: the int8 experts on gfx1151
  ([#643](https://github.com/mw00/project-maya/pull/643)), conversation slots and prompt caching
  ([#356](https://github.com/mw00/project-maya/pull/356)), and libevt as its scheduler
  ([#587](https://github.com/mw00/project-maya/pull/587)). Findings go back to Maya: `MAYA-UPSTREAM-PATCHES.md`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT):
  - the quant formats
  - the IQ grid tables (generated from its `gguf-py`)
  - `qwen4exp`, the reference Qwen's forward is held to
- [DeepSeek](https://github.com/deepseek-ai), [Qwen](https://huggingface.co/Qwen) and
  [Thinking Machines](https://huggingface.co/thinkingmachines) (Inkling-2-Small and Inkling-2-Mini, Apache 2.0): the
  models.
- [Liquid AI](https://huggingface.co/LiquidAI) (LFM3-1.7B, the LFM Open License v1.0): osmos1.7b's base model, and its
  modeling code the reference LFM3's forward is held to (`oracle/lfm3_forward.py`).
- [osmos1.7b](https://huggingface.co/mrsirg97-rgb/osmos1.7b) (the same author): the fine-tune and its LSA maps.
- [PG-19](https://huggingface.co/datasets/deepmind/pg19) (DeepMind, Apache 2.0): the held-out novel osmos1.7b's long
  rotation is measured on.
- [Tencent](https://huggingface.co/tencent) (Hy4-flash and its DFly2 drafter) and its compression team
  [AngelSlim](https://huggingface.co/AngelSlim) (the Q3_K GGUF osmos serves): the model, and the transformers
  modeling code its forward is held to.
- [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0): DFly2's speculator, the reference osmos's drafts are
  checked against token for token.
- [unsloth](https://huggingface.co/unsloth): the quantizations (Inkling's UD-IQ1_S and its importance matrix among
  them), the MXFP8 Engram table and Qwen's MTP head.
- [eaddario/imatrix-calibration](https://huggingface.co/datasets/eaddario/imatrix-calibration) (MIT): the text
  Inkling's picks are counted on.
- [huihui-ai](https://huggingface.co/huihui-ai): the abliteration patches.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): GSQ-RCO, the quant this was first built on (still supported).

## Contributing and license

`CONTRIBUTING.md` covers the process and how numbers are measured. `CHANGELOG.md` covers each release. MIT, see
`LICENSE`; third-party material keeps its own license.
