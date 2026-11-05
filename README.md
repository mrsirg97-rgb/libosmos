# libosmos

A deterministic inference engine for **DeepSeek-V4.1-Flash** (~768B parameters: ~557B in routed experts, ~196B in
Engram tables) on a **single AMD Strix Halo** (Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB).

It runs these quantizations of the model:

- **unsloth's UD-IQ1_M and UD-IQ1_S** ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF),
  5 shards each): experts in 1-2-bit IQ types, attention Q6_K. The two packs below run both with unsloth's lossless
  MXFP8 Engram table (one 203 GB file the two share) and, optionally, huihui-ai's abliteration patch. Against Maya
  each engine reads the same files (UD-IQ1_M with its stock IQ4_NL table, UD-IQ1_S with the MXFP8 one).
- **ISTA-DASLab's GSQ-RCO IQ2_XXS** ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)):
  experts IQ2_XXS / IQ2_XS / IQ1_M, its Engram a shared file linked as shard 6 (llama.cpp's layout since 2026-10-27;
  their earlier four-shard layout too). What the engine was first built on; measured below, not one of the packs.

**The guarantee:** the same input gives the same output, bit for bit, however it is run:

- alone or batched with other tokens or sequences,
- at any memory budget, whichever experts happen to be resident in GPU memory,
- decoded one token at a time or drafted (DSpark speculative decoding) and verified,
- with an expert computed on the GPU or on the CPU (the CPU lane's AVX-512 kernels give the GPU's exact bits).

Each of these is a test that compares bits, not tolerances: fingerprints of every output stream must match across
the configurations (`tests/gpu/test_lane_gpu`, `test_drafter_gpu`, `test_model_gpu`, `test_forward_gpu`), and the
kernels are checked against reference replays with deliberate mutations that must fail them.

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

Both packs run one context length: **1,048,576 tokens**, with up to 256K tokens a reply (its reasoning and answer).

## Performance

The short version, for people who want one page: [`docs/STRIX-HALO.md`](docs/STRIX-HALO.md).

One Strix Halo (128 GB, ~113 GiB GPU memory), 2026-11-04, nothing else running. The comparison is
[Project Maya](https://github.com/mw00/project-maya) **1.1.53** on the same model files: Maya as its setup configures it (its pack, its own 8 GB DSpark, its CPU lane
calibrated, its expert profile warmed on this machine first), osmos as it ships (the budget sized to the GPU,
`--budget auto`; the router's guess off; our 4.7 GB IQ2_XS DSpark). Greedy decoding throughout. Every run and its
command: `DESIGN.md`.

### Agents and long prompts

What an agent waits on is mostly reading: Maya's agent bench (4 agents at once x 8 turns, seed 1, 131K context,
DSpark on) sends ~99K new prompt tokens against ~8K answer tokens. UD-IQ1_M; power is the APU package's (amdgpu PPT:
CPU + GPU + NPU, not the wall), over the bench's own window, AMD's stock limits. The 128K prompt: real text, osmos at
its 1M context, Maya at the 139,264 the prompt needs (the setting kindest to it); both then write the same 8 tokens:

| | osmos | Maya 1.1.53 | |
| --- | --- | --- | --- |
| Wall, all 32 turns | **440 s** | 547 s | 20% less |
| First token p50 / p95 | **5.8 / 58.3 s** | 6.6 / 90.9 s | p95 36% lower |
| Turn done p50 / p95 | **44.1 / 100.1 s** | 55.9 / 124.9 s | 21% / 20% lower |
| Package power, mean | 134 W | 124 W | |
| Energy an answer token | **7.38 J** | 8.44 J | 13% less |
| A 131,072-token prompt read (one run each) | **259 s** (506 tok/s) | 274 s (479 tok/s) | 5% less |

### Both packs against Maya 1.1.53

64K context (Maya's configuration), 256 tokens an answer to Maya's three bench questions and its 2K / 8K prompts,
each engine in a fresh process, two interleaved rounds; tok/s, the mean (the range):

| | UD-IQ1_M osmos | UD-IQ1_M Maya | UD-IQ1_S osmos | UD-IQ1_S Maya |
| --- | --- | --- | --- | --- |
| Prompt, 2,048 tokens | **298** (296-300) | 237 (231-243) | **330** (326-333) | 261 (259-262) |
| Prompt, 8,192 tokens | **450** (443-456) | 404 (399-409) | **482** (477-486) | 429 (427-431) |
| Decode, DSpark drafted | **29.7** (22.9-35.7) | 28.1 (23.7-32.4) | **38.2** (28.0-49.1) | 35.4 (27.6-44.2) |
| Decode, plain | 16.5 (16.4-16.6) | 16.5 (15.9-16.8) | 19.2 (19.0-19.3) | 19.3 (18.8-20.0) |
| Output reproducible bit for bit | yes | not measured (1.1.46: no) | yes | not measured |

Plain decode is a tie; reading prompts, drafting and several conversations at once are where osmos leads. The sized
budget is part of it: at the hand-picked budgets these runs used before `--budget auto` (110 GiB plain, 100 drafted,
106 prompts) osmos drafted 28.6 (UD-IQ1_M) and 36.5 tok/s (UD-IQ1_S), its prompts 290-297 / 443-452 and 312-313 /
469-482, its plain decode the same. osmos's answers hash the same in every run and configuration; Maya's greedy decode
varied run to run when measured on 1.1.46 (DESIGN.md), not re-measured since.

At the packs' own 1M context (fixed budgets then: plain 106.5, drafted 101.5 / 100.5 GiB), decode is UD-IQ1_S 18.8
plain / 37.5 drafted, UD-IQ1_M 16.0 / 28.6 tok/s: a 1M context's attention state is ~0.85 GiB.

### The two packs: quality and power

64K context, plain at 110 GiB, drafted at 103; perplexity over 8,191 tokens of mixed text and code (`bin/ppl`):

| | UD-IQ1_M (quality) | UD-IQ1_S (speed) |
| --- | --- | --- |
| Perplexity (abliterated) | 2.459 (2.470) | 2.501 (2.505) |
| Package power decoding, plain / drafted | - | 122 / 127 W |
| Energy a token, plain / drafted | - | 6.4 / 3.3 J |
| Expert shards on disk (plus the shared table) | 140 GB | 122 GB |

UD-IQ1_S's experts are 12.5% smaller than UD-IQ1_M's, so more of them stay in GPU memory: it reads fewer from the SSD
and decodes faster; UD-IQ1_M scores better on unsloth's six benchmarks (85.55 against 83.53). The abliterated packs
decode at the same speed. Both run unsloth's MXFP8 Engram table, speed- and quality-neutral against the IQ4_NL one
(DESIGN.md, "the MXFP8 Engram pack"). ISTA-DASLab's GSQ-RCO IQ2_XXS, the quant osmos was first built on, decodes as
fast as UD-IQ1_S but scores worst of the three (perplexity 2.546): measured, not packed.

## How it runs a 250 GB model in 128 GB

- **Experts by residency.** The dense weights and as many experts as the budget holds live in GPU memory, chosen by
  how often a calibration text picks them (a profile beside the model). The rest are read from the GGUF files when
  the router picks them, guessed a layer ahead, and the resident set follows the text (hot experts swapped in).
- **Int8 experts.** A routed expert's input is rounded to int8 a block of 32, and its 1-2-bit weights are summed with
  exact integer dot products (`v_dot4` on the GPU, `vpdpbusd` on the CPU), then scaled in one fixed order - the
  contract in `kernels/lane.h` that makes the CPU and the GPU agree bit for bit.
- **Engram** rows are read from their file at each step's start, every row a reader of its own (~7 KB a token).
- **The attention's state in the model's own formats**: the window's keys FP8, the compressed rows and the indexer's
  keys FP4, as DeepSeek's design keeps them - ~0.9 KB a token, so a 1M context (~0.85 GiB) fits beside a 106 GiB expert budget.
  A packed row read back is exactly the F32 it was rounded to; a prompt's step unpacks the rows once for all its
  tokens.
- **DSpark**, DeepSeek's own drafter, proposes up to 5 tokens a round; one batched forward verifies them.

## Requirements

- AMD Ryzen AI MAX+ 395 (gfx1151) with 128 GB. The CPU lane (optional) needs AVX-512 VNNI and VBMI.
- Linux with ROCm 7.x (developed on HIP 7.15, kernel 7.2, gcc 14).
- GPU memory raised to ~113 GiB: `options ttm pages_limit=29637703` in `/etc/modprobe.d/ttm.conf` (a system setting:
  yours to make).
- The model, on an NVMe SSD (the faster its random reads, the faster the decode), one of:
  - UD-IQ1_M's or UD-IQ1_S's 5 shards (`DeepSeek-V4.1-Flash-1017-UD-IQ1_M-0000N-of-00005.gguf`; shard 5 is the
    Engram table: the stock IQ4_NL one, 110 GB, or the MXFP8 one - below),
  - GSQ-RCO's IQ2_XXS shards: llama.cpp's layout (5 shards, the shared Engram file linked as shard 6) or their earlier
    one (4 shards and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`). Its GGUFs name no DSpark target layers: the
    drafter's `dspark.json` does (`target_layers`, ours carries it),
- for drafting, DSpark's weights: our IQ2_XS requant of Maya's pack (4.7 GB,
  [mrsirg97-rgb/osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)) or Maya's own (8 GB),
- optionally (UD-IQ1_M's decode ~12-13 -> ~16 tok/s with one), an expert pick profile beside the model (`osmos-picks.bin`: every
  pick on calibration text, `bin/dbench` with `PICKS_OUT` writes one).

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

`dbench budget_gib|auto max_tokens n_new warmup q...`: `auto` sizes the budget to the GPU as `osmos-serve` does;
`MODEL` a model's first shard (GSQ-RCO: `MODEL_DIR` and `ENGRAM`),
`PICKS` the profile, `DRAFT` DSpark's directory, `PICKS_OUT` writes a profile from the run. Run each configuration in
a fresh process and interleave them; a first run after the machine has been idle is slower (a cold page cache). Each
answer prints a hash of its tokens: the same hash in every configuration is the guarantee.

## Scope

One model (DeepSeek-V4.1-Flash), one machine (Strix Halo, 128 GB), greedy decoding, one sequence at a time in the
benchmarks. The quant kernels (Q8_0, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ4_NL, BF16, MXFP4, MXFP8, from one
type table, `include/osmos/model/qtype.h`; the int8 expert path and the CPU lane: the IQ1 and IQ2 types) and
the expert streaming are model-independent; the attention, hyper-connections, Engram and DSpark are DeepSeek's.

## Layout

| | |
| --- | --- |
| `include/osmos/` | the public headers, each with its contract in a comment at the end |
| `src/model/` | GGUF, weights, Engram, DSpark index |
| `src/mem/` | the expert pool, the resident set, the file readers |
| `src/run/` | the forward: steps, layers, the lane, drafting |
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

- [libevt](https://github.com/mrsirg97-rgb/libevt) (MIT, the same author): the event scheduler under
  `osmos-serve --parallel` - several conversations at once on one engine, the loop osmos-api's requests run on
  (`third_party/libevt`, a submodule).
- [Project Maya](https://github.com/mw00/project-maya) (MIT): the reference this engine is measured against and
  learned from - the int8 expert approach (`ds4_iq.cu`), the DeepSeek reference tools in `oracle/` (copied, with
  `oracle/LICENSE.maya`), DSpark's packing. Findings from this work go back to Maya: `MAYA-UPSTREAM-PATCHES.md`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the IQ grid tables in `kernels/quant_grids.h` are
  generated from its `gguf-py` (`tools/gen_grids.py`), and the quant formats are ggml's.
- [DeepSeek](https://github.com/deepseek-ai) (MIT): the model, its inference reference, and the Engram hashing that
  `oracle/engram_hash.py` follows.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): the GSQ-RCO quantization this was developed on.
- [unsloth](https://huggingface.co/unsloth): the UD-IQ1_M and UD-IQ1_S quantizations and the MXFP8 Engram table.
- [huihui-ai](https://huggingface.co/huihui-ai): the abliteration patches.

## Contributing and changes

`CONTRIBUTING.md` (the process, the house rules, how numbers are measured); `CHANGELOG.md` (each release).

## License

MIT - see `LICENSE`. Third-party material keeps its own license (`oracle/LICENSE.maya`).
