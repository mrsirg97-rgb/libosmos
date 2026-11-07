# osmos on Strix Halo

**DeepSeek-V4.1-Flash, a 768B model, and Qwen3.8-Flash-Next on one Ryzen AI MAX+ 395 (gfx1151, 128 GB).**

osmos is an open-source inference engine built for this one chip. It runs DeepSeek-V4.1-Flash at 1–2 bits a weight
with a 1M-token context, streams the experts that do not fit from the SSD, and gives the same output bit for bit
however it runs. Below, osmos 0.5.2 is measured against Project Maya 1.1.56 on the same box and the same model files.

`MIT` · `1,048,576-token context` · `one-command install` · `OpenAI-compatible API` · `stock AMD power profiles`

## Measured against Project Maya 1.1.56

Same machine, same quantized model files, same session (2026-11-06), nothing else running, AMD's `performance`
profile. Maya as its own setup configures it (rebuilt, its pack, its drafter, its CPU lane calibrated, its expert
profile warmed first); osmos 0.5.2 as it ships. Maya's own bench questions and prompts, each engine in a fresh process,
two interleaved rounds, the order swapped in the second. Greedy decoding. Every run and its command:
[`DESIGN.md`](../DESIGN.md).

| | UD-IQ1_M osmos | UD-IQ1_M Maya | | UD-IQ1_S osmos | UD-IQ1_S Maya | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reading a 2,048-token prompt | **300** | 241 | +24% | **327** | 258 | +27% |
| Reading an 8,192-token prompt | **445** | 408 | +9% | **488** | 436 | +12% |
| Writing, with DSpark drafting | **33.5** | 27.6 | +21% | **40.6** | 34.5 | +18% |
| Writing, one token at a time | 17.1 | 17.0 | tie | 19.4 | 19.3 | tie |
| Perplexity (8,191 tokens, lower is better) | **2.4579** | 2.4649 | | **2.5006** | 2.5023 | |

## Agents: the workload this machine is for

| 4 agents × 8 turns, UD-IQ1_M | osmos 0.5.2 | Maya 1.1.56 |
| --- | ---: | ---: |
| Wall time, all 32 turns | **7.3 min** | 9.1 min |
| First token, p95 | **55.5 s** | 100.3 s |
| Turn done, p95 | **89.7 s** | 127.9 s |
| Package power, mean | 126 W | 127 W |
| Energy per answer token | **6.96 J** | 8.81 J |
| Peak GPU memory | **109.7 GiB** | 112.5 GiB |

An agent turn is mostly reading: Maya's agent bench sends about 99K new prompt tokens for 8K tokens of answers. osmos
finishes it 20% sooner on 21% less energy for each token it writes, at the same power, holding ~3 GiB less memory -
and reads its prompts at Maya's accuracy or better. With its prompts read in 8,192-token steps (the next release's
packs), 7.0 minutes and a 32.8 s first-token p95.

## Qwen3.8-Flash-Next: the second model

osmos 0.5.1 runs Qwen3.8-Flash-Next (177B, a preview of the Qwen4 architecture: Gated DeltaNet layers, sparse
attention, 512 experts) from unsloth's UD-IQ3_XXS, every weight in GPU memory, the model's own MTP head drafting, a
262,144-token context (or 1M with YaRN: a fact recalled 355K tokens back) - under the same guarantee: the same output bit for bit however it runs. Against llama.cpp on
the same file, the same day (2026-11-06, `performance`):

| | osmos 0.5.1 | llama.cpp | |
| --- | ---: | ---: | ---: |
| Reading a 2,048-token prompt | **790** | 520 | +52% |
| Reading a 16,384-token prompt | **696** | 463 | +50% |
| Writing, MTP drafting | **40.5** | not measured | |
| Writing, one token at a time | 23.7 | **26.2** | -10% |
| Agents, 4 × 8 turns (0.5.0) | **292 s** | not measured | |

On the agents bench Qwen finishes in 292 s - DeepSeek's 440 s on the same workload - its first token at a 3.0 s
median: each expert read once for every conversation's tokens, and the prompts read at its sparse attention's speed.

## Energy: the low-power profile

The same osmos at the box's `low-power` profile (a stock AMD profile), UD-IQ1_M: **2.35 J a drafted token** at 30.7
tok/s and 71 W - faster than 0.2.0 drafted at `performance` - and **3.79 J an answer token** on the agents bench (481
s at 63 W): 39-46% less energy than at `performance` for 8-9% less speed. 0.3.0's gain came from reading the experts
the GPU streams straight from the SSD into memory it reads (no CPU copy); 0.3.1's and 0.3.2's from less work a token
(drafting floors tuned for energy, the expert kernels' registers), the same bits out.

Power is the APU package's (amdgpu PPT: CPU, GPU and NPU), not the wall, sampled over the bench's own window. In a
live coding session through the rig agent, osmos wrote at about 30–37 tok/s while reusing the conversation's earlier
tokens, estimated from that session's own timestamps.

## One person's stack, chip to agent

| Layer | What it is |
| --- | --- |
| **[rig](https://github.com/mrsirg97-rgb/rig)** | the agent harness: tools, sessions, launches the model |
| **osmos-api** | OpenAI-compatible server in C: the model's own tokenizer and chat encoding, streaming, tool calls |
| **[libevt](https://github.com/mrsirg97-rgb/libevt)** | the event scheduler: several conversations at once on one engine |
| **osmos engine** | HIP kernels for gfx1151, expert streaming from the SSD, a CPU lane that matches the GPU bit for bit, the budget sized to the GPU at start |
| ROCm 7 · amdgpu | AMD's runtime and driver |
| DeepSeek · unsloth | the model (MIT) and its 1–2-bit quantizations |

The bold layers are built for this project, all open source. The stack builds on itself: running on osmos in rig,
DeepSeek wrote the fix for rig 3.5.1 ([rig#221](https://github.com/mrsirg97-rgb/rig/pull/221)), spec and tests
first, in rig's own process.

## What it shows about Strix Halo

- One 128 GB APU runs a 768B-parameter frontier model for real agent work: unified memory lets the GPU hold 113 GiB of
  it and read the rest from an NVMe SSD as needed.
- A 1M-token context fits in about 0.85 GiB, because the attention state stays in the model's own FP8 and FP4 formats.
- It installs in one command on stock settings (only the kernel's GPU-memory limit raised), sizes itself to the GPU it
  finds, and every pinned file is checked by SHA-256.

## What would help

- Early access to the next Halo-class APU, to port and measure osmos on it before launch.
- A ROCm contact for gfx1151 issues found along the way.
- Support to keep the engine, its benchmarks and its installer maintained in the open.

[github.com/mrsirg97-rgb/libosmos](https://github.com/mrsirg97-rgb/libosmos) ·
[huggingface.co/mrsirg97-rgb/osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017) ·
[github.com/mrsirg97-rgb/rig](https://github.com/mrsirg97-rgb/rig)
