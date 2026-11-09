# osmos on Strix Halo

**DeepSeek-V4.1-Flash, a 768B model, and Qwen3.8-Flash-Next on one Ryzen AI MAX+ 395 (gfx1151, 128 GB).**

osmos is an open-source inference engine built for this one chip:
- It runs DeepSeek-V4.1-Flash at 1-2 bits a weight with a 1M-token context.
- The experts that don't fit in memory are streamed from the SSD.
- It gives the same output bit for bit however it runs.

Below, osmos 0.5.9 is measured against Project Maya 1.1.56 and llama.cpp on the same box and the same model files.

`MIT` · `1,048,576-token context` · `one-command install` · `OpenAI-compatible API` · `stock AMD power profiles`

## Measured against Project Maya 1.1.56

The test conditions:
- one session (2026-11-08), nothing else running, AMD's `performance` profile, greedy decoding
- the same quantized model files for both engines
- Maya as its own setup configures it: rebuilt, its pack, its drafter, its CPU lane and expert profile calibrated
- Maya's own bench questions and prompts at 64K context
- each engine in a fresh process, two interleaved rounds with the order swapped in the second

Every run and its command: [`DESIGN.md`](../DESIGN.md).

| | UD-IQ2_XXS osmos | Maya | | UD-IQ1_M osmos | Maya | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Writing, with drafting | **26.5** | 12.0 | 2.2x | **35.4** | 27.4 | +29% |
| Writing, one token at a time | **14.4** | 10.3 | +39% | **17.5** | 16.5 | +6% |
| Reading a 2,048-token prompt | **316** | 261 | +21% | **373** | 242 | +54% |
| Reading an 8,192-token prompt | 429 | 433 | tie | **519** | 412 | +26% |
| SSD read per token written | **0.067 GB** | ~0.17 GB | 2.5x less | | | |
| Perplexity (8,191 tokens, lower is better) | **2.4492** | 2.4512 | | **2.4579** | 2.4621 | |

UD-IQ2_XXS is the quant that fits memory the least: its experts are 157 GB against a ~107 GiB budget. There,
osmos reads 2.5x fewer bytes from the SSD for each token it writes, because every expert lives in one cache that keeps
the ones decode comes back to. That's where its lead comes from. UD-IQ1_S, the smallest quant, is closest to a tie
on plain decode (19.5 against 19.3) and still +23% drafted.

## Agents: the workload this machine is for

| 4 agents × 8 turns, UD-IQ1_M | osmos 0.5.9 | Maya 1.1.56 |
| --- | ---: | ---: |
| Wall time, all 32 turns | **6.7 min** | 9.2 min |
| First token, p95 | **53.8 s** | 73.3 s |
| Turn done, p95 | **83.3 s** | 124.5 s |
| Package power, mean | **104 W** | 123 W |
| Energy per answer token | **7.03 J** | 8.87 J |
| RAM left free, lowest | **8.2 GiB** | 2.8 GiB |

An agent turn is mostly reading: Maya's agent bench sends about 99K new prompt tokens for 8K tokens of answers. osmos
finishes it 26% sooner, using 21% less energy for each token it writes and less power. It also leaves 8 GiB of RAM
to the rest of the system by design.

## Qwen3.8-Flash-Next: the second model

osmos runs Qwen3.8-Flash-Next (177B, a preview of the Qwen4 architecture: Gated DeltaNet layers, sparse attention,
512 experts) from unsloth's UD-IQ3_XXS:
- every weight in GPU memory
- the model's own MTP head drafting
- a 262,144-token context, or 1M with YaRN
- the same bit-for-bit guarantee

Against llama.cpp 7d7f028 on the same file, the same session:

| | osmos 0.5.9 | llama.cpp | |
| --- | ---: | ---: | ---: |
| Reading a 2,048-token prompt | **873** | 569 | +53% |
| Reading a 16,384-token prompt | **811** | 521 | +56% |
| Reading a 65,536-token prompt | **709** | 373 | 1.9x |
| Writing, MTP drafting | **45.6** | not measured | |
| Writing, one token at a time | **27.1** | 25.7 | +5% |
| Writing at 16K context | **25.8** | 22.5 | +15% |
| Perplexity, 16,384 tokens | 2.2732 | 2.2693 (f16 cache) / 2.2718 (q8_0) | |

Power is the APU package's (amdgpu PPT: CPU, GPU and NPU), not the wall, sampled over each run's own window.

## One person's stack, chip to agent

| Layer | What it is |
| --- | --- |
| **[rig](https://github.com/mrsirg97-rgb/rig)** | the agent harness: tools, sessions, launches the model |
| **osmos-api** | an OpenAI-compatible server in C: the model's own tokenizer and chat encoding, streaming, tool calls |
| **[libevt](https://github.com/mrsirg97-rgb/libevt)** | the event scheduler: several conversations at once on one engine |
| **osmos engine** | HIP kernels for gfx1151, one expert cache streaming from the SSD, a CPU lane that matches the GPU bit for bit, sized to the GPU and RAM it finds |
| ROCm 7 · amdgpu | AMD's runtime and driver |
| DeepSeek · Qwen · unsloth | the models and their quantizations |

The bold layers are built for this project, all open source. The stack builds on itself: running on osmos in rig,
DeepSeek wrote the fix for rig 3.5.1 ([rig#221](https://github.com/mrsirg97-rgb/rig/pull/221)), spec and tests
first, in rig's own process.

## What it shows about Strix Halo

- **A frontier model on one APU.** One 128 GB APU runs a 768B-parameter model for real agent work. Unified memory
  lets the GPU hold 113 GiB of it, and the rest is read from an NVMe SSD as needed.
- **A 1M-token context in under 1 GiB.** It fits in about 0.85 GiB, because the attention state stays in the model's
  own FP8 and FP4 formats.
- **Stock settings.** It installs in one command, with only the kernel's GPU-memory limit raised. It sizes itself
  to the GPU it finds, and every pinned file is checked by SHA-256.

## What would help

- Early access to the next Halo-class APU, to port and measure osmos on it before launch.
- A ROCm contact for gfx1151 issues found along the way.
- Support to keep the engine, its benchmarks and its installer maintained in the open.

[github.com/mrsirg97-rgb/libosmos](https://github.com/mrsirg97-rgb/libosmos) ·
[huggingface.co/mrsirg97-rgb/osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017) ·
[github.com/mrsirg97-rgb/rig](https://github.com/mrsirg97-rgb/rig)
