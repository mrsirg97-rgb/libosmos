# osmos on Strix Halo

**DeepSeek-V4.1-Flash, a 768B model, on one Ryzen AI MAX+ 395 (gfx1151, 128 GB).**

osmos is an open-source inference engine built for this one chip. It runs DeepSeek-V4.1-Flash at 1–2 bits a weight
with a 1M-token context, streams the experts that do not fit from the SSD, and gives the same output bit for bit
however it runs. On the same box and the same model files, it is faster than the fastest engine we know of for this
hardware.

`MIT` · `1,048,576-token context` · `one-command install` · `OpenAI-compatible API` · `stock AMD power limits`

## Measured against Project Maya 1.1.53

Same machine, same quantized model files, same day, nothing else running. Maya as its own setup configures it (its
pack, its drafter, its CPU lane calibrated, its expert profile warmed first); osmos as it ships. Two interleaved
rounds of Maya's own bench questions and prompts, each engine in a fresh process. Greedy decoding. Every run and its
command: [`DESIGN.md`](../DESIGN.md).

| tok/s, mean of two rounds | UD-IQ1_M osmos | UD-IQ1_M Maya | | UD-IQ1_S osmos | UD-IQ1_S Maya | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reading a 2,048-token prompt | **298** | 237 | +26% | **330** | 261 | +26% |
| Reading an 8,192-token prompt | **450** | 404 | +11% | **482** | 429 | +12% |
| Writing, with DSpark drafting | **29.7** | 28.1 | +6% | **38.2** | 35.4 | +8% |
| Writing, one token at a time | 16.5 | 16.5 | tie | 19.2 | 19.3 | tie |

## Agents: the workload this machine is for

| 4 agents × 8 turns, UD-IQ1_M | osmos | Maya 1.1.53 |
| --- | ---: | ---: |
| Wall time, all 32 turns | **440 s** | 547 s |
| First token, p95 | **58.3 s** | 90.9 s |
| Turn done, p95 | **100.1 s** | 124.9 s |
| Package power, mean | 134 W | 124 W |
| Energy per answer token | **7.38 J** | 8.44 J |
| A 131,072-token prompt | **259 s** | 274 s |

An agent turn is mostly reading: Maya's agent bench sends about 99K new prompt tokens for 8K tokens of answers. osmos
finishes it 20% sooner and uses 13% less energy for each token it writes, drawing slightly more power for less time.

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
