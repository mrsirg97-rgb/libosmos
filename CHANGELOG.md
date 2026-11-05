# Changelog

## [0.1.3]: the budget sizing keeps the headroom it promises

The first rig session on 0.1.2 left 0.61 GiB of GPU memory free, under the 1 GiB the sizing promises. Two things
the sizing did not expect: a resident expert takes about 6% more memory than its bytes, so a step could overshoot;
and the memory here moves in steps of about 0.93 GiB, so no budget lands inside a narrow window. The sizing now
measures what a GiB of budget really costs between builds, aims each step with that, keeps a bracket of the largest
budget that left the headroom and the smallest that did not, and never keeps a build short of the headroom when one
that left it is known. On this box the UD-IQ1_S pack sizes to 100.85 GiB with 1.54 GiB free; finding it takes more
builds, so the engine is ready in about 90 s instead of 40. Each sizing build is a line in the log now.

## [0.1.2]: rig thinks at max by default

The rig row the installer writes now carries `"effort": "max"` beside its efforts (`low`, `high`, `max`): rig sends
it as `reasoning_effort`, and osmos-api maps it to DeepSeek's highest reasoning effort (100). rig's `-effort` (or its
session's effort) picks another; osmos-api's own default for a client that names none stays `high` (75).

## [0.1.1]: the installer sets up rig when it is not there

`osmos_install.py --rig` used to write the pack's row into rig's config and assume rig was installed. Now, when no
rig is on PATH or in `~/.local/bin`, it installs it first: `tools/install/packs.json` pins a rig release (its tag and
its `install.sh`'s SHA-256), the installer fetches that exact script, checks the hash before running it - a script
that is not the pinned one is refused and never run - and runs it at the pinned version (rig's script verifies the
binary it downloads against the release's checksums). Then it registers the pack as before. Checked from a fresh
clone on a machine with rig removed: rig v3.5.0 installed, the pack registered as its default, and rig started
osmos-api, sized to the GPU, and answered - a plain prompt and a tool call.

## [0.1.0]: DeepSeek-V4.1-Flash on one Strix Halo, installed in one command, served over an OpenAI API

The first release. osmos runs DeepSeek-V4.1-Flash-1017 (~768B parameters) on a single AMD Strix Halo (Ryzen AI MAX+
395, gfx1151, 128 GB), deterministically: the same input gives the same output, bit for bit, whether it runs alone or
batched with other sequences, at any memory budget, decoded a token at a time or drafted, with an expert on the GPU or
the CPU. Each of those is a test that compares bits.

Two packs, both on unsloth's quants and their shared MXFP8 Engram table, both at a 1M-token context:
`ud-iq1_s` (speed: 18.8 tok/s plain, 37.5 drafted) and `ud-iq1_m` (quality: 16.0 / 28.6). huihui-ai's abliteration
is an opt-in overlay (`--abliterated`), applied exactly at load time. The drafter is DeepSeek's own DSpark, its experts
requantized to IQ2_XS (4.7 GB, down from 8): it and the packs' expert pick profiles are published at
huggingface.co/mrsirg97-rgb/osmos-evo-1017 (MIT) and pinned by commit and SHA-256 in `tools/install/packs.json`.

`tools/install/osmos_install.py ud-iq1_s --serve` checks the machine (the GPU, the GPU memory the kernel lets it map,
the tools, the disk), builds the engine, fetches every file at its pinned commit or uses the ones already on disk
(`--have`), checks every hash, lays out the pack and starts **osmos-api**: an OpenAI-compatible server in C
(`/v1/chat/completions`, streamed or whole, with tool calls and reasoning; `/v1/models`; `/health`), with the model's
own tokenizer and DeepSeek-V4.1's chat encoding byte for byte against their references. `--rig` registers the pack in
[rig](https://github.com/mrsirg97-rgb/rig) and asks before making it the default model.

The engine sizes its expert budget to the GPU it finds (`--budget auto`): it builds itself at its dense weights, then
again with the room left, but a 1 GiB headroom, given to experts, until it fills the machine - on any Halo, context,
number of conversations or drafter. A 1M context costs ~0.85 GiB of attention state: the indexer scores a prompt's
step in chunks, so no buffer grows with the step times the context.

Measured on one Halo against Project Maya 1.1.53, the fastest engine we know of for this model on this hardware, on
the same files (README, DESIGN.md): prompts read 26% faster at 2K, 11-12% at 8K, 6% at 128K; drafted decode 6-8%
faster; plain decode a tie; Maya's 4-agent bench in 20% less time on 13% less energy an answer token.

`osmos-serve` keeps several conversations at once on one engine through [libevt](https://github.com/mrsirg97-rgb/libevt),
a submodule. CI builds and tests what runs without a Halo - the host library under AddressSanitizer, UBSan and
ThreadSanitizer, the tokenizer and chat encoding against their fixtures, osmos-api against a scripted engine, the
installer - and compiles the GPU code for gfx1151 in ROCm's image; the GPU and whole-model tests run on the box.
