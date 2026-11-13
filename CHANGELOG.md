# Changelog

## [0.7.2]: IQ1_S's prompt decode, faster

- **IQ1_S's experts decode 6-10% faster in prompt steps**, the same bits. IQ1_S was the slowest expert type a value
  (60 GB/s through mm_group against IQ2_XXS's 115): it took the generic path, a value at a time in F32. A value is
  dl (g + delta) with g of -1, 0 or 1, so a chunk's 8 values are three exact products rounded once - made once, each
  value's bits then copied by a byte permute. In DeepSeek's UD packs and osmos-evo-A (their gate / up). osmos-evo-A
  at `performance`: prompts 373-380 -> 384-385 tok/s at 2K and 604-607 -> 620 at 8K on the GPU (with the NPU 630-635
  -> 644-647); decode unchanged (19.0 / 35.3 tok/s against 18.9 / 35.6 plain / drafted) - it reads the experts
  through other decoders. The README's osmos-evo-A table has these; its others stay 0.7.0's.
- **Measured, not changed**: the NPU at low-power. On Qwen it no longer pays past a 0.05 share; on osmos-evo-A the
  default 0.15 still gives +2.8% on an 8K prompt for the same power. The NPU's own power mode (`xrt-smi --pmode`)
  changes nothing: the package's power, shared by the CPU, GPU and NPU, is what bounds it. A 2-bit IQ1_S grid (4 KB,
  not 16) was tried again beside the new decode and gave nothing: the table is not what holds IQ1_S.

## [0.7.1]: an audit of the runtime - the protocol's edges, the state between requests, the shared facts

- **DeepSeek's answer no longer depends on what came before it.** A turn read whole and the same turn continued after
  its earlier answer gave different answers (in 0.7.0's replay of 12 real agent turns, 4 alike): the earlier answer's
  tokens had been decoded - a decode step computes the routed experts otherwise than a prompt step (the lane's int8
  against mm's f16), and a small prompt step (under 64 tokens) took the decode's math too. Now every prompt step
  takes the prompt's math at any size (`osmos_read`), and osmos-serve keeps each conversation's state at its prompt's
  end: a later turn reads the earlier answer again as part of its prompt instead of continuing on its decoded state,
  and the slot folder keeps prompts' states alone. The cost: a turn re-reads the previous answer (prompt speed, ~600
  tok/s at `performance`). Qwen's decode and prompt steps were already one math - unchanged. States kept by 0.7.0
  are not taken back (their state version raised).
- **Answers a client can always send back.** A DeepSeek answer holding the image placeholder's text (its token, or
  its text in plain pieces) is now refused with its completion attached (#34's path), where before it came back as a
  200 that every later request refused as history. A Qwen tool argument of `1e999` comes back as its text, valid JSON,
  not `Infinity`; and any number past a double's range in a request is a 400 by name - before, `1e400` in a tool's
  schema dropped every tool silently, and in a message read as "out of memory".
- **The stream and the whole answer agree** in two more cases cut by `max_tokens` on Qwen (the whitespace before
  `</think>` and after it).
- **The connection.** A request that sends nothing is closed after 10 s and a stream nobody reads after 60 s, so
  sixteen idle connections no longer lock every client out (`/health` too). A non-streamed request whose client
  leaves is now stopped as tokens land, not only in a half-second pause - before, it ran to its end and the next
  request waited behind it. A refusal sent before the body was read (413, 400, the connection limit's 503) now reaches
  the client instead of a reset. `Content-Length` is one number or a 400. Sampling's other fields (penalties,
  `logit_bias`, `logprobs`, `top_logprobs`, `n` as a float) are refused by name as `temperature` is, their greedy
  values taken.
- **A stop keeps the conversations.** SIGTERM or SIGINT to osmos-api (rig's stop) now QUITs osmos-serve, which sets
  every conversation aside and writes RAM's to the slot folder, then exits; before, the engine was killed and the
  conversations held in RAM were lost on every restart. A conversation too short to keep no longer evicts kept ones
  on its way to being refused.
- **Forged structure.** A Qwen tool call's name or argument key holding `<` or a newline is refused: the template's
  own `>` after `f<|im_end|` made a real `<|im_end|>` - a turn forged. With `preserve_thinking: false`, a user turn of
  tool responses as text is no longer a 400.
- **Text kept whole.** DeepSeek's user text, text parts, tool results and reasoning were cut at a `\u0000`; they are
  rendered whole, as the reference encoder keeps them. Against the reference, also: a tool result after an empty user
  turn is one user turn, a falsy `response_format` renders no section, a numeric tool result is its number, and the
  image placeholder in text parts or tool content is refused as the reference refuses it.
- **The tokenizer** merges in linear-logarithmic time: a 256 KB word took minutes (a 1 MB one, hours of a core: one
  request could pin it), now 0.2-0.3 s - the same ids, checked against llama.cpp on 20,000 fuzzed cases a vocabulary.
- **The runtime.** A warm start whose read fails no longer leaves its experts pinned for good (every later decode
  refused until a reload). A failed DeepSeek step marks its sequence broken instead of leaving its n-gram history
  past its position. A split model's first shard named otherwise than `-00001-of-NNNNN` is refused by name (a short
  name overran a buffer). `dspark.json` dims that wrap 64 bits are refused, and the drafter checks its vectors'
  lengths. Leaks on Qwen's create failures, a 2 GiB offset's int, `split.count`'s type.
- **One source for shared facts.** CI now fetches Qwen's tokenizer shard and the template's renderer, so the Qwen
  tests run there (they all skipped). The bench tools build with `gpu-build`; the KLD tools share numpy's quantile
  (`bench_common.h`) and `kld` reads small steps as the engine does; `q4x_bench`'s prompt time ends at the prompt;
  `dbench`'s warm-up is out of the energy windows. Python reads the ggml types from `qtype.h` (`tools/convert/qtypes.py`;
  the assembler's copy lacked Q5_1 and Q2_0). The NPU's 128-token part is asserted to be the tables' last run. The
  installer looks for hipcc where make runs it. Pack titles no longer carry speeds (the README's are the numbers).

## [0.7.0]: the NPU beside the GPU in long prompts

- **The NPU now helps with prompts, as an opt-in.** Strix Halo's NPU takes a fixed share of a prompt step's
  routed-expert work while the GPU runs the rest: about 2.5% faster prompts on both models (8K prompt at
  `performance`: DeepSeek UD-IQ1_M 626 -> 642.5 tok/s, Qwen UD-IQ3_XXS 930 -> 955). Decode doesn't use it. Every
  run's output is the same, bit for bit, but NPU mode's numbers are its own: its sums round differently than the
  GPU's. Its distance from the model, measured: DeepSeek's KL against GPU-only within the noise on wiki and the
  agent hold-outs (UD-IQ1_M and osmos-evo-A); Qwen's KL against its own GPU run 0.002-0.004 on the hold-outs.
- **How to turn it on:** `osmos_install.py --npu` (it checks for the NPU, XRT and its headers, builds with `NPU=1`,
  and fetches the pack's kernels, a release asset of this tag), or `npu` / `npu_share` in osmos.json, or
  `osmos-serve --npu <dir> [--npu-share F]`. The defaults are each model's best measured share: 0.15 for DeepSeek,
  0.05 for Qwen. Packs: the four DeepSeek packs and `qwen38-ud-iq3_xxs`; `qwen38-ud-q4_k_xl` has no kernels yet.
- **The kernels** decode the GGUF's own IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ4_NL blocks on the NPU's
  cores and read the weights and activations in place (no copies between the GPU and the NPU). `tools/npu/build.sh`
  compiles them with mlir-aie's IRON, checks each against the model's own weights, and makes the release tarballs.
- **Tools:** `NPU=` / `NPU_SHARE=` for dbench, ppl, kld, q4x_bench, q4x_ppl and q4x_kld.

## [0.6.10]: steady DeepSeek prompts, one Qwen context

- **DeepSeek's prompt speed no longer swings.** On the same build, an 8K prompt ran anywhere from 420 to 559 tok/s,
  run to run. The cause was 0.6.9's lent steps: giving the expert cache's tail back to the system and allocating a
  4 GiB block for each long prompt set the kernel compacting memory under the GPU's mappings, and the driver
  stopped the GPU's queues for 0.2-5 s a prompt while it caught up (`/sys/class/kfd/kfd/proc/<pid>/stats_*/
  evicted_ms`). The packs now read prompts 4,096 tokens a step in buffers of their own, as 0.6.8 did at 2,048: at
  `low-power` on UD-IQ1_M, auto-sized, an 8K prompt 548 / 548 tok/s (lent steps: 517 / 550), decode 15.0 / 15.1
  (14.9 / 15.0), 0.2 GiB less expert cache, the GPU stopped for 8 ms a prompt. The lent-step code is gone. Every
  output the same, bit for bit.
- **Qwen ships one context, 512K**, for both of its packs (YaRN, factor 2: its perplexity the native 262K's to the
  digits measured). The separate 1M pack is gone. If you installed `qwen38-ud-iq3_xxs-1m`, install `qwen38-ud-iq3_xxs`.
- **`qwen38-ud-q4_k_xl --abliterated`:** huihui-ai's patch made into an overlay for Q4_K_XL's own shards, as the
  UD-IQ3_XXS pack has had.
- **Tools:** `tools/bench/hostbw.hip` (the GPU's bandwidth on each kind of host memory it can map),
  `q4x_layercmp.hip` (Qwen's residual a layer against llama.cpp's), `ref_vs_ref.py` (one engine's log-probabilities
  scored against another's). `llama_ref` maps its model's weights rather than copying them into pinned memory, which
  a model larger than RAM can't survive (llama.cpp turns mmap off on iGPUs by default since #28160).
- **Not in this release:** scoring Qwen against its BF16 weights. llama.cpp can't run the 330 GB reference on this
  machine at a usable speed (mapped, it computes on one CPU thread; copied, it runs out of memory). It waits for osmos
  to stream Qwen's experts itself (0.6.11).

## [0.6.9]: DeepSeek's long prompts in 8,192-token steps, their memory lent by the expert cache

- **Long prompts 15-20% faster on DeepSeek, decode as fast as before.** The packs now take a prompt 8,192 tokens a
  step, not 2,048: each expert does four times the work a launch. A step that large needs 4 GiB of buffers a decode
  never touches, and holding them always cost the expert cache 4 GiB and drafted decode 6-10%. Now the cache lends
  them while a long prompt runs: it gives up the experts at its end, hands their pages back to the system, and takes
  them back when the answer starts. At `low-power` on UD-IQ1_M: a 64K prompt 553-586 tok/s (471-508 before), an 8K
  prompt 484-521 (459-473), decode within 1-3% of before (the evicted experts read back over the answer's first
  tokens). Every output the same as 0.6.8's, bit for bit.
- **Qwen scored as DeepSeek is:** `tools/bench/q4x_kld` (0.6.6's KL protocol for Qwen) and `tools/bench/llama_ref`
  (a reference's log-probabilities from llama.cpp, for a model too large for osmos's own reference path).

## [0.6.8]: osmos-evo-A, osmos's own DeepSeek pack, is the default

- **`osmos-evo-a`: UD-IQ1_M's quality at about UD-IQ1_S's speed.** UD-IQ1_M with the routed experts' down
  projections of layers 20-39 requantized to Q2_K from DeepSeek's MXFP4 originals, and the attention and head
  requantized to Q5_K by GPTQ; every other tensor UD-IQ1_M's. Against the model's own weights (0.6.6's reference) it is
  level with UD-IQ1_M on agent traces (KL 0.0485 against 0.0479) and closer on Wiki text (0.175 against 0.190); at
  `low-power` it decodes 8% faster than UD-IQ1_M plain and 15% faster drafted (17.9 / 29.7 tok/s), within 2-6% of
  UD-IQ1_S (18.3 / 31.7), and at 64K of context the fastest of the three (16.4 tok/s). Prompts are 5-9% slower than the
  UD packs'. `--abliterated` takes huihui-ai's UD-IQ1_M patch.
- **It ships as a 34 GB delta on UD-IQ1_M** ([osmos-evo-1017](https://huggingface.co/mrsirg97-rgb/osmos-evo-1017)).
  The installer makes the pack from UD-IQ1_M's files and the delta and checks each shard's SHA-256; on btrfs or XFS
  every byte the two share is cloned, so beside UD-IQ1_M it takes almost no disk. `tools/convert/gguf_delta.py` and
  `gguf_assemble.py` make and apply such deltas for any GGUF set.
- **Measured, not kept:** dropping the least-picked 10% of each layer's experts (for a pack wholly in memory) nearly
  doubles the distance from the model (KL 0.177 -> 0.327).

## [0.6.7]: osmos starts where ROCm is not on the system's library path

No engine change: every output the same as 0.6.6's.

- **osmos-serve and the GPU tests find ROCm's libraries where they were built against.** They linked
  `libhsa-runtime64` with no search path, so they started only where ROCm's lib directory is in the system's
  library path (`/opt/rocm` in ld.so.conf). With a vendored ROCm SDK (`ROCM=...`) they built and then failed to
  start: osmos-api's engine tests exited 127 (reported by @ucotman in #34's follow-up - thank you). The binaries now
  carry `$(ROCM)/lib` as their RUNPATH; `LD_LIBRARY_PATH` still overrides it.

## [0.6.6]: a pack scored against the model itself

No engine output changes: the DeepSeek and Qwen paths are as 0.6.5's. What changes is how a quant is judged.

- **The reference: DeepSeek's own weights, run on this box.** The routed experts as deepseek-ai stores them (FP4
  codes, an E8M0 scale every 32: 296 GB, MXFP4 in GGUF, no value changed) and the dense layers BF16 (exactly the
  FP8 originals; the attention's wo_a Q8_0, its one approximation), an overlay on any DeepSeek GGUF. osmos now
  reads MXFP4 from a GGUF file and runs it in the prompt's matmul. A reference pass streams the experts from the
  SSD: 10,240 tokens in 166 s. Built with `tools/convert/ds4_experts_mxfp4.py` and `ds4_ref_dense.py` from the
  repo's files (the dense shard rebuilt byte for byte).
- **The measure: KL divergence over the whole vocabulary, a token at a time** (`tools/bench/kld`), with top-1
  agreement; runs paired against a baseline with standard errors and both halves of the text
  (`kld_compare.py`); where a run's divergence lies, by token kind and chat span (`kld_tokens.py`). An agent
  trace is scored on its assistant tokens alone.
- **The protocol** (DESIGN.md, "0.6.6"): choices on a Wiki text's working rows, its first 2,048 rows scored once
  per final candidate, the agent hold-out only for a release's finalists, anything fitted on calibration text
  apart from all three.
- **The packs' numbers** (README, and `osmos_install.py --list`): KLD on Wiki text / agent traces - UD-IQ2_XXS
  0.126 / 0.036, UD-IQ1_M 0.190 / 0.048, UD-IQ1_S 0.259 / 0.066. UD-IQ2_XXS's agent-trace number matches an
  independent measurement made elsewhere (0.034). Perplexity misranks these packs: on the agent traces UD-IQ1_M's
  is below UD-IQ2_XXS's.
- **Tried, measured, not kept:** a per-layer gain on the routed experts (Local Code's correction for their own
  2-bit quant). Unsloth's gate and up show no bias; their IQ2 downs overshoot 4-7%, but scaling them back changes
  nothing end to end (+0.0007 +- 0.0018) - the overshoot answers a smaller hidden upstream.
- **`osmos_install.py --doctor`**: the machine as osmos will run on it - the GPU and its memory limit, the driver
  and ROCm, the power profile (balanced is slower than the profiles osmos is measured at), the RAM, `/tmp` as RAM,
  direct reads on the models' folder, an SSD under it, a wireless download - each line what to change.

## [0.6.5]: Qwen's experts on the membrane; the membrane on the largest quant, measured

No output changes: every ids hash the same (Qwen UD-IQ3_XXS plain and MTP; DeepSeek UD-IQ1_M and UD-IQ2_XXS
drafted), at low-power and at `performance`.

- **Qwen's experts are a membrane's units.** They had been uploaded as three whole tensors a layer; now 24,576
  units are read with direct I/O into memory the GPU reads in place, and a table of their addresses is written to
  the GPU once - a step asks the membrane nothing. On a machine that holds them all (this one) that is the same
  path as before: decode and prompts level with 0.6.4 (low-power: decode 27.3-27.5 tok/s both, an 8K prompt 801
  both; `performance`: decode 28.6 -> 28.8, MTP 50.4 -> 50.2, a 2K prompt ~1,000 both), and a load about 5 s
  shorter. It is the first half of running the same quant on a 64 GB machine with part of its experts streamed;
  the streamed half is not in this release. `q4x_weights_experts`, `membrane_whole`, `q4x_forward_whole`.
- **A Qwen build now counts the system's RAM, not only the GPU's memory.** With the experts in host memory the old
  check (1 GiB of GPU memory left) no longer bounded RAM. The headroom is now the GPU's free memory or the RAM
  past the 8 GiB a build leaves the system, whichever is less (DeepSeek's rule), and a forward that cannot fit is
  refused before it takes any memory. A build that left the system under 8 GiB is now refused.
- **A weight at the very end of a file is read.** An expert that ends a shard whose size is not a multiple of 4 KiB
  was refused by the pool (its aligned read passed the file's end); such a read now ends at the file's end.
- **DeepSeek keeps a decode step's logits, not a prompt step's.** The engine's buffer held 2,048 rows of 129,280
  (1 GiB) though a server reads a prompt's last row and a decode step's few; it now holds 63. The gigabyte goes
  to resident experts: on UD-IQ2_XXS 2.8% fewer reads waited on, the answers after a prompt about 1% faster
  (13.8 -> 14.0, 21.4 -> 21.5 tok/s at `performance`); UD-IQ1_M level. `ForwardOptions.logit_rows`.
- **Reads ahead can be bounded** (`ahead_limit`, off by default): at most that many bytes of speculative reads in
  flight, so a read that is waited on never queues behind them. With it DeepSeek's routing guess stops costing
  12% on UD-IQ2_XXS (21.2 -> 23.4 tok/s against 24.3 with no guess); a 2-name guess read 1-3% ahead in single
  runs. No pack's settings change: the guess stays off.
- **Measured, written down** (DESIGN.md, "0.6.5"): on UD-IQ2_XXS the cache's rule is within 8% of every online
  policy simulated, the reads are at the SSD's ceiling with no copy, a prompt takes 3-7% of the next answer's
  misses from the residents, and a GiB of room is 2.7% of the misses - the membrane is at or near what this box
  allows, and room is the lever.

Not tested in this release: osmos-serve's own paths on the GPU; UD-Q4_K_XL.

## [0.6.4]: a special token's text the model says is text, and the API's answer the stream's

osmos-api correctness, no engine change: every ids hash the same. Issue #34 (reported by @ucotman with a repro, a
control and the source lines - thank you), then an audit of every path a request and an answer take.

- **`</think>` the model writes in its answer is text** (#34). Asked about osmos-serve's `--answer-after </think>`,
  DeepSeek wrote the flag's argument in plain pieces (` </` `think` `>`, as a client's text reaches it - measured on
  UD-IQ2_XXS), and the parse refused it as the token: a 500, and greedy, every retry the same. The stream and the
  parse now read markers in a shape of the answer where a special token's text spelled in plain pieces is defused;
  what goes out is the text. The token itself where the format has none is still refused, as DeepSeek's reference
  refuses it. Qwen the same.
- **A refused completion goes with its error** (#34): the 500's body and a stream's error event carry `completion`,
  the text the parse refused (the EOS dropped), so a client can show the model what it wrote - a duplicated
  parameter too. Never the answer: no content, no [DONE]. The message names what failed (the model's tool call, or
  its answer).
- **The stream is the answer's text.** DeepSeek's stream trimmed whitespace its parse keeps (before `</think>`, the
  EOS, all of it before a calls block where the parse drops one blank line): a client's history differed from the
  model's - so the prompt cache's reuse broke there each turn. Qwen's trimmed ASCII where its parse strips Python's
  whitespace (U+00A0, U+3000, ...). A run cut by max_tokens in its reasoning streamed the rest as content.
- **A character cut by max_tokens** (an emoji's first byte token) was a 500 "out of memory" - and, streaming, an
  HTTP status line written into the open event stream. JSON strings now write an invalid UTF-8 sequence as U+FFFD
  (Python's "replace"), and an error after the stream opens is an error event.
- **Qwen's tool arguments typed by `anyOf` / `oneOf`** (pydantic's `Optional[str]`): a schema with no `type` let a
  string argument become a number, a boolean, null or an object. Its types are read from `type`, `anyOf` and
  `oneOf`; a parameter with none - or no schema - is a string, as vLLM's Qwen parser reads it.
- **Text parts can't forge a turn.** Qwen joins a message's text parts with nothing between, so `</thi` + `nk>` in
  two parts was the token; parts are defused as one text now.
- **`response_format`**: DeepSeek tells the model the schema as its encoding does (`json_schema`'s schema, `json_object`
  any object), on the first system message; it was ignored. Qwen's template has none: refused by name.
- `max_tokens: 0` is refused (it meant no limit).

## [0.6.3]: Qwen's prompts 4% faster, every bit the same

Every kernel set against a ceiling measured on the box (a pure read of its weights' bytes: 220-239 GB/s; WMMA's
47-49 TFLOP/s), then the cheapest of what that showed. At low-power, UD-IQ3_XXS, interleaved runs; no output
changes: every ids hash the same. DeepSeek untouched.

- **Qwen runs the output head for a prompt's last row alone.** osmos-serve tells a sequence where its prompt ends and
  reads one row of its logits; Qwen had ignored that and computed all of a step's rows (2,048 or 4,096 of them). A
  4,096-token prompt: 822-832 -> 859-860 tok/s at 2,048-token steps, 853-866 -> 896-901 at 4,096 (the pack's).
  `q4x_seq_prompt`, `q4x_kept`; a row not kept gives no logits and no pick. test_q4x_prompt_gpu, test_engine_q4x_gpu.
  One difference: in a step of more than 2,048 tokens, an activation past f16's range in a skipped row's last mixer
  output is no longer found - nothing reads that value, so no token can come of it, but the step is no longer refused.
- **Decode +0.8%** (27.2-27.4 -> 27.4-27.6 tok/s): the norm kernels' loads issued together (buffers `__restrict__`,
  loops unrolled, one division a thread for the f16 copy's place), and the mixer matmuls' silu and gated-mean
  epilogues shared by a block's waves, where one wave had done them all.
- **Measured, written down** (DESIGN.md, "0.6.3"): the table of both models' kernels against their ceilings; the
  mixer's up at a prompt's step taken apart (its gate input's load 21% and already at the read ceiling, the
  320-column matmul 75%); what has no lever at these bits (the mixers' block counts are their shapes').
- q4x_bench tells the prompt's end as the server does (`Q4X_BENCH_ALL_ROWS=1`: as before).

## [0.6.2]: the membrane a module, kernels both families share

Structure, and a first pass over DeepSeek's decode. No output changes: every ids hash the same (DeepSeek UD-IQ1_M,
plain and drafted; Qwen UD-IQ3_XXS).

- **The membrane is a module of its own** (`mem/membrane`, no HIP inside): the two pools, the reads, the holds and
  the visit clock, promotion, the warm start and the guess behind one interface. DeepSeek's forward is its first
  caller, with the same reads. test_membrane (CPU, real files and readers).
- **A failed promotion's unit goes back to cold.** A copy that failed had left the unit pinned with unwritten bytes
  (also so in 0.6.1). Now the membrane refuses every call until the step ends, and `pool_drop` returns the unit.
  test_pool.
- **Kernels both families share.** The launch helpers are one header (`kernels/launch.h`); `mm_tokens` is behind
  both matmul dispatches; `ops_argmax` is both greedy picks; `wave_pick` is both routers' top k (ops_router
  9.3 -> 7.0 us). q4x_gather and q4x_argmax are gone.
- **Qwen's greedy pick refuses a NaN row**, as DeepSeek's does (it had returned an id). **A router's pick is always
  an expert's id**: q4x_route gave 2147483647 on NaN logits. test_argmax_gpu, test_router_gpu,
  test_q4x_overflow_gpu.
- **DeepSeek's decode: 1,263 -> 1,222 launches a token**, -0.18% kernel + gap time. The lane's int8 input is written
  in the FFN's mix (q8_round's bits), and the overflow flag is cleared where found set, not every step.
  test_overflow_gpu: the step after an overflow is accepted, the same token as a clean forward.
- **Measured, nothing to build** (DESIGN.md): DeepSeek's misses are an expert's first pick of the run (96.5%), not
  the cache's rule, and the guess off is right. The token boundary's idle in the kernel traces is rocprofv3's own
  (a launch after a host wait 250-340 us under it, 2.3 us without): a wait that slept to a mark was built, gained
  nothing (Qwen plain 28.18 -> 28.22 tok/s) and was removed.

## [0.6.1]: fail closed - a failed step, a full GPU, a lost GPU

An audit of 0.6.0 (its diff, and the engine's error paths). No output changes: every ids hash the same (UD-IQ3_XXS,
UD-Q4_K_XL, DeepSeek UD-IQ1_M), a 4096-token prompt's kernels within 1% of 0.6.0's.

- **A Qwen sequence after a failed step is refused, not read on.** A step that failed once its GPU work began (a
  GPU call, or the f16 overflow check after the whole step) had already written its DeltaNet and n-gram rings - a
  step of 9 tokens or more every slot - but left the sequence usable: another step, a rollback or a save gave wrong
  logits or a wrong state, silently. Now such a sequence is refused (steps, rollback, save, the drafters) until a
  state is loaded into it or another is made; `q4x_seq_fail` does the same for a caller's batch that failed past it
  (the engine's batches of more than 64 sequences). osmos-serve was not exposed: it makes a new sequence after any
  failure. test_q4x_overflow_gpu.
- **Qwen refuses a build that leaves the GPU under 1 GiB.** The runtime loads a kernel's code at its first launch;
  with no memory left it segfaulted inside hipLaunchKernel (UD-Q4_K_XL with its n-gram table resident, an 8K context
  in 4096-token steps: 0.02 GiB left after its sequence). A forward or sequence that would leave under FIT_HEADROOM
  is now refused with the GiB left and the levers: stream the n-gram table (the pack does: 26.8 GiB left), or a
  smaller max_tokens or max_context. test_q4x_headroom_gpu.
- **osmos-serve exits when a failure left the GPU unusable** (`FATAL <reason>`): after a kernel's fault every call
  fails the same way, and it had stayed up answering ERR to every request. osmos-api then ends each run as "the
  engine exited".
- **DeepSeek releases a failed step's expert units.** A failure in layer 0 had left them held, the next visit adding
  to a list with no bound.
- **Smaller:** DeepSeek's expert-gemm input buffer has the 8-tile tail Qwen's has (a block reads its every tile's
  input: up to ~0.7 MB past the last expert's, discarded, a page fault at worst; guarding the loads instead cost the
  gemms 11-31%); IQ4_XS's run decode in aligned 8-byte loads; mm_x16_rows's refusals documented and tested; a buffer
  reuse's size checked, not assumed; upload errors name what failed; out of memory reported with a reason.

## [0.6.0]: Qwen's prompts faster, every bit the same

At AMD's `performance` profile against 0.5.12 (two interleaved rounds, every output's hash the same): UD-IQ3_XXS's
prompts +9-14% (2K 862-863 -> 956-987 tok/s, 8K 788-822 -> 899), UD-Q4_K_XL's +30-35% (2K 624-633 -> 813-850, 8K
649-652 -> 865-869); decode +1.5-2.5% plain, +3-4% drafted by the MTP head; DeepSeek the same (its speed the
SSD's). Every change below keeps the bits: each has a test against the kernel it replaced, mutation-checked.


- **The experts' prompt-sized matmuls faster, every bit the same.** A step's expert with 17-64 tokens runs 2 row tiles
  a wave (each activation fragment feeding two matrix-core ops, not one), 65-128 tokens as the dense gemm's 64 x 128
  blocks - each in the same sums' order as before (test_mm_gpu: mm_group at 24 / 37 / 64 / 100 / 128 tokens, mm_run's
  bits, every expert type). By type, measured on real experts (low-power): Q4_K +17-42%, Q5_1 +6-42%, IQ4_XS +6-9%,
  IQ4_NL and Q2_0 +16% / +60% at 128 tokens. UD-Q4_K_XL's 512-token prompt 507 -> 566 tok/s, UD-IQ3_XXS's 2K
  780-795 -> 802-804; every ids hash the same. DeepSeek's types measured too: the gemm blocks for its IQ3_XXS
  (+41%), Q2_K (+35%), IQ2_S (+16%) and IQ1_S (+6-10%) experts at 65-128 tokens, Q2_K's 2 row tiles (+7-30%); the
  codebook types' 2 row tiles slower (not taken). Its 2K prompt unchanged (the SSD's, its experts' ~32 tokens a step
  the old kernel's); its answer's hash the same.
- **Q4_K's prompt decode cheaper, every bit the same.** A lane decodes a whole 32-value group of a row (its 6-bit
  scales unpacked once, not once an 8-value chunk), the codes masked 4 at a time, and (d sc) q - dmin m one fma
  ((d sc) q exact in F32: 21 bits, so the fma rounds as the product then the difference did); the 65-128-token
  gemm stages 64 values a barrier, not 32. test_decode_run_gpu: the new decodes decode<Q4_K>'s bits on 2^20 random
  blocks (subnormal and largest scales among them), mutation-checked. Q4_K experts (low-power): 32 tokens -29%,
  64 -19%, 128 -8%, 8-16 -10%; UD-Q4_K_XL's 2K prompt 719-727 -> 737-738 tok/s; both packs' ids hashes the same.
- **The IQ3 pack's expert decodes cheaper, every bit the same.** IQ4_NL / IQ4_XS's 16 values by v_perm_b32 (4 codes
  an instruction pair, not a load a value); IQ3_XXS a group's scale word and d once a lane (decode_hn's runs, the
  token-tile and 128-token kernels giving a lane a whole group); test_decode_run_gpu covers them, mutation-checked.
  IQ3_XXS gate / up -16% at 8-16 tokens, IQ4_NL down -2 .. -8%; UD-IQ3_XXS's 2K prompt 754-789 -> 796-800 tok/s;
  both packs' ids hashes the same.
- **The experts' f16 gather 2.3x faster, every bit the same.** mm_x16_rows (each step's tokens copied into their
  experts' slots as f16) a thread 8 adjacent columns - two 16-byte loads, one 16-byte store, each value x16_put's
  rounding - not a value a thread, its 2-byte stores scattered; a stride that breaks the rows' 16-byte alignment
  refused. test_x16_rows_gpu: a host reference's bytes (padding slots, wider strides, every rounding edge, the
  overflow flag), mutation-checked. A 2048-token step on the IQ3 pack 2.51 -> 1.09 ms a layer (a 4096-token
  prompt's 245 -> 112 ms); both Qwen packs' and DeepSeek's (UD-IQ1_M) hashes the same.
- **The mixers' gated mean in their up's launch, every bit the same.** mm_run_gated_mean: the hyper-connection
  mixer's up computes a 16-row tile of each of the 4 streams in turn (tile's own sums) and keeps sum += xn
  sigmoid(up) in stream order - q4x_hc_gated_mean's expression and order - writing the mean and its f16; the up's
  F32 (10,240 values a token: 84 MB written and read back a mixer at a 2K step) no longer stored, one launch fewer
  a mixer. test_gated_mean_gpu: the two calls' bytes (F32, f16, overflow) at 1-2,048 tokens, mutation-checked.
  A 4096-token prompt's up + gated mean 366 -> 281 ms. With the gather above, at serve's 2048-token steps
  (low-power, three interleaved runs each): UD-IQ3_XXS's 4K prompt 751-756 -> 768-782 tok/s, UD-Q4_K_XL's 681-701
  -> 737-745; decode the same or ~1% up; every ids hash the same.
- **DeltaNet's conv and the MoE's sum faster, every bit the same.** q4x_conv at 4 taps, dilation 1: a thread a
  channel over 8 tokens, x read once into a register window and the weights once (not 4 loads a token); q4x_moe_sum
  4 values a thread, a pick's row one 16-byte load (a width not 4s, or a misaligned row: the old kernel). Each
  value the same products in the same order. test_q4x_conv_sum_gpu: the old kernels' bytes, mutation-checked. A
  4096-token prompt at 2048-token steps (low-power): conv 131 -> 57 ms, moe_sum 166 -> 114 ms, the GPU's time
  5.47 -> 5.27 s; every ids hash the same.
- **F32 matrices' prompt tiles, every bit the same.** A prompt's F32 matvec (Qwen3.8's router, gates, the mixers'
  inject; DeepSeek UD-IQ1_M's router) at 3 rows a wave x 12 tokens a block, not 4 x 8 (the router 1.52 -> 1.27-1.34
  ms at 2,048 tokens), and a matrix of at most 4 rows on a narrow tile - a wave all the rows of its own tokens, not 8
  waves of which 7 summed rows past the end (the inject 520 -> 456 us, the shared expert's gate 68 -> 30). A tile is
  rows' and tokens' own sums: test_mv_f32_wide_gpu, the 4 x 8 tile's bits, mutation-checked. A 4096-token prompt's
  F32 matvecs 406 -> 349 ms; Qwen's and DeepSeek's hashes the same.
- **The mixers' inject in their norm's launch, every bit the same.** q4x_hc_norm_inject: a block a token, its 4
  streams in turn - each combine_norm's (or hc_norm's) update, norm and f16 as before - the normed values kept in
  LDS, and the inject's 4 rows a wave each, lane l's 8-value chunks in order as each stream is done, then mv's tree:
  mv_strided's bits without reading xn back (84 MB a mixer at a 2K step) or the inject's launch. A prompt's step
  alone (past 16 tokens): a small step keeps the inject beside on the second stream (one block a token would be
  decode's 27.4 -> 25.6 tok/s). test_hc_norm_inject_gpu: the two calls' bytes, the inject in place and apart,
  mutation-checked. A 4096-token prompt's norms and injects 479 -> 361 ms; every ids hash the same.
- **The delta rule a head over 4 blocks, every bit the same.** q4x_delta's 48 value heads a block each left 8 of the
  40 CUs two; a head's 128 state columns now over 4 blocks (a column's sums its own, whatever the block).
  test_q4x_delta_split_gpu: the one-block-a-head kernel's outputs and states, mutation-checked. A 2048-token step
  4.2 -> 3.9 ms; a 4096-token prompt's 355 -> 306 ms; decode the same.
- Tried, not taken: the experts on the int8 matrix cores (kernels/mmq.hip, test_mmq_gpu - bit-exact against a host
  copy, but iu8's rate is f16's on gfx1151: even at <= 32 tokens, 64-128 tokens 0.87x / 0.68x) and int4 gate / up
  activations (simulated: twice the noise's per-token error, top-1 5.4% against 2.8%); IQ2_S's decode through
  v_perm_b32 (its grid's 3 values a scale; a prompt-sized bench -10%, the model none: its 1-16-token experts read
  at DRAM's rate) and a cheaper Q6_K matvec (decode's Q6_K runs on mm, not mv); the attention's K and V two blocks
  ahead (no change: its time the score chain, the softmax and the barriers a 16-key block, its 128-key chunks and
  their order fixed by the bits).

## [0.5.12]: the larger Qwen, DFlash2 as an option, a race in the MoE fixed

- **Qwen3.8's UD-Q4_K_XL runs** (unsloth's, 111.3 GB): its expert types Q4_K and Q5_1 read (each bit for bit
  gguf-py's), Q5_K on the experts' matvec too; its weights resident (~82 GB of GPU memory) with the n-gram table
  streamed from the SSD. Its perplexity is llama.cpp's on the same file (the first 512 tokens 6.4340 against 6.4331).
  Its matmul tuned: Q4_K and Q5_1 on the register-A path (each lane decoding its row of the weights straight into
  registers: the same bits, 25-33% faster) and their own shapes (Q4_K's long rows at 32 x 16 x 2 - 16 waves a block,
  never swept before - Q5_1 at 32 x 8 x 2); a 512-token prompt 247 -> 507 tok/s, decode 20.6 -> 23.2 (low-power). On
  this calibration text UD-IQ3_XXS is a little ahead past 2K tokens on both engines (4K: osmos 6.7522 against 6.7730,
  llama.cpp 6.7683 against 6.8126). **A pack: `qwen38-ud-q4_k_xl`** (rig `qwen3.8-flash-next-l`, alias `qwen-l`), its
  n-gram table streamed - for whoever wants the larger quant: on 16 points of real agent work (rig replay, MTP-drafted,
  low-power) 10-11% slower decode and 16-18% slower prompts than UD-IQ3_XXS, 11-24% more energy a token (its dense weights
  all Q8_0: 0.8 GB more read a token). No abliteration overlay for it yet.
- **Qwen's decode: fewer launches, a faster router.** The router's top-k picks by one wave's shuffles instead of a
  block-wide argmax a pick (19.3 -> 13.9 us a token's layer); the mixers' down matmul applies its silu and writes the
  up's activations itself (1,744 -> 1,647 launches a token). Every bit the same (the ids hash of 512 + 128 tokens,
  plain and MTP-drafted). Plain decode 26.88 -> 27.01 tok/s at low-power: a launch here costs ~2 us however little it
  does (measured), so the rest of a token's launches are 0.6.0's to cut.
- **The installer finds the GPU by its PCI ids** (sysfs: AMD's 1002, Strix Halo's Radeon 1586), not by running
  rocminfo - which hung on a box without system ROCm (issue #25's report, @ucotman's suggestion: thank you). Anything
  else is refused, the AMD devices it did find named.
- **Measured, and left as they were** (DESIGN.md): every other matmul shape - the dense Q6_K / Q8_0 of both models and
  UD-IQ3_XXS's experts at 16 waves and 32-value slabs (no gain on the model; Qwen's head at 4 waves +0.7% of a decode
  token, but its rounding error doubled: not taken); HIP graphs for a decode token's launches (1.86 against 2.05 us a
  launch: not worth a captured, updated graph).
- **DFlash2 as Qwen's drafter, an option** (incoai's Qwen3.8-Flash-Next-DFlash2, Apache-2.0; tools/convert/
  dflash2_gguf.py makes its 0.64 GB Q8_0 GGUF). A pack whose `drafter` is that file drafts with it: a block of 7 drafts
  from one drafter pass, fed the target's own layers 5-45. Its drafts are the reference implementation's (511 of 511,
  test_q4x_dflash_gpu) and the answer's bits are the target's whatever drafts. The MTP head stays the default - on real
  agent work it wins. Measured at low-power, 256 tokens decoded each, two rounds: on a chat prompt DFlash2 54.9 tok/s
  and 1.29 J a token against MTP's 45.4 and 1.57 (+21%); on prose 39.4 / 1.97 against 44.0 / 1.73 (-10%); on 16
  points of real rig sessions (tools/bench/rig_replay.py: an agent's turn, its tool calls and results, Qwen's chat
  template) 41.7 / 1.83 against 45.8 / 1.67 (-9%, ahead on 3 of 16).
- **Fixed: Qwen's steps of 9 to 16 tokens were not always the same bits.** The MoE's expert tables (one kernel's
  block) read each scan's total from the last expert's place while that expert's own thread made it exclusive - no
  barrier between. Read after, the totals came up short, and the last expert's slots and entries were cleared as past
  the live ones: the tokens that picked expert 511 got its stale output. It happened at random, a few times in a pass
  of such steps, beside the shared expert on the second stream. Steps of 4, 6 and 8 tokens (decode's and MTP's
  sizes) and prompt-sized steps were not seen to in the runs made, but the race was in the kernel for any step.
  Since 0.5.10. A barrier (`test_q4x_forward_gpu`: the 256 tokens as steps of 9, each one's rows the
  one batch's bits - 3 of 3 runs failed without it). Found building DFlash2's rollback test.
- **tools/bench/energy.py takes a window shorter than the power samples' spacing** (a 43-token prompt's quarter
  second): the two samples around it, when they are one spacing apart. A window with none near it is still refused
  (the csv not the run's).
- **Qwen's n-gram table can stay on the SSD** (osmos-serve `--stream-ngram`, `"stream_ngram": true` in osmos.json),
  as DeepSeek's Engram does. The table (`per_layer_token_embd`, 26.8 GiB of UD-IQ3_XXS) is read a step's rows at a time
  (16 rows of 90 bytes a token) by 64 readers while the GPU runs the layer before the one that needs them, then
  gathered as the held table's: the same bits (`test_q4x_ngram_gpu`: a 3,000-token prompt and 8 decode steps). The
  GPU memory it took is left to the system: 86.3 -> 59.4 GiB for UD-IQ3_XXS. Measured at low-power (two rounds):
  prompts the same (2K 731-768 tok/s against 748 held, 16K 695-699 against 677), decode 0.5-2.5% slower (26.2-26.3
  against 26.5; after a 2K prompt 25.8-25.9 against 26.4; MTP-drafted 44.5-45.2 against 45.2), the page cache warm or
  cold alike. Off by default for UD-IQ3_XXS; the larger Qwen quant's pack will need it.
- **`--fill`: the RAM osmos keeps free is the user's to set.** 8 GiB by default, as before (room for an editor, a
  browser, language servers and the agent harness beside the model); osmos-serve `--fill GiB`, or `"fill_gib"` in a
  pack's `osmos.json`, sets another, 0 or more. Past what the machine can keep free, the smallest build runs (the
  dense weights, every expert streamed) and a NOTE says so. The saved conversations' RAM stops at the same line.
  Measured on UD-IQ1_M: `--fill 4` sized the cache 4.2-4.6 GiB larger. The output's bits are the same at any value.
- **Measured, and left as they were** (DESIGN.md "0.5.12"): a decode token's memory traffic by kernel (the dense
  weights read exactly once a token, once a round drafted); the expert cache registered coarse-grained (7.5% fewer
  requests into the memory fabric, ~1% faster at best, reverted: not worth a coherence hazard); a verify's rejected
  drafts' experts demoted in the cache (4.5% slower: they are the next round's experts); the drafting floors on
  UD-IQ2_XXS (0.1 and 5 a round still the best).

## [0.5.11]: nothing out of the format reaches a client; measured at low-power

The first outside report (#25): an agent's tool call that DeepSeek's own parser takes was refused, and osmos-api
answered with its markup. Fixing it led to an audit of every path a completion takes to the client, and of the
prompt's tokens. The release is measured at `low-power` (~70 W), where an agent's work costs half the energy.

- **DeepSeek's tool calls parse as DeepSeek's own parser does** (issue #25, reported by @ucotman with a repro, a
  control experiment and numbers from a second gfx1151 - thank you). An invoke with a blank line after its
  name (`name="mcp">\n\n`) is taken, as the reference's regex takes it (Python's `$` matches before one trailing
  newline), and a parameter's `<\n` too. Nothing more: the blank lines and spaces the reference refuses are refused.
- **A turn the parse refuses is an error, never an answer.** osmos-api sent a completion out of the format as its
  text (content, finish "stop"): a refused calls block's markup, a stray special token's text, an empty answer when
  the turn ended before `</think>`. An agent read that as the answer and sent the turn again. Now any turn that ends
  out of the format is a 500 with the parse's reason (a stream's error event, no `[DONE]`), DeepSeek's and Qwen's.
  The stream never sends a special token's text the parse refuses: it stops at it as at a calls block. A completion
  cut by max_tokens is still "length", its text as it is - streamed, now the same text as the whole answer's (the
  bytes held back before a cut were dropped from it).
- **A client's text can't forge the prompt's turns.** The prompt was rendered and then tokenized with every special
  token's text matched, so `<｜User｜>` or `<|im_start|>` in a message or a tool's result (a fetched page quoting a
  chat) became the real token. Now a special token is cut only where the template put one: the request is rendered
  again from its strings defused (one bit of each special token's text flipped, `tokenizer_defuse`), and that
  render's tokens are the prompt's (`tokenizer_encode_shaped`). A call's arguments sent as a JSON string are defused
  as the object the template reads. A prompt with no such text is the same tokens as before.
- **The MTP head's drafts are the same every run.** Their confidences varied run to run in their low bits (the
  output never did, nor the main model's logits: every verify's row 0 the same bits at every position). Two causes:
  - **A race on the pinned token buffer.** The head's seed copies its tokens to the GPU from the buffer the next
    step writes too, and returned without waiting. With the GPU busy, the next draft's token was written over the
    seed's before its copy ran. Now each copy marks its end (`pin_free`), and a writer waits for it first. The wait
    returns at once unless the copy truly hasn't run, so speed is unchanged (low-power, 3 rounds: plain 26.28
    against 26.39 tok/s, drafted 44.43 against 44.29).
  - **Unwritten memory read at position 0.** A conversation's first seeded row took the head's pending h when
    `pend_pos == p - 1`. A new sequence has no pending h (`pend_pos -1`), which at position 0 is `p - 1` too, so
    it read memory never written, in place of zeros.
  - `test_mtp_seed_gpu`: a GPU held before the seed, and a sequence made where another was let go, each the
    reference's drafts bit for bit. Each fix taken out fails its case. Four traced runs of the bench now draft
    the same, round for round.
  - `Q4X_BENCH_TRACE` also prints each verify's row-0 logits hash and the whole verify's.
- **Qwen's small steps keep the GPU busier** (decode, a verify). Same output bits; plain decode +1.8%, drafted +1.4 to
  +2.8% (low-power, 3 interleaved rounds against 0.5.10: 26.76 against 26.30, 44.89-45.53 against 44.28):
  - **Latency-bound work beside the main stream** (the second stream the shared expert already uses): each mixer's
    inject (4 rows), the DeltaNet's alpha, beta and gates, the attention's QSA indexer (its keys and picks). Each
    waited for right before its first reader. Only a matvec's work goes there: an mm would write the f16 buffer
    the main stream's matmuls read (a model whose weights take the mm path keeps them on the main stream).
  - **The greedy pick split across blocks** (`q4x_argmax`): one block a row scanned 248,320 logits in ~160 us. The
    same id on every row (ties, NaNs, a row of -inf: `test_q4x_mtp_ops_gpu`).
  - **F32 and BF16 matvecs at 2 to 8 tokens** keep 2 chunks in flight a lane (1 before): the mixers' inject 13.2
    -> 9.1 us at 4 tokens. The R / U parameters change no bit.
- **The tests' GPU delay was ~20x short** (`q4x_spin` counted the shader clock as 100 MHz): its "2 ms" was ~100 us,
  so the second stream's delay tests could pass without a wait. Now `wall_clock64` at the device's rate. With it,
  each new wait taken out fails `test_q4x_forward_gpu`.
- **Measured at low-power** (2026-11-09, means of interleaved rounds; README), against Maya 1.1.57 (DeepSeek) and
  llama.cpp 7d7f028 (Qwen). Agents (UD-IQ1_M, drafted): 485 s against 653, 3.42 J an answer token against 4.75 at
  56 W, 8.2 GiB of RAM left free against 2.2. UD-IQ2_XXS decode plain 13.5 tok/s against 8.9, drafted 24.8 against
  9.9, 0.064 / 0.082 GB of SSD a token against ~0.17 / ~0.18. Qwen prompts +36 to +40% (1.8x at 64K), decode at 16K
  +19%, MTP-drafted 45.5. Every answer and perplexity the same bits as 0.5.9's. Against 0.5.9 at `performance`,
  the agents bench 20% slower at half the energy an answer token (7.03 J); Qwen's decode the same at both.
- **0.5.9's README had llama.cpp's two Qwen cache types swapped:** f16 is 2.7169 / 2.2718, q8_0 2.7110 / 2.2693
  (llama.cpp's own log lines, K (f16) / K (q8_0)). osmos's 2.2732 at 16K is above both, not between them.
- `test_q4x_overflow_gpu`: a step whose activations pass f16's range (an overlay with every scale f16's largest)
  gives no token, and a clean forward after it the clean token.

## [0.5.10]: a turn doesn't end before its answer

A session on 2026-11-08 hit an empty turn: Qwen3.8 closed its thinking and ended the turn at once, 18 tokens of
reasoning and no reply. osmos decodes greedily, so each of rig's resamples was the same empty turn.

- **No end of turn before the answer has begun** (osmos-serve `--answer-after`, `osmos/text/answer.h`). While a
  turn's tokens since `</think>` are whitespace alone, an end-of-turn token isn't taken: the row's best other token
  is (`osmos_greedy_except`). It's one rule on a row's tokens and logits, applied at every pick (the plain step, a
  drafted round's verify, a `--parallel` round, the first pick after a prompt), so drafting keeps its bits.
  osmos-api passes its dialect's `</think>`, so every pack has it. A prompt in plain mode ends in an empty think
  block, so the rule covers it too.
- `tools/serve/answer_test.py` makes the failure on purpose on real Qwen: the model's natural next token made an
  end. Without the rule the turn ends there; with it the turn goes on, the same tokens one conversation and
  `--parallel`, plain and drafted. It covers the pick after the prompt and mid-decode, and each of the five call
  sites taken out fails it. `tests/test_answer.c`: the rule on a hand-made vocabulary and on both real tokenizers
  (DeepSeek 129 whitespace tokens, Qwen3.8 422).
- **Qwen's rig efforts are its template's names:** `low`, `medium`, `xhigh` (the default). `high` was the
  template's alias for `xhigh`.

## [0.5.9]: Qwen abliterated, ready for another shape; measured against the latest Maya and llama.cpp

Every stream the same bits as 0.5.8 (DeepSeek's and Qwen's).

- **Qwen abliterated** (`--abliterated` on both Qwen packs): huihui-ai's patch for unsloth's Qwen3.8 quants, turned
  into an overlay the engine reads (`tools/install/overlay_q4x.py`: the variants of our types, XOR-decoded, under the
  model's names - 96 tensors, 703 MB). The installer makes it from the patch and its key at their pinned commit and
  checks the result's SHA-256 (a manifest file can now be derived). Perplexity +0.33% (8K) / +0.59% (16K).
  `osmos-api`'s `overlay`, `q4x_bench` / `q4x_ppl`'s `OVERLAY` take it; a conversation kept with it is not loaded
  without it (the state's identity).
- **The Qwen loader refuses what the kernels cannot run**, the key named, instead of a step failing later: an
  attention head other than 256, a DeltaNet state other than 128, an indexer head other than 128 or more than 4 of
  them, more than 16 query heads a key head, a conv past 8 taps, an expert width not whole tiles, n-gram heads not
  the hidden width, an n-gram table in a type its kernel does not decode. Two of these were latent reads past a
  buffer (the indexer's width, the n-gram width) - for a model of other shapes than Qwen3.8's (Qwen4).
- Qwen's decode profiled (DESIGN.md, "0.5.9"): at its bits it is near what this design reads it at - the dense
  matmuls at 170-220 GB/s, the router's 40 us the shared expert beside it; a 3.5% faster expert path declined (a
  decoded token would no longer be a prompt's bits).
- **Measured against the latest of each** (2026-11-08, `performance`, the means of two interleaved rounds; README):
  - **Maya 1.1.56 (7bf15ac, rebuilt and calibrated), DeepSeek at 64K:**
    - Prompts +24 to +54% on UD-IQ1_M and UD-IQ1_S, drafted decode +23 to +29%. Plain decode +6% on UD-IQ1_M, a tie
      on UD-IQ1_S. Perplexity 2.4579 / 2.5006 against Maya's 2.4621 / 2.5065.
    - UD-IQ2_XXS, the quant that fits memory the least: plain decode +39% (14.4 against 10.3), drafted 2.2x (26.5
      against 12.0), 2K prompts +21%, 8K a tie. osmos reads 2.5x fewer SSD bytes a token (1.9x drafted).
      Perplexity 2.4492 against 2.4512.
    - Agents (UD-IQ1_M): 405 s against 551, 7.03 J against 8.87 an answer token, with 8.2 GiB of RAM left free
      against 2.8.
  - **llama.cpp 7d7f028, Qwen3.8:**
    - Prompts +50 to +56% at 2K to 16K, 1.9x at 64K. Plain decode +5%, and +15% at 16K depth.
    - Perplexity between llama.cpp's f16 and q8_0 caches.
    - Qwen's prompts in that session were 873 tok/s at 2K and 709 at 64K, against 0.5.5's 903 and 727 (16K level,
      811 against 810). No regression: 0.5.8 and 0.5.9 measured again afterwards, three interleaved rounds of the same
      command, give 900-908 each. The session read low for a reason not found (it ran right after the DeepSeek
      runs). The llama.cpp table keeps the session's numbers, since llama.cpp was measured under the same conditions.
- **DeepSeek loads read the warm experts once.** The `auto` budget is found by building the forward several times
  (`fit_sized`), and since 0.5.8 each build read the whole warm set from the SSD. On UD-IQ2_XXS that was about 600 GB
  and 125 s at load for a 107 GiB budget. The warm set is now read at the first step, so sizing builds read only the
  dense weights. From start to the first answer: 54 s, against 123-131 s. The same cache at the first step, so the same speed and the same bits
  (`test_warm_gpu`).
- **rig names the packs by model.** `deepseek-1017-ud-s`, `-m` and `-l` (UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS), and
  `qwen3.8-flash-next` (`-1m` for the 1M pack), in place of `osmos-<pack>`. Aliases: `ds-s` / `ds-m` / `ds-l` (the
  three DeepSeek rows no longer share `osmos`), `qwen` / `qwen-1m`. Installing again replaces a 0.5.8 row, keeps its
  port, and moves rig's default if it was on the old row.
- `oracle/qwen_dump` refuses an output directory it can't write. Before, it crashed after computing its perplexity,
  and the number was lost.
- `q4x_bench`: `Q4X_BENCH_TRACE` prints each round's drafts and their probabilities exactly, to compare two runs.

## [0.5.8]: the membrane - one cache for every expert

Every stream the same bits as 0.5.7 (each benchmark's hash, every pack). Against 0.5.7's build the same session
(2026-11-08, `performance`, each pack's 1M context, auto budgets, two interleaved rounds):

| | 0.5.7 | 0.5.8 | |
| --- | ---: | ---: | ---: |
| UD-IQ2_XXS drafted, the three answers | 31.1 / 31.7 s | 29.9 / 29.9 s | -4 to -6% |
| UD-IQ1_M drafted | 23.2 / 23.3 s | 22.3 / 22.3 s | -4% |
| UD-IQ1_S drafted | 19.1 / 19.0 s | 18.7 / 18.7 s | -2% |
| UD-IQ2_XXS, agents' mix (prompts and answers) | 91.2 / 92.1 s | 85.7 / 85.5 s | **-6 to -7%** (8K prompt +11 to +15%) |
| UD-IQ1_M, agents' mix | 65.4 / 65.5 s | 65.4 / 65.6 s | level (answers -2%, prompts -3 to -5%) |
| RAM left free (lowest) | 6.9-7.7 GiB | 8.1-8.9 GiB | the 8 GiB rule kept |

The mix: q1, an 8K document, q2, a 2K document, q3, prompts in 2K steps. At each pack's 1M context: UD-IQ1_S 19.5
plain / 42.7 drafted tok/s, UD-IQ1_M 17.6 / 35.1, UD-IQ2_XXS 14.3 / 26.0 (the installer's titles).

- **Every expert in one cache, by recency** (the membrane). The resident set keeps the dense weights alone; the
  budget's experts are read into the pool at create and stay or go by use - stale first, then the least recently
  released - an expert of the read's size first, its memory reused (the layers' experts come in 5-7 sizes: by
  size, no block has to be found). Today's fixed set and its swaps are gone. The same 128 decoded tokens a second
  time wait on no read (0.5.7: 36).
- **Prompts pass through.** A prompt's step reads into a pool of its own (0.5.7's: two layers' streamed experts), uses
  the cache's experts in place without reordering them (`pool_pass`), and a decode copies an expert it finds there
  into the cache (`pool_place`) - a 2K prompt between two decode steps costs the second none of its experts.
- **A prompt reads ahead only what the next layer picks**: the next router on this layer's input, every token's 6,
  their union (`ahead.h`) - 98.9% of the experts it will read, against a read ahead of every cold expert of the next
  layer, 41% of which were never picked. A prompt is SSD-bound (7.1 GB/s): the 8K prompt reads 132 GB, not 154.
- **The budget leaves 8 GiB of RAM free** (`fit_room`, `FIT_RAM_FREE`): on this APU the GPU's free memory counts the
  pool's registered pages, so 0.5.7's sizing counted the pool twice and still left under 8 GiB. auto budgets are
  0.9-1.8 GiB smaller; the cache's better use of them is the speed above.
- Reads complete into the pool that asked (`reads_fetcher(r, pool)`, `reads_pump`'s pools); `pool_unit_bytes`.
- API: `forward_swaps` and `OsmosStats.swaps` removed (no swaps); osmos-serve's `STATS` line and dbench's lose them.
- Measured and not kept (DESIGN.md, "0.5.8"): admission (2Q: +16% misses; TinyLFU, ARC, LIRS: no better than LRU),
  one LRU for all sizes (the arena's holes: 3-4 evictions a read), prompts borrowing the cache's cold end (by size,
  it took the warm start), a protected share of the warm start for prompts (decode +56% misses at a quarter).
  Belady's optimum, the future known, is 56-62% fewer misses than 0.5.8 - out of reach without knowing it.

## [0.5.7]: stale experts out first

Every stream the same bits as 0.5.6 (each benchmark's hash, every pack, plain and drafted). Measured 2026-11-08,
`performance`, each pack's own 1M context, the three questions x 256 tokens, the old and new builds interleaved over
two rounds:

| | 0.5.6 | 0.5.7 | |
| --- | ---: | ---: | ---: |
| UD-IQ2_XXS drafted, tok/s (the three answers' decode) | 21.2 (36.4 s) | 24.6 (31.3 s) | **+16%** |
| UD-IQ2_XXS plain | 13.7 | 13.8 | the same |
| UD-IQ1_M drafted / plain | 33.1 / 17.2 | 33.9 / 17.1 | +2.4% / the same |
| UD-IQ1_S drafted / plain | 41.8 / 19.5 | 41.7 / 19.3 | the same / -1% (both rounds) |

(0.5.6's column is the old build in the same session; its installer titles, single runs, read lower: UD-IQ2_XXS
drafted 19.3.)

- **Stale experts go first** (`pool_now`). The pool's room went to the expert wanted furthest ahead - Belady's
  choice for a prompt, whose every step visits each layer's experts. A decode's expert not picked again on the next
  token kept its past "wanted at", so it counted as wanted soonest: those piled up, evicted last, while the experts
  just used went first - and a guessed read, which may only take room from experts wanted after it, could take none
  of theirs (28% of UD-IQ2_XXS's refused). An expert past its time is now stale: out before any still wanted, the
  least recently released of them first; a read ahead is stamped with the next layer's visit, not this one's. With
  the guess off, UD-IQ2_XXS drafted waits on 18% fewer reads (10,555 -> 8,673) and decodes 12% faster on this alone.
- **UD-IQ2_XXS's pack turns the guess off** (as the IQ1 packs run): on the new pool the guess no longer pays - its
  top 6 read 14,000 experts a run, 65% never used, their reads in the SSD's queue ahead of the ones waited on; off,
  top 1, 2 and 3 within 2% of each other, top 6 9% behind. The guess stays a pack setting.
- Measured and not kept (DESIGN.md, "0.5.7"): a wider guess (top 10: -40%), an expert-transition table (learned in
  run order: its trusted predictions are experts already resident - no miss caught the router did not), votes across a
  verify's tokens (rank alone does as well).

## [0.5.6]: conversations kept by default, and UD-IQ2_XXS

Every 0.5.5 stream the same bits (DeepSeek's three questions, both packs; Qwen's prompts and decode). Measured the
same session (2026-11-08, `performance`, 64K context, three questions x 256 tokens; perplexity over 8,191 tokens):

| | UD-IQ2_XXS (new, slow) | UD-IQ1_M (quality) | UD-IQ1_S (speed) |
| --- | ---: | ---: | ---: |
| Decode, plain / drafted, tok/s | 14.0 / 21.3 | 17.3 / 33.6 | 19.5 / 40.5 |
| Drafts accepted | 67% | 70% | 72% |
| Energy a token, plain / drafted | 8.39 / 4.76 J | 7.67 / 3.85 J | 7.02 / 3.37 J |
| Perplexity (abliterated) | 2.4492 (2.4571) | 2.4579 | 2.5006 |

Qwen against 0.5.5: prompts 2K / 4K / 16K 905 / 875 / 806 tok/s, decode 27.1 plain and 46.1 with MTP (0.5.5: 27.1 /
45.9).

- **The UD-IQ2_XXS pack** (`ud-iq2_xxs`, the slow one): unsloth's UD-IQ2_XXS with the MXFP8 Engram table, DSpark,
  UD-IQ1_M's pick profile, the guess on, huihui-ai's abliteration overlay optional. The best perplexity of the three
  and the slowest decode: its experts 18% larger than UD-IQ1_M's, so more misses read from the SSD, and a drafted
  round waits on them most (the power falls to 100 W as the GPU idles; the energy a token rises). 0.5.7's work.
  At the packs' own 1M context: UD-IQ2_XXS 13.9 / 19.3 tok/s, UD-IQ1_M 17.2 / 33.2, UD-IQ1_S 19.4 / 41.5 (the
  installer's titles, the IQ1 packs' last taken in 0.5.x's first builds: 16.0 / 28.6, 18.8 / 37.5).
- **Kept conversations on by default** (osmos-serve, so osmos-api and the packs; `OSMOS_SLOT_KEEP=0` turns them
  off). A conversation that loses its lane to another is set aside, then taken back reading only its new tokens -
  the bits of one never set aside (`tools/serve/slot_test.py`, both models). A 12K-token turn taken back: DeepSeek
  3.5 s against 30.5 read from token 0, Qwen 2.8 against 19.5 (low-power). maya's logs of the operator's agents:
  4,533 conversations resumed, most 32K-190K tokens - each a full re-read with slots off.
- **The store holds RAM by bytes, and a conversation past it streams**: RAM available after the build less the 8 GiB
  kept free; a set-aside that fits goes to RAM (the store takes the saved state over, no copy), one that does not is
  written straight to its slot file in 4 MiB pieces, and a take-back reads it back the same way - a state of any size
  through 4 MiB of host memory (`oseq_save_to` / `oseq_load_from`; `OSMOS_SLOT_RAM_GB` lowers the RAM room). With
  DeepSeek loaded the room above 8 GiB is 0-5 GiB, so its long conversations (~320 MB at 190K tokens) would otherwise
  not have been kept. Qwen's state is ~118 MB fixed plus ~14.6 KB a token (2.9 GB at 190K tokens).
- **UD-IQ2_XXS's types on every expert path**: Q2_K (new) and IQ3_XXS's int8 lanes (GPU and CPU), each bit for
  bit against gguf-py or an independent replay of the lane's contract, on random blocks and the quant's own tensors.
  Q2_K is the lane contract's first type with a minimum (`lane_q8_part_min`). The quant loads and answers sanely
  (its numbers in the bake-off, at performance).
- **IQ2_XXS / IQ2_XS f16 decode**: a block whose f16 scale is -0 decoded +0 where gguf-py has -0 (256 of 3.67 M
  values on UD-IQ2_XXS's IQ2_XS; a multiply fused with its rounding). Fixed with the sign folded into the existing
  xor - no barrier: DeepSeek prompts +~1% (377 against ~374 tok/s), every stream the same.
- **The CPU lane runs** (opt-in, off by default): since 0.3.0 it crashed on its first job - it mapped the experts'
  buffered files while the pool reads through their O_DIRECT twins, so no expert's file matched. Either now matches,
  one it never mapped refuses the lane by name, and `test_lane_gpu` runs on the UD quants. Its threads can be pinned
  (`cpu_lane_create`'s cores; dbench's `CPU_CORES`, e.g. `0-1,8-9`: two cores a CCD). Measured on UD-IQ2_XXS and not
  turned on: a miss is still an SSD read wherever it is computed (DESIGN.md, "the CPU, the box's layout").
- `agent_bench.py --lanes N --slots on|off` (more agents than lanes; the bench's slots on the SSD, removed after),
  `slot_test.py --b-file` (Qwen's prompt files overlap: the test refuses texts that share a start).
- Measured and not kept (DESIGN.md, "0.5.6"): the drafter's rows chained on the GPU (+0.2%), Q6_K's wide loader, the
  small tiles' variants; drafts copied from the context (3-5% at most against the drafters: priced by the measured
  verify widths).
- `test_overlay_gpu` now skips on this box (its reference, huihui's patched copy, removed).

## [0.5.5]: faster kernels, the same bits

Every token the same bits as 0.5.4 (each benchmark's hash, both models). Against a 0.5.4 build, the same session
(2026-11-07, `performance`, two rounds):

| | 0.5.4 | 0.5.5 | |
| --- | ---: | ---: | ---: |
| Qwen prompt, 2K / 8K / 16K / 64K tokens | 835 / 744 / 717 / 652 | 903 / 831 / 810 / 727 | +8 to +13% |
| Qwen decode, plain / MTP drafted | 23.5 / 40.1 | 27.1 / 45.9 | +15% / +14% |
| DeepSeek UD-IQ1_M decode, plain | 16.3 | 17.1 | +5% |

- **The prompt's GEMM** (`kernels/mm.h`'s gemm_kernel): a block 64 rows x 256 tokens, each weight slab decoded once
  into LDS and shared by its waves, double-buffered; the K split's partitions walked in turn into a running sum -
  the order of the sums the tile's, so the same bits. Q6_K / Q8_0 prompt matrices +21-40%.
- **A token's A operand in registers**: at a decode token the weights decoded straight into WMMA's operand, no
  LDS - more waves resident, the same chain (Q6_K's long rows, Q8_0, IQ4_NL / IQ4_XS; by type and slab).
- **Qwen: the shared expert on a second stream** beside the routed experts at small steps (decode, a draft's
  verify); a test forces the order so a missing wait cannot pass.
- **Qwen: a step past 2,048 a layer's part at a time** (the cache-friendly 2K sub-batches) and its routed experts over
  the whole step; the experts' activation written straight as the down's f16. The Qwen pack reads prompts in 4K
  steps (+6 to +14% on long prompts, decode the same, +4 GiB).
- Measured and not kept (DESIGN.md, "0.5.5"): a fused gate + up expert kernel, a post-prompt resident-set settle
  for DeepSeek, HIP graphs, wider loads, an Infinity Cache prefetch stream, wave64 kernels (WMMA's bits the same,
  the speed a wash), DeepSeek's shared expert on a second stream; and int8 (its per-block scales cap it below f16
  on RDNA 3.5).

## [0.5.4]: Qwen's long prompts, and what does not help

- **Qwen: the experts' tables placed an expert a block** for a step of more than 256 picks (a prompt's): a 16K prompt
  +2.8% (against a v0.5.3 build, two interleaved rounds: 701 / 649 -> 716 / 673 tok/s); decode and 2K / 4K prompts
  unchanged (decode 23.65 against 23.67 tok/s, a 2K prompt 785 against 786); every hash the same as 0.5.3's.
- **osmos-api: a pack's `max_tokens`** (osmos-serve's `--max-tokens`, a prompt step's tokens), unset by default. The
  DeepSeek packs keep 2K steps: at 4K a 4K prompt +10% and a 16K +18%, drafted answers -1 to -2%; at 8K +8% / +27%,
  but -6 to -10% drafted (the step's buffers ~4 GiB of the experts' budget at 1M context).
- Measured and not kept (DESIGN.md, "0.5.4"): the dense mm with more rows a block, with a decoded slab shared by more
  tokens, and on weights decoded to f16 first (each the same bits, each slower); the matmuls' f16 a step's whole
  (parked: 8K steps level with 2K for Qwen, their activations past the cache).

## [0.5.3]: measured against Maya 1.1.56

- Docs: DeepSeek measured against Project Maya 1.1.56 on the same files and session (README, the one-pager): prompts
  +9 to +27%, drafting +18 to +21%, plain decode a tie, the agents bench 20% less time and 21% less energy at ~3 GiB
  less memory, perplexity at or below Maya's (2.4579 / 2.4649, 2.5006 / 2.5023). The credits name the author's Maya
  pull requests the engine carries (#643, #356, #587).

## [0.5.2]: Qwen at a million tokens

Qwen3.8-Flash-Next's context to **1,048,576 tokens** with YaRN - the model card's static YaRN, factor 4 over the
native 262,144 - as a pack of its own, `qwen38-ud-iq3_xxs-1m` (the same files; its rig row's alias `qwen-1m`). The
262K pack is unchanged: its rotation, and every token, 0.5.1's bits.

- **One rotation** (`q4x_rope`, `kernels/rope.h`): plain to the native context; past it llama.cpp's YaRN (ext 1,
  beta 32 / 1, the pair's magnitude 1 + 0.1 ln(factor)) for every rotation - the attention's, the indexer's, the MTP
  head's. Held to llama.cpp's own at positions 0 and 700,000 (`qwen_dump`'s `QWEN_DUMP_YARN`, `QWEN_DUMP_POS0`).
- **The context's own forward**: its rotation in its identity (a 262K state is not a 1M one's); 1M's cache ~13.6 GB,
  ~93 GB of GPU memory in all.
- `q4x_ppl` and `q4x_bench` take a forward's context (`Q4X_PPL_CTX`, `Q4X_BENCH_CTX`).

Measured (2026-11-06; DESIGN.md, "0.5.2: YaRN to a million positions"): a 394,045-token document's fact recalled
355K tokens back, word for word (the prompt read at 414 tok/s). On short text YaRN costs what the card says it may:
perplexity 8K / 16K 2.7256 / 2.2839 (plain 2.7143 / 2.2732; llama.cpp's YaRN on its Q8_0 cache 2.7196 / 2.2911), a 2K
prompt 745 tok/s (810), drafting 34.9 (40.4: the MTP head's acceptance 59 -> 50%) - the 262K pack the default.

## [0.5.1]: Qwen's prompts, faster

Qwen3.8-Flash-Next's prompts **13% (2K) to 28% (64K) faster**, its decode 2.6%, every token the same bits as 0.5.0
(each benchmark's hash, every length). Profiled with rocprofv3, each change A/B'd at low-power against a v0.5.0 build
(DESIGN.md, "0.5.1: decode and prompts, profiled"):

- **The attention's shared-memory bank conflicts.** Its V^T stores sent a wave's 32 lanes to two banks - a 16-way
  conflict on every store; V's lanes now run along the keys (conflict cycles 31x fewer, the kernel's cycles -45%). Its
  cache read 8 values a load and a block ahead in registers.
- **The DeltaNet's**: a lane's q / k half and its pair's on one bank; one float of padding apart.
- A block's output added to the residual in the next mixer's norm launch (`q4x_hc_combine_norm`): ~100 launches a
  token fewer.
- `test_mm_gpu` holds and times Qwen's mixer matrices.

Measured and not kept: the attention's scores once a 64-key round (slower: half the blocks a CU), its dims split over
the waves (moved the bits for 1.4%), the Q8_0 mixers' K split across blocks (slower than unsplit). DeepSeek
unchanged - its code and its outputs.

At `performance` (2026-11-06; README):

| Qwen3.8 UD-IQ3_XXS | 0.5.0 | 0.5.1 | llama.cpp |
| --- | ---: | ---: | ---: |
| Prompt 2K / 8K / 16K | 700 / 591 / 561 | **790 / 714 / 696** | 520 / 489 / 463 |
| Prompt 64K | 501 | **640** | not measured |
| Decode plain / drafted | 23.1 / 39.5 | **23.7 / 40.5** | 26.2 / not measured |

## [0.5.0]: Qwen3.8-Flash-Next

A second model family behind the engine interface: **Qwen3.8-Flash-Next** (`qwen4exp`, ~177B, 512 experts), unsloth's
UD-IQ3_XXS with the model's MTP head drafting, through osmos-serve, osmos-api and rig. Held as DeepSeek is: each block
against llama.cpp's own tensors, perplexity at or below llama.cpp's at 2K and 8K (16K: +0.05%, within the spread of
its own two caches), and a token's bits the same however it is batched, drafted or shared with other conversations.

- **The forward** (`src/q4x/`, `include/osmos/q4x/`): the Gated DeltaNet (36 layers, their states in rings 8 steps
  back), the gated residual streams, the n-gram embedding, the MoE (10 of 512, a step's experts read once for up to
  128 of its tokens), every matmul on the matrix cores with f16 activations at every token count.
- **QSA**, Qwen's sparse attention: a token's top 512 blocks of 4 keys by its indexer's score - llama.cpp's picks
  exactly; the dense attention's bits below 2,052 positions.
- **The cache in Q8_0** (~13 KB a token): the native **262,144-token** context. YaRN to 1M: 0.5.1.
- **Shared steps and drafts**: a batch's conversations one step, each expert read once for all of them; the MTP head
  drafts every lane at once.
- **Text**: Qwen's tokenizer (llama.cpp's qwen35 pre-tokenizer) and chat template (tools, reasoning), osmos-api's
  dialect by family.
- The engine: `oseq_draft_all` (a family's drafter on several sequences at once; DeepSeek's one at a time, as before).
- **The installer's pack** `qwen38-ud-iq3_xxs`: unsloth's 4 shards and the MTP head pinned at a commit with their
  SHA-256s, laid out with the head as its one GGUF (`drafter_kind: mtp`); each pack's rig row its own (packs.json's
  `rig`: Qwen's 262K window with 64K replies and its efforts low / medium / high, alias `qwen`; DeepSeek's rows as
  before).

Measured (2026-11-06, `performance`; README): prompts 2K / 8K / 16K **672 / 588 / 567 tok/s** (llama.cpp 520 / 489 /
463), 64K 500; decode MTP drafted **~40 tok/s**, plain 23.2 (llama.cpp 26.2); the agents bench (4 x 8) **292 s**
(DeepSeek UD-IQ1_M: 446 s).

DeepSeek unchanged: osmos-serve's token streams the v0.4.0 binary's on 16 requests (the five questions greedy and
drafted, a 2K prompt, a continued turn, four at once - `~/bench/v050-streams.log`); its GPU tests pass; UD-IQ1_M
plain 17.1 tok/s, drafted 33.6 (0.3.2's).

## [0.4.0]: one engine interface

osmos-serve now runs a model through one interface (`osmos/run/engine.h`: `Osmos`, `OsmosSeq`, `osmos_*`, `oseq_*`)
instead of DeepSeek's forward by name - the ground for a second model family (Qwen3.8-Flash-Next, designed in
DESIGN.md). `osmos_open` picks the family by the file's `general.architecture` and refuses any other by name.
DeepSeek-V4.1-Flash is the one family behind it, unchanged: osmos-serve's token streams are 0.3.2's on every request
checked (single and `--parallel`, greedy and drafted, conversation reuse), the agents bench level (447.6 s, 7.07 J an
answer token against 442.5 / 448.7 s, 7.07 J). No model, output or speed changes.

## [0.3.2]: the low-power round

Work cut at AMD's low-power profile, each change A/B'd there (interleaved builds, the answers' hashes compared) and
kept in its own commit only if no worse; then the whole release measured at `performance` against 0.3.0.

- The expert kernels (`kernels/q8.h`) ask for a SIMD's 16 waves (`amdgpu_waves_per_eu`): at 5-6 tokens an expert they
  took 98-107 registers and ran at 12, their memory's latency less hidden. 19 experts at 5 tokens, gate+up: 1,143 ->
  798 us; at 6: 1,543 -> 1,004; down at 6: 909 -> 563. 1-4 tokens unchanged; no bit moves. `tools/bench/q8bench` times
  the kernels on a layer's own experts.
- The drafter's greedy pick runs on the GPU (`ops_argmax`): a draft row's 129,280 logits (517 KB) are no longer copied
  to the host and scanned; the same drafts, ties to the lowest token as before. A row with a NaN now ends the drafts.

Measured and not kept (DESIGN.md): the resident set's swaps off (-39% drafted: they keep the text's experts in GPU
memory), `hc_post_dot` reading its weights once a block of tokens (slower: the re-reads hit the cache), the drafter's
head through `mm` (slower at 5 rows).

At `performance`, against 0.3.0 (0.3.1's drafting floors included), UD-IQ1_M / UD-IQ1_S, 256 tokens an answer to the
three bench questions:

| | 0.3.0 | 0.3.2 |
| --- | ---: | ---: |
| UD-IQ1_M drafted | 32.0 tok/s, 4.10 J | **33.6 tok/s, 3.87 J** |
| UD-IQ1_S drafted | 37.8 tok/s | **40.8 tok/s**, 3.35 J |
| plain (M / S) | 17.1 / 19.3 | 17.1 / 19.3 |
| prompts 2K / 8K (M) | 296 / 443 | 293 / 441 |
| agents (4 x 8, M) | 444.6 s, 7.01 J | 442.5 / 448.7 s, 7.07 J |

0.3.0's agents rerun beside 0.3.2's two, the same evening: level. 0.3.0's 433.2 s that afternoon was the box's state
then, ~2.5% faster (the profile switched twice between).

At low-power, UD-IQ1_M: drafted 29.3 -> 30.7 tok/s, 2.50 -> 2.35 J a token; plain 15.9 / 4.45 J -> 15.8 / 4.49 J;
agents 479.0 s / 3.78 J -> 481.3 s / 3.79 J an answer token. The answers hash as 0.3.0's throughout.

## [0.3.1]: drafting tuned for energy

DSpark's drafts are kept while each confidence is at least 0.1 and their product too (`DRAFTER_CONF_MIN` /
`DRAFTER_CUM_MIN` in `osmos/run/drafter.h`, one setting for osmos-serve and the bench), down from Maya's 0.5 / 0.5.
Every draft is verified by the trunk, so the setting changes only the cost: the answers' hashes are the same at every
value swept. Swept at the low-power profile (UD-IQ1_M, drafted, 0.0 to 0.8): 0.1 the least energy a token - lower
floors let more drafts reach a verify pass that runs anyway, and a floor above 0 still lets the drafter stop a hopeless
row early (0.0, no floor, costs as much as 0.5). Confirmed over two interleaved rounds against 0.5: 29.4 -> 30.65
tok/s, 2.51 -> 2.345 J a token (-6.6%). The agents bench, mostly reading, measured level (3.78 -> 3.84 J an answer
token, 479 -> 483 s: within a run's spread). `dbench` takes `DRAFT_CONF`, `DRAFT_CUM` and `DRAFT_MAX` for sweeps.

## [0.3.0]: less energy a token - the experts read straight into memory the GPU reads

The streamed experts are now read with `O_DIRECT` straight into registered host memory the GPU reads (the SSD's DMA
lands them; no CPU copy), instead of a buffered read into the page cache and a CPU copy into GPU memory. The readers
used ~48% of a core each, twelve of them, on an APU whose CPU and GPU share one package power limit: at that limit the
watts they no longer spend go to the GPU. The pool's memory is one registered arena (`mem/host_slots`, `mem/arena`),
each expert part read as its 4 KiB-aligned range with the kernels reading the slice inside it; a filesystem that
refuses `O_DIRECT` refuses loudly. The host's waits for the GPU spin in decode (short, 40 a step; a wake costs ~0.2 ms)
and sleep in a prompt's steps (long), by the step's size: the device is set to blocking sync at a process's start
(an event's own flag is ignored on ROCm 7.15).

No output changes: the bench's answer hashes are 0.2.0's. Measured on UD-IQ1_M, budget auto, at the box's low-power
profile (power-profiles-daemon's power-saver: a stock AMD profile) and at performance, against 0.2.0 the same day;
energy is the APU package's (amdgpu PPT), not the wall's:

| | 0.2.0 low-power | 0.3.0 low-power | 0.2.0 performance | 0.3.0 performance |
| --- | ---: | ---: | ---: | ---: |
| plain decode | 14.8 tok/s, 4.80 J | **15.9, 4.45 J** | 16.4 | **17.1**, 7.71 J |
| drafted decode | 25.8, 2.83 J | **29.3, 2.50 J** | 29.8 | **32.0**, 4.10 J |
| agents (4 x 8) | 527.2 s, 4.12 J | **479.0 s, 3.78 J** | 450.6 s, 7.25 J | **433.2 s, 6.98 J** |

an answer token's energy for the agents. Prompts at performance about level within the bench's spread (2K 281-300 ->
296, 8K 458-466 -> 443 tok/s), 0.31 -> 0.29 J a prompt token. At low-power osmos now drafts as fast as 0.2.0 did at
performance, at ~71 W against ~125. Direct reads take no page-cache hits: the forward waits on more reads (the
agents' 44,058 against 37,598) - a RAM tier for streamed experts is next.

`tools/bench/energy.py` turns `tools/power.py`'s samples and a bench's windows (`dbench` now prints them) into joules a
token; `OSMOS_STATS=1` prints osmos-serve's read waits at exit.

## [0.2.0]: the forward's reads on libevt

The forward's expert reads now run on [libevt](https://github.com/mrsirg97-rgb/libevt), the scheduler osmos-serve's
loop already used. The readers take the read needed soonest from libevt's priority queue - a miss the forward waits on
first, then reads ahead by when the forward will reach them - and a read ahead that becomes waited on moves to the top
in place (libevt's new `queue_update`). A finished read is an event the forward's thread runs. A miss is split across
the readers unless another miss is out; reads side by side are read whole, keeping the file's readahead. This replaces
a hand-built inbox (a mutex, a condition, an array and a "broken" flag) and two FIFO queues with a `hurry` that moved a
read between them - one scheduler for the project, not two. libevt's sources are built into `libosmos.a`, so every
test and sanitizer run covers them; libevt itself gained `cap_submit` and `queue_push` reporting refusals and an
in-place `queue_update` (c70c58c, 32988ad).

No output changes: which experts are read when is outside the arithmetic; the fingerprint tests and the bench's answer
hashes are unchanged. Measured against 0.1.3 on the same box and sizing (UD-IQ1_M, `performance`, budget auto):
decode level (16.4 plain, 29.8 drafted tok/s), prompts faster (292 -> 300 tok/s at 2K, 444 -> 466 at 8K); the agents bench (4 agents x 8 turns) 450.6 s against 0.1.3's 443.8 and
451.8 s, the forward waiting on reads 11.5% less (85.4 s against 96.5 s), 7.25 J an answer token against 7.30.

The first build held reads back in libevt's `cap` until the forward thread pumped - level alone, but 4% slower with
four agents (461 s), and neither a deeper cap nor letting misses past it fixed that: ordering the reads where the
readers take them did. `OSMOS_STATS=1` makes osmos-serve print the forward's read counters at exit, which found it.

Also measured, not built: a forward-chaining rule on routing transitions (expert e at one layer predicting f at the
next), from a recorded trace (`dbench`'s new `ROUTES_OUT`). It catches ~8% of the next layer's misses against the
router guess's 56%, at 20x the reads: 82% of misses are an expert's first pick in the run, which no learned count can
predict. `test_pool_gpu` runs on UD-IQ1_M again (it had skipped since the original files left the test box).

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

Measured on one Halo against Project Maya 1.1.53, on the same files (README, DESIGN.md): prompts read 26% faster at 2K, 11-12% at 8K, 6% at 128K; drafted decode 6-8%
faster; plain decode a tie; Maya's 4-agent bench in 20% less time on 13% less energy an answer token.

`osmos-serve` keeps several conversations at once on one engine through [libevt](https://github.com/mrsirg97-rgb/libevt),
a submodule. CI builds and tests what runs without a Halo - the host library under AddressSanitizer, UBSan and
ThreadSanitizer, the tokenizer and chat encoding against their fixtures, osmos-api against a scripted engine, the
installer - and compiles the GPU code for gfx1151 in ROCm's image; the GPU and whole-model tests run on the box.
