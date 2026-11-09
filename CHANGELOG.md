# Changelog

## [0.5.12]: (in progress)

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
