# oracle

maya's DeepSeek-V4.1 references and dump tools, copied from [Project Maya](https://github.com/mw00/project-maya)
`tools/ds4/` (MIT, `LICENSE.maya`) at the bar's commit (v1.1.36 + #643 + #587). the numerics substrate is held to
(DESIGN.md, "the oracle"); a tool, never the runtime.

- `ref.py`, `ref_gguf.py`: the forward in torch (F32), over the GGUF's own blocks.
- `engram_hash.py`, `exl3_torch.py`, `st.py`: what `ref.py` imports.
- `compare.py`, `dumpcmp.py`: dumps and logits side by side (maya-ds4's record formats, which substrate writes too).

changed here (each marked `# substrate:` in the file):

- `ref_gguf.py`: Engram rows of any type gguf-py decodes (this model's are IQ4_NL), not Q6_K / MXFP8 only.
- `ref_gguf.py`: the embedding rows by their type (this model's are Q5_K), not Q8_0 only.
- `ref_gguf.py`: the Engram gate folded from `q_weight` and `k_weight` where the GGUF keeps them apart (as `ref.py`).

run in `.oracle-venv` (`python3 -m venv .oracle-venv && .oracle-venv/bin/pip install pyyaml && .oracle-venv/bin/pip install torch numpy --index-url
https://download.pytorch.org/whl/cpu`), with `PYTHONPATH=<maya>/third_party/llama.cpp/gguf-py`.
