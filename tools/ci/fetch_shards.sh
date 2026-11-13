#!/bin/bash
# the tests' model shards: each first shard is its model's metadata and tokenizer (DeepSeek's 5.8 MB, Qwen's 10.9 MB),
# fetched at tools/install/packs.json's pinned commit and checked against its SHA-256 - so the tests that read them
# never skip in CI (0.7.1: only DeepSeek's was fetched, every Qwen test skipped). where the tests look by default:
#   ud-iq1_m-1      -> ~/Models/ds1017/UD-IQ1_M/      (tests' MODEL, OSMOS_TEST_SHARD)
#   qwen38-iq3xxs-1 -> ~/Models/qwen38/UD-IQ3_XXS/    (tests' QWEN, OSMOS_TEST_QWEN)
set -eu
cd "$(dirname "$0")/../.."
fetch() {
  read -r repo rev path sha < <(python3 -c 'import json, sys; e = json.load(open("tools/install/packs.json"))["files"][sys.argv[1]]; print(e["repo"], e["revision"], e["path"], e["sha256"])' "$1")
  dst="$2/$(basename "$path")"
  mkdir -p "$2"
  if [ -f "$dst" ] && echo "$sha  $dst" | sha256sum -c --status; then
    echo "$1: there already"
    return
  fi
  curl -fsSL -o "$dst" "https://huggingface.co/$repo/resolve/$rev/$path"
  echo "$sha  $dst" | sha256sum -c
}
fetch ud-iq1_m-1 "$HOME/Models/ds1017/UD-IQ1_M"
fetch qwen38-iq3xxs-1 "$HOME/Models/qwen38/UD-IQ3_XXS"
