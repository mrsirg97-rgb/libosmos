#!/bin/bash
# CI's host and sanitizer jobs (.github/workflows/ci.yml) in a clean ubuntu:24.04 under podman, on a commit as it is
# (git archive + the libevt submodule's files): the operator's rule before a push. the two tokenizer shards fetched as
# CI fetches them (fetch_shards.sh: the first shards alone - a box's whole model folder mounted is not CI's).
# pipefail: a step's failure not hidden by its tail
#   tools/ci/podman.sh <repo worktree> [commit]   (the copy under ${TMPDIR:-/tmp}: not a RAM tmpfs's room for a big one)
set -eu
repo=$1; rev=${2:-HEAD}
w=$(mktemp -d "${TMPDIR:-/tmp}/osmos-ci.XXXX")
trap 'rm -rf "$w"' EXIT
git -C "$repo" archive "$rev" | tar -x -C "$w"
cp -a "$repo/third_party/libevt/." "$w/third_party/libevt/"; rm -f "$w/third_party/libevt/.git"
echo "ci: $(git -C "$repo" rev-parse --short "$rev") in $w"
podman run --rm -v "$w:/src:Z" -w /src docker.io/library/ubuntu:24.04 bash -c '
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git make gcc g++ python3 python3-pip curl ca-certificates > /dev/null
echo "== the tokenizers shards"; tools/ci/fetch_shards.sh
python3 -m pip install --quiet --break-system-packages jinja2==3.1.4 gguf==0.17.1
echo "== tests"; make -j"$(nproc)" tests 2>&1 | tail -15
echo "== osmos-api"
export OSMOS_GGUF_PY="$(python3 -c "import gguf, os; print(os.path.dirname(os.path.dirname(gguf.__file__)))")"
make -j"$(nproc)" api > /dev/null && python3 tests/api/test_api.py 2>&1 | tail -4
echo "== the installer"; python3 tools/install/test_install.py 2>&1 | tail -3
echo "== asan + ubsan"; make clean > /dev/null; make asan 2>&1 | tail -6
echo "== tsan"; make clean > /dev/null; make tsan 2>&1 | tail -6
echo "== ci: all passed"'
