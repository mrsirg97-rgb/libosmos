#!/bin/bash
# 0.7.0: the NPU's kernels (spikes/npu/fused/fused_ds.py) compiled and checked, each against its model's own weights
# (the driver's run: its correlation printed), into <out>/ds/<sub>/ and <out>/qwen/<sub>/ (final.xclbin, insts.bin -
# npu_open's layout), then a tarball a set - <out>/osmos-npu-<set>-<version>.tar.gz, its members the set's
# directories - and their sha256s: the release assets a pack's npu names (--npu unpacks one into the pack's npu/).
#   tools/npu/build.sh <out> [ds|qwen ...]   (mlir-aie's IRON in ~/npu-venv, its env_setup.sh; the models' GGUFs)
# each kernel compiled in its own IRON cache (NPU_CACHE_HOME), so its files are its own, whatever the cache's names
set -euo pipefail
OUT=$(realpath -m "$1"); shift
SETS=("${@:-ds qwen}"); SETS=(${SETS[@]})
HERE=$(cd "$(dirname "$0")/../.." && pwd)
VER=$(sed -n 's/.*OSMOS_VERSION "\(.*\)".*/\1/p' "$HERE/include/osmos/version.h" 2>/dev/null || echo dev)
# shellcheck disable=SC1090
set +u; source ~/npu-venv/bin/activate && source ~/Projects/mlir-aie/utils/env_setup.sh >/dev/null; set -u
declare -A KERNELS=(
  [ds]="iq2xxs:gateup iq1s:gateup iq2xs:gateup iq2xxs:down iq1s:down iq2xs:down iq2s:down"
  [qwen]="iq3xxs:gateup iq2s:gateup iq4nl:down")
for set in "${SETS[@]}"; do
  for k in ${KERNELS[$set]}; do
    fmt=${k%:*} kind=${k#*:}
    sub=$fmt; [ "$kind" = down ] && sub=$fmt-down
    cache=$OUT/.cache/$set/$sub; rm -rf "$cache"; mkdir -p "$cache" "$OUT/$set/$sub"
    echo "== $set $sub"
    (cd "$HERE/spikes/npu/fused" && NPU_CACHE_HOME=$cache MODEL=$set FMT=$fmt KIND=$kind python3 fused_ds.py) | tail -n 1
    x=$(find "$cache" -name final.xclbin); i=$(find "$cache" -name insts.bin)
    [ "$(echo "$x" | wc -l)" = 1 ] && [ "$(echo "$i" | wc -l)" = 1 ] && [ -n "$x" ] || { echo "$sub: not one kernel in its cache"; exit 1; }
    cp "$x" "$i" "$OUT/$set/$sub/"
  done
done
rm -rf "$OUT/.cache"
for set in "${SETS[@]}"; do
  a=osmos-npu-$set-$VER.tar.gz
  tar -C "$OUT/$set" --sort=name --owner=0 --group=0 --numeric-owner --mtime=@0 -cf - . | gzip -n > "$OUT/$a"
  (cd "$OUT" && sha256sum "$a" | tee "$a.sha256" && stat -c '%s bytes' "$a")
done
