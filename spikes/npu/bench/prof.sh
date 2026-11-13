#!/bin/bash
# kernel traces (rocprofv3) of the 8K prompt, GPU alone and with the NPU at 0.15, 106 GiB, under the watchdog
B=~/bench/npu-070; O=~/osmos-070; M=~/Models/osmos/ud-iq1_m; P=~/bench/osmos-prompts
echo "profile $(cat /sys/firmware/acpi/platform_profile), $(date)"
guard() { "$@" & local p=$!; while kill -0 $p 2>/dev/null; do
  [ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -lt 4194304 ] && { pkill -9 -x dbench_npu; echo "watchdog: killed"; }; sleep 1; done; wait $p; }
for tag in gpu npu; do
  rm -rf $B/prof_$tag; mkdir -p $B/prof_$tag
  ( cd $O && guard env $( [ $tag = npu ] && echo NPU=$O/build/npu-ds NPU_SHARE=0.15 ) MODEL=$M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf \
      PICKS=$M/osmos-picks.bin MAX_CTX=16384 NO_GUESS=1 GREEDY=1 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv \
      -d $B/prof_$tag -o run -- ./bin/dbench_npu 106 8192 8 $P/warmup.txt $P/p8192.txt > $B/prof_$tag/dbench.log 2>&1 )
  echo "$tag | $(grep -E 'prompt 8192' $B/prof_$tag/dbench.log | sed -E 's/.*\((.*)\)/\1/') | $(grep -E 'the NPU' $B/prof_$tag/dbench.log | sed 's/^ *so far //')"
  sleep 30
done
