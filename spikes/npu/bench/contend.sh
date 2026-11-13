#!/bin/bash
# what slows the GPU beside the NPU: DeepSeek's 8K prompt (GPU alone, 90 GiB budget) alone, beside the NPU's gate + up
# kernel looped with its weights in XRT's own buffer (bo), and beside it with them in a 12 GiB HIP-registered region
# read through user pointers (userptr: the expert cache's kind). each under the MemAvailable watchdog, 30 s between
B=~/bench/npu-070; O=~/osmos-070; M=~/Models/osmos/ud-iq1_m; P=~/bench/osmos-prompts
echo "profile $(cat /sys/firmware/acpi/platform_profile), $(date)"
guard() { "$@" & local p=$!; while kill -0 $p 2>/dev/null; do
  [ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -lt 4194304 ] && { pkill -9 -x dbench_npu; pkill -9 -x ds_loop; echo "watchdog: killed"; }; sleep 1; done; wait $p; }
run() { local tag=$1 r=$2 log=$B/ct_$1_$2.log
  if [ $tag != alone ]; then rm -f $B/loop.stop; $O/bin/ds_loop $O/build/npu-ds/iq2xxs $tag 12 $B/loop.stop > $B/ct_loop_$1_$2.log 2>&1 & sleep 15; fi
  ( cd $O && guard env MODEL=$M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf PICKS=$M/osmos-picks.bin MAX_CTX=16384 NO_GUESS=1 GREEDY=1 \
      ./bin/dbench_npu 90 8192 32 $P/warmup.txt $P/p8192.txt > $log 2>&1 )
  [ $tag != alone ] && { touch $B/loop.stop; wait; }
  echo "R$r $tag | $(grep -E 'prompt 8192' $log | sed -E 's/.*\((.*)\)/\1/') | $(grep -E 'tok/s \(' $log | sed -E 's/ \(32 tokens\).*//') | $( [ $tag != alone ] && cat $B/ct_loop_$1_$2.log )"
  sleep 30
}
for r in 1 2; do run alone $r; run bo $r; run userptr $r; done
echo all done
