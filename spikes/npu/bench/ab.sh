#!/bin/bash
# 0.7.0's first A/B: DeepSeek UD-IQ1_M, an 8K prompt + 32 tokens, GPU alone against GPU + NPU (forward_npu, share
# NPU_SHARE), one budget for both (106 GiB: the NPU run's own sizing), low-power; each run under a MemAvailable
# watchdog (4 GiB: every dbench_npu killed), 30 s between
B=~/bench/npu-070; O=~/osmos-070; M=~/Models/osmos/ud-iq1_m; P=~/bench/osmos-prompts
echo "profile $(cat /sys/firmware/acpi/platform_profile), $(date)"
guard() { "$@" & local p=$!; while kill -0 $p 2>/dev/null; do
  [ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -lt 4194304 ] && { pkill -9 -x dbench_npu; echo "watchdog: dbench_npu killed"; }; sleep 1; done; wait $p; }
run() { local tag=$1 r=$2 log=$B/ab_$1_$2.log
  rm -f $B/stop; python3 $B/sample.py $B/ab_$1_$2.smp $B/stop & local sp=$!
  ( cd $O && guard env $( [ $tag != gpu ] && echo NPU=$O/build/npu-ds NPU_SHARE=$tag ) MODEL=$M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf \
      PICKS=$M/osmos-picks.bin MAX_CTX=16384 NO_GUESS=1 GREEDY=1 IDS=$B/ab_$1_$2.ids ./bin/dbench_npu 106 8192 32 $P/warmup.txt $P/p8192.txt > $log 2>&1 )
  touch $B/stop; wait $sp
  echo "R$r $tag | $(python3 $B/window.py $log $B/ab_$1_$2.smp) | $(grep -E 'prompt 8192' $log | sed -E 's/.*\((.*)\)/\1/') | $(grep -E 'tok/s \(' $log | sed -E 's/ \(32 tokens\) stream ([0-9a-f]+).*/ \1/') | $(grep -E 'lent steps' $log | sed 's/^ *so far //') | $(grep -E 'the NPU:' $log | sed 's/^ *so far //')"
  sleep 30
}
for r in ${ROUNDS:-1 2}; do [ -z "$NO_GPU" ] && run gpu $r; for sh in ${SHARES:-0.15}; do run $sh $r; done; done
echo all done
