#!/bin/bash
# usage: run.sh "<query>" <logfile> <max_seconds> [page]   -- stops at DUMPED; a finished run is not redone.
# Waits while the screen is LOCKED (WebGPU stalls behind a lock); relaunches if the step counter is still for 600 s.
grep -q -E "DUMPED|DUMP failed" "$2" 2>/dev/null && exit 0
PAGE=${4:-h2_beta.html}; HERE=$(cd "$(dirname "$0")" && pwd)
locked() { ioreg -n Root -d1 -a | grep -A1 CGSSessionScreenIsLocked | tail -1 | grep -q true; }
T=0
while [ $T -lt $3 ]; do
  while locked; do sleep 60; done
  PROF=$(mktemp -d $HERE/chprof.XXXX)
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --enable-unsafe-webgpu \
    --enable-logging=stderr --v=0 --user-data-dir=$PROF --no-first-run --new-window --window-size=1100,800 \
    "http://127.0.0.1:8765/$PAGE?$1" 2>&1 | grep --line-buffered -E 'CONSOLE|CONVERGED|DUMP|rror' | grep --line-buffered -oE '"[^"]*"|CONVERGED.*|[Ee]rror.*' > $2 &
  last=""; still=0; done=0; stalled=0
  while [ $T -lt $3 ]; do
    sleep 5; T=$((T+5))
    if grep -q -E "DUMPED|DUMP failed" $2; then done=1; break; fi
    s=$(grep -o 'step=[0-9]*' $2 | tail -1)
    if [ "$s" = "$last" ]; then still=$((still+5)); else still=0; last=$s; fi
    if [ $still -ge 600 ]; then stalled=1; break; fi
  done
  pkill -f "user-data-dir=$PROF"; sleep 2; pkill -9 -f "user-data-dir=$PROF"; rm -rf $PROF
  [ $done = 1 ] && break
  [ $stalled = 1 ] && { echo "STALL at $s, relaunching" >> $2.stalls; continue; }
done
