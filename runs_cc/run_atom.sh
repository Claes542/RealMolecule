#!/bin/bash
# usage: run_atom.sh "<query>" <logfile> <max_seconds>  -- Atom Simulator one-atom mode; stops at 'CONVERGED one'.
HERE=$(cd "$(dirname "$0")" && pwd); PROF=$(mktemp -d $HERE/chprof.XXXX)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --enable-logging=stderr --v=0 --user-data-dir=$PROF \
  --no-first-run --new-window --window-size=900,700 --disable-renderer-backgrounding --disable-background-timer-throttling \
  "http://127.0.0.1:8765/atom_simulator.html?$1" 2>&1 | grep --line-buffered -E 'CONSOLE|CONVERGED|rror' | grep --line-buffered -oE '"[^"]*"|CONVERGED.*|[Ee]rror.*' > $2 &
T=0; while [ $T -lt $3 ]; do sleep 5; T=$((T+5)); grep -q "CONVERGED one" $2 && break; done
sleep 2; pkill -f "user-data-dir=$PROF"; sleep 1; pkill -9 -f "user-data-dir=$PROF"; rm -rf $PROF
