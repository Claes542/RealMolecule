#!/bin/bash
killprof() { for s in TERM KILL; do ps -axo pid,command | grep -F -- "user-data-dir=$1" | grep -v grep | awk '{print $1}' | xargs kill -$s 2>/dev/null; sleep 2; done; }
# usage: run_atom.sh "<query>" <logfile> <max_seconds>  -- Atom Simulator one-atom mode; stops at 'CONVERGED one'.
HERE=$(cd "$(dirname "$0")" && pwd); PROF=$(mktemp -d $HERE/chprof.XXXX)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --enable-logging=stderr --v=0 --user-data-dir=$PROF \
  --no-first-run --new-window --window-size=900,700 --disable-renderer-backgrounding --disable-background-timer-throttling \
  "http://127.0.0.1:8765/atom_simulator.html?$1" 2>&1 | grep --line-buffered -E 'CONSOLE|CONVERGED|rror' | grep --line-buffered -oE '"[^"]*"|CONVERGED.*|[Ee]rror.*' > $2 &
T=0; while [ $T -lt $3 ]; do sleep 5; T=$((T+5)); grep -q "CONVERGED one" $2 && break; done
sleep 2; killprof "$PROF"; rm -rf $PROF
