#!/bin/bash
# CH4 with the s-NODE in 3D (molecule.js USER_SNODE + frozen-core K, 2026-11-24). Needs the page server on 8765 and the dump
# server on 8767. C: +4 pseudo with r_c = HF 2s node 0.380, frozen hydrogenic 1s^2 zeta 5.6727 (zval 4), four 2hom C-H pairs
# over bare protons, sectors imposed. h 0.05, box 10 (grid 200: the node sphere is ~7.6 cells; at h 0.1 it is under 4).
# 1) C atom (mode=c) vs -(IE1..IE4) = -5.4398;  2) CH4 Td at 2.054 and 1.954 (C-H force);  3) square-planar at 2.054.
R=/Users/cgjoh/Development/H2O/runs_cc; D=${DUMPDIR:-/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps}
C="h=0.05&box=10&nmax=200&inner=1&win=2000&conv=0.00003&blog=120&reset&rcX=0.380&el=C&snode=1&fzeta=5.6727&fzval=4"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 40000
  echo "== $2 solver $(grep -E 'CONVERGED [a-z]' $R/$2.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$2.log) err $(grep -c 's-node ON' $R/$2.log) snode"
  python3 $R/forces.py $R/$2.log 2>&1 | tail -1
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $2 $D 2>&1 | grep -E "E indep" | cut -c1-90); }
run "mode=c" sn_c
run "mode=ch4&Hbare=1&bpair=1&R=2.054" sn_ch4_td
run "mode=ch4&Hbare=1&bpair=1&R=1.954" sn_ch4_td1954
run "mode=ch4&planar=1&Hbare=1&bpair=1&R=2.054" sn_ch4_sq
echo CH4SNODEDONE
