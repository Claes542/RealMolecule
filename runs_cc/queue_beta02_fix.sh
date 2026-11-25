#!/bin/bash
# Redo the article's constant-beta paragraph (beta 0.2, ?betaiso=1, h0.1) after the 2026-11-24 factor-2 fix in molecule.js
# (the psi update used 2*beta; energies were recomputed at beta). Configs copied from the 2026-11-19/20 queue15/queue24.
# E indep (tools/check_energy.py) + E_surf (surf.py, iso area). H atom has no wall: -0.48208 at h0.1 stands.
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps
while ! grep -q BETADDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bp6drd6vw.output 2>/dev/null; do sleep 60; done
B=0.2
run() { $R/run.sh "$1&beta=$B&betaiso=1&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&dump=$2" $R/$2.log 30000
  echo "== $2 solver $(grep -E 'CONVERGED [a-z]' $R/$2.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$2.log) err"
  python3 $R/forces.py $R/$2.log 2>&1 | tail -2
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $2 $D 2>&1 | grep -E "E indep" | cut -c1-100); python3 $R/surf.py $2 $D $B; }
E6="mode=c2h6&rcX=1.0&el=C&Hbare=1&bpair=1&box=14"
CH4="mode=ch4&rcX=1.0&el=C&Hbare=1&bpair=1&box=12"
for CC in 2.30 2.50 2.901 3.30; do run "$E6&CC=$CC" fx_c2h6z_CC$CC; done
for RR in 1.90 2.054 2.25; do run "$CH4&R=$RR" fx_ch4_R$RR; done
run "$CH4&R=2.054&planar=1" fx_ch4sq_R2.054
run "mode=c&rcX=1.0&el=C&box=12" fx_atomC
run "mode=xh&anchor=1&ZX=3&bb=2,1&rcX=1.08&el=B&Hbare=1&bpair=1&box=12&R=2.329" fx_bh_R2.329
run "mode=x&ZX=3&bb=2,1&rcX=1.08&el=B&box=12" fx_atomB
echo FXDONE
