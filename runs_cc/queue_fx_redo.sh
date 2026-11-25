#!/bin/bash
# Redo ethane 2.30/2.50 (beta 0.2 fixed): their dumps failed because the dump directory was missing. After queue_betad_only.
while ! grep -q BDODONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bs0v5bbwi.output 2>/dev/null; do sleep 60; done
R=/Users/cgjoh/Development/H2O/runs_cc; D=/private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/scratchpad/dumps; B=0.2
for CC in 2.30 2.50; do N=fx_c2h6z_CC$CC
  $R/run.sh "mode=c2h6&rcX=1.0&el=C&Hbare=1&bpair=1&box=14&CC=$CC&beta=$B&betaiso=1&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset&dump=$N" $R/$N.log 30000
  echo "== $N solver $(grep -E 'CONVERGED [a-z]' $R/$N.log | grep -o 'E=[-0-9.]*' | head -1) $(grep -ci error $R/$N.log) err"
  (cd /Users/cgjoh/Development/H2O && python3 -W ignore tools/check_energy.py $N $D 2>&1 | grep -E "E indep" | cut -c1-100); python3 $R/surf.py $N $D $B
done; echo FXREDONE
