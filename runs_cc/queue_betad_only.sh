#!/bin/bash
# Per-domain beta WITHOUT betanowall (wall force on): must reproduce he2_3d_beta_R3.0 E=-5.49523. Runs after the beta=0.2 queue.
while ! grep -q FXDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bjkvqfvbp.output 2>/dev/null; do sleep 60; done
R=/Users/cgjoh/Development/H2O/runs_cc
C="h=0.05&box=10&nmax=200&inner=1&win=2000&conv=0.00003&blog=120&reset&curv=0"
$R/run.sh "mode=he2&R=3.0&beta=1.355&betainter=1&betad=1.355,1.355&$C" $R/he2_3d_betad_R3.0.log 20000
echo "== $(grep -E 'CONVERGED [a-z]' $R/he2_3d_betad_R3.0.log | grep -o 'E=[-0-9.]*' | head -1)"; echo BDODONE
