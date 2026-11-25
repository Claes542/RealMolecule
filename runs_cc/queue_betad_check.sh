#!/bin/bash
# Regression check for per-domain beta (2026-11-25): He2 R=3 with ?betad=1.355,1.355&betanowall=1 must reproduce
# the global-beta run he2_3d_beta_R3.0 (E=-5.49523): same beta on both domains, wall pinned by mirror symmetry.
R=/Users/cgjoh/Development/H2O/runs_cc
C="h=0.05&box=10&nmax=200&inner=1&win=2000&conv=0.00003&blog=120&reset&curv=0"
$R/run.sh "mode=he2&R=3.0&beta=1.355&betainter=1&betad=1.355,1.355&betanowall=1&$C" $R/he2_3d_betad_R3.0.log 20000
echo "== $(grep -E 'CONVERGED [a-z]' $R/he2_3d_betad_R3.0.log | grep -o 'E=[-0-9.]*' | head -1) err $(grep -ci error $R/he2_3d_betad_R3.0.log)"
echo BETADDONE
