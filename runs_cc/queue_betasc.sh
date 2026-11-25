#!/bin/bash
# Self-consistent Pauli-wall beta (?betasc=1000: beta_n = sqrt(-2 eps_n) every 1000 steps), after the water-dimer queue.
# 1) He2 R3 (betainter=1): eps of the He pair (HF -0.918) and E vs betad=1.355 (he2_3d_betad_R3.0.log).
# 2) CH4 1.90/2.054/2.25, walls between the four 2hom bond pairs as in the beta=0.2 runs: F-zero bond length.
R=/Users/cgjoh/Development/H2O/runs_cc
while ! grep -q WDDONE /private/tmp/claude-501/-Users-cgjoh-Development-H2O/ec531292-9ff8-4ec4-877b-d412946b52fd/tasks/bojpn1t63.output 2>/dev/null; do sleep 60; done
$R/run.sh "mode=he2&R=3.0&beta=1.355&betainter=1&betasc=1000&h=0.05&box=10&nmax=200&inner=1&win=2000&conv=0.00003&blog=120&reset&curv=0" $R/he2_3d_betasc_R3.0.log 20000
echo "== he2 $(grep -E 'CONVERGED [a-z]' $R/he2_3d_betasc_R3.0.log | grep -o 'E=[-0-9.]*' | head -1)  $(grep -o 'BETASC.*' $R/he2_3d_betasc_R3.0.log | tail -1)"
CH4="mode=ch4&rcX=1.0&el=C&Hbare=1&bpair=1&box=12"
for RR in 1.90 2.054 2.25; do N=sc_ch4_R$RR
  $R/run.sh "$CH4&R=$RR&beta=0.2&betaiso=1&betasc=1000&h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset" $R/$N.log 30000
  echo "== $N $(grep -E 'CONVERGED [a-z]' $R/$N.log | grep -o 'E=[-0-9.]*' | head -1)  $(grep -o 'BETASC.*' $R/$N.log | tail -1)"
  python3 $R/forces.py $R/$N.log 2>&1 | tail -2
done; echo SCDONE
