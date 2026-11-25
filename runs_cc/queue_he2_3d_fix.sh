#!/bin/bash
# He2 in 3D (molecule.js, h2_beta.html mode=he2, 2026-11-24): free wall vs evanescent Robin beta = sqrt(-2 eps_He)
# = sqrt(2*0.918) = 1.355 on walls between different atoms (?betainter=1). Reference: He atom (mode=h&Z=2), same mesh.
# Axisym (h0.035): free -0.55/-0.18/-0.05, beta +0.056/+0.011/+0.0025 at R=2/3/4; RHF +0.121/+0.014/+0.001.
R=/Users/cgjoh/Development/H2O/runs_cc
C="h=0.05&box=10&nmax=200&inner=1&win=2000&conv=0.00003&blog=120&reset&curv=0"
run() { $R/run.sh "$1&$C&dump=$2" $R/$2.log 20000
  echo "== $2 $(grep -E 'CONVERGED [a-z]' $R/$2.log | grep -o 'E=[-0-9.]*' | head -1) err $(grep -ci error $R/$2.log)"; }
# atoms already done (beta cannot act without a wall)

for RR in 2.0 3.0 4.0; do

  run "mode=he2&R=$RR&beta=1.355&betainter=1" he2_3d_beta_R$RR
done
echo HE23DDONE
