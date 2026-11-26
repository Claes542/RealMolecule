#!/bin/bash
# Pauli walls ALSO between one atom's own pairs (2026-11-25): beta_n = sqrt(-2 eps_n), self-consistent (betasc=1000),
# on every wall (no betainter). Test: does it give a SHAPE? With all walls free, H2O has no bend restoring force
# (F_open +0.0063/+0.0209/+0.0230 at 99.5/104.5/109.5, fq_h2o_*), CH4 flattens. Water dimer paused (resumes: run.sh skips DUMPED runs).
R=/Users/cgjoh/Development/H2O/runs_cc
C="h=0.1&inner=1&win=2000&conv=0.00003&blog=120&nmax=300&reset"
B="beta=1&betaiso=1&betasc=1000"
run() { $R/run.sh "$1&$B&$C&dump=$2" $R/$2.log 30000; echo "== $2 $(grep -o 'BETASC.*' $R/$2.log | tail -1 | cut -c1-170)"; python3 $R/forces.py $R/$2.log 2>&1 | tail -2; }
H="mode=h2o&c2v=1&half=1&rcX=0.71&el=O&bpair=1&Hbare=1&box=12&R=1.809"
for A in 104.5 99.5 109.5; do run "$H&ang=$A" wb_h2o_a$A; done
CH4="mode=ch4&rcX=1.0&el=C&Hbare=1&bpair=1&box=12"
for RR in 2.054 1.90 2.25; do run "$CH4&R=$RR" wb_ch4_R$RR; done
run "$CH4&R=2.054&planar=1" wb_ch4sq_R2.054
echo WBDONE
